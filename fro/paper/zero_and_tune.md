# 阶段 2：攻击场景下的语音克隆流程

你前面已经理解了正常零样本克隆：
$$
x_{\text{ref}}[n] + Y_{\text{target}}
\rightarrow
\hat{x}[n]
$$
其中：
$$
x_{\text{ref}}[n]
$$
是参考语音，提供“像谁说”。
$$
Y_{\text{target}}
$$
是目标文本，提供“说什么”。

攻击场景的区别是：**参考语音不是授权提供的，而是从公开环境中收集的。**

所以攻击场景的起点不是：
$$
\text{用户主动提供干净参考语音}
$$
而是：
$$
\text{公开音视频中的目标人声音}
$$

------

## 1. 攻击者拿到的最初对象：公开视频或公开音频

攻击者最开始拿到的通常不是干净语音文件，而是一个公开媒体文件：
$$
V_{\text{public}}
$$
例如视频、直播回放、播客、课程录音、社交平台语音。

如果是视频，里面包含图像轨和音频轨：
$$
V_{\text{public}}
=
(\text{video stream}, \text{audio stream})
$$
攻击者真正需要的是音频轨：
$$
V_{\text{public}}
\rightarrow
a_{\text{raw}}[n]
$$
这里的：
$$
a_{\text{raw}}[n]
$$
就是从公开媒体中抽取出来的一维音频序列。

这一步得到的音频通常不是理想参考语音，它可能包含：
$$
\text{目标人语音} + \text{背景音乐} + \text{环境噪声} + \text{其他人说话} + \text{压缩失真}
$$
所以可以写成：
$$
a_{\text{raw}}[n]
=
s_{\text{target}}[n]
+
b[n]
+
u[n]
$$
其中：
$$
s_{\text{target}}[n]
$$
是目标说话人的真实声音。
$$
b[n]
$$
是背景噪声或背景音乐。
$$
u[n]
$$
是其他干扰，例如其他说话人、混响、平台压缩。

攻击者后续所有操作，都是为了从这个混合音频里构造出可用于克隆的数据。

------

## 2. 预处理：从公开音频中整理出目标说话人片段

输入：
$$
a_{\text{raw}}[n]
$$
攻击者希望得到若干段目标说话人的语音：
$$
X_{\text{target}}
=
\{x_1[n],x_2[n],\dots,x_K[n]\}
$$
这一阶段通常包括几个概念上的步骤。

第一，重采样和归一化：
$$
a_{\text{raw}}[n]
\rightarrow
a_1[n]
$$
例如统一采样率、统一响度范围。

第二，语音活动检测：
$$
a_1[n]
\rightarrow
\{r_1,r_2,\dots,r_m\}
$$
这里 $r_i$ 是检测出来的“有语音”的时间片段。它还没有保证一定是目标人在说话，只是排除了纯静音或明显非语音区域。

刚刚这个机制对应：

- **VAD — Voice Activity Detection — 语音活动检测**
   判断一段音频中哪些时间区域有人声。

第三，说话人筛选：
$$
\{r_1,r_2,\dots,r_m\}
\rightarrow
X_{\text{target}}
$$
如果音频里只有目标人，这一步简单；如果有多人，就要尽量筛出目标人的片段。概念上可以理解为：把“同一个人说的话”聚到一起，把别人说的话排除。

这一步的输出是：
$$
X_{\text{target}}
=
\{x_1[n],x_2[n],\dots,x_K[n]\}
$$
也就是一组目标人的语音片段。

到这里还没有发生语音克隆。只是攻击者把公开音频整理成“可用参考语音材料”。

------

# 3. 攻击场景有两条主要克隆路径

从这里开始，攻击者有两种路线。

第一种是**零样本克隆路线**：
$$
\text{少量参考语音} + \text{目标文本}
\rightarrow
\text{直接生成伪造语音}
$$
第二种是**微调式克隆路线**：
$$
\text{一批目标人语音} + \text{对应文本}
\rightarrow
\text{训练/微调目标人声音}
\rightarrow
\text{生成伪造语音}
$$
E2E-VGuard 之所以强调现实场景，是因为很多系统已经把第二条路线做成了端到端流程：用户只上传音频，系统后台自动用 ASR 得到文本，再用文本和音频训练目标声音。论文明确指出，商业 API 往往只接受音频输入，并在后台依赖 ASR；现实中攻击者也常从 YouTube、Bilibili 这类平台收集没有人工标注文本的公开音频，因此 ASR 比人工标注更现实。

下面两条路线分别讲。

------

# 路线 A：零样本攻击式语音克隆

这条路线和你之前做的 Tacotron2 实验最接近。

## A1. 选择一段参考语音

输入：
$$
X_{\text{target}}
=
\{x_1[n],x_2[n],\dots,x_K[n]\}
$$
攻击者选择其中一段或几段作为参考：
$$
x_{\text{ref}}[n]
$$
这里：
$$
x_{\text{ref}}[n] \in X_{\text{target}}
$$
如果系统支持多段参考，也可以把多段拼接或作为多参考输入：
$$
X_{\text{ref}}
=
\{x_{r_1}[n],x_{r_2}[n],\dots\}
$$
输出：
$$
x_{\text{ref}}[n]
\quad \text{或} \quad
X_{\text{ref}}
$$

------

## A2. 从参考语音提取说话人条件

这一步和你实验一致。

输入：
$$
x_{\text{ref}}[n]
$$
先转声学特征：
$$
x_{\text{ref}}[n]
\rightarrow
M_{\text{ref}}(t,f)
$$
再送入说话人编码器：
$$
M_{\text{ref}}
\rightarrow
e_{\text{spk}}
$$
输出：
$$
e_{\text{spk}}
$$
它表示：
$$
\text{目标人声音身份}
$$
你的实验报告里也写到，参考音频经说话人编码器得到说话人嵌入，再与输入文本一起进入 Tacotron2 合成器生成梅尔谱图，最后由 WaveRNN 转为可播放波形。

------

## A3. 攻击者提供伪造文本

输入：
$$
Y_{\text{fake}}
$$
例如攻击者希望目标人“说出”的一段文本。这里不讨论具体滥用内容，只从系统角度看，它就是目标文本。

文本经过规范化和编码：
$$
Y_{\text{fake}}
\rightarrow
T_{\text{fake}}
\rightarrow
V_{\text{fake}}
\rightarrow
H_{\text{fake}}
$$
输出：
$$
H_{\text{fake}}
$$
它表示：
$$
\text{伪造文本的发音上下文表示}
$$

------

## A4. 合成器融合“伪造文本”和“目标人声音”

现在有两个上一步输出：
$$
H_{\text{fake}}
$$
和：
$$
e_{\text{spk}}
$$
合成器逐帧生成目标梅尔谱图：
$$
(H_{\text{fake}},e_{\text{spk}})
\rightarrow
\hat{M}_{\text{fake}}
$$
更细一点：
$$
q_t
\rightarrow
\alpha_t
\rightarrow
c_t
$$
循环得到：
$$
\hat{M}_{\text{fake}}
=
(\hat{m}_1,\hat{m}_2,\dots,\hat{m}_T)
$$
输出：
$$
\hat{M}_{\text{fake}}
$$
它是：
$$
\text{“目标人声音 + 伪造文本”的声学特征}
$$

------

## A5. 声码器恢复波形

输入：
$$
\hat{M}_{\text{fake}}
$$
声码器生成波形：
$$
\hat{M}_{\text{fake}}
\rightarrow
\hat{x}_{\text{fake}}[n]
$$
输出：
$$
\hat{x}_{\text{fake}}[n]
$$
这就是零样本攻击式语音克隆的最终结果。

完整链路是：
$$
V_{\text{public}}
\rightarrow
a_{\text{raw}}[n]
\rightarrow
X_{\text{target}}
\rightarrow
x_{\text{ref}}[n]
\rightarrow
M_{\text{ref}}
\rightarrow
e_{\text{spk}}
$$
这条路线的特点是：**不一定重新训练目标人模型，只用参考语音作为条件。**

E2E-VGuard 也指出，zero-shot TTS 使用 reference audio 作为 prompt 来克隆声音；而 fine-tuning-based TTS 则需要几分钟语音来更好地复制目标说话人。

------

# 路线 B：端到端微调式攻击语音克隆

这条路线是 E2E-VGuard 特别关注的重点。

和零样本路线不同，微调式路线不是只提取一个 $e_{\text{spk}}$，而是要构造一个目标说话人的训练集：
$$
D_{\text{target}}
=
\{(x_i[n], y_i)\}_{i=1}^{K}
$$
其中：
$$
x_i[n]
$$
是第 $i$ 段目标人语音。
$$
y_i
$$
是这段语音对应的文本。

关键问题是：公开音频通常没有 $y_i$。

所以系统引入 ASR。

------

## B1. 攻击者已有目标人语音片段

从前面预处理得到：
$$
X_{\text{target}}
=
\{x_1[n],x_2[n],\dots,x_K[n]\}
$$
这些是目标人的公开语音片段。

每一段 $x_i[n]$ 都是一维波形序列。

------

## B2. ASR 自动听写，得到伪标签文本

输入：
$$
x_i[n]
$$
ASR 系统输出文字：
$$
\tilde{y}_i = ASR(x_i[n])
$$
对所有片段做：
$$
X_{\text{target}}
\rightarrow
\tilde{Y}
=
\{\tilde{y}_1,\tilde{y}_2,\dots,\tilde{y}_K\}
$$
输出：
$$
\tilde{D}_{\text{target}}
=
\{(x_i[n],\tilde{y}_i)\}_{i=1}^{K}
$$
这里的 $\tilde{y}_i$ 不是人工标注，而是 ASR 自动生成的文本。

E2E-VGuard 论文里明确说，真实工业语音合成产品可以通过 API 只上传音频，后台使用补充 ASR 系统自动识别；它也举了 ByteDance API 和 GPT-SoVITS WebUI 的例子，说明系统先用 ASR 得到文本，再用文本和音频进行训练/微调。

刚刚这个自动得到的文本可以叫：

- **Pseudo Transcript — 伪转录文本 / 自动转写文本**
   不是人工标注，而是 ASR 对音频自动生成的文本标签。

------

## B3. 把音频和 ASR 文本变成训练样本

现在每条样本是：
$$
(x_i[n],\tilde{y}_i)
$$
但 TTS 模型训练通常不会直接用原始形式，而是继续变换。

语音侧：
$$
x_i[n]
\rightarrow
M_i(t,f)
$$
文本侧：
$$
\tilde{y}_i
\rightarrow
T_i
\rightarrow
V_i
\rightarrow
H_i
$$
于是训练样本变成：
$$
(H_i,M_i)
$$
如果是多说话人或目标说话人微调，还会带有说话人 ID 或说话人条件：
$$
(H_i,M_i,\text{spk}=u)
$$
输出训练集：
$$
D_{\text{train}}
=
\{(H_i,M_i,\text{spk}=u)\}_{i=1}^{K}
$$
这里 $u$ 表示目标说话人。

------

## B4. 微调 TTS 模型：让模型适配目标人声音

输入：
$$
D_{\text{train}}
$$
已有预训练 TTS 模型参数：
$$
\theta_0
$$
微调目标是让模型根据文本表示 $H_i$ 生成对应目标人的梅尔谱图 $M_i$：
$$
G_{\theta}(H_i,\text{spk}=u)
\approx
M_i
$$
训练损失可以抽象成：
$$
\mathcal{L}_{tts}
=
\sum_i
\left\|
G_{\theta}(H_i,u)-M_i
\right\|
$$
通过优化得到目标人适配后的模型：
$$
\theta_0
\rightarrow
\theta_u
$$
输出：
$$
G_{\theta_u}
$$
它已经学到了目标人的声音特征。

这里的本质不是只提取一个固定向量，而是模型参数、说话人表征或适配模块都可能被目标人数据影响。

刚刚这个过程对应：

- **Fine-tuning — 微调**
   在已有预训练模型基础上，用目标人数据继续训练，使模型适配该目标人的声音。

------

## B5. 输入伪造文本，生成目标人声学特征

攻击者输入目标文本：
$$
Y_{\text{fake}}
$$
文本处理：
$$
Y_{\text{fake}}
\rightarrow
T_{\text{fake}}
\rightarrow
V_{\text{fake}}
\rightarrow
H_{\text{fake}}
$$
送入微调后的模型：
$$
H_{\text{fake}}
\rightarrow
\hat{M}_{\text{fake}}
=
G_{\theta_u}(H_{\text{fake}})
$$
输出：
$$
\hat{M}_{\text{fake}}
$$

------

## B6. 声码器恢复最终伪造语音

输入：
$$
\hat{M}_{\text{fake}}
$$
声码器生成：
$$
\hat{x}_{\text{fake}}[n]
=
Vocoder(\hat{M}_{\text{fake}})
$$
输出：
$$
\hat{x}_{\text{fake}}[n]
$$
完整端到端微调攻击链路是：
$$
V_{\text{public}}
\rightarrow
a_{\text{raw}}[n]
\rightarrow
X_{\text{target}}
=
\{x_i[n]\}_{i=1}^{K}
$$
这就是 E2E-VGuard 重点防御的真实攻击场景。

------

# 4. 这条链路里，ASR 为什么变成关键节点？

在你的 Tacotron2 实验里，文本是你手动输入的。那是推理阶段文本。

但在微调式攻击中，模型需要训练样本：
$$
\text{音频} + \text{对应文本}
$$
也就是：
$$
(x_i[n],y_i)
$$
如果没有 $y_i$，模型不知道这段音频到底对应哪些字、哪些音素、哪些发音位置。

TTS 训练要学习：
$$
\text{文本位置}
\leftrightarrow
\text{声学帧}
$$
如果文本缺失，训练很难直接进行。

ASR 的作用就是自动补出：
$$
x_i[n]\rightarrow \tilde{y}_i
$$
所以攻击者不必人工标注：
$$
\text{公开音频}
\rightarrow
\text{ASR}
\rightarrow
\text{自动训练数据}
$$
这就是端到端攻击更现实的原因。

E2E-VGuard 在“Pronunciation Prevention”部分也明确说，TTS 微调需要文本和音频数据对齐；文本可以来自人工标注或 ASR，而人工方式耗费人力和成本，因此更常见的是使用 ASR。

------

# 5. 如果 ASR 文本错了，会发生什么？

假设真实语音是：
$$
x_i[n] = \text{“今天下午开会”}
$$
正确文本应该是：
$$
y_i = \text{“今天下午开会”}
$$
但 ASR 得到：
$$
\tilde{y}_i = \text{“今天下午开车”}
$$
那么训练样本变成：
$$
(x_i[n],\text{“今天下午开车”})
$$
模型会被迫学习一个错误对应：
$$
\text{“开车”这个文本}
\leftrightarrow
\text{“开会”这个声音}
$$
如果这种错配大量存在，模型的文本—发音对齐会被破坏。

对于 Tacotron2 类模型，就是注意力对齐会乱；对于现代模型，就是文本 token、语音 token、语义 token 之间的对应关系会乱。

所以 E2E-VGuard 后面做“发音保护”的基本思路就是：**让 ASR 先听错，使训练数据从源头错配。**

