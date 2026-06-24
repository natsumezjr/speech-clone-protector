# 1. 总输入：参考语音和目标文本

LLM-based 语音克隆仍然有两个输入。

第一个输入是参考语音：
$$
x_{\text{ref}}[n]
$$
它是一维离散波形序列。它提供的是：
$$
\text{谁在说、怎么说、声音环境是什么}
$$
第二个输入是目标文本：
$$
Y=(y_1,y_2,\dots,y_N)
$$
它提供的是：
$$
\text{要说什么}
$$
最终输出是新的语音波形：
$$
\hat{x}[n]
$$
目标是：
$$
(x_{\text{ref}}[n],Y)\rightarrow \hat{x}[n]
$$
也就是：用参考语音中的声音风格去读目标文本。

------

# 2. 参考语音分支：从波形到语音 token

这部分是 LLM-based 语音克隆和 Tacotron2 最大的不同。Tacotron2 更偏向于从参考语音里提取一个固定说话人向量 $e$。LLM-based TTS 通常会把参考语音变成一串离散语音 token，让语言模型像处理文本 token 一样处理语音。

------

## 2.1 原始参考语音波形

输入：
$$
x_{\text{ref}}[n]
$$
其中 $n$ 是采样点编号。

如果采样率是 $f_s=24000$，3 秒语音就是：
$$
3\times 24000=72000
$$
个采样点。

所以原始参考语音是：
$$
x_{\text{ref}}[0],x_{\text{ref}}[1],\dots,x_{\text{ref}}[L-1]
$$
输出仍然是：
$$
x_{\text{ref}}[n]
$$
这一步只是明确输入对象。

------

## 2.2 波形预处理

输入：
$$
x_{\text{ref}}[n]
$$
系统通常先做采样率统一、音量归一化、静音裁剪等：
$$
x_{\text{ref}}[n]\rightarrow x'_{\text{ref}}[n]
$$
输出：
$$
x'_{\text{ref}}[n]
$$
这一步和你 Tacotron2 实验中的预处理类似，只是后面的编码方式不同。

------

## 2.3 语音编码器：波形变成连续隐表示

输入：
$$
x'_{\text{ref}}[n]
$$
语音 tokenizer 的 encoder 会把一维波形压缩成一串连续向量：
$$
z_{\text{ref}}[t]=E_{\text{aud}}(x'_{\text{ref}})[t]
$$
其中：
$$
z_{\text{ref}}[t]\in \mathbb{R}^{d}
$$
$t$ 不再是原始采样点编号，而是压缩后的帧编号。

例如原始波形可能有 72000 个采样点，但经过 encoder 后可能变成 150 个左右的帧级表示。具体帧率取决于 tokenizer 或 codec 设计。

输出：
$$
Z_{\text{ref}}=(z_{\text{ref}}[1],z_{\text{ref}}[2],\dots,z_{\text{ref}}[T_{\text{ref}}])
$$
这一步的本质是：
$$
\text{高频率的一维波形}
\rightarrow
\text{低帧率的连续语音表示}
$$
T-SemAttack 论文中也把 speech tokenizer 描述为连续语音信号和离散表示之间的关键接口，通常包含 encoder 和 quantizer：encoder 先把语音信号编码成表示，quantizer 再离散化成 token。

刚刚这个对象对应：

- **Audio Encoder — 音频编码器**
   把波形压缩成连续语音表示。

------

## 2.4 量化器：连续表示变成离散 token

现在输入是：
$$
Z_{\text{ref}}=(z_{\text{ref}}[1],\dots,z_{\text{ref}}[T_{\text{ref}}])
$$
LLM 直接处理连续向量比较困难，所以系统会把每个连续向量映射到离散 codebook 里的某个编号。

可以先用最简单的一层 codebook 理解。

假设有一个 codebook：
$$
C=\{c_1,c_2,\dots,c_K\}
$$
每个 $c_k$ 是一个可学习向量。对于某一帧 $z_{\text{ref}}[t]$，量化器寻找最近的 code：
$$
q_t=\arg\min_k \|z_{\text{ref}}[t]-c_k\|
$$
输出的 $q_t$ 是一个整数编号，例如：
$$
q_t=317
$$
而不是连续向量。

对所有帧做量化：
$$
Z_{\text{ref}}
\rightarrow
Q_{\text{ref}}=(q_1,q_2,\dots,q_{T_{\text{ref}}})
$$
输出：
$$
Q_{\text{ref}}
$$
这就是参考语音的 token 序列。

刚刚这个过程对应：

- **VQ — Vector Quantization — 向量量化**
   把连续向量映射到有限 codebook 中的离散编号。
- **Speech Token / Audio Token — 语音 token / 音频 token**
   语音被离散化之后得到的符号序列。

CosyVoice 论文也强调，LLM-based TTS 范式中，语音信号会被离散化成 token 序列，然后由 LLM 与文本 prompt 一起建模，再由 token-based vocoder 恢复为波形。

------

## 2.5 多层量化：为什么会有多组 token？

很多现代语音 tokenizer 不只输出一层 token，而是输出多层 codebook 的 token。比如 RVQ。

一层 codebook 可能无法精细表示语音。于是系统先用第一层 codebook 表示主要信息，再用第二层表示残差，再用第三层继续表示残差。

可以写成：
$$
z_t \approx c^{(1)}_{q^{(1)}_t}
+
c^{(2)}_{q^{(2)}_t}
+
\cdots
+
c^{(R)}_{q^{(R)}_t}
$$
其中 $R$ 是 codebook 层数。

于是每个时间帧 $t$ 不只有一个 token，而是一组 token：
$$
(q^{(1)}_t,q^{(2)}_t,\dots,q^{(R)}_t)
$$
整段参考语音得到：
$$
Q_{\text{ref}}
=
\{q^{(r)}_t\}_{t=1,r=1}^{T_{\text{ref}},R}
$$
T-SemAttack / ROSETok 材料里也提到，tokenizer 可以包含 RVQ quantizer，例如 S3 Tokenizer / ROSETok 中都涉及多层 RVQ 量化结构。

刚刚这个机制对应：

- **RVQ — Residual Vector Quantization — 残差向量量化**
   用多层 codebook 逐步表示连续向量，后一层补前一层没表示好的残差。

到这里，参考语音分支完成：
$$
x_{\text{ref}}[n]
\rightarrow
x'_{\text{ref}}[n]
\rightarrow
Z_{\text{ref}}
\rightarrow
Q_{\text{ref}}
$$
其中：
$$
Q_{\text{ref}}
$$
就是后面 LLM 看到的“参考声音提示”。

------

# 3. 目标文本分支：从文字到文本 token

现在处理目标文本。

------

## 3.1 原始文本

输入：
$$
Y=(y_1,y_2,\dots,y_N)
$$
例如一句目标文本。

输出仍是：
$$
Y
$$

------

## 3.2 文本规范化

输入：
$$
Y
$$
文本规范化得到：
$$
Y_{\text{norm}}
$$
例如数字、标点、大小写、缩写、多音字、语种标记等处理。

输出：
$$
Y_{\text{norm}}
$$

------

## 3.3 文本 tokenizer

输入：
$$
Y_{\text{norm}}
$$
文本 tokenizer 把它转为离散 token：
$$
P_{\text{text}}=(p_1,p_2,\dots,p_N)
$$
这里 $p_i$ 可以是字符、音素、BPE token、拼音 token 或其他文本单位。

输出：
$$
P_{\text{text}}
$$

------

## 3.4 文本嵌入

输入：
$$
P_{\text{text}}=(p_1,\dots,p_N)
$$
每个文本 token 通过 embedding 表变成向量：
$$
u_i=\text{Embed}_{text}(p_i)
$$
得到：
$$
U_{\text{text}}=(u_1,u_2,\dots,u_N)
$$
输出：
$$
U_{\text{text}}
$$
这一步和普通 LLM 处理文本类似。

------

# 4. 把参考语音 token 和目标文本 token 组织成 LLM 输入

现在我们有两条上游输出。

参考语音分支输出：
$$
Q_{\text{ref}}
$$
目标文本分支输出：
$$
P_{\text{text}}
\quad \text{或} \quad
U_{\text{text}}
$$
接下来要把它们组织成一个条件生成任务。

LLM-based TTS 的基本思想是：
$$
\text{给定参考语音 token 和目标文本 token，预测目标语音 token}
$$
类似普通 LLM：
$$
\text{给定前文 token，预测后文 token}
$$
但这里的“前文”包含语音和文本两种 token。

------

## 4.1 参考语音 token 作为 acoustic prompt

输入：
$$
Q_{\text{ref}}
$$
它可以被看成 acoustic prompt，也就是“声音提示”。

它告诉模型：
$$
\text{说话人的音色、韵律、语速、情绪、录音环境}
$$
在 VALL-E 中，3 秒 enrolled recording 的离散 codec codes 可以作为 acoustic prompt，使模型合成未见说话人的个性化语音。

输出仍然是：
$$
Q_{\text{ref}}
$$
但是语义上它已经变成了 prompt。

刚刚这个对象对应：

- **Acoustic Prompt — 声学提示**
   作为条件输入的一小段参考语音 token，用来提示模型“像谁、用什么风格说”。

------

## 4.2 目标文本 token 作为 content prompt

输入：
$$
P_{\text{text}}
$$
它告诉模型：
$$
\text{要说什么}
$$
输出仍然是：
$$
P_{\text{text}}
$$

------

## 4.3 拼接成条件序列

系统会把这些 token 按一定格式组织起来。例如抽象写成：
$$
S_{\text{in}}
=
[\text{BOS},
P_{\text{text}},
\text{SEP},
Q_{\text{ref}},
\text{GEN}]
$$
这里：

- $P_{\text{text}}$：目标文本 token；
- $Q_{\text{ref}}$：参考语音 token；
- $\text{GEN}$：提示模型开始生成目标语音 token。

不同系统的顺序和特殊符号不一定一样，有的会把参考语音和其转写文本一起放进去，有的会分 semantic token / acoustic token 多阶段处理。但理论上都是：
$$
(P_{\text{text}},Q_{\text{ref}})
\rightarrow
S_{\text{in}}
$$
输出：
$$
S_{\text{in}}
$$
这一步的意义是把“文本条件”和“声音条件”放到同一个可建模序列里。

------

# 5. LLM 生成新的语音 token

这是 LLM-based 语音克隆的核心。

输入：
$$
S_{\text{in}}
$$
模型要生成：
$$
Q_{\text{gen}}
=
(\hat{q}_1,\hat{q}_2,\dots,\hat{q}_{T_{\text{gen}}})
$$
也就是目标语音的 token 序列。

------

## 5.1 自回归生成的理论形式

LLM 每一步预测下一个语音 token：
$$
P(Q_{\text{gen}}\mid P_{\text{text}},Q_{\text{ref}})
=
\prod_{t=1}^{T_{\text{gen}}}
P(\hat{q}_t
\mid
P_{\text{text}},Q_{\text{ref}},\hat{q}_{<t})
$$
意思是：

第 $t$ 个新语音 token 由三部分决定：
$$
\text{目标文本}
+
\text{参考声音}
+
\text{已经生成的新语音 token}
$$
每一步输入都来自前面：
$$
(P_{\text{text}},Q_{\text{ref}})
\rightarrow
\hat{q}_1
$$
直到生成结束 token。

输出：
$$
Q_{\text{gen}}
$$
VALL-E 把 TTS 建模为条件语言建模任务，而不是传统连续信号回归；这正是这里的核心。

------

## 5.2 这里的 token 到底表示什么？

不同模型设计不完全一样，但可以分成两类。

第一类是 **语义 token**：
$$
Q^{sem}
$$
更偏向表达：
$$
\text{说了什么、音素内容、语义/发音骨架}
$$
第二类是 **声学 token**：
$$
Q^{aco}
$$
更偏向表达：
$$
\text{音色、细节、环境、音质、韵律}
$$
CosyVoice 使用 supervised semantic tokens，并提出 LLM 负责 text-to-token generation，再用 conditional flow matching 做 token-to-speech synthesis；它的目标就是提升零样本语音克隆中的内容一致性和说话人相似度。

所以更完整地写，LLM 可能先生成中层语义 token：
$$
(P_{\text{text}},Q_{\text{ref}})
\rightarrow
Q^{sem}_{\text{gen}}
$$
然后另一个模块再把语义 token 和参考声学条件转成低层声学特征：
$$
(Q^{sem}_{\text{gen}},Q_{\text{ref}})
\rightarrow
A_{\text{gen}}
$$
也可能像 neural codec language model 那样直接生成 codec token：
$$
(P_{\text{text}},Q_{\text{ref}})
\rightarrow
Q^{codec}_{\text{gen}}
$$
这两种设计细节不同，但共同点是：
$$
\text{LLM 不直接输出波形采样点}
$$
而是输出某种离散语音表示。

------

# 6. 从新语音 token 到连续声学表示

现在输入是 LLM 输出：
$$
Q_{\text{gen}}
$$
但这还不是波形。它只是离散 token 序列。

接下来需要把 token 还原成连续表示。

------

## 6.1 token 查表恢复为 codebook 向量

如果 $Q_{\text{gen}}$ 是 codec token，那么每个 token $\hat{q}_t$ 对应 codebook 里的一个向量：
$$
\hat{q}_t
\rightarrow
c_{\hat{q}_t}
$$
如果是多层 RVQ，则每个时间帧是一组 token：
$$
(\hat{q}^{(1)}_t,\hat{q}^{(2)}_t,\dots,\hat{q}^{(R)}_t)
$$
恢复连续 latent：
$$
\hat{z}_t
=
c^{(1)}_{\hat{q}^{(1)}_t}
+
c^{(2)}_{\hat{q}^{(2)}_t}
+
\cdots
+
c^{(R)}_{\hat{q}^{(R)}_t}
$$
对所有时间帧恢复：
$$
Q_{\text{gen}}
\rightarrow
\hat{Z}_{\text{gen}}
=
(\hat{z}_1,\hat{z}_2,\dots,\hat{z}_{T_{\text{gen}}})
$$
输出：
$$
\hat{Z}_{\text{gen}}
$$

------

## 6.2 解码器把 latent 变成声学特征或波形前表示

输入：
$$
\hat{Z}_{\text{gen}}
$$
codec decoder 或声学解码器输出：
$$
\hat{A}_{\text{gen}}
$$
这里 $\hat{A}_{\text{gen}}$ 可能是：
$$
\hat{M}
$$
也就是梅尔谱图；

也可能是更底层的 acoustic representation。

抽象写：
$$
\hat{Z}_{\text{gen}}
\rightarrow
\hat{A}_{\text{gen}}
$$
输出：
$$
\hat{A}_{\text{gen}}
$$
如果是 CosyVoice 类结构，LLM 生成的 supervised semantic tokens 会进一步由 conditional flow matching 模型解码到低层声学特征，再由 vocoder 生成波形。

刚刚这个模块对应：

- **Token-to-Speech Decoder — token 到语音解码器**
   把离散语音 token 还原成连续声学表示或可生成波形的中间表示。

------

# 7. 从声学表示恢复最终波形

现在输入是：
$$
\hat{A}_{\text{gen}}
$$
如果它是梅尔谱图：
$$
\hat{A}_{\text{gen}}=\hat{M}
$$
则声码器生成波形：
$$
\hat{M}\rightarrow \hat{x}[n]
$$
如果它是 codec latent，也可能由 codec decoder 直接生成波形：
$$
\hat{Z}_{\text{gen}}\rightarrow \hat{x}[n]
$$
无论是哪种，最后都得到：
$$
\hat{x}[n]
$$
这就是新的克隆语音序列。

------

# 8. 完整链路，不跳步骤地串起来

现在把 LLM-based 语音克隆完整写一遍。

参考语音分支：
$$
x_{\text{ref}}[n]
\rightarrow
x'_{\text{ref}}[n]
$$
目标文本分支：
$$
Y
\rightarrow
Y_{\text{norm}}
$$
条件序列构造：
$$
(P_{\text{text}},Q_{\text{ref}})
\rightarrow
S_{\text{in}}
$$
LLM 生成：
$$
S_{\text{in}}
\rightarrow
\hat{q}_1
$$
token 还原：
$$
Q_{\text{gen}}
\rightarrow
\hat{Z}_{\text{gen}}
$$
声学解码：
$$
\hat{Z}_{\text{gen}}
\rightarrow
\hat{A}_{\text{gen}}
$$
波形恢复：
$$
\hat{A}_{\text{gen}}
\rightarrow
\hat{x}[n]
$$
最终：
$$
(x_{\text{ref}}[n],Y)
\rightarrow
\hat{x}[n]
$$

------

# 9. 它和 Tacotron2 流程的关键区别

你原来实验中的核心链路是：
$$
x_{\text{ref}}[n]
\rightarrow
M_{\text{ref}}
\rightarrow
e
$$
也就是：
$$
\text{参考语音主要被压缩成一个说话人向量}
$$
而 LLM-based TTS 更像：
$$
x_{\text{ref}}[n]
\rightarrow
Q_{\text{ref}}
$$
也就是：
$$
\text{参考语音被保留为一串语音 token，作为上下文提示}
$$
区别可以这样理解：

Tacotron2 路线更像：
$$
\text{先提取一个“这个人是谁”的向量，再让合成器照着这个人说}
$$
LLM-based 路线更像：
$$
\text{把一小段这个人的声音也变成 token，连同目标文本一起喂给语言模型，让模型续写一段符合这个声音风格的新语音 token}
$$
这就是为什么它会有更强的 in-context learning 味道。

刚刚这个机制对应：

- **ICL — In-Context Learning — 上下文学习**
   模型不一定更新参数，而是在推理时根据上下文中的参考语音 token 学会当前要模仿的说话风格。

------

# 10. 攻击场景下，这条链为什么危险？

在攻击场景下，攻击者只需要从公开视频里得到：
$$
x_{\text{ref}}[n]
$$
系统就可以得到：
$$
x_{\text{ref}}[n]\rightarrow Q_{\text{ref}}
$$
而 $Q_{\text{ref}}$ 里不只是“音色”，还可能包含：
$$
\text{说话人身份}
+
\text{语速}
+
\text{情绪}
+
\text{韵律}
+
\text{录音环境}
$$
然后攻击者输入目标文本：
$$
Y_{\text{fake}}\rightarrow P_{\text{text}}
$$
模型生成：
$$
(P_{\text{text}},Q_{\text{ref}})
\rightarrow
Q_{\text{fake}}
\rightarrow
\hat{x}_{\text{fake}}[n]
$$
这就是现代 LLM-based 语音克隆的风险核心：

**参考语音不再只是提取一个 speaker embedding，而是作为一段丰富的声学上下文，被模型拿来条件生成。**

所以后面 E2E-VGuard 和 T-SemAttack 的保护点才会不同。

E2E-VGuard 会问：
$$
x_{\text{ref}}[n]\rightarrow Q_{\text{ref}}
$$
这一步里，能不能让机器得到的说话人/音色信息偏掉？

同时它还会问：
$$
x_i[n]\rightarrow ASR(x_i[n])
$$
这一步里，能不能让自动转写文本错掉？

T-SemAttack 会更进一步问：
$$
x[n]\rightarrow Z\rightarrow Q
$$
这一步里，能不能让 speech tokenizer 从一开始就把语义编码错？

T-SemAttack 论文明确指出，speech tokenizer 是现代 ASR 和大音频语言模型中的前端桥梁；一旦 tokenizer 的编码输出被轻微扰动大幅改变，后面的 LLM 很难恢复原始语义。

------

# 11. 这一部分你应该先记住的最小链路

LLM-based 语音克隆最小链路是：
$$
\text{参考语音波形}
\rightarrow
\text{连续语音表示}
\rightarrow
\text{离散参考语音 token}
$$
压成一行就是：
$$
x_{\text{ref}}[n]
\rightarrow
Z_{\text{ref}}
\rightarrow
Q_{\text{ref}}
$$
这就是后面理解两篇论文的基础。