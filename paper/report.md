### 2.1 总体设计思路

本作品面向黑箱语音克隆与 LLM-based 端到端语音合成场景，设计一种主动式音频保护方法。从语音克隆防护视角看，当前需要重点考虑三类相关威胁：第一类是零样本语音克隆，攻击者仅利用短时参考音频和目标文本即可生成相似说话人语音；第二类是少样本或微调式语音克隆，攻击者通过收集多条目标语音样本进一步学习目标说话人的稳定声音特征；第三类是 LLM-based 端到端语音合成，系统通常将连续语音编码为离散语音 token，并利用语言模型完成 text-to-token 或 token-to-token 建模，再由声学生成模块还原为语音波形[1-4]。因此，本文将语音克隆防护场景概括为零样本克隆、微调式克隆和 LLM-based 端到端克隆三个层次：前两者体现攻击者使用目标语音的方式，后者体现当前语音合成系统的模型架构变化。

上述三类场景虽然流程不同，但都依赖同一个基础事实：攻击者必须从参考音频中提取机器可用的说话人信息和语音条件信息。零样本克隆需要从参考音频中提取音色、说话人身份、韵律、情感和声学环境等条件；微调式克隆需要通过多条语音样本学习目标说话人的稳定发音和风格模式；LLM-based 语音合成则进一步把语音条件编码为 speech token、semantic token 或 acoustic token，使其能够与文本 token 一起进入后续建模流程[1-4]。因此，只要攻击者的目标是复刻原始说话人的声音，防御方就可以从“破坏机器对参考音频的编码结果”这一角度设计保护方法，而不必假设能够访问攻击者使用的具体 TTS 模型。

本作品采用黑箱防御设定。现实攻击中，攻击者可能使用商业语音合成 API、闭源语音克隆服务或未知开源模型，防御方通常无法获得目标系统的模型参数、梯度和完整内部结构。已有端到端语音合成研究也指出，商业语音克隆 API 可以只接收音频输入，并在服务器后端完成文本识别、说话人训练和语音合成流程[1]。因此，本作品不直接依赖目标 TTS 模型，而是在本地构建可微替代编码器集合，用于近似不同语音系统中较普遍存在的关键编码环节。该设定更符合真实语音隐私保护场景，也要求微小扰动具有跨模型迁移能力。

在 LLM-based 端到端语音克隆场景中，系统流程可以进一步拆成两条关键链路。第一条是 ASR 侧自动标注链路。当攻击者获得的是公开音频而非人工标注的文本—音频对时，通常需要先利用 ASR 系统将语音转写为文本标签，再将该文本标签用于后续训练或微调[1,5-6]。该链路主要回答“这段语音说了什么”，其核心在于 ASR 侧语义编码器对语音内容的提取。第二条是 TTS 侧音频条件链路。该链路不直接生成文本，而是从参考音频中提取说话人身份、音色、风格、韵律、节奏和声学条件等信息，主要回答“是谁在说、怎么说”[1-4]。需要说明的是，ASR 侧语义编码器与 TTS 侧音频条件编码器在底层声学前端、网络结构或预训练语音表征上可能存在相似性，甚至可能使用同类编码模型作为本地替代模型；但二者在宏观系统中的任务不同，前者服务于语义识别，后者服务于语音合成条件建模。因此，本文分别称其为 ASR 侧语义编码器和 TTS 侧音频条件编码器。

基于上述流程分析，本作品的核心设计是同时对两类编码链路施加微小扰动。在 ASR 侧，本作品不再直接使用面向最终文本输出的固定目标损失。原因在于，ASR 的最终结果表现为离散文本序列，若直接围绕最终文本构造 loss，通常会受到具体 ASR 模型结构、解码方式、输出概率形式以及文本长度等因素影响，构造稳定且可迁移的优化目标较为困难[1,5-6]。因此，本作品将语义扰动目标前移到 ASR 内部的连续语义编码层，不要求 ASR 输出某个固定句子，而是使受保护语音在语义编码空间中偏离原始语音表示。根据本项目实验观察，语义编码层的微小扰动会在后续 token 化和识别过程中被进一步放大，从而间接提高自动转写错误率，破坏端到端训练阶段的文本—音频对应关系。

在 TTS 侧，本作品保留并强化音频条件扰动目标。语音克隆模型对参考音频的依赖并不局限于文本内容，而是高度依赖音色、说话人身份、韵律、节奏、风格和声学环境等条件表示[1-4]。若这些条件表示发生偏移，即使人耳仍能基本理解原始语音，TTS 模型在合成阶段也可能得到错误的说话人条件或不稳定的说话方式表示，从而降低合成语音与原始说话人的相似度。与此同时，本作品引入感知约束，使微小扰动尽量分布在人耳相对不敏感的时频区域，在保持语音正常听感可用性的同时，提高其对机器编码器的影响[1]。整体而言，本作品生成的是输入相关的主动式保护样本：从生成机制上看，它具有对抗样本式特征；从端到端微调场景看，它会污染自动文本标签和音频条件学习过程，因此具有类不可学习样本的数据保护效果[1]。



### 2.1.2 保护算法设计思路

​	基于 2.1.1 对黑箱语音克隆场景的分析，本文将原始语音记为 (x)，将受保护语音记为 (x'=x+\delta)，其中 (\delta) 表示施加在原始语音上的微小扰动。本文的目标不是破坏语音本身的正常可听性，而是在保持语音正常听感可用的前提下，使机器模型在关键编码环节获得偏移后的表示。由此，本文将保护算法的总体优化目标设计为：
$$
L=
\lambda_{sem}L_{sem}
+
\lambda_{fea}L_{fea}
+
\lambda_{psy}L_{psy}
+
\lambda_{2}|\delta|_2
$$
其中，(L_{sem}) 面向 ASR 侧语义编码链路，(L_{fea}) 面向 TTS 侧特征编码器 / tokenizer，(L_{psy}) 用于约束微小扰动的人耳感知影响，(|\delta|_2) 用于限制微小扰动的整体能量。该公式并不是经验性地拼接多个损失项，而是由 LLM-based 语音克隆的训练流程、推理流程、离散 token 化机制以及人耳感知机制共同推出。

首先，从原始波形进入模型的角度看，语音信号 (x) 本质上是随时间变化的一维采样序列。虽然部分模型可以直接从波形中学习表示，但在 ASR、TTS 和声学建模中，系统通常会先通过分帧、加窗、短时频谱分析、Mel 滤波或 log-Mel 变换等方式，将原始波形转换为更适合建模的时频声学表示[5,7]。这一过程可以抽象为：
$$
x'
\rightarrow
a(t,f)
$$
其中，(a(t,f)) 表示受保护语音在时间 (t) 和频率 (f) 上的声学表示，可以理解为 Mel 频谱图、log-Mel 频谱图或其他声学前端表示。后续 ASR 侧和 TTS 侧并不一定直接使用完全相同的声学前端，而是在 (a(t,f)) 的基础上形成各自任务相关的输入表示：
$$
\Phi_{asr}(x')=\Phi_{asr}(a(t,f))
$$

$$
\Phi_{fea}(x')=\Phi_{fea}(a(t,f))
$$

其中，(\Phi_{asr}) 表示 ASR 侧声学前端或特征变换，主要服务于语义识别；(\Phi_{fea}) 表示 TTS 侧特征前端或 tokenizer 前端，主要服务于说话人、音色、韵律、风格和声学条件建模。也就是说，(\Phi_{asr}(x')) 和 (\Phi_{fea}(x')) 都来源于原始语音的时频声学结构，但由于后续任务不同，它们会进入不同的编码链路。

在端到端克隆训练或微调阶段，攻击者通常需要构造目标说话人的文本—音频训练对。当攻击者只获得公开音频而没有人工标注文本时，系统需要先通过 ASR 自动生成文本标签，再将文本标签与音频特征共同用于后续微调或适配[1,5-6]。这一阶段可以抽象为两条并行链路。第一条是 ASR 侧自动标注链路：
$$
x'
\rightarrow
a(t,f)
\rightarrow
\Phi_{asr}(x')
\rightarrow
E_{sem}^{ASR}(\Phi_{asr}(x'))
\rightarrow
h_{sem}'
\rightarrow
D_{ASR}(h_{sem}')
\rightarrow
\hat{y}'
\rightarrow
T_{text}(\hat{y}')
\rightarrow
z_{text}'
$$
其中，(E_{sem}^{ASR}) 表示 ASR 侧语义编码器，(h_{sem}') 表示 ASR 编码器输出的连续语义表示，(D_{ASR}) 表示 ASR 解码器，(\hat{y}') 表示由受保护语音自动转写得到的文本标签，(T_{text}) 表示文本 tokenizer，(z_{text}') 表示送入后续 LLM-based TTS 训练流程的文本 token 或文本表示。该链路回答的是“这段语音说了什么”。因此，如果 (h_{sem}') 相对于原始语音对应的语义表示 (h_{sem}) 发生偏移，后续自动文本标签就可能出现错误，从而污染端到端训练阶段的文本—音频对应关系。

第二条是 TTS 侧特征编码链路：
$$
x'
\rightarrow
a(t,f)
\rightarrow
\Phi_{fea}(x')
\rightarrow
E_{fea}^{TTS}(\Phi_{fea}(x'))
\rightarrow
h_{fea}'
\rightarrow
Q_{fea}(h_{fea}')\ \text{或}\ P_{fea}(h_{fea}')
\rightarrow
z_{fea}'
$$
其中，(E_{fea}^{TTS}) 表示 TTS 侧特征编码器 / tokenizer。这里的 (fea) 取自 feature，表示该链路服务于语音合成中的特征条件建模，而不是最终文本识别。(h_{fea}') 表示特征侧编码器输出的连续隐藏表示；(Q_{fea}) 表示离散量化器，适用于使用 speech tokenizer、semantic token、acoustic token 或 codec token 的系统；(P_{fea}) 表示连续投影层或 speaker / style embedding 层，适用于使用连续说话人条件、风格条件或声学条件的系统。最终得到的 (z_{fea}') 表示受保护语音中的说话人身份、音色、韵律、节奏、风格和声学条件等特征表示。该链路回答的是“是谁在说、怎么说”。

当两条链路完成后，端到端克隆训练或微调过程可以进一步表示为：
$$
(z_{text}',z_{fea}')
\rightarrow
\text{LLM token modeling / alignment}
\rightarrow
\text{fine-tuning / adaptation}
\rightarrow
\theta'
$$
其中，(\theta') 表示攻击者经过微调或适配后得到的目标说话人模型参数。该流程说明，在训练或微调阶段，ASR 侧语义编码链路决定自动文本标签是否可靠，TTS 侧特征编码链路决定说话人条件和说话方式是否稳定。因此，如果受保护语音能够同时影响 (h_{sem}') 和 (h_{fea}')，就可能同时破坏文本—音频对齐关系和说话人特征学习过程。

在推理阶段，流程与训练阶段不同。常规零样本语音克隆或 LLM-based TTS 推理通常不再依赖 ASR 自动标注，因为目标文本一般由攻击者或用户直接输入[2-4]。此时，文本侧流程可以表示为：
$$
y_{tar}
\rightarrow
T_{text}(y_{tar})
\rightarrow
z_{text}
$$
其中，(y_{tar}) 表示攻击者希望合成的目标文本，(z_{text}) 表示目标文本对应的 token 或文本表示。参考音频侧流程则仍然需要经过 TTS 侧特征编码器 / tokenizer：


$$
{x^{ref}}'
\rightarrow
a^{ref}(t,f)
\rightarrow
\Phi_{fea}({x^{ref}}')
\rightarrow
E_{fea}^{TTS}(\Phi_{fea}({x^{ref}}'))
\rightarrow
h'_{fea}
\rightarrow
Q_{fea}(h'_{fea}) \ \text{或} \ P_{fea}(h'_{fea})
\rightarrow
z'_{fea}
$$
随后，文本 token 和参考音频特征共同进入 LLM-based TTS 推理链路：
$$
(z_{text},z_{fea}')
\rightarrow
\text{LLM token modeling}
\rightarrow
z_{speech}
\rightarrow
G_{aco}(z_{speech})
\rightarrow
\hat{x}_{tts}
$$
其中，(z_{speech}) 表示由 LLM 预测出的语音 token 或中间语音表示，(G_{aco}) 表示声学生成器或 vocoder，(\hat{x}_{tts}) 表示最终合成语音。该流程说明，ASR 不是所有 TTS 推理流程的共同依赖点，而是端到端克隆训练、微调适配和自动标注场景中的关键依赖点；相对地，TTS 侧特征编码器 / tokenizer 是训练与推理阶段共同存在的核心依赖点[1-4]。

由训练流程可知，(L_{sem}) 的设计来源是 ASR 侧自动标注链路。ASR 的最终结果表现为离散文本序列，若直接围绕最终文本输出构造 loss，通常会受到具体 ASR 模型结构、解码方式、输出概率形式以及文本长度等因素影响，构造稳定且可迁移的优化目标较为困难[1,5-6]。因此，本文不直接要求 ASR 输出某个固定文本，而是将优化入口前移到 ASR 侧连续语义编码层。其设计思想可以概括为：
$$
x' \rightarrow \Phi_{asr}(x') \rightarrow E_{sem}^{ASR}(\Phi_{asr}(x'))

h_{sem}'
$$

$$
h_{sem}' \neq h_{sem}
\rightarrow
\hat{y}' \neq y
$$

其中，(h_{sem}) 表示原始语音对应的 ASR 侧语义表示，(y) 表示原始语音对应的真实文本。(L_{sem}) 的作用不是直接规定 (\hat{y}') 必须变成某个指定句子，而是促使 (h_{sem}') 在连续语义空间中偏离 (h_{sem})，从而间接提高自动转写错误率。根据本项目实验观察，语义编码层的微小扰动会在后续 token 化和识别过程中被进一步放大。因此，连续语义表示比最终离散文本更适合作为黑箱防御中的可微替代入口。

由训练流程和推理流程共同可知，(L_{fea}) 的设计来源是 TTS 侧特征编码链路。无论攻击者采用微调式克隆还是零样本推理，只要其目标是复刻目标说话人的声音，就必须从参考音频中提取说话人身份、音色、韵律、节奏、风格和声学环境等特征条件[1-4]。因此，本文将 TTS 侧特征编码器 / tokenizer 作为另一个核心防护目标，其设计思想可以概括为：
$$
x' \rightarrow \Phi_{fea}(x') \rightarrow E_{fea}^{TTS}(\Phi_{fea}(x'))

h_{fea}'
$$

$$
h_{fea}' \neq h_{fea}
\rightarrow
z_{fea}' \neq z_{fea}
$$

其中，(h_{fea}) 和 (z_{fea}) 分别表示原始语音对应的特征侧连续表示和特征条件表示。当 (z_{fea}') 偏离原始说话人的稳定特征表示时，后续 LLM token modeling 和声学生成模块即使仍能接收目标文本，也难以准确复刻原始说话人的音色和说话方式。因此，(L_{fea}) 主要服务于降低合成语音与原始说话人的相似度，是语音克隆防御中针对“个人声音特征”的核心损失项。

需要说明的是，ASR 侧语义编码器和 TTS 侧特征编码器并不是完全隔离的两类神经结构。二者都以原始语音的时频声学结构为基础，可能共享低层声学特征、预训练语音表征或相似的 encoder 架构；在具体实现中，甚至可能出现局部替代编码器重合的情况。例如，低层声学特征可以同时服务于语义识别和特征条件建模，部分预训练语音模型也可能同时携带语义、音色、韵律和声学线索[3-4]。因此，本文对 (L_{sem}) 和 (L_{fea}) 的区分主要是宏观功能区分：(L_{sem}) 面向“说了什么”的 ASR 语义链路，(L_{fea}) 面向“是谁在说、怎么说”的 TTS 特征链路。在不经过 ASR 的 TTS 推理场景中，语义侧微小扰动仍可能通过 speech tokenizer、semantic token 或语音表征纠缠影响 TTS 条件建模，但这一作用应理解为实验观察和机制解释，而不是简单归因于 ASR 转写错误。

进一步地，LLM-based 语音合成对离散 token 的依赖，使微小扰动具有被放大的可能性。大语言模型擅长处理 token 序列，而连续语音信号需要先经过编码器、tokenizer 或 neural codec，被转换为离散 speech token、semantic token、acoustic token 或 codec token 后，才能更自然地进入 LLM token modeling 流程[2-4,10-11]。对于采用向量量化或残差向量量化的 tokenizer，可将其量化过程抽象为一个 codebook 最近邻选择问题。设编码器输出的连续表示为 (h)，codebook 为：

或者更紧凑：
$$
\mathcal{C}={e_1,e_2,\cdots,e_K}
$$
对于采用残差向量量化的 speech tokenizer 或 neural codec，量化器 (Q) 通常不是只从单个 codebook 中选择一次最近向量，而是由多层 codebook 逐级逼近编码器输出。设编码器输出的连续表示为 $h$，第 $l$ 层 codebook 记为：
$$
\mathcal{C}^{(l)}=\{e^{(l)}_1,e^{(l)}_2,\cdots,e^{(l)}_{K_l}\},
\quad l=1,2,\cdots,L
$$
其中，$e^{(l)}_k$ 表示第 $l$ 层 codebook 中的第 $k$ 个向量，$L$ 表示残差量化层数。残差向量量化从初始残差 $r^{(0)}=h$ 开始，在第 $l$ 层选择与当前残差最接近的 codebook 向量：
$$
k_l^*(h)=
\arg\min_{1\leq k\leq K_l}
\left\|r^{(l-1)}-e^{(l)}_k\right\|_2^2
$$
然后更新下一层残差：
$$
r^{(l)}=r^{(l-1)}-q^{(l)}(h)
$$
经过 $L$ 层量化后，最终的量化表示可以写为：
$$
Q_{\mathrm{RVQ}}(h)
=
\sum_{l=1}^{L} q^{(l)}(h)
=
\sum_{l=1}^{L} e^{(l)}_{k_l^*(h)}
$$
对应的离散 token 序列可以表示为：
$$
\mathbf{k}^*(h)
=
\left(k_1^*(h),k_2^*(h),\cdots,k_L^*(h)\right)
$$
当 $h$ 或某一层残差 $r^{(l-1)}$ 位于多个 codebook 向量的边界附近时，即使连续空间中的偏移 $\Delta h$ 很小，也可能导致某一层或多层最近邻索引发生变化：
$$
k_l^*(h+\Delta h)\neq k_l^*(h)
$$
从而使整体离散 token 序列发生变化：
$$
\mathbf{k}^*(h+\Delta h)\neq \mathbf{k}^*(h)
$$
因此，在 RVQ 结构中，微小扰动不仅可能影响单层最近邻选择，还可能沿多层残差量化过程逐级改变 token 组合。也就是说，连续表示空间中的较小偏移，经过编码器和多层 codebook 量化后，可能在离散 token 空间中表现为更明显的 token 序列变化。这为本文从可微编码器表示层施加微小扰动、进而影响 LLM-based 语音系统的离散 token 建模提供了理论动机。

除 (L_{sem}) 和 (L_{fea}) 外，本文还引入 (L_{psy}) 和 (|\delta|*2) 作为约束性目标。原因在于，本作品关注的是个人语音信息安全中的主动防护，而不是单纯破坏音频信号。如果微小扰动过强，虽然可能提高模型错误率，但会降低原始语音的正常可听性和使用价值。从声学角度看，语音系统通常会将原始波形转换为包含时间、频率和强度信息的声学特征表示，例如 Mel 频谱图或 log-Mel 频谱图[5,7]。这些表示中的时间 (t)、频率 (f) 和能量强度共同影响语音的内容、音色和响度感知。人耳对语音的辨识也依赖类似的时频结构：不同频率成分及其随时间变化的强弱关系，会影响人对音素、音色、响度和发音方式的判断[8-9]。由于 auditory masking，即听觉掩蔽效应，强能量语音成分会掩蔽邻近时间或频率区域内较弱的噪声，因此同样大小的微小扰动在不同 ((t,f)) 区域的可感知性并不相同[8-9]。在原始语音能量较强、掩蔽阈值较高的时频单元中，较小噪声更容易被原始语音掩蔽；而在静音、弱能量或人耳敏感区域，同等大小的噪声更容易被察觉。因此，(L*{psy}) 的作用是引导微小扰动更多分布在人耳相对不敏感的位置，(|\delta|_2) 则用于限制整体微小扰动能量，避免保护样本偏离原始语音过大。

综上，本文的保护算法设计框架可以概括为：在端到端训练或微调阶段，(L_{sem}) 主要影响 ASR 侧语义编码和自动标注链路，(L_{fea}) 主要影响 TTS 侧特征编码和说话人条件学习；在推理阶段，(L_{fea}) 继续作用于参考音频特征提取，(L_{sem}) 则可能通过语义 token 或语音表征纠缠影响部分 LLM-based TTS 系统；(L_{psy}) 与 (|\delta|_2) 共同保证微小扰动仍处于正常听感可接受范围内。最终，本文通过语义侧和特征侧的联合微小扰动，在黑箱条件下实现对语音克隆流程的主动防护。由于本作品关注的是个人声音信息在未授权语音克隆中的保护问题，其问题属性更接近个人信息安全与语音隐私保护，而不是传统网络攻防场景。



[1] Zhang Z, Wang D, Mi Y, et al. E2E-VGuard: adversarial prevention for production LLM-based end-to-end speech synthesis[C]. Advances in Neural Information Processing Systems. 2025.

[2] Wang C, Chen S, Wu Y, et al. Neural codec language models are zero-shot text to speech synthesizers[EB/OL]. arXiv preprint arXiv:2301.02111, 2023.

[3] Du Z, Chen Q, Zhang S, et al. CosyVoice: a scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens[EB/OL]. arXiv preprint arXiv:2407.05407, 2024.

[4] Zhang X, Zhang D, Li S, et al. SpeechTokenizer: unified speech tokenizer for speech large language models[EB/OL]. arXiv preprint arXiv:2308.16692, 2023.

[5] Radford A, Kim J W, Xu T, et al. Robust speech recognition via large-scale weak supervision[C]. Proceedings of the 40th International Conference on Machine Learning. 2023: 28492-28518.

[6] Prabhavalkar R, Hori T, Sainath T N, et al. End-to-end speech recognition: a survey[EB/OL]. arXiv preprint arXiv:2303.03329, 2023.

[7] Shen J, Pang R, Weiss R J, et al. Natural TTS synthesis by conditioning WaveNet on Mel spectrogram predictions[C]. IEEE International Conference on Acoustics, Speech and Signal Processing. 2018: 4779-4783.

[8] Qin Y, Carlini N, Goodfellow I, et al. Imperceptible, robust, and targeted adversarial examples for automatic speech recognition[EB/OL]. arXiv preprint arXiv:1903.10346, 2019.

[9] Schönherr L, Kohls K, Zeiler S, et al. Adversarial attacks against automatic speech recognition systems via psychoacoustic hiding[C]. Network and Distributed System Security Symposium. 2019.

[10] Zeghidour N, Luebs A, Omran A, et al. SoundStream: an end-to-end neural audio codec[EB/OL]. arXiv preprint arXiv:2107.03312, 2021.

[11] Stevens S S, Volkmann J, Newman E B. A scale for the measurement of the psychological magnitude pitch[J]. The Journal of the Acoustical Society of America, 1937, 8(3): 185-190.

