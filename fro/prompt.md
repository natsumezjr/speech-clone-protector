你是一个资深前端工程师。请从零创建一个前端项目，项目名称为“语音克隆防护平台”。这是一个面向“全国大学生信息安全竞赛作品赛”评委展示的 Web 前端原型，用于展示主动式语音克隆防护系统。你没有读过项目文档和论文，因此下面会给出完整背景、功能范围、技术栈、页面设计、接口设计和实现边界。请严格按要求实现。

一、项目背景

本项目是一个“发布前源头防护”的语音安全平台。目标不是检测已经生成的深度伪造音频，而是在用户公开发布自己的语音前，对原始语音注入人耳不明显感知的保护扰动，使未授权语音克隆系统、ASR 系统或基于 LLM 的语音系统难以稳定提取原始说话人的语义内容和音色身份特征。

平台面向评委展示，因此重点是“系统闭环”和“可解释可视化”，不是普通的上传文件工具。前端需要清晰展示：

1. 原始音频进入系统。
2. 系统执行语义防护和音色防护。
3. 系统加入心理声学约束，使扰动尽量不影响人耳听感。
4. 输出保护音频。
5. 展示机器语义理解被干扰、声纹相似度下降、听感质量仍可接受。
6. 支持下载保护音频。
7. 支持后续快速对接真实后端 API。

二、技术背景说明

请把下面的概念转化为前端页面中的模块、文案、指标和图表，不需要实现真实算法。

1. 语义防护

现代 ASR、语音大模型和 LLM-based TTS 通常会先通过 speech tokenizer 或语义编码器，把连续语音波形转换为语义表示或离散 token。研究表明，微小扰动可能导致 tokenizer 编码结果发生明显偏移，使 ASR 或 LALM 的语义理解结果出现错误。

前端表达方式：

* 使用“语义防护”“语义分支”“ASR / Tokenizer / LALM 理解干扰”等词。
* 展示 ASR 转写对比。
* 展示 WER、CER、Token 变化率、Semantic Drift Score 等 mock 指标。
* 页面中可以出现 S3 Tokenizer、HuBERT、Whisper、MFCC 等作为“代理语义编码器”标签，但只作为展示，不做真实推理。

2. 音色防护

语音克隆系统需要从参考音频中提取说话人的音色、声纹、风格或 speaker embedding。防护系统通过扰动音色相关特征，使机器难以稳定建模原始说话人身份。

前端表达方式：

* 使用“音色防护”“声纹相似度下降”“Speaker Embedding”“Timbre Encoder”等词。
* 展示原始音频和保护音频的声纹相似度变化。
* 展示 SIM 分数、Embedding Distance、雷达图等 mock 指标。
* 可出现 WavLM、ECAPA-TDNN、CosyVoice Encoder、Style Encoder 等展示标签，但不做真实推理。

3. 心理声学约束

防护扰动不能明显破坏人耳听感。系统会尽量把扰动隐藏在人耳不敏感的时间-频率区域内，兼顾安全性和可用性。

前端表达方式：

* 使用“心理声学约束”“听感保真”“掩蔽阈值”“不可感知性”等词。
* 展示 SNR、PESQ、MOS-LQO 等 mock 指标。
* 展示扰动谱低于掩蔽阈值的可视化曲线。

4. E2E-VGuard 相关思想

该方向强调端到端语音克隆场景。攻击者可能只上传无标注音频，后端自动 ASR 转写，再进行 TTS 训练或零样本克隆。因此防护需要同时考虑：

* 音色身份保护。
* ASR / pronunciation / 语义理解干扰。
* 心理声学约束。
* 黑盒迁移展示。
* 商业 API / 开源 WebUI 风险场景。

5. T-SemAttack 相关思想

该方向强调 speech tokenizer 本身是语音大模型链路中的安全瓶颈。微小扰动会造成语义编码器表示漂移、token 变化、LALM 注意力失衡，最终导致 ASR 或 LALM 语义理解错误。

前端需要借鉴这个思想，将“语义分支”从单纯 ASR 攻击扩展为：

* Speech Tokenizer Encoder
* HuBERT
* Whisper
* MFCC
* Representation Loss
* Semantic Drift
* Error Tokens
* ASR / LALM Understanding

但注意：只做前端展示，不做真实攻击算法，不实现真实模型调用。

三、项目目标

从零创建一个可运行的 Vite 前端项目。技术栈固定为：

* React
* TypeScript
* Vite
* pnpm
* Tailwind CSS
* shadcn/ui
* lucide-react
* @tanstack/react-query
* zustand
* react-hook-form
* zod
* axios
* wavesurfer.js
* echarts

项目必须支持两种数据模式：

1. mock 模式

   * 完全使用前端 mock 数据。
   * 不调用任何后端 API。
   * 所有页面、图表、任务状态、结果数据都来自本地 mock。
   * 下载保护音频必须实现，可以通过前端生成一个 mock WAV Blob 或使用 public 目录中的 demo wav。
   * mock 就是 mock，不要混用用户真实上传音频和 mock 分析结果。

2. backend 模式

   * 所有数据通过后端 API 获取。
   * 前端只留出接口和 API client，不需要后端真实存在。
   * 后端不可用时显示清晰错误提示。
   * 不要自动 fallback 到 mock，避免混淆。
   * 下载保护音频通过后端下载接口获取。

模式通过环境变量控制：

VITE_DATA_MODE=mock
VITE_API_BASE_URL=http://localhost:8000

要求：

* 如果 VITE_DATA_MODE=mock，使用 mockClient。
* 如果 VITE_DATA_MODE=backend，使用 backendClient。
* 禁止在一次任务中同时使用 mock 数据和 backend 数据。
* UI 可以显示当前模式，但不要让用户误以为 mock 和 backend 可以混合运行。

四、页面范围

实现 4 个页面：

1. 首页 `/`
2. 防护工作台 `/workspace`
3. 结果分析 `/results/:taskId`
4. 历史任务 `/history`

不做登录页。右上角可以保留“评委用户”作为展示身份。登录接口可以预留类型和 API client 方法，但不实现页面和认证流程。

五、整体视觉风格

采用“A 深色赛博安全 + B 企业级隐私保护平台”的混合风格。

关键词：

* 深色背景
* 深蓝、黑蓝、青色、蓝色、绿色为主
* 少量紫色和橙色强调
* 卡片式布局
* 玻璃态 / 半透明 panel
* 细边框
* 轻微 glow
* 声波、盾牌、token、网络节点、频谱图元素
* 评委视角：专业、可信、完整、可解释
* 不要做成游戏 UI 或过度黑客风
* 不要使用夸张 emoji
* 不要使用花哨动效影响可读性

页面应有统一顶栏：

左侧：

* shield / waveform 风格 logo
* 标题：语音克隆防护平台
* 小版本 badge：V2.0

中间导航：

* 首页
* 防护工作台
* 结果分析
* 历史任务

右侧：

* 状态 badge：系统防护中
* 通知图标
* 用户：评委用户

六、首页设计

首页目标：让评委 10 秒内理解项目是什么、为什么重要、系统能力是什么。

主要模块：

1. Hero 区

标题：
发布前保护你的声音，降低语音克隆风险

副标题：
融合语义防护与音色防护的双重机制，在保证听感质量的同时干扰语音理解与音色建模，有效抵御非授权的语音克隆与滥用。

按钮：

* 开始防护：跳转 `/workspace`
* 查看演示：跳转 `/results/demo-task`

标签：

* 端到端可验证
* 多模型自适应
* 听感友好
* 高效易用

2. 中央流程图

用原创 UI 画出流程，不要复制论文图。

流程：
原始音频 → 保护性扰动 → 保护音频 → 下游影响

下游影响包括：

* ASR 语音识别：识别准确率下降
* Tokenizer 分词器：表示偏移增大
* LLM 语言模型：理解偏差增大
* 克隆系统 / TTS：声纹相似度下降

3. 三个核心策略卡片

* 语义防护

  * 干扰语义表示
  * 降低 ASR / LALM 理解准确率
  * 多模型语义编码器
  * token 变化与语义漂移

* 音色防护

  * 削弱声纹特征
  * 降低说话人相似度
  * 抑制音色建模
  * 阻断克隆条件提取

* 心理声学约束

  * 控制扰动可感知性
  * 掩蔽阈值建模
  * 听感保真优化

4. KPI 卡片

使用 mock 指标：

* ASR 干扰：WER 68.7%
* 声纹相似度下降：86.2%
* 听感保真：PESQ 3.67
* 对抗评估：多模型平均成功率 85.6%
* 任务通过率：98.7%
* 平均处理时长：72s

5. 作品亮点卡片

标题：作品亮点

内容：

* 提出语义与音色双重防护框架，兼顾安全与可用。
* 引入心理声学约束，扰动不可感知、听感友好。
* 通过 ASR、Tokenizer、LLM、TTS 多层指标展示防护效果。
* 前端支持 Mock / Backend 快速切换，便于竞赛演示和真实后端对接。

七、防护工作台页面设计

路径：`/workspace`

目标：展示用户如何提交防护任务。

布局：左中右三栏 + 底部任务状态。

1. 左侧：音频接入

标题：音频接入

tab：

* 上传音频
* 录音输入（仅展示，提示“接口预留”）

上传区域：

* 拖拽音频文件到此处，或点击上传
* 支持 .wav / .mp3 / .flac / .m4a
* 单文件 ≤ 200MB

mock 模式要求：

* 上传行为可以只保存文件名和前端 object URL 作为展示，但不要将该真实上传音频与 mock 分析结果混用。
* mock 任务使用固定 mock 结果。
* 如果用户上传了文件，显示“Mock 模式下仅展示文件接入状态，分析结果来自固定演示数据”。

文件卡片：

* 文件名
* 时长
* 采样率
* 声道
* 位深
* 文件大小
* 上传时间
* 文件指纹 mock hash
* 波形预览
* 播放按钮

可以使用 wavesurfer.js 画波形；如果实现复杂，可先用 SVG / CSS mock waveform，但代码结构要支持后续替换为 wavesurfer。

2. 中间：防护策略配置

标题：防护策略配置

保护模式：

* 标准保护：平衡安全与听感
* 强保护：更强安全性，略降听感
* 高保真：更优听感，安全性适中
* 高级自定义：自由调整参数

防护目标：

* 语义防护
* 音色防护
* 联合防护（推荐）

默认选中：联合防护（推荐）

参数配置：

* epsilon / 扰动强度，默认 0.08
* 优化轮数 Steps，默认 20
* 心理声学权重 lambdaPsy，默认 0.15
* ASR 模型，默认 Whisper-large-v3 或 Paraformer-large
* 语义编码器集合，显示 S3 Tokenizer / HuBERT / Whisper / MFCC
* Timbre 模式，默认 Untargeted，可选 Targeted
* 音色编码器集合，显示 WavLM / ECAPA-TDNN / CosyVoice Encoder / Style Encoder

高级选项可以折叠展示。

模式说明：

* Mock 模式：使用本地固定模拟数据，快速展示平台流程与评估结果，不调用后端。
* Backend 模式：调用后端服务执行真实防护流程，结果来自后端返回。
* 两种模式互斥，不混合数据。

任务执行：

* 主按钮：开始生成保护音频
* 次按钮：使用 Mock 数据演示

mock 模式下：

* 点击“开始生成保护音频”或“使用 Mock 数据演示”都创建 mock task。
* 展示任务状态进度。
* 进度完成后跳转 `/results/mock-task-001`。

backend 模式下：

* 调用 `POST /api/tasks/protect` 创建任务。
* 轮询 `GET /api/tasks/{taskId}` 或通过 SSE `GET /api/tasks/{taskId}/events` 更新状态。
* 完成后跳转结果页。

3. 右侧：系统架构概览

用卡片画一个小型原创架构图：

输入音频 x → 防护优化引擎 → 保护音频 x'

防护优化引擎下面分三支：

语义分支：

* ASR 系统
* 多模型语义编码
* S3 / HuBERT / Whisper / MFCC
* 表示空间约束
* 语义漂移评估

音色分支：

* Timbre Encoder
* Speaker Embedding
* 声纹特征约束
* 说话人不可恢复

听感约束：

* 心理声学模型
* 掩蔽阈值建模
* 听感优化
* 最小化可感知差异

底部显示联合优化目标的展示公式，不需要真实计算：
L = λ_sem L_sem + λ_timbre L_timbre + λ_psy L_psy + λ_2 ||δ||_2

4. 底部：任务状态

阶段：

1. 文件预处理
2. 编码器加载
3. 扰动优化
4. 心理声学约束
5. 结果评估
6. 报告生成

每个阶段显示：

* 等待开始
* 进行中
* 已完成
* 失败

右侧结果产物：

* 保护音频 .wav：必须可下载
* 评估报告 .pdf：接口预留
* 详细数据 .csv：接口预留
* 证据包 .zip：接口预留

八、结果分析页面设计

路径：`/results/:taskId`

目标：展示防护是否有效，给评委形成证据链。

1. 顶部 summary strip

字段：

* 任务 ID
* 任务状态：已完成
* 完成时间
* 处理耗时
* 防护模式：联合防护（推荐）
* 综合判定：防护有效
* 综合得分：92.6

2. 原始音频 vs 保护音频

两个音频卡片：

原始音频：

* 文件名 target_speech_demo.wav
* 波形
* 播放按钮
* 时长 12.34s
* 采样率 16kHz
* 格式 WAV
* 大小 1.88MB
* 描述：原始录音，包含清晰语义内容与可克隆声纹特征。

保护音频：

* 文件名 protected_voice_mock.wav
* 波形
* 播放按钮
* 时长 12.34s
* 采样率 16kHz
* 格式 WAV
* 大小 1.90MB
* 描述：防护后音频，语义受保护，声纹相似度显著降低，听感基本保持。

mock 模式下：

* 保护音频必须能下载。
* 可以通过前端生成一个 mock WAV Blob，文件名 `protected_voice_mock.wav`。
* 如果做播放器也使用这个 mock WAV Blob。

backend 模式下：

* 从后端 result 返回 audio URL 或调用下载接口。

3. 机器理解分析：ASR 转写对比

左侧：原始转写
示例：
今天天气很好，我们一起去公园散步吧。沿着湖边走，你可以看到很多漂亮的花，微风吹过来，感觉非常舒服。我们找个地方坐下，聊聊最近的生活和工作，放松一下心情。

右侧：保护后转写
示例：
今天石头很蓝，我们一路去公元散不唬。船长胡边走，你可以买到很多漂多的画，未分叫过来，甘觉非等似醒。我们转个地放坐下，聊聊最没的生高和工件，放松一下先青。

中间指标：

* WER：68.7%
* CER：54.1%
* Token 变化率：72.9%
* Semantic Drift：0.81

实现一个简单 diff 高亮组件：

* 新增内容：绿色
* 删除内容：红色
* 替换内容：橙色
  如果真实 diff 太复杂，可以使用预标注 mock segments。

4. 声纹 / 音色分析

指标：

* 防护前声纹相似度：0.912
* 防护后声纹相似度：0.126
* 下降：86.2%
* Embedding 距离：0.214 → 1.387
* 提升：548.1%

图表：

* 雷达图：音色相似度、基频特征、共振峰特征、韵律特征、声道特征
* 柱状图或卡片展示前后变化

5. 感知质量评估

指标：

* SNR：21.8 dB，优秀
* PESQ：3.67，良好
* MOS-LQO：3.82 / 5，良好

图表：

* 心理声学阈值分析
* x 轴：频率
* y 轴：强度 dB
* 两条曲线：掩蔽阈值、保护干扰谱
* 表达“扰动大多低于掩蔽阈值”

6. 综合指标趋势

用 ECharts 展示 4 到 5 个小图：

* ASR 干扰 WER
* 声纹相似度
* 听感质量 MOS-LQO
* PESQ
* 任务耗时

7. 结果解读

标题：结果解读（自动生成）

内容：

* 语义层面：WER 68.7%，关键语义被显著干扰，机器理解难度提升。
* 声纹层面：相似度从 0.912 降至 0.126，已有效破坏可克隆性。
* 听感层面：PESQ=3.67，MOS-LQO=3.82，整体听感保持良好，满足可用性要求。
* 综合结论：各项指标达到演示阈值，判定为“防护有效”。

8. 操作与导出

按钮：

* 下载保护音频：必须实现
* 导出评估报告 PDF：接口预留，点击显示 toast “后端接口预留”
* 导出详细数据 CSV：接口预留，点击显示 toast “后端接口预留”
* 下载证据包 ZIP：接口预留，点击显示 toast “后端接口预留”
* 重新执行任务：跳转 `/workspace`

九、历史任务页面设计

路径：`/history`

轻量版即可。用于展示完整平台闭环。

内容：

* 搜索框
* 状态筛选：全部 / 已完成 / 处理中 / 失败
* 模式筛选：标准保护 / 强保护 / 高保真 / 联合防护
* 任务表格

表格字段：

* 任务 ID
* 文件名
* 防护模式
* 数据模式：Mock / Backend
* 状态
* WER
* 声纹相似度下降
* PESQ
* 创建时间
* 操作：查看结果 / 下载保护音频 / 删除

mock 模式：

* 使用固定历史任务数组。
* 查看结果跳转 `/results/mock-task-001`。
* 下载保护音频必须可用。

backend 模式：

* 调用 `GET /api/tasks` 获取列表。
* 删除调用 `DELETE /api/tasks/{taskId}`。
* 下载调用后端下载接口。

十、接口设计

请在前端定义完整 TypeScript 类型和 API client。即使后端不存在，也要保证接口结构清晰。

目录建议：

src/
app/
assets/
components/
layout/
charts/
audio/
cards/
common/
config/
runtime.ts
data/
mockData.ts
hooks/
lib/
pages/
HomePage.tsx
WorkspacePage.tsx
ResultsPage.tsx
HistoryPage.tsx
services/
apiClient.ts
mockClient.ts
backendClient.ts
store/
appStore.ts
taskStore.ts
types/
api.ts
task.ts
audio.ts
utils/
download.ts
mockWav.ts
format.ts
main.tsx
router.tsx
index.css

环境配置：

.env.example:
VITE_DATA_MODE=mock
VITE_API_BASE_URL=http://localhost:8000

runtime.ts：

* 读取 import.meta.env.VITE_DATA_MODE
* 只允许 mock 或 backend
* 导出 dataMode、apiBaseUrl、isMockMode、isBackendMode

统一 service：

apiClient.ts：

* 根据 runtime 选择 mockClient 或 backendClient
* 不允许混合
* 导出 uploadFile、createProtectionTask、getTaskStatus、getTaskResult、listTasks、deleteTask、downloadProtectedAudio、exportReport、exportCsv、downloadEvidenceZip 等函数

后端接口预留：

POST /api/files/upload
请求：multipart/form-data
响应：
{
"fileId": "file_001",
"filename": "target_speech_demo.wav",
"durationSec": 12.34,
"sampleRate": 16000,
"channels": 1,
"bitDepth": 16,
"sizeBytes": 1880000,
"format": "wav",
"audioUrl": "..."
}

POST /api/tasks/protect
请求：
{
"fileId": "file_001",
"mode": "standard | strong | high_fidelity | custom",
"targets": ["semantic", "timbre"],
"semantic": {
"enabled": true,
"asrModel": "whisper-large-v3",
"encoders": ["s3-tokenizer", "hubert", "whisper", "mfcc"],
"lambdaSemantic": 1.0
},
"timbre": {
"enabled": true,
"mode": "untargeted | targeted",
"encoders": ["wavlm", "ecapa-tdnn", "cosyvoice", "styletts2"],
"lambdaTimbre": 1.0
},
"psychoacoustic": {
"enabled": true,
"lambdaPsy": 0.15
},
"optimization": {
"epsilon": 0.08,
"steps": 20
}
}

响应：
{
"taskId": "task_001",
"status": "queued"
}

GET /api/tasks/{taskId}
响应：
{
"taskId": "task_001",
"status": "queued | running | completed | failed",
"progress": 0.65,
"stage": "perturbation_optimization",
"message": "正在进行扰动优化",
"createdAt": "...",
"updatedAt": "...",
"error": null
}

GET /api/tasks/{taskId}/events

* SSE 接口预留
* 前端可以先不实现 SSE，只实现轮询
* 代码结构要方便以后接入 SSE

GET /api/tasks/{taskId}/result
响应：
{
"taskId": "task_001",
"verdict": "protected",
"score": 92.6,
"originalAudio": {...},
"protectedAudio": {...},
"asr": {
"originalText": "...",
"protectedText": "...",
"wer": 0.687,
"cer": 0.541,
"tokenChangeRate": 0.729,
"semanticDrift": 0.81
},
"speaker": {
"simBefore": 0.912,
"simAfter": 0.126,
"simDropRate": 0.862,
"embeddingDistanceBefore": 0.214,
"embeddingDistanceAfter": 1.387
},
"quality": {
"snr": 21.8,
"pesq": 3.67,
"mosLqo": 3.82
},
"charts": {
"psychoacoustic": [...],
"trend": [...]
}
}

GET /api/tasks/{taskId}/download/protected-audio

* 返回 wav blob
* mock 模式必须实现本地 wav blob 下载
* backend 模式请求后端 blob

GET /api/tasks
响应：历史任务列表

DELETE /api/tasks/{taskId}
删除任务

POST /api/reports/export
接口预留

GET /api/tasks/{taskId}/export/csv
接口预留

GET /api/tasks/{taskId}/download/evidence
接口预留

十一、Mock 数据要求

mock 数据必须完整、可信、适合评委展示。

固定 mock task：

taskId: mock-task-001
filename: target_speech_demo.wav
protectedFilename: protected_voice_mock.wav
status: completed
mode: joint
dataMode: mock
duration: 12.34s
sampleRate: 16000
language: 中文普通话
createdAt: 2024-06-01 14:21:36
completedAt: 2024-06-01 14:23:18
elapsed: 72s
score: 92.6
verdict: 防护有效

ASR:
originalText:
今天天气很好，我们一起去公园散步吧。沿着湖边走，你可以看到很多漂亮的花，微风吹过来，感觉非常舒服。我们找个地方坐下，聊聊最近的生活和工作，放松一下心情。

protectedText:
今天石头很蓝，我们一路去公元散不唬。船长胡边走，你可以买到很多漂多的画，未分叫过来，甘觉非等似醒。我们转个地放坐下，聊聊最没的生高和工件，放松一下先青。

wer: 0.687
cer: 0.541
tokenChangeRate: 0.729
semanticDrift: 0.81

Speaker:
simBefore: 0.912
simAfter: 0.126
simDropRate: 0.862
embeddingDistanceBefore: 0.214
embeddingDistanceAfter: 1.387

Quality:
snr: 21.8
pesq: 3.67
mosLqo: 3.82

Trends:
生成 20 个点，用于趋势图，数值要合理，不要完全随机失真。

十二、下载保护音频实现

这是必须实现的功能。

mock 模式：

* 实现 `createMockProtectedWavBlob()`。
* 生成一个合法 WAV Blob，16kHz，mono，16-bit PCM，时长可为 3 到 5 秒。
* 内容可以是低幅度合成音 + 轻微调制，不需要是真实语音。
* 下载文件名：`protected_voice_mock.wav`。
* 结果页和历史任务页的“下载保护音频”都调用该函数。

backend 模式：

* 调用 `GET /api/tasks/{taskId}/download/protected-audio`。
* 使用 axios 获取 blob。
* 下载文件名从 response header 或 result metadata 获取。
* 后端失败时显示 toast。

PDF / CSV / ZIP：

* 不需要真实生成。
* 保留按钮和函数。
* mock 模式点击显示 toast：“该导出项为后端接口预留。”
* backend 模式调用接口，失败时显示错误。

十三、组件要求

请尽量拆分组件，避免把所有代码写在一个页面里。

建议组件：

Layout:

* AppShell
* TopNav
* PageHeader
* StatusBadge

Audio:

* AudioUploadPanel
* AudioWaveform
* AudioCompareCard
* MockAudioPlayer

Cards:

* MetricCard
* StrategyCard
* FeatureCard
* EvidenceCard
* TaskSummaryStrip

Charts:

* TrendChart
* RadarChart
* PsychoacousticChart
* MiniSparkline
* SimilarityBar

Workspace:

* ProtectionModeSelector
* ProtectionTargetSelector
* ParameterForm
* ArchitectureOverview
* TaskProgressStepper

Results:

* AsrDiffPanel
* SpeakerAnalysisPanel
* QualityPanel
* ResultInterpretation
* ExportActions

History:

* TaskTable
* TaskFilters

十四、状态管理

使用 zustand 保存：

* 当前 dataMode
* 当前上传文件信息
* 当前任务状态
* 当前任务结果
* 历史任务列表

使用 @tanstack/react-query 管理异步请求：

* listTasks
* getTaskStatus
* getTaskResult
* backend API 请求
* mock 请求也可以用 Promise + setTimeout 模拟延迟

十五、表单校验

使用 react-hook-form + zod。

校验：

* epsilon: 0.01 到 0.2
* steps: 1 到 500
* lambdaPsy: 0 到 1
* 防护目标至少选择一个
* backend 模式下必须有 fileId 后才能创建任务
* mock 模式下可直接创建 mock task，但仍应显示用户是否上传文件

十六、交互细节

* 所有按钮有 hover / active 状态。
* 使用 toast 展示成功、失败、接口预留。
* 任务执行进度用模拟定时器推进。
* 进度完成后自动跳转结果页。
* 历史任务页点击“查看结果”跳转结果页。
* 顶栏导航 active 状态要正确。
* 空状态要好看。
* 错误状态要清晰。
* loading skeleton 或 spinner 要有。
* 页面要响应式，至少保证 1440px 桌面下效果最佳；平板和小屏可以纵向堆叠。

十七、样式实现建议

Tailwind 主题色：

background:

* #020617
* #07111f
* #0b1220

primary cyan:

* #06b6d4
* #22d3ee
* #38bdf8

green:

* #22c55e
* #10b981

blue:

* #3b82f6

purple:

* #8b5cf6

orange:

* #f59e0b

danger:

* #ef4444

卡片：

* bg-slate-900/60
* border-cyan-500/20
* backdrop-blur
* rounded-2xl
* shadow with subtle cyan glow

字体：

* 中文优先使用系统 sans-serif。
* 不要引入外部在线字体，避免网络依赖。

十八、README 要求

生成 README.md，内容包括：

1. 项目简介
2. 技术栈
3. 安装方式

使用 pnpm：

pnpm install
pnpm dev
pnpm build

4. 环境变量说明

VITE_DATA_MODE=mock
VITE_API_BASE_URL=http://localhost:8000

5. Mock / Backend 模式说明

强调：

* mock 模式完全使用前端模拟数据。
* backend 模式完全使用后端 API。
* 两者不混合。

6. 页面说明
7. API 对接说明
8. 已实现功能
9. 预留接口
10. 真实算法不在前端实现

十九、验收标准

请保证：

1. `pnpm install` 后可以正常运行。
2. `pnpm dev` 可以打开页面。
3. 4 个页面都可访问。
4. 首页视觉接近高质量竞赛展示大屏。
5. 防护工作台可以创建 mock 任务并跳转结果页。
6. 结果页展示完整评估证据链。
7. 历史任务页可以查看 mock 历史任务。
8. 下载保护音频功能真实可用。
9. PDF / CSV / ZIP 导出按钮保留接口并给出提示。
10. `.env` 可以切换 mock / backend。
11. backend 模式下不会混用 mock 数据。
12. TypeScript 类型完整，没有明显 any 滥用。
13. 组件拆分合理。
14. README 完整。

二十、实现边界

不要做：

* 不要实现真实语音防护算法。
* 不要调用真实 ASR / TTS / LLM。
* 不要写 Python 后端。
* 不要做登录系统。
* 不要混合 mock 数据和 backend 数据。
* 不要把论文原图直接塞进页面。
* 不要把页面做成简单后台模板。

必须做：

* 从零创建完整 Vite + React + TypeScript 项目。
* 使用 pnpm。
* 使用 Tailwind + shadcn/ui 风格。
* 使用 ECharts 展示图表。
* 使用 lucide-react 图标。
* 使用严格类型定义。
* 支持 mock/backend 双 client。
* 实现保护音频下载。
* 输出完整 README。
