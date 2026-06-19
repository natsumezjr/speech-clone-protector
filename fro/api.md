1. 当前前后端模式约定

前端通过环境变量选择运行模式：

VITE_DATA_MODE=mock
VITE_API_BASE_URL=http://localhost:8000

VITE_DATA_MODE 只允许 mock 或 backend；README 明确说明 mock 模式完全使用前端固定模拟数据，backend 模式完全使用后端 API，不会自动 fallback 到 mock，两者互斥。

后端默认 Base URL 是：

http://localhost:8000

api.md 中也明确写了 Base URL 和接口列表。

当前阶段进度（2026-06-19）：

- 当前仓库仍定位为前端原型，不新增后端工程，不实现真实算法。
- mock/backend 双模式保留；backendClient 继续作为后端 API 预留调用层。
- WorkspacePage 已删除任务状态轮询、定时推进和周期性 getTaskStatus 调用。
- mock 模式下创建任务成功后直接进入结果页；backend 模式下创建任务成功后也直接跳转结果页，不在前端主动轮询状态。
- ResultsPage 的 ASR 转写对比区域已更新为 6 个指标：WER、CER、Token 错误率、SD、IR、DR。
- PDF / CSV / ZIP、SSE、历史任务持久化、真实 ASR/TTS/声纹/PESQ/SNR/心理声学算法仍不在当前阶段实现范围内。

2. 前后端接口总表

当前 api.md 已列出这些后端预留接口：文件上传、创建防护任务、查询任务状态、查询任务结果、下载保护音频、历史任务、删除任务、导出报告、导出 CSV、下载证据包，以及 SSE 事件流预留。

接口	方法	前端用途	当前状态
/api/files/upload	POST	上传音频文件，获得 fileId 和音频元信息	后端预留，前端已写调用
/api/tasks/protect	POST	创建语音防护任务	后端预留，前端已写调用；创建成功后直接跳结果页
/api/tasks/{taskId}	GET	查询任务状态	后端预留；当前前端不轮询、不定时查询
/api/tasks/{taskId}/events	GET	SSE 实时任务进度	仅文档预留，前端尚未接入
/api/tasks/{taskId}/result	GET	查询任务分析结果	后端预留，前端已写调用
/api/tasks/{taskId}/download/protected-audio	GET	下载保护音频	前端 mock 已实现；backend 预留
/api/tasks	GET	查询历史任务列表	后端预留，前端已写调用
/api/tasks/{taskId}	DELETE	删除历史任务	后端预留，前端已写调用
/api/reports/export	POST	导出 PDF 报告	按钮有，接口预留
/api/tasks/{taskId}/export/csv	GET	导出 CSV 明细	按钮有，接口预留
/api/tasks/{taskId}/download/evidence	GET	下载 ZIP 证据包	按钮有，接口预留
3. 接口详细定义与示例 JSON
3.1 上传音频文件
请求
POST /api/files/upload
Content-Type: multipart/form-data

字段：

file: File

前端上传组件限制单文件不超过 200MB，并展示支持 .wav / .mp3 / .flac / .m4a / .webm；mock 模式下只展示文件接入状态，分析结果仍来自固定 mock 数据。

响应示例
{
  "fileId": "file_20260601_0001",
  "filename": "target_speech_demo.wav",
  "durationSec": 12.34,
  "sampleRate": 16000,
  "channels": 1,
  "bitDepth": 16,
  "sizeBytes": 1880000,
  "format": "WAV",
  "audioUrl": "http://localhost:8000/static/audio/file_20260601_0001.wav",
  "uploadedAt": "2026-06-01 14:21:36",
  "fingerprint": "sha256:8f21c9a4"
}
后端需要返回的关键点

fileId 很重要。因为 backend 模式下，防护任务创建必须依赖上传后返回的 fileId。前端 ParameterForm 中已经写了：backend 模式下如果没有 uploadedFile.fileId，不能创建任务。

3.2 创建防护任务
请求
POST /api/tasks/protect
Content-Type: application/json
请求 JSON 示例
{
  "fileId": "file_20260601_0001",
  "mode": "standard",
  "targets": ["semantic", "timbre"],
  "semantic": {
    "enabled": true,
    "asrModel": "Whisper-large-v3",
    "encoders": ["s3-tokenizer", "hubert", "whisper", "mfcc"],
    "lambdaSemantic": 1.0
  },
  "timbre": {
    "enabled": true,
    "mode": "untargeted",
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
字段说明

mode 可取：

standard | strong | high_fidelity | custom

前端任务结果里还用了 joint 表示联合防护结果状态，但创建任务时建议后端仍按 standard / strong / high_fidelity / custom 接收，联合防护通过 targets: ["semantic", "timbre"] 表达。

targets 可取：

semantic
timbre

默认建议：

["semantic", "timbre"]

也就是联合防护。

响应 JSON 示例
{
  "taskId": "task_20260601_142136",
  "status": "queued"
}
3.3 查询任务状态
请求
GET /api/tasks/{taskId}
响应 JSON 示例
{
  "taskId": "task_20260601_142136",
  "status": "running",
  "progress": 0.62,
  "stage": "perturbation_optimization",
  "message": "正在进行扰动优化",
  "createdAt": "2026-06-01 14:21:36",
  "updatedAt": "2026-06-01 14:22:41",
  "error": null
}
状态枚举
queued
running
completed
failed
阶段枚举
file_preprocess
encoder_loading
perturbation_optimization
psychoacoustic_constraint
result_evaluation
report_generation
后端实现建议

progress 建议使用 0 到 1 的浮点数。
stage 用固定枚举，不要直接返回中文，中文由前端映射。

当前前端进度

该接口仍保留在 ApiClient 类型与 backendClient/mockClient 中，但 WorkspacePage 当前不再主动轮询 GET /api/tasks/{taskId}，也不再用定时器模拟任务进度。

当前阶段的页面行为是：

- mock 模式：createProtectionTask 成功后，前端直接设置本地 completed 状态并跳转结果页；
- backend 模式：createProtectionTask 成功后，前端直接跳转 /results/{taskId}，由结果页读取 /api/tasks/{taskId}/result；
- 后续任务进度机制会统一设计，本阶段不接轮询、不接 SSE。

3.4 SSE 任务事件流
请求
GET /api/tasks/{taskId}/events
Accept: text/event-stream

这个接口目前只在 api.md 中作为“预留接口”出现。当前前端没有接入 SSE，也没有接入轮询或定时推进。后续任务进度机制会统一设计。

SSE 事件示例
event: task_progress
data: {"taskId":"task_20260601_142136","status":"running","progress":0.35,"stage":"encoder_loading","message":"正在加载语义编码器与音色编码器"}

event: task_progress
data: {"taskId":"task_20260601_142136","status":"running","progress":0.72,"stage":"psychoacoustic_constraint","message":"正在进行心理声学约束优化"}

event: task_completed
data: {"taskId":"task_20260601_142136","status":"completed","progress":1,"stage":"report_generation","message":"任务完成"}
3.5 查询任务结果
请求
GET /api/tasks/{taskId}/result
响应 JSON 示例
{
  "taskId": "task_20260601_142136",
  "status": "completed",
  "mode": "joint",
  "dataMode": "backend",
  "verdict": "防护有效",
  "score": 92.6,
  "completedAt": "2026-06-01 14:23:18",
  "elapsedSec": 72,
  "originalAudio": {
    "fileId": "file_20260601_0001",
    "filename": "target_speech_demo.wav",
    "durationSec": 12.34,
    "sampleRate": 16000,
    "channels": 1,
    "bitDepth": 16,
    "sizeBytes": 1880000,
    "format": "WAV",
    "audioUrl": "http://localhost:8000/static/audio/file_20260601_0001.wav",
    "uploadedAt": "2026-06-01 14:21:36",
    "fingerprint": "sha256:8f21c9a4"
  },
  "protectedAudio": {
    "fileId": "protected_20260601_0001",
    "filename": "protected_voice.wav",
    "durationSec": 12.34,
    "sampleRate": 16000,
    "channels": 1,
    "bitDepth": 16,
    "sizeBytes": 1900000,
    "format": "WAV",
    "audioUrl": "http://localhost:8000/static/audio/protected_20260601_0001.wav",
    "uploadedAt": "2026-06-01 14:23:18",
    "fingerprint": "sha256:42ae19d0"
  },
  "asr": {
    "originalText": "今天天气很好，我们一起去公园散步吧。沿着湖边走，你可以看到很多漂亮的花，微风吹过来，感觉非常舒服。我们找个地方坐下，聊聊最近的生活和工作，放松一下心情。",
    "protectedText": "今天石头很蓝，我们一路去公元散不唬。船长胡边走，你可以买到很多漂多的画，未分叫过来，甘觉非等似醒。我们转个地放坐下，聊聊最没的生高和工件，放松一下先青。",
    "wer": 0.687,
    "cer": 0.541,
    "tokenChangeRate": 0.729,
    "tokenErrorRate": 0.729,
    "semanticDrift": 0.81,
    "insertRate": 0.236,
    "deleteRate": 0.184
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
    "psychoacoustic": [
      {
        "frequency": 80,
        "maskingThreshold": 24,
        "perturbation": 10
      },
      {
        "frequency": 160,
        "maskingThreshold": 31,
        "perturbation": 14
      },
      {
        "frequency": 315,
        "maskingThreshold": 36,
        "perturbation": 18
      },
      {
        "frequency": 1000,
        "maskingThreshold": 47,
        "perturbation": 33
      }
    ],
    "trend": [
      {
        "step": 1,
        "wer": 0.21,
        "sim": 0.86,
        "mos": 4.24,
        "pesq": 4.07,
        "elapsed": 10.2
      },
      {
        "step": 2,
        "wer": 0.25,
        "sim": 0.81,
        "mos": 4.20,
        "pesq": 4.03,
        "elapsed": 13.5
      }
    ],
    "radarBefore": [0.92, 0.87, 0.82, 0.78, 0.84],
    "radarAfter": [0.14, 0.22, 0.24, 0.31, 0.28]
  }
}
注意

当前 mock 数据中已经有这些字段：asr、speaker、quality、charts.psychoacoustic、charts.trend、charts.radarBefore、charts.radarAfter。 后端需要尽量对齐这个结构，否则结果页组件会进入兼容 fallback。

ASR 指标当前展示规则

结果页 ASR 转写对比区域中间展示 6 个指标，两行三列：

1. WER（词错率）
2. CER（字错率）
3. Token 错误率
4. SD（语义漂移）
5. IR（插入率）
6. DR（删除率）

字段优先级：

- WER：优先 result.asr.wer；没有字段时显示“无”。
- CER：优先 result.asr.cer；没有字段但 originalText/protectedText 都存在时，前端用字符级编辑距离计算 fallback；否则显示“无”。
- Token 错误率：优先 result.asr.tokenErrorRate；没有时兼容 result.asr.tokenChangeRate；两个字段都没有时显示“无”。前端不会从文本自行计算 Token 错误率。
- SD：优先 result.asr.semanticDrift；没有字段时显示“无”。前端不会用 WER/CER 伪造语义漂移。
- IR：优先 result.asr.insertRate；没有字段但 originalText/protectedText 都存在时，前端用字符级 diff 计算 fallback；否则显示“无”。
- DR：优先 result.asr.deleteRate；没有字段但 originalText/protectedText 都存在时，前端用字符级 diff 计算 fallback；否则显示“无”。

当前 TypeScript 兼容字段：

{
  "originalText": "string",
  "protectedText": "string",
  "wer": "number | optional",
  "cer": "number | optional",
  "tokenChangeRate": "number | optional",
  "tokenErrorRate": "number | optional",
  "semanticDrift": "number | optional",
  "insertRate": "number | optional",
  "deleteRate": "number | optional"
}

3.6 下载保护音频
请求
GET /api/tasks/{taskId}/download/protected-audio
响应
Content-Type: audio/wav
Content-Disposition: attachment; filename="protected_voice.wav"

返回二进制 Blob。

前端行为

当前前端已经实现下载逻辑：结果页和历史页都会调用 downloadProtectedAudio(taskId)，拿到 { blob, filename } 后执行浏览器下载。

mock 模式下，mockClient 会生成合法 WAV Blob，文件名是：

protected_voice_mock.wav

这一点在 README 的“已实现功能”中也写明了。

3.7 查询历史任务列表
请求
GET /api/tasks
响应 JSON 示例
[
  {
    "taskId": "task_20260601_142136",
    "filename": "target_speech_demo.wav",
    "protectedFilename": "protected_voice.wav",
    "mode": "joint",
    "dataMode": "backend",
    "status": "completed",
    "wer": 0.687,
    "simDropRate": 0.862,
    "pesq": 3.67,
    "createdAt": "2026-06-01 14:21:36"
  },
  {
    "taskId": "task_20260602_091703",
    "filename": "campus_interview.wav",
    "protectedFilename": "protected_voice.wav",
    "mode": "strong",
    "dataMode": "backend",
    "status": "completed",
    "wer": 0.714,
    "simDropRate": 0.881,
    "pesq": 3.44,
    "createdAt": "2026-06-02 09:17:03"
  }
]
前端页面行为

历史任务页已有搜索、状态筛选、模式筛选，并调用 listTasks() 获取数据。

3.8 删除任务
请求
DELETE /api/tasks/{taskId}
推荐响应
204 No Content

或者：

{
  "success": true,
  "taskId": "task_20260601_142136",
  "message": "任务已删除"
}
当前前端行为

历史任务表格里已经调用 deleteTask(taskId)；mock 模式下只是展示交互，backend 模式应调用真实 DELETE 接口。

3.9 导出评估报告 PDF
请求
POST /api/reports/export
Content-Type: application/json
请求 JSON 示例
{
  "taskId": "task_20260601_142136"
}
响应
Content-Type: application/pdf
Content-Disposition: attachment; filename="voice_protection_report_task_20260601_142136.pdf"

返回 PDF Blob。

当前状态

按钮存在，backendClient 已经按 POST /api/reports/export 调用；但是 README 说明 PDF / CSV / ZIP 导出按钮目前是接口预留并给 toast 提示。

3.10 导出详细数据 CSV
请求
GET /api/tasks/{taskId}/export/csv
响应
Content-Type: text/csv
Content-Disposition: attachment; filename="task_20260601_142136_metrics.csv"

CSV 示例内容：

metric,value
wer,0.687
cer,0.541
tokenChangeRate,0.729
tokenErrorRate,0.729
semanticDrift,0.81
insertRate,0.236
deleteRate,0.184
simBefore,0.912
simAfter,0.126
simDropRate,0.862
snr,21.8
pesq,3.67
mosLqo,3.82

当前也是接口预留。

3.11 下载证据包 ZIP
请求
GET /api/tasks/{taskId}/download/evidence
响应
Content-Type: application/zip
Content-Disposition: attachment; filename="task_20260601_142136_evidence.zip"

建议 ZIP 包结构：

task_20260601_142136_evidence/
  original.wav
  protected.wav
  result.json
  asr_comparison.json
  speaker_metrics.json
  quality_metrics.json
  charts/
    psychoacoustic.json
    trend.json

当前也是接口预留。

4. 当前阶段 API 对接进度

当前阶段只维护前端原型和 API 契约，不新增后端目录，不实现 FastAPI / Express / SQLite / 文件存储 / SSE 服务。

已完成：

- mock/backend 双 client 结构保留。
- backendClient 继续保留 API 调用路径，便于后续接入真实后端。
- uploadFile(file) 链路保留，backend 模式下会调用 POST /api/files/upload。
- createProtectionTask(payload) 链路保留，backend 模式下会调用 POST /api/tasks/protect。
- getTaskResult(taskId) 链路保留，结果页会调用 GET /api/tasks/{taskId}/result。
- downloadProtectedAudio(taskId) 链路保留，结果页和历史页可调用 GET /api/tasks/{taskId}/download/protected-audio。
- 结果页 ASR 指标已改成 6 项安全展示，并兼容缺字段。
- mock 数据已补齐 wer、cer、tokenChangeRate、semanticDrift、insertRate、deleteRate，保证演示完整。

当前刻意不做：

- 不实现后端服务。
- 不实现真实算法。
- 不接 SSE。
- 不接轮询。
- 不实现 PDF / CSV / ZIP 导出文件生成。
- 不实现历史任务持久化。
- 不实现登录与权限。

任务状态接口当前只作为契约保留：

- /api/tasks/{taskId} 仍在 ApiClient 中存在；
- WorkspacePage 当前不主动调用该接口；
- 后续任务进度机制会统一设计后再接入。
5. 当前还没有实现或存在问题的功能
5.1 没有真实后端

仓库当前没有后端工程目录。fro 是前端项目，README 也明确写“不提供 Python 后端”。

因此以下内容都还没有真实服务支撑：

文件真正上传保存；
任务队列；
任务状态持久化；
真实结果查询；
后端音频文件下载；
PDF / CSV / ZIP 文件生成；
SSE 实时事件流。
5.2 没有真实语音防护算法

README 明确说明前端不实现真实语音防护算法，不调用真实 ASR / TTS / LLM。

也就是说，现在没有实现：

E2E-VGuard 真实扰动生成；
T-SemAttack 真实语义扰动；
真实 ASR 转写；
真实 WER / CER 计算；
真实 speech tokenizer token change rate；
真实 speaker embedding / 声纹相似度；
真实 PESQ / MOS-LQO / SNR；
真实心理声学阈值计算。

当前这些都是 mock 展示数据。

5.3 录音输入已接入前端闭环

当前前端已实现浏览器录音输入：

浏览器麦克风权限申请；
MediaRecorder 录音；
停止录音后生成 File；
生成本地 object URL 供即时预览；
复用 uploadFile(file) 上传链路；
Backend 模式下上传成功后保留后端返回的 fileId/audioUrl；
录音文件可作为后续创建保护任务的输入。

仍未实现：

真实录音波形分析；
录音分段上传；
服务端录音格式转码。
5.4 任务进度机制暂未实现

api.md 只把 /api/tasks/{taskId}/events 标为 SSE 事件流预留，/api/tasks/{taskId} 状态查询接口也仅作为后端契约保留。

未实现：

前端 EventSource 接入；
后端事件推送；
前端任务状态轮询；
前端定时模拟任务推进；
任务阶段实时日志；
loss / progress / stage 实时流式更新。
5.5 PDF / CSV / ZIP 导出没有真实实现

结果页的导出按钮已经有，但 PDF、CSV、ZIP 当前只是调用预留接口；mock 模式下会提示接口预留。README 也明确写“PDF / CSV / ZIP 导出按钮保留接口并给出 toast 提示”。

目前只有下载保护音频是前端 mock 里真实可用的。

5.6 登录与权限系统没有实现

README 明确说“不包含登录系统”。

未实现：

登录页；
token；
用户身份；
权限控制；
API 鉴权；
用户任务隔离。

目前右上角“评委用户”只是展示。

5.7 历史任务查看结果已使用真实 taskId

历史任务表格里，点击查看结果现在会跳转到：

navigate(`/results/${task.taskId}`)

删除任务成功后也会 invalidate/refetch 历史任务列表。

仍可继续增强：

删除二次确认；
批量删除；
历史任务详情抽屉；
按创建时间/评分排序。
5.8 后端错误与任务失败细节后续可增强

当前任务状态接口有 error 字段，但前端展示还比较粗。建议后端未来返回更细的错误结构：

{
  "error": {
    "code": "MODEL_LOAD_FAILED",
    "message": "语义编码器加载失败",
    "detail": "Whisper-large-v3 checkpoint not found",
    "recoverable": true
  }
}

这样前端可以区分：

文件格式错误；
音频过长；
显存不足；
模型加载失败；
后端任务超时；
结果文件丢失。
5.9 真实文件持久化与任务持久化当前不实现

当前接口定义了 fileId 和 taskId，但还没有后端存储规范。

如果后续进入后端阶段，可再设计类似结构：

uploads/
  fileId.wav

tasks/
  taskId/
    request.json
    status.json
    result.json
    original.wav
    protected.wav

如果后续需要数据库，可以先用 SQLite：

files
tasks
task_events
task_results
6. 下一步前端优先级

当前不规划后端实现。下一步如果继续做前端，建议只围绕展示稳定性和契约清晰度推进：

1. 清理页面中仍然写死的 mock-task-001 入口，区分演示入口和真实任务入口。
2. 将结果页 artifacts 展示从 fallback 文案逐步改成后端字段优先、缺失时清晰显示“未生成”。
3. 将导出按钮在 mock/backend 不可用时统一成明确禁用态或明确 toast，不让用户误以为文件已经真实生成。
4. 将 API 状态卡片从“可切换假状态”改成只读展示当前 VITE_DATA_MODE 与 VITE_API_BASE_URL。
5. 继续保持 pnpm build 通过，并用缺字段结果数据验证“无”值展示。
