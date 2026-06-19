1. 当前前后端模式约定

前端通过环境变量选择运行模式：

VITE_DATA_MODE=mock
VITE_API_BASE_URL=http://localhost:8000

VITE_DATA_MODE 只允许 mock 或 backend；README 明确说明 mock 模式完全使用前端固定模拟数据，backend 模式完全使用后端 API，不会自动 fallback 到 mock，两者互斥。

后端默认 Base URL 是：

http://localhost:8000

api.md 中也明确写了 Base URL 和接口列表。

2. 前后端接口总表

当前 api.md 已列出这些后端预留接口：文件上传、创建防护任务、查询任务状态、查询任务结果、下载保护音频、历史任务、删除任务、导出报告、导出 CSV、下载证据包，以及 SSE 事件流预留。

接口	方法	前端用途	当前状态
/api/files/upload	POST	上传音频文件，获得 fileId 和音频元信息	后端预留，前端已写调用
/api/tasks/protect	POST	创建语音防护任务	后端预留，前端已写调用
/api/tasks/{taskId}	GET	查询任务状态	后端预留，前端已写调用
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

前端上传组件限制单文件不超过 200MB，并展示支持 .wav / .mp3 / .flac / .m4a；mock 模式下只展示文件接入状态，分析结果仍来自固定 mock 数据。

响应示例
{
  "fileId": "file_20240601_0001",
  "filename": "target_speech_demo.wav",
  "durationSec": 12.34,
  "sampleRate": 16000,
  "channels": 1,
  "bitDepth": 16,
  "sizeBytes": 1880000,
  "format": "WAV",
  "audioUrl": "http://localhost:8000/static/audio/file_20240601_0001.wav",
  "uploadedAt": "2024-06-01 14:21:36",
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
  "fileId": "file_20240601_0001",
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
  "taskId": "task_20240601_142136",
  "status": "queued"
}
3.3 查询任务状态
请求
GET /api/tasks/{taskId}
响应 JSON 示例
{
  "taskId": "task_20240601_142136",
  "status": "running",
  "progress": 0.62,
  "stage": "perturbation_optimization",
  "message": "正在进行扰动优化",
  "createdAt": "2024-06-01 14:21:36",
  "updatedAt": "2024-06-01 14:22:41",
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

3.4 SSE 任务事件流
请求
GET /api/tasks/{taskId}/events
Accept: text/event-stream

这个接口目前只在 api.md 中作为“预留接口”出现。 当前前端主要还是用定时推进 / 查询状态的方式，尚未正式接入 SSE。

SSE 事件示例
event: task_progress
data: {"taskId":"task_20240601_142136","status":"running","progress":0.35,"stage":"encoder_loading","message":"正在加载语义编码器与音色编码器"}

event: task_progress
data: {"taskId":"task_20240601_142136","status":"running","progress":0.72,"stage":"psychoacoustic_constraint","message":"正在进行心理声学约束优化"}

event: task_completed
data: {"taskId":"task_20240601_142136","status":"completed","progress":1,"stage":"report_generation","message":"任务完成"}
3.5 查询任务结果
请求
GET /api/tasks/{taskId}/result
响应 JSON 示例
{
  "taskId": "task_20240601_142136",
  "status": "completed",
  "mode": "joint",
  "dataMode": "backend",
  "verdict": "防护有效",
  "score": 92.6,
  "completedAt": "2024-06-01 14:23:18",
  "elapsedSec": 72,
  "originalAudio": {
    "fileId": "file_20240601_0001",
    "filename": "target_speech_demo.wav",
    "durationSec": 12.34,
    "sampleRate": 16000,
    "channels": 1,
    "bitDepth": 16,
    "sizeBytes": 1880000,
    "format": "WAV",
    "audioUrl": "http://localhost:8000/static/audio/file_20240601_0001.wav",
    "uploadedAt": "2024-06-01 14:21:36",
    "fingerprint": "sha256:8f21c9a4"
  },
  "protectedAudio": {
    "fileId": "protected_20240601_0001",
    "filename": "protected_voice.wav",
    "durationSec": 12.34,
    "sampleRate": 16000,
    "channels": 1,
    "bitDepth": 16,
    "sizeBytes": 1900000,
    "format": "WAV",
    "audioUrl": "http://localhost:8000/static/audio/protected_20240601_0001.wav",
    "uploadedAt": "2024-06-01 14:23:18",
    "fingerprint": "sha256:42ae19d0"
  },
  "asr": {
    "originalText": "今天天气很好，我们一起去公园散步吧。沿着湖边走，你可以看到很多漂亮的花，微风吹过来，感觉非常舒服。我们找个地方坐下，聊聊最近的生活和工作，放松一下心情。",
    "protectedText": "今天石头很蓝，我们一路去公元散不唬。船长胡边走，你可以买到很多漂多的画，未分叫过来，甘觉非等似醒。我们转个地放坐下，聊聊最没的生高和工件，放松一下先青。",
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

当前 mock 数据中已经有这些字段：asr、speaker、quality、charts.psychoacoustic、charts.trend、charts.radarBefore、charts.radarAfter。 后端需要尽量对齐这个结构，否则结果页组件会缺字段。

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
    "taskId": "task_20240601_142136",
    "filename": "target_speech_demo.wav",
    "protectedFilename": "protected_voice.wav",
    "mode": "joint",
    "dataMode": "backend",
    "status": "completed",
    "wer": 0.687,
    "simDropRate": 0.862,
    "pesq": 3.67,
    "createdAt": "2024-06-01 14:21:36"
  },
  {
    "taskId": "task_20240602_091703",
    "filename": "campus_interview.wav",
    "protectedFilename": "protected_voice.wav",
    "mode": "strong",
    "dataMode": "backend",
    "status": "completed",
    "wer": 0.714,
    "simDropRate": 0.881,
    "pesq": 3.44,
    "createdAt": "2024-06-02 09:17:03"
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
  "taskId": "task_20240601_142136",
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
  "taskId": "task_20240601_142136"
}
响应
Content-Type: application/pdf
Content-Disposition: attachment; filename="voice_protection_report_task_20240601_142136.pdf"

返回 PDF Blob。

当前状态

按钮存在，backendClient 已经按 POST /api/reports/export 调用；但是 README 说明 PDF / CSV / ZIP 导出按钮目前是接口预留并给 toast 提示。

3.10 导出详细数据 CSV
请求
GET /api/tasks/{taskId}/export/csv
响应
Content-Type: text/csv
Content-Disposition: attachment; filename="task_20240601_142136_metrics.csv"

CSV 示例内容：

metric,value
wer,0.687
cer,0.541
tokenChangeRate,0.729
semanticDrift,0.81
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
Content-Disposition: attachment; filename="task_20240601_142136_evidence.zip"

建议 ZIP 包结构：

task_20240601_142136_evidence/
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

4. 后端最小实现建议

如果现在要补一个最小 FastAPI 后端，建议先实现这 5 个接口即可：

POST /api/files/upload
POST /api/tasks/protect
GET /api/tasks/{taskId}
GET /api/tasks/{taskId}/result
GET /api/tasks/{taskId}/download/protected-audio

剩下的历史任务、删除、PDF、CSV、ZIP、SSE 可以第二步补。

最小后端数据流：

上传音频
→ 返回 fileId
→ 创建任务
→ 返回 taskId
→ 后端模拟/执行防护
→ 生成 protected wav
→ 返回 result JSON
→ 提供 protected-audio 下载
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

5.3 录音输入没有实现

上传面板里有“录音输入”tab，但点击后只是提示“录音输入接口预留”，页面文案也写“后续可接入浏览器录音与实时上传能力”。

未实现：

浏览器麦克风权限申请；
MediaRecorder 录音；
录音波形；
录音上传；
录音转任务。
5.4 SSE 实时进度没有实现

api.md 只把 /api/tasks/{taskId}/events 标为 SSE 事件流预留。

未实现：

前端 EventSource 接入；
后端事件推送；
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

5.7 历史任务“查看结果”存在硬编码问题

历史任务表格里，点击查看结果会固定跳转到：

/results/mock-task-001

而不是跳转到当前行的 task.taskId。

这会导致：

mock-task-002 / mock-task-003 无法查看自己的详情；
backend 模式下历史任务列表即使返回真实 taskId，也无法打开对应结果；
多任务闭环不完整。

建议修改为：

navigate(`/results/${task.taskId}`)
5.8 后端错误与任务失败细节还不够细

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
5.9 缺少真实文件持久化与任务持久化规范

当前接口定义了 fileId 和 taskId，但还没有后端存储规范。

建议后端补：

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
6. 下一步实现优先级

建议按这个顺序补：

修复历史任务查看结果硬编码
navigate('/results/mock-task-001') 改成 navigate(/results/${task.taskId})。
写最小 FastAPI 后端
先返回与 mockResult 同结构的数据。
先跑通 backend 模式闭环。
实现真实文件上传和保护音频下载
上传保存原始音频。
生成或复制一个 protected wav。
返回真实 blob。
实现任务状态轮询
queued → running → completed。
先不用 SSE。
接入真实算法
先接一个最小保护脚本。
后续再接 E2E-VGuard / T-SemAttack 分支。
最后补 PDF / CSV / ZIP / SSE
这些是展示增强，不是第一优先级。