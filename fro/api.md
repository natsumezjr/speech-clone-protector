# API 接口文档

## 概述

Base URL: http://localhost:8000

详见以下接口定义文件:
- src/types/api.ts - ApiClient 接口
- src/types/task.ts - 任务相关类型
- src/types/audio.ts - 音频元信息类型
- src/services/backendClient.ts - 后端请求实现

## 接口列表

### POST /api/files/upload
上传音频文件 (multipart/form-data)
响应: AudioFileMeta

### POST /api/tasks/protect
创建防护任务
请求体: ProtectionTaskRequest
响应: ProtectionTaskCreated

### GET /api/tasks/{taskId}
查询任务状态
响应: TaskStatusResponse

### GET /api/tasks/{taskId}/result
查询任务结果
响应: TaskResult

### GET /api/tasks
查询历史任务列表
响应: HistoryTask[]

### DELETE /api/tasks/{taskId}
删除任务

### GET /api/tasks/{taskId}/download/protected-audio
下载保护音频 (blob)

### POST /api/reports/export
导出评估报告 PDF (blob)

### GET /api/tasks/{taskId}/export/csv
导出详细数据 CSV (blob)

### GET /api/tasks/{taskId}/download/evidence
下载证据包 ZIP (blob)

## 预留接口

### GET /api/tasks/{taskId}/events
SSE 事件流
