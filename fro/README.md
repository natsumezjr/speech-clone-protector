# 语音克隆防护平台 V2.0

面向“全国大学生信息安全竞赛作品赛”评委展示的 Web 前端原型，用于演示发布前源头防护式语音克隆风险降低流程。平台展示原始音频接入、语义防护、音色防护、心理声学约束、结果评估和保护音频下载的完整闭环。

## 技术栈

- React + TypeScript + Vite
- pnpm
- Tailwind CSS 与 shadcn/ui 风格组件
- lucide-react
- @tanstack/react-query
- zustand
- react-hook-form + zod
- axios
- wavesurfer.js 依赖预留，当前波形为可替换的前端 mock 组件
- ECharts / echarts-for-react

## 安装与运行

```bash
pnpm install
pnpm dev
pnpm build
```

## 环境变量

复制 `.env.example` 为 `.env` 后配置：

```bash
VITE_DATA_MODE=mock
VITE_API_BASE_URL=http://localhost:8000
```

`VITE_DATA_MODE` 只允许 `mock` 或 `backend`。非法值会在前端启动时报错。

## Mock / Backend 模式

- `mock` 模式完全使用前端固定模拟数据，不调用任何后端 API。
- `backend` 模式完全使用后端 API，不会自动 fallback 到 mock。
- 两种模式互斥，避免一次任务中混用演示数据和真实后端数据。
- 当前阶段不接入任务状态轮询，也不接入 SSE。任务创建成功后直接进入结果页，后续进度机制会统一设计。

## 页面说明

- `/`：首页，展示项目定位、系统流程、三类核心策略、KPI 与作品亮点。
- `/workspace`：防护工作台，支持音频上传/录音接入、策略配置与任务创建；当前不做轮询或定时进度推进。
- `/results/:taskId`：结果分析，展示音频对比、ASR 转写 diff、六项 ASR 指标、声纹分析、心理声学曲线、综合趋势和导出动作。
- `/history`：历史任务，支持搜索、状态筛选、模式筛选、查看结果、下载保护音频和删除接口预留。

## API 对接说明

统一入口在 `src/services/apiClient.ts`。该文件根据 `src/config/runtime.ts` 的 `dataMode` 选择：

- `src/services/mockClient.ts`
- `src/services/backendClient.ts`

后端预留接口包括：

- `POST /api/files/upload`
- `POST /api/tasks/protect`
- `GET /api/tasks/{taskId}`
- `GET /api/tasks/{taskId}/events`，SSE 预留
- `GET /api/tasks/{taskId}/result`
- `GET /api/tasks/{taskId}/download/protected-audio`
- `GET /api/tasks`
- `DELETE /api/tasks/{taskId}`
- `POST /api/reports/export`
- `GET /api/tasks/{taskId}/export/csv`
- `GET /api/tasks/{taskId}/download/evidence`

当前 API 进度：

- `POST /api/files/upload`：backend 模式下上传音频时调用；mock 模式只保存本地预览状态。
- `POST /api/tasks/protect`：已保留调用；创建成功后直接跳转 `/results/:taskId`。
- `GET /api/tasks/{taskId}`：接口契约保留，但当前前端不轮询、不定时查询。
- `GET /api/tasks/{taskId}/events`：SSE 仅文档预留，当前未接入。
- `GET /api/tasks/{taskId}/result`：结果页读取任务结果，ASR 指标按字段优先、可计算 fallback、否则显示“无”。
- `POST /api/reports/export`、`GET /api/tasks/{taskId}/export/csv`、`GET /api/tasks/{taskId}/download/evidence`：按钮和 client 预留，当前不生成真实 PDF / CSV / ZIP。

结果页 ASR 区域展示六项指标：

- `WER（词错率）`
- `CER（字错率）`
- `Token 错误率`
- `SD（语义漂移）`
- `IR（插入率）`
- `DR（删除率）`

## 已实现功能

- Vite + React + TypeScript 项目结构。
- 深色竞赛展示风格 UI 与响应式布局。
- 四个页面完整路由。
- mock/backend 双 client 与严格运行时模式。
- Zustand 保存数据模式、上传文件、当前任务、结果和历史任务。
- React Query 管理结果和历史任务请求。
- react-hook-form + zod 校验防护策略参数。
- ECharts 图表：趋势图、雷达图、心理声学阈值曲线。
- 浏览器录音输入已接入前端闭环：MediaRecorder 录音、生成 File、本地预览，并在 backend 模式下复用上传链路。
- 结果页 ASR 转写对比区域已展示 6 项指标；缺字段时安全显示“无”，CER / IR / DR 可基于文本 diff 做 fallback。
- Mock 模式生成合法 WAV Blob，并支持结果页与历史页下载 `protected_voice_mock.wav`。
- PDF / CSV / ZIP 导出按钮保留接口并给出 toast 提示。

## 实现边界

前端不实现真实语音防护算法，不调用真实 ASR / TTS / LLM，不包含登录系统，也不提供 Python 后端。当前不实现任务状态轮询、SSE、历史任务持久化、PDF / CSV / ZIP 文件生成。页面中的语义漂移、声纹相似度和心理声学指标均为前端演示数据，用于说明系统闭环和评估证据链。
