# Speech Clone Protector Frontend

React + TypeScript + Vite frontend for the SemE2E speech protection backend.

The frontend is backend-only. It does not include a mock client or mock task data.
All task creation, status polling, result loading,
ASR evaluation, clone evaluation, exports, and history operations call the API
configured by `VITE_API_BASE_URL`.

## Requirements

- Node.js 20+
- pnpm
- Running SemE2E backend, normally started from the repository root with
  `start.ps1` on Windows or `start.sh` on Unix-like systems.

## Environment

Create `fro/.env` from `fro/.env.example` if needed:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

Do not commit `.env` or `.api_key`; both are ignored by git.

## Commands

```bash
pnpm install
pnpm dev
pnpm build
```

## Main Routes

- `/`: project entry page.
- `/workspace`: upload or record audio, configure protection, submit real backend task.
- `/results/:taskId`: audio comparison, ASR test, clone test, metrics, details, exports.
- `/history`: backend task history separated by protection, ASR, and clone views.

## Backend Contract

The frontend expects these endpoints:

- `GET /api/capabilities`
- `POST /api/files/upload`
- `POST /api/tasks/protect`
- `GET /api/tasks/{taskId}/status`
- `GET /api/tasks/{taskId}/result`
- `GET /api/tasks/{taskId}/details`
- `GET /api/tasks/{taskId}/events`
- `POST /api/tasks/{taskId}/asr-eval`
- `POST /api/tasks/{taskId}/clone-voice`
- `GET /api/tasks`
- `DELETE /api/tasks/{taskId}`
- `GET /api/tasks/{taskId}/download/protected-audio`
- `GET /api/tasks/{taskId}/export/report`
- `GET /api/tasks/{taskId}/export/csv`
- `GET /api/tasks/{taskId}/download/evidence`

Unsupported or unavailable metrics must be returned as `null` or explicit
unavailable status by the backend. The frontend should not fabricate scores,
PESQ, SIM, MOS, trend lines, radar values, or progress.

## Portability Notes

- API base URL is environment-driven through `VITE_API_BASE_URL`.
- No local absolute user path is required by frontend source code.
- Browser recording requires HTTPS or localhost because of MediaRecorder.
