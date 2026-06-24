import type { AudioFileMeta } from './audio'
import type {
  CloneVoiceRequest,
  CloneVoiceResult,
  HistoryTask,
  ProtectionTaskRequest,
  TaskResult,
  TaskStatusResponse,
} from './task'

export interface ProtectionTaskCreated {
  taskId: string
  status: TaskStatusResponse['status']
}

export interface ApiClient {
  uploadFile(file: File): Promise<AudioFileMeta>
  createProtectionTask(payload: ProtectionTaskRequest): Promise<ProtectionTaskCreated>
  getTaskStatus(taskId: string): Promise<TaskStatusResponse>
  getTaskResult(taskId: string): Promise<TaskResult>
  cloneVoice(taskId: string, payload: CloneVoiceRequest): Promise<CloneVoiceResult>
  listTasks(): Promise<HistoryTask[]>
  deleteTask(taskId: string): Promise<void>
  downloadProtectedAudio(taskId: string): Promise<{ blob: Blob; filename: string }>
  exportReport(taskId: string): Promise<Blob>
  exportCsv(taskId: string): Promise<Blob>
  downloadEvidenceZip(taskId: string): Promise<Blob>
}
