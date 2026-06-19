import axios from 'axios'
import { apiBaseUrl } from '@/config/runtime'
import type { ApiClient } from '@/types/api'
import type { AudioFileMeta } from '@/types/audio'
import type { HistoryTask, ProtectionTaskRequest, TaskResult, TaskStatusResponse } from '@/types/task'

const http = axios.create({
  baseURL: apiBaseUrl,
  timeout: 15000,
})

function filenameFromDisposition(header: string | undefined, fallback: string) {
  if (!header) return fallback
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(header)
  return match ? decodeURIComponent(match[1]) : fallback
}

export const backendClient: ApiClient = {
  async uploadFile(file: File): Promise<AudioFileMeta> {
    const form = new FormData()
    form.append('file', file)
    const response = await http.post<AudioFileMeta>('/api/files/upload', form)
    return response.data
  },
  async createProtectionTask(payload: ProtectionTaskRequest) {
    const response = await http.post('/api/tasks/protect', payload)
    return response.data
  },
  async getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
    const response = await http.get(`/api/tasks/${taskId}`)
    return response.data
  },
  async getTaskResult(taskId: string): Promise<TaskResult> {
    const response = await http.get(`/api/tasks/${taskId}/result`)
    return response.data
  },
  async listTasks(): Promise<HistoryTask[]> {
    const response = await http.get('/api/tasks')
    return response.data
  },
  async deleteTask(taskId: string): Promise<void> {
    await http.delete(`/api/tasks/${taskId}`)
  },
  async downloadProtectedAudio(taskId: string) {
    const response = await http.get(`/api/tasks/${taskId}/download/protected-audio`, {
      responseType: 'blob',
    })
    return {
      blob: response.data,
      filename: filenameFromDisposition(response.headers['content-disposition'], 'protected_voice.wav'),
    }
  },
  async exportReport(taskId: string): Promise<Blob> {
    const response = await http.post('/api/reports/export', { taskId }, { responseType: 'blob' })
    return response.data
  },
  async exportCsv(taskId: string): Promise<Blob> {
    const response = await http.get(`/api/tasks/${taskId}/export/csv`, { responseType: 'blob' })
    return response.data
  },
  async downloadEvidenceZip(taskId: string): Promise<Blob> {
    const response = await http.get(`/api/tasks/${taskId}/download/evidence`, { responseType: 'blob' })
    return response.data
  },
}
