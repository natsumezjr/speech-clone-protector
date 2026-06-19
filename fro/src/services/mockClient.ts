import { mockHistoryTasks, mockResult, mockTaskStatus } from '@/data/mockData'
import type { ApiClient } from '@/types/api'
import type { AudioFileMeta } from '@/types/audio'
import { createMockProtectedWavBlob } from '@/utils/mockWav'
import { shortHash } from '@/utils/format'

const delay = (ms = 450) => new Promise((resolve) => window.setTimeout(resolve, ms))

export const mockClient: ApiClient = {
  async uploadFile(file: File): Promise<AudioFileMeta> {
    await delay(250)
    return {
      fileId: `mock-file-${Date.now()}`,
      filename: file.name,
      durationSec: 12.34,
      sampleRate: 16000,
      channels: 1,
      bitDepth: 16,
      sizeBytes: file.size,
      format: file.name.split('.').pop()?.toUpperCase() ?? 'WAV',
      audioUrl: URL.createObjectURL(file),
      uploadedAt: new Date().toLocaleString('zh-CN', { hour12: false }),
      fingerprint: shortHash(`${file.name}-${file.size}`),
    }
  },
  async createProtectionTask() {
    await delay(350)
    return { taskId: 'mock-task-001', status: 'queued' }
  },
  async getTaskStatus() {
    await delay()
    return mockTaskStatus
  },
  async getTaskResult() {
    await delay()
    return mockResult
  },
  async listTasks() {
    await delay()
    return mockHistoryTasks
  },
  async deleteTask() {
    await delay(180)
  },
  async downloadProtectedAudio() {
    await delay(120)
    return { blob: createMockProtectedWavBlob(), filename: 'protected_voice_mock.wav' }
  },
  async exportReport() {
    throw new Error('该导出项为后端接口预留。')
  },
  async exportCsv() {
    throw new Error('该导出项为后端接口预留。')
  },
  async downloadEvidenceZip() {
    throw new Error('该导出项为后端接口预留。')
  },
}
