import { dataMode } from '@/config/runtime'
import type { ApiClient } from '@/types/api'
import { backendClient } from './backendClient'
import { mockClient } from './mockClient'

export const apiClient: ApiClient = dataMode === 'mock' ? mockClient : backendClient

export const uploadFile = apiClient.uploadFile
export const createProtectionTask = apiClient.createProtectionTask
export const getTaskStatus = apiClient.getTaskStatus
export const getTaskResult = apiClient.getTaskResult
export const listTasks = apiClient.listTasks
export const deleteTask = apiClient.deleteTask
export const downloadProtectedAudio = apiClient.downloadProtectedAudio
export const exportReport = apiClient.exportReport
export const exportCsv = apiClient.exportCsv
export const downloadEvidenceZip = apiClient.downloadEvidenceZip
