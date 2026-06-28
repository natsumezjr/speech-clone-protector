import type { ApiClient } from '@/types/api'
import { backendClient } from './backendClient'

export const apiClient: ApiClient = backendClient

export const getCapabilities = apiClient.getCapabilities
export const uploadFile = apiClient.uploadFile
export const createProtectionTask = apiClient.createProtectionTask
export const getTaskStatus = apiClient.getTaskStatus
export const getTaskResult = apiClient.getTaskResult
export const getPsychoacousticSlice = apiClient.getPsychoacousticSlice
export const getTaskDetails = apiClient.getTaskDetails
export const runAsrEval = apiClient.runAsrEval
export const cloneVoice = apiClient.cloneVoice
export const listTasks = apiClient.listTasks
export const deleteTask = apiClient.deleteTask
export const downloadProtectedAudio = apiClient.downloadProtectedAudio
export const exportReport = apiClient.exportReport
export const exportCsv = apiClient.exportCsv
export const downloadEvidenceZip = apiClient.downloadEvidenceZip
