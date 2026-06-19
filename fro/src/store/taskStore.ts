import { create } from 'zustand'
import type { HistoryTask, TaskResult, TaskStatusResponse, UploadedFileState } from '@/types/task'

interface TaskState {
  uploadedFile: UploadedFileState | null
  currentTaskStatus: TaskStatusResponse | null
  currentTaskResult: TaskResult | null
  historyTasks: HistoryTask[]
  setUploadedFile: (file: UploadedFileState | null) => void
  setCurrentTaskStatus: (status: TaskStatusResponse | null) => void
  setCurrentTaskResult: (result: TaskResult | null) => void
  setHistoryTasks: (tasks: HistoryTask[]) => void
}

export const useTaskStore = create<TaskState>((set) => ({
  uploadedFile: null,
  currentTaskStatus: null,
  currentTaskResult: null,
  historyTasks: [],
  setUploadedFile: (uploadedFile) => set({ uploadedFile }),
  setCurrentTaskStatus: (currentTaskStatus) => set({ currentTaskStatus }),
  setCurrentTaskResult: (currentTaskResult) => set({ currentTaskResult }),
  setHistoryTasks: (historyTasks) => set({ historyTasks }),
}))
