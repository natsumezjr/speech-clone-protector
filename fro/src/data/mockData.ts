import type { HistoryTask, TaskResult, TaskStatusResponse } from '@/types/task'

const originalText =
  '今天天气很好，我们一起去公园散步吧。沿着湖边走，你可以看到很多漂亮的花，微风吹过来，感觉非常舒服。我们找个地方坐下，聊聊最近的生活和工作，放松一下心情。'

const protectedText =
  '今天石头很硬，我们一躺去公元跳呀。船长胡边走，你可以买到很多漂多的面，未分叫过来，甘戈羊等似腿。我们转个地放坐下，聊聊最没的生离和工作，放松一下先青。'

export const mockTaskStatus: TaskStatusResponse = {
  taskId: 'mock-task-001',
  status: 'completed',
  progress: 1,
  stage: 'report_generation',
  message: '报告生成完成',
  createdAt: '2026-06-01 14:21:36',
  updatedAt: '2026-06-01 14:23:18',
  error: null,
}

export const mockResult: TaskResult = {
  taskId: 'mock-task-001',
  status: 'completed',
  mode: 'joint',
  dataMode: 'mock',
  verdict: '防护有效',
  score: 92.6,
  completedAt: '2026-06-01 14:23:18',
  elapsedSec: 72,
  originalAudio: {
    filename: 'target_speech_demo.wav',
    durationSec: 12.34,
    sampleRate: 16000,
    channels: 1,
    bitDepth: 16,
    sizeBytes: 1880000,
    format: 'WAV',
    fingerprint: 'sha256:8f21c9a4',
  },
  protectedAudio: {
    filename: 'protected_timbre_20260601.wav',
    durationSec: 12.34,
    sampleRate: 16000,
    channels: 1,
    bitDepth: 16,
    sizeBytes: 1900000,
    format: 'WAV',
    fingerprint: 'sha256:42ae19d0',
  },
  asr: {
    originalText,
    protectedText,
    wer: 0.687,
    cer: 0.541,
    tokenChangeRate: 0.729,
    semanticDrift: 0.81,
    insertRate: 0.236,
    deleteRate: 0.184,
  },
  speaker: {
    simBefore: 0.912,
    simAfter: 0.126,
    simDropRate: 0.862,
    embeddingDistanceBefore: 0.214,
    embeddingDistanceAfter: 1.387,
  },
  quality: {
    snr: 21.8,
    pesq: 3.67,
    mosLqo: 3.82,
  },
  charts: {
    psychoacoustic: [
      [80, 24, 10],
      [160, 31, 14],
      [315, 36, 18],
      [630, 42, 26],
      [1000, 47, 33],
      [2000, 45, 32],
      [4000, 39, 27],
      [8000, 32, 20],
      [12000, 26, 14],
    ].map(([frequency, maskingThreshold, perturbation]) => ({
      frequency,
      maskingThreshold,
      perturbation,
    })),
    trend: Array.from({ length: 20 }, (_, index) => {
      const step = index + 1
      return {
        step,
        wer: 0.18 + step * 0.026 + Math.sin(step / 2) * 0.012,
        sim: 0.91 - step * 0.039 - Math.sin(step / 3) * 0.01,
        mos: 4.25 - step * 0.021 + Math.cos(step / 3) * 0.025,
        pesq: 4.08 - step * 0.019 + Math.sin(step / 4) * 0.018,
        elapsed: 7 + step * 3.25,
      }
    }),
    radarBefore: [0.92, 0.87, 0.82, 0.78, 0.84],
    radarAfter: [0.14, 0.22, 0.24, 0.31, 0.28],
  },
}

export const mockHistoryTasks: HistoryTask[] = [
  {
    taskId: 'mock-task-001',
    filename: 'target_speech_demo.wav',
    protectedFilename: 'protected_timbre_20260601.wav',
    mode: 'joint',
    dataMode: 'mock',
    status: 'completed',
    wer: 0.687,
    simDropRate: 0.862,
    pesq: 3.67,
    createdAt: '2026-06-01 14:21:36',
  },
  {
    taskId: 'mock-task-002',
    filename: 'campus_interview.wav',
    protectedFilename: 'protected_voice_mock.wav',
    mode: 'strong',
    dataMode: 'mock',
    status: 'completed',
    wer: 0.714,
    simDropRate: 0.881,
    pesq: 3.44,
    createdAt: '2026-06-02 09:17:03',
  },
  {
    taskId: 'mock-task-003',
    filename: 'lecture_clip.flac',
    protectedFilename: 'protected_voice_mock.wav',
    mode: 'high_fidelity',
    dataMode: 'mock',
    status: 'running',
    wer: 0.493,
    simDropRate: 0.721,
    pesq: 4.02,
    createdAt: '2026-06-03 16:08:44',
  },
  {
    taskId: 'mock-task-004',
    filename: 'phone_call_sample.m4a',
    protectedFilename: 'protected_voice_mock.wav',
    mode: 'standard',
    dataMode: 'mock',
    status: 'failed',
    wer: 0,
    simDropRate: 0,
    pesq: 0,
    createdAt: '2026-06-04 11:30:12',
  },
]
