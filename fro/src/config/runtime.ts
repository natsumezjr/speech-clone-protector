export type DataMode = 'mock' | 'backend'

const rawMode = import.meta.env.VITE_DATA_MODE ?? 'mock'

if (rawMode !== 'mock' && rawMode !== 'backend') {
  throw new Error(`VITE_DATA_MODE 只允许 mock 或 backend，当前值为 ${rawMode}`)
}

export const dataMode: DataMode = rawMode
export const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
export const isMockMode = dataMode === 'mock'
export const isBackendMode = dataMode === 'backend'
