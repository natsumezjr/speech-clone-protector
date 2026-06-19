import { Download, FileArchive, FileDown, FileSpreadsheet, RotateCcw } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/common/Button'
import { Panel } from '@/components/common/Panel'
import { downloadEvidenceZip, downloadProtectedAudio, exportCsv, exportReport } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import { downloadBlob } from '@/utils/download'

export function ExportActions({ taskId }: { taskId: string }) {
  const navigate = useNavigate()
  const pushToast = useAppStore((state) => state.pushToast)

  const downloadAudio = async () => {
    try {
      const { blob, filename } = await downloadProtectedAudio(taskId)
      downloadBlob(blob, filename)
      pushToast({ kind: 'success', title: '保护音频已开始下载' })
    } catch (error) {
      pushToast({ kind: 'error', title: '下载失败', description: error instanceof Error ? error.message : '请检查后端接口。' })
    }
  }

  const reserved = async (type: 'pdf' | 'csv' | 'zip') => {
    try {
      if (type === 'pdf') await exportReport(taskId)
      if (type === 'csv') await exportCsv(taskId)
      if (type === 'zip') await downloadEvidenceZip(taskId)
    } catch (error) {
      pushToast({ kind: 'info', title: '后端接口预留', description: error instanceof Error ? error.message : '该导出项为后端接口预留。' })
    }
  }

  return (
    <Panel>
      <div className="flex flex-wrap gap-3">
        <Button icon={<Download className="h-4 w-4" />} onClick={downloadAudio}>
          下载保护音频
        </Button>
        <Button variant="secondary" icon={<FileDown className="h-4 w-4" />} onClick={() => void reserved('pdf')}>
          导出评估报告 PDF
        </Button>
        <Button variant="secondary" icon={<FileSpreadsheet className="h-4 w-4" />} onClick={() => void reserved('csv')}>
          导出详细数据 CSV
        </Button>
        <Button variant="secondary" icon={<FileArchive className="h-4 w-4" />} onClick={() => void reserved('zip')}>
          下载证据包 ZIP
        </Button>
        <Button variant="ghost" icon={<RotateCcw className="h-4 w-4" />} onClick={() => navigate('/workspace')}>
          重新执行任务
        </Button>
      </div>
    </Panel>
  )
}
