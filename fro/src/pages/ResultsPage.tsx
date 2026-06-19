import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { AudioCompareCard } from '@/components/audio/AudioCompareCard'
import { MetricCard } from '@/components/cards/MetricCard'
import { EvidenceCard } from '@/components/cards/EvidenceCard'
import { TaskSummaryStrip } from '@/components/cards/TaskSummaryStrip'
import { TrendChart } from '@/components/charts/TrendChart'
import { Spinner } from '@/components/common/Spinner'
import { PageHeader } from '@/components/layout/PageHeader'
import { AsrDiffPanel } from '@/components/results/AsrDiffPanel'
import { ExportActions } from '@/components/results/ExportActions'
import { QualityPanel } from '@/components/results/QualityPanel'
import { ResultInterpretation } from '@/components/results/ResultInterpretation'
import { SpeakerAnalysisPanel } from '@/components/results/SpeakerAnalysisPanel'
import { getTaskResult } from '@/services/apiClient'
import { useTaskStore } from '@/store/taskStore'
import { percent } from '@/utils/format'

export function ResultsPage() {
  const { taskId = 'mock-task-001' } = useParams()
  const setCurrentTaskResult = useTaskStore((state) => state.setCurrentTaskResult)
  const { data, isLoading, error } = useQuery({
    queryKey: ['task-result', taskId],
    queryFn: async () => {
      const result = await getTaskResult(taskId)
      setCurrentTaskResult(result)
      return result
    },
  })

  if (isLoading) {
    return (
      <div className="grid min-h-[420px] place-items-center">
        <div className="text-center">
          <Spinner className="mx-auto h-8 w-8" />
          <p className="mt-4 text-slate-300">正在加载结果证据链...</p>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div>
        <PageHeader title="结果分析" description="结果加载失败。Backend 模式下请确认后端 API 已启动，前端不会自动 fallback 到 Mock 数据。" />
        <div className="rounded-2xl border border-red-400/25 bg-red-950/30 p-6 text-red-100">
          {error instanceof Error ? error.message : '无法获取任务结果。'}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Result Evidence"
        title="结果分析"
        description="以 ASR 转写、语义漂移、声纹相似度和心理声学指标形成可解释的评估证据链。"
      />
      <TaskSummaryStrip result={data} />
      <div className="grid gap-5 lg:grid-cols-2">
        <AudioCompareCard title="原始音频" audio={data.originalAudio} variant="cyan" description="原始录音，包含清晰语义内容与可克隆声纹特征。" />
        <AudioCompareCard title="保护音频" audio={data.protectedAudio} variant="green" description="防护后音频，语义受保护，声纹相似度显著降低，听感基本保持。" />
      </div>
      <div className="grid gap-4 md:grid-cols-4">
        <MetricCard label="WER" value={percent(data.asr.wer)} />
        <MetricCard label="Token 变化率" value={percent(data.asr.tokenChangeRate)} tone="blue" />
        <MetricCard label="声纹相似度下降" value={percent(data.speaker.simDropRate)} tone="green" />
        <MetricCard label="MOS-LQO" value={data.quality.mosLqo.toFixed(2)} tone="orange" />
      </div>
      <AsrDiffPanel asr={data.asr} />
      <SpeakerAnalysisPanel result={data} />
      <QualityPanel result={data} />
      <EvidenceCard title="综合指标趋势">
        <TrendChart data={data.charts.trend} />
      </EvidenceCard>
      <ResultInterpretation result={data} />
      <ExportActions taskId={data.taskId} />
    </div>
  )
}
