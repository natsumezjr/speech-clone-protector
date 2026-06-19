import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import {
  CheckCircle2,
  ClipboardList,
  Clock3,
  Copy,
  Download,
  FileArchive,
  FileAudio,
  FileJson,
  FileText,
  Info,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Volume2,
} from 'lucide-react'
import { downloadEvidenceZip, downloadProtectedAudio, exportCsv, exportReport, getTaskResult } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import type { TaskResult, TrendPoint } from '@/types/task'
import type { AudioFileMeta } from '@/types/audio'
import { downloadBlob } from '@/utils/download'
import { cn } from '@/lib/utils'
import { AudioPlayer } from '@/components/audio/AudioPlayer'
import { formatDurationSeconds, getAudioDuration, getAudioSource } from '@/utils/audio'

const statusText: Record<TaskResult['status'], string> = {
  queued: '排队中',
  running: '处理中',
  completed: '已完成',
  success: '已完成',
  failed: '失败',
  error: '失败',
}

const modeText: Record<TaskResult['mode'], string> = {
  standard: '标准保护',
  strong: '强保护',
  high_fidelity: '高保真',
  custom: '高级自定义',
  joint: '联合防护',
}

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
      <div className="grid min-h-[520px] place-items-center">
        <div className="text-center">
          <Loader2 className="mx-auto h-9 w-9 animate-spin text-cyan-300" />
          <p className="mt-4 text-slate-300">正在加载结果证据链...</p>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return <div className="ui-card p-6 text-red-100">{error instanceof Error ? error.message : '无法获取任务结果。'}</div>
  }

  return (
    <div className="-mx-5 max-w-none space-y-5 pb-6">
      <SummaryBar result={data} />

      <div className="grid grid-cols-[minmax(0,1fr)_380px] items-stretch gap-5 max-xl:grid-cols-1">
        <main className="space-y-5">
          <AudioCompare result={data} />
          <AsrCompare result={data} />
          <div className="grid grid-cols-[1.08fr_0.92fr] items-stretch gap-5 max-lg:grid-cols-1">
            <SpeakerPanel result={data} />
            <QualityPanel result={data} />
          </div>
        </main>
        <Aside result={data} />
      </div>

      <div className="grid min-h-[230px] grid-cols-[0.58fr_0.42fr] gap-5 max-lg:grid-cols-1">
        <TrendPanel result={data} />
        <InterpretationPanel result={data} />
      </div>
    </div>
  )
}

function SummaryBar({ result }: { result: TaskResult }) {
  const isEffective = result.score >= 80
  const verdict = result.verdict || (isEffective ? '防护有效' : '防护无效')

  return (
    <section className="ui-card grid h-[74px] grid-cols-[250px_180px_250px_170px_230px_minmax(260px,1fr)] items-center px-5 max-2xl:grid-cols-[1.05fr_0.78fr_1.08fr_0.75fr_1fr_1.42fr] max-xl:h-auto max-xl:grid-cols-3 max-xl:gap-y-4 max-xl:py-4">
      <SummaryItem icon={<ClipboardList />} label="任务 ID" value={result.taskId} copy />
      <SummaryItem icon={<ShieldCheck />} label="任务状态" value={statusText[result.status] ?? result.status} green={result.status === 'completed' || result.status === 'success'} />
      <SummaryItem icon={<Clock3 />} label="完成时间" value={result.completedAt} />
      <SummaryItem icon={<Clock3 />} label="处理耗时" value={formatElapsed(result.elapsedSec)} />
      <SummaryItem icon={<Sparkles />} label="防护模式" value={modeText[result.mode] ?? result.mode} green />
      <div className="flex h-full items-center justify-end gap-4 border-l border-cyan-300/10 pl-5 whitespace-nowrap">
        <span className="text-xs text-slate-500">综合判定</span>
        <ShieldCheck className={cn('h-11 w-11', isEffective ? 'text-emerald-300' : 'text-red-300')} />
        <div className="text-left">
          <p className={cn('text-[27px] font-black leading-none', isEffective ? 'text-emerald-300' : 'text-red-300')}>{verdict}</p>
          <p className="mt-1 text-xs text-slate-400">{isEffective ? '满足当前安全性阈值' : '未达到当前安全性阈值'}</p>
        </div>
        <div className="grid h-[58px] w-[58px] place-items-center rounded-full border-4 border-dashed border-emerald-400/70 text-center text-sm font-black text-emerald-300">
          {result.score.toFixed(1)}
        </div>
      </div>
    </section>
  )
}

function formatElapsed(seconds: number) {
  const hh = Math.floor(seconds / 3600)
  const mm = Math.floor((seconds % 3600) / 60)
  const ss = seconds % 60
  return [hh, mm, ss].map((value) => String(value).padStart(2, '0')).join(':')
}

function SummaryItem({ icon, label, value, green, copy }: { icon: ReactNode; label: string; value: string; green?: boolean; copy?: boolean }) {
  return (
    <div className="flex min-w-0 items-center justify-center gap-3 border-r border-cyan-300/10 px-4 whitespace-nowrap">
      <span className="text-slate-500 [&_svg]:h-4 [&_svg]:w-4">{icon}</span>
      <div className="min-w-0">
        <p className="text-xs text-slate-500">{label}</p>
        <p className={cn('mt-1 truncate text-[14px] font-bold text-slate-200', green && 'text-emerald-300')}>{value}</p>
      </div>
      {copy ? <Copy className="h-4 w-4 shrink-0 text-slate-500" /> : null}
    </div>
  )
}

function SectionTitle({ children, info }: { children: ReactNode; info?: boolean }) {
  return (
    <h2 className="flex h-6 items-center gap-2 whitespace-nowrap text-[16px] font-black leading-none text-white">
      {children}
      {info ? <Info className="h-3.5 w-3.5 text-slate-500" /> : null}
    </h2>
  )
}

function AudioCompare({ result }: { result: TaskResult }) {
  const uploadedFile = useTaskStore((state) => state.uploadedFile)
  const [protectedObjectUrl, setProtectedObjectUrl] = useState<string>()
  const originalAudio = {
    ...result.originalAudio,
    objectUrl: result.originalAudio.objectUrl ?? uploadedFile?.objectUrl,
    audioUrl: result.originalAudio.audioUrl ?? uploadedFile?.audioUrl,
  }
  const protectedAudio = { ...result.protectedAudio, objectUrl: result.protectedAudio.objectUrl ?? protectedObjectUrl }

  useEffect(() => {
    return () => {
      if (protectedObjectUrl?.startsWith('blob:')) URL.revokeObjectURL(protectedObjectUrl)
    }
  }, [protectedObjectUrl])

  useEffect(() => {
    const originalUrl = result.originalAudio.objectUrl
    const protectedUrl = result.protectedAudio.objectUrl
    return () => {
      if (originalUrl?.startsWith('blob:')) URL.revokeObjectURL(originalUrl)
      if (protectedUrl?.startsWith('blob:')) URL.revokeObjectURL(protectedUrl)
    }
  }, [result.originalAudio.objectUrl, result.protectedAudio.objectUrl])

  const loadProtectedAudio = async () => {
    if (getAudioSource(protectedAudio)) return getAudioSource(protectedAudio)
    const { blob } = await downloadProtectedAudio(result.taskId)
    const objectUrl = URL.createObjectURL(blob)
    setProtectedObjectUrl((current) => {
      if (current?.startsWith('blob:')) URL.revokeObjectURL(current)
      return objectUrl
    })
    return objectUrl
  }

  return (
    <section className="ui-card p-5">
      <SectionTitle info>原始音频 vs 保护音频</SectionTitle>
      <div className="mt-5 grid grid-cols-[1fr_58px_1fr] items-center gap-6">
        <AudioCard title="原始音频（未保护）" audio={originalAudio} color="#00aef0" />
        <div className="grid h-12 w-12 place-items-center rounded-full border border-cyan-300/28 bg-slate-950/70 text-[18px] font-black text-white shadow-[0_0_24px_rgba(56,189,248,0.12)]">VS</div>
        <AudioCard title="保护音频（已防护）" audio={protectedAudio} color="#22c55e" green onPlayRequest={loadProtectedAudio} />
      </div>
    </section>
  )
}

function AudioCard({
  title,
  audio,
  color,
  green,
  onPlayRequest,
}: {
  title: string
  audio: AudioFileMeta
  color: string
  green?: boolean
  onPlayRequest?: () => Promise<string | undefined>
}) {
  const src = getAudioSource(audio)
  const duration = getAudioDuration(audio)

  return (
    <div className={cn('flex h-[252px] flex-col rounded-[9px] border p-5', green ? 'border-emerald-400/18 bg-emerald-400/8' : 'border-cyan-300/14 bg-[#07192d]/80')}>
      <p className="flex items-center gap-2 whitespace-nowrap text-sm font-black text-slate-200">
        {green ? <ShieldCheck className="h-4 w-4 text-emerald-300" /> : <Volume2 className="h-4 w-4 text-sky-300" />}
        {title}
      </p>
      <p className="ml-6 mt-0.5 flex min-w-0 text-xs text-slate-400">
        <span className="truncate">{audio.filename.replace(/\.[^.]+$/, '')}</span>
        <span className="shrink-0">{audio.filename.match(/\.[^.]+$/)?.[0] ?? ''}</span>
      </p>
      <TinyWave color={color} className="h-[58px]" />
      <div className="mt-4 h-1 rounded-full bg-slate-700">
        <div className="h-full w-[35%] rounded-full" style={{ background: color }} />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[10px] text-slate-400">
        <span>00:00</span>
        <span>{formatDurationSeconds(duration)}</span>
      </div>
      <div className="mt-auto">
        <AudioPlayer
          src={src}
          title={title}
          filename={audio.filename}
          disabledReason={green ? '点击播放时将从后端下载保护音频' : '暂无原始音频 URL'}
          downloadable={Boolean(src)}
          downloadFilename={audio.filename}
          onPlayRequest={onPlayRequest}
        />
      </div>
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2 border-t border-cyan-300/10 pt-3 pb-1 text-[12px] text-slate-400">
        <span>时长 {duration ? `${duration.toFixed(2)}s` : '待解析'}</span>
        <span>采样率 {audio.sampleRate ? `${audio.sampleRate / 1000}kHz` : '待解析'}</span>
        <span>声道 {audio.channels ?? '待解析'}</span>
        <span>格式 {audio.format}</span>
        <span>大小 {formatFileSize(audio.sizeBytes)}</span>
      </div>
    </div>
  )
}

function AsrCompare({ result }: { result: TaskResult }) {
  const editStats = getTextEditStats(result.asr.originalText, result.asr.protectedText)
  const wer = result.asr.wer
  const cer = result.asr.cer ?? editStats.cer
  const tokenErrorRate = result.asr.tokenErrorRate ?? result.asr.tokenChangeRate
  const semanticDrift = result.asr.semanticDrift
  const insertRate = result.asr.insertRate ?? editStats.insertRate
  const deleteRate = result.asr.deleteRate ?? editStats.deleteRate

  return (
    <section className="ui-card min-h-0 p-5">
      <SectionTitle>机器理解分析（ASR 转写对比）</SectionTitle>
      <div className="mt-5 grid grid-cols-[400px_318px_minmax(0,1fr)] items-stretch gap-5 max-2xl:grid-cols-[390px_318px_minmax(0,1fr)]">
        <TextBox title="原始转写（ASR）" text={result.asr.originalText} foot="原始音频经 ASR 识别得到的参考转写文本" />
        <div className="grid grid-cols-3 content-center gap-3">
          <ScoreBox label="WER（词错率）" value={formatOptionalPercent(wer)} red />
          <ScoreBox label="CER（字错率）" value={formatOptionalPercent(cer)} red />
          <ScoreBox label="Token 错误率" value={formatOptionalPercent(tokenErrorRate)} red />
          <ScoreBox label="SD（语义漂移）" value={formatOptionalPercent(semanticDrift)} red />
          <ScoreBox label="IR（插入率）" value={formatOptionalPercent(insertRate)} red />
          <ScoreBox label="DR（删除率）" value={formatOptionalPercent(deleteRate)} red />
        </div>
        <TextBox title="保护音频转写（ASR）" text={result.asr.protectedText} foot="红色为新增内容，绿色删除线为原文缺失内容" content={buildTextDiff(result.asr.originalText, result.asr.protectedText)} />
      </div>
    </section>
  )
}

function TextBox({ title, text, foot, content }: { title: string; text: string; foot: string; content?: ReactNode }) {
  return (
    <div className="flex h-[226px] flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/18 p-4">
      <h3 className="mb-3 whitespace-nowrap text-sm font-bold text-slate-300">{title}</h3>
      <div className="min-h-[122px] overflow-y-auto rounded-[7px] border border-cyan-300/8 bg-slate-950/22 p-4 text-[13px] leading-6 text-slate-200">
        {content ?? text}
      </div>
      <p className="mt-auto truncate pt-4 text-[12px] text-slate-500">{foot}</p>
    </div>
  )
}

function getTextEditStats(original: string, next: string) {
  const a = Array.from(original)
  const b = Array.from(next)
  if (a.length === 0) {
    return {
      cer: undefined,
      insertRate: undefined,
      deleteRate: undefined,
    }
  }

  const dp = Array.from({ length: a.length + 1 }, () => Array<number>(b.length + 1).fill(0))

  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }

  let i = 0
  let j = 0
  let insertions = 0
  let deletions = 0

  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      i += 1
      j += 1
    } else if (j < b.length && (i === a.length || dp[i][j + 1] >= dp[i + 1]?.[j])) {
      insertions += 1
      j += 1
    } else if (i < a.length) {
      deletions += 1
      i += 1
    }
  }

  const base = a.length

  return {
    cer: (insertions + deletions) / base,
    insertRate: insertions / base,
    deleteRate: deletions / base,
  }
}

function buildTextDiff(original: string, next: string) {
  const a = Array.from(original)
  const b = Array.from(next)
  const dp = Array.from({ length: a.length + 1 }, () => Array<number>(b.length + 1).fill(0))

  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }

  const nodes: ReactNode[] = []
  let i = 0
  let j = 0
  let key = 0

  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      nodes.push(a[i])
      i += 1
      j += 1
    } else if (j < b.length && (i === a.length || dp[i][j + 1] >= dp[i + 1]?.[j])) {
      nodes.push(
        <span key={`ins-${key}`} className="text-red-300">
          {b[j]}
        </span>,
      )
      j += 1
      key += 1
    } else if (i < a.length) {
      nodes.push(
        <span key={`del-${key}`} className="text-emerald-300 line-through decoration-emerald-300/70">
          {a[i]}
        </span>,
      )
      i += 1
      key += 1
    }
  }

  return nodes
}

function ScoreBox({ label, value, red }: { label: string; value: string; red?: boolean }) {
  return (
    <div className="h-[82px] rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-3 text-center">
      <p className="whitespace-nowrap text-[11px] leading-4 text-slate-400">{label}</p>
      <div className="mt-2 grid justify-items-center">
        <span className={cn('text-[24px] font-black leading-none', red ? 'text-red-300' : 'text-cyan-300')}>
          {value}
        </span>
      </div>
    </div>
  )
}

function SpeakerPanel({ result }: { result: TaskResult }) {
  const simDropRate = result.speaker.simDropRate || relativeDrop(result.speaker.simBefore, result.speaker.simAfter)
  const embeddingIncrease = relativeIncrease(result.speaker.embeddingDistanceBefore, result.speaker.embeddingDistanceAfter)

  return (
    <section className="ui-card h-full overflow-x-auto p-5">
      <div className="mb-5 border-b border-cyan-300/10 pb-3">
        <SectionTitle>Feature / 声学特征分析</SectionTitle>
      </div>

      <div className="grid grid-cols-[minmax(230px,1fr)_220px_minmax(230px,1fr)] items-stretch gap-5">
        <FeatureStatCard
          title="Feature 相似度（越低越好）"
          before={result.speaker.simBefore.toFixed(3)}
          after={result.speaker.simAfter.toFixed(3)}
          delta={`↓ ${formatPercent(simDropRate)}`}
          foot="计算方法：x-vector 余弦相似度"
          tone="green"
        />

        <div className="flex min-w-[220px] flex-col items-center justify-start rounded-[9px] border border-cyan-300/12 bg-slate-950/12 px-3 py-3">
          <h3 className="mb-2 whitespace-nowrap text-sm font-bold text-slate-300">
            声学特征分布对比
          </h3>
          <RadarChart before={result.charts.radarBefore} after={result.charts.radarAfter} />
        </div>

        <FeatureStatCard
          title="Embedding 距离（越大越好）"
          before={result.speaker.embeddingDistanceBefore.toFixed(3)}
          after={result.speaker.embeddingDistanceAfter.toFixed(3)}
          delta={`↑ ${formatPercent(embeddingIncrease)}`}
          foot="计算方法：ECAPA-TDNN 向量距离"
          tone="red"
        />
      </div>
    </section>
  )
}

function FeatureStatCard({ title, before, after, delta, foot, tone }: { title: string; before: string; after: string; delta: string; foot: string; tone: 'green' | 'red' }) {
  return (
    <div className="h-[168px] rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-4">
      <h3 className="whitespace-nowrap text-[13px] font-bold leading-5 text-slate-300">{title}</h3>
      <div className="mt-3 grid grid-cols-[1fr_28px_1fr] items-center text-center text-[20px]">
        <span className="text-slate-200">{before}</span>
        <span className="text-slate-400">→</span>
        <span className="text-emerald-300">{after}</span>
      </div>
      <div className={cn('mt-2 rounded-[5px] py-2 text-center font-black', tone === 'green' ? 'bg-emerald-400/14 text-emerald-300' : 'bg-red-400/12 text-red-300')}>{delta}</div>
      <p className="mt-2 truncate text-[11px] leading-4 text-slate-500">{foot}</p>
    </div>
  )
}

function QualityPanel({ result }: { result: TaskResult }) {
  return (
    <section className="ui-card h-full overflow-hidden p-5">
      <SectionTitle>感知质量评估</SectionTitle>
      <div className="mt-5 grid grid-cols-3 gap-3">
        <QualityMetric label="SNR（信噪比）" value={`${result.quality.snr.toFixed(1)} dB`} tag="优秀" tone="green" />
        <QualityMetric label="PESQ" value={result.quality.pesq.toFixed(2)} tag="良好" tone="blue" />
        <QualityMetric label="听感保真（MOS-LQO）" value={`${result.quality.mosLqo.toFixed(2)} / 5`} tag="良好" tone="orange" />
      </div>
      <div className="mt-5 h-[126px] overflow-hidden rounded-[9px] border border-cyan-300/12 bg-slate-950/16 px-4 py-3">
        <div className="mb-1 flex items-center justify-between">
          <h3 className="whitespace-nowrap text-[13px] font-bold text-slate-300">心理声学阈值分析（关键频段）</h3>
          <div className="flex gap-4 text-[10px] text-slate-400">
            <span className="text-emerald-300">— 掩蔽阈值</span>
            <span className="text-lime-300">— 防护扰动谱</span>
          </div>
        </div>
        <LineChart result={result} />
      </div>
    </section>
  )
}

function QualityMetric({ label, value, tag, tone }: { label: string; value: string; tag: string; tone: 'green' | 'blue' | 'orange' }) {
  return (
    <div className="h-[86px] rounded-[9px] border border-cyan-300/12 bg-slate-950/16 p-3 text-center">
      <p className="whitespace-nowrap text-[11px] text-slate-400">{label}</p>
      <p className={cn('mt-1 text-[20px] font-black leading-none', tone === 'green' && 'text-emerald-300', tone === 'blue' && 'text-cyan-300', tone === 'orange' && 'text-orange-300')}>{value}</p>
      <span className={cn('mt-1.5 inline-block rounded px-3 py-0.5 text-[11px] font-bold', tone === 'green' && 'bg-emerald-400/14 text-emerald-300', tone === 'blue' && 'bg-cyan-400/14 text-cyan-300', tone === 'orange' && 'bg-orange-400/14 text-orange-300')}>{tag}</span>
    </div>
  )
}

function TrendPanel({ result }: { result: TaskResult }) {
  const cards = [
    ['ASR 干扰（WER %）', formatOptionalPercent(result.asr.wer), trendDelta(result.charts.trend, 'wer'), 'blue', 'wer'],
    ['Feature 相似度', result.speaker.simAfter.toFixed(3), trendDelta(result.charts.trend, 'sim'), 'green', 'sim'],
    ['听感质量（MOS-LQO）', result.quality.mosLqo.toFixed(2), trendDelta(result.charts.trend, 'mos'), 'orange', 'mos'],
    ['PESQ', result.quality.pesq.toFixed(1), trendDelta(result.charts.trend, 'pesq'), 'purple', 'pesq'],
    ['总耗时（秒）', `${result.elapsedSec}s`, trendDelta(result.charts.trend, 'elapsed'), 'blue', 'elapsed'],
  ] as const

  return (
    <section className="ui-card h-full overflow-hidden p-3.5">
      <SectionTitle>综合指标趋势</SectionTitle>
      <div className="mt-3 grid grid-cols-5 gap-3">
        {cards.map(([title, value, delta, tone, metric]) => (
          <div key={title} className="h-[118px] min-w-0 rounded-[7px] border border-cyan-300/12 bg-slate-950/18 p-2">
            <p className="truncate whitespace-nowrap text-[12px] text-slate-400">{title}</p>
            <div className="mt-1 flex items-end justify-between">
              <p className={cn('text-[22px] font-black leading-none', tone === 'green' && 'text-emerald-300', tone === 'orange' && 'text-orange-300', tone === 'purple' && 'text-violet-300', tone === 'blue' && 'text-sky-300')}>{value}</p>
              <p className={cn('text-[11px] font-bold', delta.startsWith('↑') ? 'text-red-300' : 'text-cyan-300')}>{delta}</p>
            </div>
            <MiniChart points={result.charts.trend} metric={metric} tone={tone} />
          </div>
        ))}
      </div>
    </section>
  )
}

function InterpretationPanel({ result }: { result: TaskResult }) {
  const simDropRate = result.speaker.simDropRate || relativeDrop(result.speaker.simBefore, result.speaker.simAfter)
  const verdict = result.verdict || (result.score >= 80 ? '防护有效' : '防护无效')
  const editStats = getTextEditStats(result.asr.originalText, result.asr.protectedText)
  const cer = result.asr.cer ?? editStats.cer
  const tokenErrorRate = result.asr.tokenErrorRate ?? result.asr.tokenChangeRate
  const items = [
    `语义层面：WER ${formatOptionalPercent(result.asr.wer)}，CER ${formatOptionalPercent(cer)}，Token 错误率 ${formatOptionalPercent(tokenErrorRate)}。`,
    `Feature 层面：相似度从 ${result.speaker.simBefore.toFixed(3)} 降至 ${result.speaker.simAfter.toFixed(3)}（↓${formatPercent(simDropRate)}）。`,
    `听感层面：PESQ=${result.quality.pesq.toFixed(2)}，MOS-LQO=${result.quality.mosLqo.toFixed(2)}，整体听感保持良好，满足可用性要求。`,
    `综合结论：综合评分 ${result.score.toFixed(1)}，判定为「${verdict}」。`,
  ]

  return (
    <section className="ui-card h-full overflow-hidden p-3.5">
      <SectionTitle>
        结果解读 <span className="text-sm font-normal text-slate-500">（自动生成）</span>
      </SectionTitle>
      <div className="mt-3 grid grid-cols-2 gap-x-5 gap-y-2 rounded-[7px] border border-cyan-300/10 bg-slate-950/12 p-3 text-[12px] leading-5 text-slate-300">
        {items.map((item) => (
          <p key={item} className="flex min-w-0 gap-2">
            <CheckCircle2 className="mt-1 h-3.5 w-3.5 shrink-0 text-emerald-300" />
            <span className="line-clamp-2">{item}</span>
          </p>
        ))}
      </div>
      <p className="mt-2 text-right text-[11px] text-slate-500">以上分析基于系统自动评估，仅供评审参考。</p>
    </section>
  )
}

function Aside({ result }: { result: TaskResult }) {
  const navigate = useNavigate()
  const pushToast = useAppStore((state) => state.pushToast)
  const artifacts = result.artifacts?.length
    ? result.artifacts
    : [
        { label: '原始音频', filename: result.originalAudio.filename, sizeBytes: result.originalAudio.sizeBytes },
        { label: '保护音频', filename: result.protectedAudio.filename, sizeBytes: result.protectedAudio.sizeBytes },
        { label: 'ASR 转写', filename: 'asr_comparison.json' },
        { label: 'Feature 分析', filename: 'speaker_metrics.json' },
        { label: '质量报告', filename: 'quality_metrics.json' },
      ]
  const evidenceSizeBytes = artifacts.reduce((sum, artifact) => sum + (artifact.sizeBytes ?? 0), 0)
  const taskInfo = [
    ['提交时间', result.submittedAt ?? result.createdAt ?? result.originalAudio.uploadedAt ?? '-'],
    ['输入来源', result.inputSource ?? '手动上传'],
    ['音频时长', formatDurationSeconds(getAudioDuration(result.originalAudio))],
    ['语言类型', result.language ?? '未标注'],
    ['处理模型', result.processingModel ?? result.asrModel ?? modeText[result.mode] ?? result.mode],
    ['优化目标', result.optimizationTarget ?? result.mode],
  ]

  const runDownload = async (kind: 'audio' | 'report' | 'csv' | 'zip') => {
    try {
      let blob: Blob
      let filename: string

      if (kind === 'audio') {
        const file = await downloadProtectedAudio(result.taskId)
        blob = file.blob
        filename = file.filename
      } else if (kind === 'report') {
        blob = await exportReport(result.taskId)
        filename = `${result.taskId}-report.pdf`
      } else if (kind === 'csv') {
        blob = await exportCsv(result.taskId)
        filename = `${result.taskId}-metrics.csv`
      } else {
        blob = await downloadEvidenceZip(result.taskId)
        filename = `${result.taskId}-evidence.zip`
      }

      downloadBlob(blob, filename)
      pushToast({ kind: 'success', title: '下载已开始', description: filename })
    } catch (error) {
      pushToast({ kind: 'error', title: '导出暂不可用', description: error instanceof Error ? error.message : '请稍后重试。' })
    }
  }

  return (
    <aside className="grid h-full grid-rows-[328px_318px_minmax(260px,1fr)] gap-5 max-xl:grid-rows-none">
      <div className="ui-card px-5 py-6">
        <h2 className="mb-5 text-[17px] font-black text-white">任务信息</h2>
        {taskInfo.map(([label, value]) => (
          <p key={label} className="mb-3 grid grid-cols-[78px_1fr] text-[13px] leading-5">
            <span className="text-slate-500">{label}</span>
            <span className="truncate font-semibold text-slate-300">{value}</span>
          </p>
        ))}
        <p className="mt-2 grid grid-cols-[78px_1fr] text-[13px] leading-5">
          <span className="text-slate-500">参数配置</span>
          <button className="text-left font-bold text-cyan-300">查看详情 ›</button>
        </p>
      </div>

      <div className="ui-card p-5">
        <h2 className="mb-5 text-[18px] font-black text-white">操作与导出</h2>
        <button onClick={() => void runDownload('audio')} className="cyan-button flex h-12 w-full items-center justify-center gap-2 rounded-[8px] text-[16px] font-black">
          <Download className="h-4 w-4" />
          下载保护音频
        </button>
        {['导出评估报告（PDF）', '导出详细数据（CSV）', '重新执行任务'].map((item, index) => (
          <button
            key={item}
            onClick={() => {
              if (index === 0) void runDownload('report')
              if (index === 1) void runDownload('csv')
              if (index === 2) navigate('/workspace')
            }}
            className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-[8px] border border-cyan-300/12 bg-white/[0.035] text-[16px] font-bold text-slate-300"
          >
            {index === 0 ? <FileText className="h-4 w-4" /> : index === 1 ? <FileJson className="h-4 w-4" /> : <RefreshCw className="h-4 w-4" />}
            {item}
          </button>
        ))}
      </div>

      <div className="ui-card p-5">
        <h2 className="mb-4 text-[19px] font-black text-white">
          证据包 <span className="text-xs font-normal text-slate-500">（可下载）</span>
        </h2>
        {artifacts.map((artifact) => {
          const Icon = artifact.label.includes('音频') ? FileAudio : FileJson
          const size = artifact.sizeLabel ?? (artifact.sizeBytes ? formatFileSize(artifact.sizeBytes) : '-')
          return (
          <p key={artifact.filename} className="mb-3 grid min-w-0 grid-cols-[20px_72px_minmax(0,1fr)_72px] items-center gap-3 text-[13px] leading-5 text-slate-400">
            <Icon className="h-5 w-5 shrink-0" />
            <span className="whitespace-nowrap">{artifact.label}</span>
            <span className="block min-w-0 truncate" title={artifact.filename}>{artifact.filename}</span>
            <span className="justify-self-end whitespace-nowrap text-slate-300">{size}</span>
          </p>
          )
        })}
        <button onClick={() => void runDownload('zip')} className="mt-4 flex h-11 w-full items-center gap-3 rounded-[8px] border border-cyan-300/12 bg-cyan-400/8 px-3 text-[16px] font-black text-cyan-300">
          <FileArchive className="h-4 w-4" />
          <span>全部下载（ZIP）</span>
          <span className="ml-auto text-xs text-slate-300">{evidenceSizeBytes ? formatFileSize(evidenceSizeBytes) : '后端生成'}</span>
        </button>
      </div>
    </aside>
  )
}

function formatFileSize(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)}MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${bytes}B`
}

function formatPercent(value: number) {
  return `${(value <= 1 ? value * 100 : value).toFixed(1)}%`
}

function formatOptionalPercent(value: unknown) {
  const numberValue = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numberValue)) return '无'
  return `${(numberValue <= 1 ? numberValue * 100 : numberValue).toFixed(1)}%`
}

function relativeDrop(before: number, after: number) {
  if (!Number.isFinite(before) || before === 0) return 0
  return Math.max(0, (before - after) / Math.abs(before))
}

function relativeIncrease(before: number, after: number) {
  if (!Number.isFinite(before) || before === 0) return 0
  return Math.max(0, (after - before) / Math.abs(before))
}

function trendDelta(points: TrendPoint[], metric: keyof TrendPoint) {
  if (points.length < 2) return ''
  const first = Number(points[0][metric])
  const last = Number(points[points.length - 1][metric])
  if (!Number.isFinite(first) || !Number.isFinite(last) || first === 0) return ''
  const change = (last - first) / Math.abs(first)
  const arrow = change >= 0 ? '↑' : '↓'
  return `${arrow} ${formatPercent(Math.abs(change))}`
}

function TinyWave({ color, className }: { color: string; className?: string }) {
  return (
    <svg viewBox="0 0 520 90" className={cn('h-full w-full', className)} preserveAspectRatio="none">
      <line x1="0" x2="520" y1="45" y2="45" stroke={color} strokeOpacity="0.18" />
      {Array.from({ length: 120 }, (_, index) => {
        const height = 5 + Math.abs(Math.sin(index * 0.52) * 33) + (index % 7) * 2.8
        return <rect key={index} x={index * 4.3} y={(90 - height) / 2} width="1.8" height={height} rx="1" fill={color} opacity={0.33 + (index % 4) * 0.13} />
      })}
    </svg>
  )
}

function RadarChart({ before, after }: { before: number[]; after: number[] }) {
  const labels = ['音色', '基频', '共振峰', '节奏', '能量', '谱包络']

  return (
    <svg
      viewBox="0 0 220 170"
      className="mx-auto h-[154px] w-[208px] overflow-visible"
      preserveAspectRatio="xMidYMid meet"
    >
      <g transform="translate(110 76)">
        {[58, 42, 26].map((r) => (
          <polygon
            key={r}
            points={radarPolygon(r, 6)}
            fill="none"
            stroke="rgba(56,189,248,.22)"
            strokeWidth="1"
          />
        ))}

        {Array.from({ length: 6 }, (_, index) => {
          const a = -Math.PI / 2 + (index * Math.PI * 2) / 6
          return (
            <line
              key={index}
              x1="0"
              y1="0"
              x2={Math.cos(a) * 58}
              y2={Math.sin(a) * 58}
              stroke="rgba(56,189,248,.16)"
              strokeWidth="1"
            />
          )
        })}

        <polygon
          points={radarPolygon(56, 6, before)}
          fill="rgba(14,165,233,.22)"
          stroke="#38bdf8"
          strokeWidth="2"
        />

        <polygon
          points={radarPolygon(56, 6, after)}
          fill="rgba(34,197,94,.28)"
          stroke="#22c55e"
          strokeWidth="2"
        />

        {labels.map((label, index) => {
          const a = -Math.PI / 2 + (index * Math.PI * 2) / 6
          const x = Math.cos(a) * 76
          const y = Math.sin(a) * 70 + 4

          return (
            <text
              key={label}
              x={x}
              y={y}
              textAnchor="middle"
              fontSize="11"
              fontWeight="800"
              fill="#c7d2fe"
            >
              {label}
            </text>
          )
        })}

        <g transform="translate(-58 72)" fontSize="10" fill="#94a3b8">
          <line x1="0" x2="18" y1="0" y2="0" stroke="#38bdf8" strokeWidth="2" />
          <text x="24" y="4">防护前</text>

          <line x1="76" x2="94" y1="0" y2="0" stroke="#22c55e" strokeWidth="2" />
          <text x="100" y="4">防护后</text>
        </g>
      </g>
    </svg>
  )
}

function radarPolygon(radius: number, sides: number, scale?: number[]) {
  return Array.from({ length: sides }, (_, i) => {
    const s = scale?.[i] ?? 1
    const a = -Math.PI / 2 + (i * Math.PI * 2) / sides
    return `${Math.cos(a) * radius * s},${Math.sin(a) * radius * s}`
  }).join(' ')
}

function LineChart({ result }: { result: TaskResult }) {
  const points = result.charts.psychoacoustic
  const width = 430
  const height = 58
  const max = Math.max(...points.flatMap((p) => [p.maskingThreshold, p.perturbation]), 1)
  const min = Math.min(...points.flatMap((p) => [p.maskingThreshold, p.perturbation]), 0)
  const span = Math.max(1, max - min)
  const toPoints = (key: 'maskingThreshold' | 'perturbation') =>
    points
      .map((point, index) => {
        const x = (index / Math.max(1, points.length - 1)) * width
        const y = height - 6 - ((point[key] - min) / span) * (height - 12)
        return `${x.toFixed(1)},${y.toFixed(1)}`
      })
      .join(' ')

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[58px] w-full overflow-hidden">
      {[10, 28, 46].map((y) => (
        <line key={y} x1="0" x2={width} y1={y} y2={y} stroke="rgba(148,163,184,.13)" />
      ))}
      <polyline points={toPoints('maskingThreshold')} fill="none" stroke="#22c55e" strokeWidth="2" />
      <polyline points={toPoints('perturbation')} fill="none" stroke="#86efac" strokeDasharray="6 5" strokeWidth="2" />
      {points.filter((_, index) => index % 4 === 0).map((point, index) => (
        <text key={point.frequency} x={index * 4 * (width / Math.max(1, points.length - 1))} y="56" fontSize="9" fill="#64748b">
          {point.frequency >= 1000 ? `${Math.round(point.frequency / 1000)}k` : point.frequency}
        </text>
      ))}
    </svg>
  )
}

function MiniChart({ points, metric, tone }: { points: TrendPoint[]; metric: keyof TrendPoint; tone: string }) {
  const color = tone === 'green' ? '#22c55e' : tone === 'orange' ? '#fb923c' : tone === 'purple' ? '#a78bfa' : '#38bdf8'
  const values = points.map((point) => Number(point[metric]))
  const width = 130
  const height = 54
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = Math.max(1, max - min)
  const path = values.map((value, index) => `${(index / Math.max(1, values.length - 1)) * width},${height - 7 - ((value - min) / span) * (height - 14)}`).join(' ')

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="mt-2 h-[50px] w-full overflow-hidden">
      {[14, 30, 46].map((y) => (
        <line key={y} x1="0" x2={width} y1={y} y2={y} stroke="rgba(148,163,184,.11)" />
      ))}
      <polyline points={`0,${height - 7} ${path} ${width},${height - 7}`} fill={`${color}20`} stroke="none" />
      <polyline points={path} fill="none" stroke={color} strokeWidth="2" />
    </svg>
  )
}
