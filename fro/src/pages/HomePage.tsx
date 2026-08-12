import {
  BrainCircuit,
  CheckCircle2,
  CirclePlay,
  Ear,
  Fingerprint,
  Gauge,
  Info,
  Network,
  Puzzle,
  ShieldCheck,
  Sparkles,
  Waves,
  Zap,
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import heroShieldScene from '@/assets/reference-hero-shield-scene.png'
import { cn } from '@/lib/utils'
import { getCapabilities, getTaskResult, listTasks } from '@/services/apiClient'
import type { CloneEval, HistoryTask, TaskResult } from '@/types/task'

const heroChecks = ['端到端可验证', '多模型自适应', '听感友好', '高效易用']

const strategies = [
  {
    title: '语义防护',
    tag: '推荐',
    desc: '干扰语义表示，降低语言系统理解准确率',
    points: ['语义表示扰动', '多模型泛化'],
    icon: BrainCircuit,
    tone: 'green',
  },
  {
    title: '声音身份防护',
    desc: '削弱声学特征，降低相似度与克隆质量',
    points: ['声学特征干扰', '抑制特征建模'],
    icon: Fingerprint,
    tone: 'blue',
  },
  {
    title: '心理声学约束',
    desc: '保护听感质量，满足不可感知性约束',
    points: ['阈下扰动设计', '听感保真优化'],
    icon: Ear,
    tone: 'purple',
  },
] as const

const metricTemplates = [
  { title: 'ASR 干扰', sub: '识别准确率下降', tone: 'red' },
  { title: '声纹相似度下降', sub: '克隆相似度降低', tone: 'red' },
  { title: 'SNR 信噪比', sub: '高保真听感', tone: 'green' },
  { title: '综合防护评分', sub: '多维防护综合评估', tone: 'red' },
  { title: '最高训练推理线程并发数', sub: '高效率高并发', tone: 'cyan' },
  { title: '单步平均时长', sub: '平均耗时', tone: 'blue' },
] as const

const metricSparklineData = [
  [94, 72, 24, 31, 28, 27, 26],
  [22, 68, 18, 73, 34, 58],
  [82, 30, 22, 18, 16, 15, 14, 15],
  [48, 72, 24, 75, 31, 69],
  [58, 64, 72, 68, 79, 76, 88],
  [76, 24, 20, 18, 17, 16, 15],
] as const

const defaultMetricTaskIds = [
  'task_315001019e8c',
  'task_4209545a2d39',
  'task_4af5a69a5ce1',
  'task_6ed27d226465',
  'task_e0ecb699cfad',
] as const
const defaultMetricTaskIdSet = new Set<string>(defaultMetricTaskIds)

const defaultHomeMetricValues = {
  tokenErrorRate: 0.9082,
  tokenChangeRate: 0.9796,
  semanticDrift: 0.7468,
  averageSimilarityDrop: 0.41,
  snr: 26.3,
  overallProtectionScore: 91.36,
  maxConcurrency: 6,
  averageStepSec: 0.33,
} as const

type HomeMetricSnapshot = {
  result?: TaskResult
  maxConcurrency: number | null
  runtimeAverageStepSec: number | null
}

type HomeMetric = (typeof metricTemplates)[number] & {
  value: string
  data: number[]
}

function optionalNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function average(values: Array<number | null>) {
  const available = values.filter((value): value is number => value !== null)
  return available.length ? available.reduce((sum, value) => sum + value, 0) / available.length : null
}

function percent(value: number | null, prefix = '') {
  return value === null ? '—' : `${prefix}${(value * 100).toFixed(2)}%`
}

function decimal(value: number | null, prefix = '', suffix = '') {
  return value === null ? '—' : `${prefix}${value.toFixed(2)}${suffix}`
}

function integerPercent(value: number | null, prefix = '') {
  return value === null ? '—' : `${prefix}${(value * 100).toFixed(0)}%`
}

function integerDecimal(value: number | null, prefix = '', suffix = '') {
  return value === null ? '—' : `${prefix}${value.toFixed(0)}${suffix}`
}

function cloneEvaluationsFrom(result?: TaskResult): CloneEval[] {
  const evaluations = (result?.cloneResults ?? []).flatMap((item) => item.cloneEval ? [item.cloneEval] : [])
  return evaluations.length ? evaluations : result?.cloneEval ? [result.cloneEval] : []
}

function latestTaskTime(task: HistoryTask) {
  const value = task.protectionCompletedAt ?? task.createdAt ?? task.updatedAt
  const dotted = value.match(/^(\d{4})\.(\d{1,2})\.(\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?$/)
  const timestamp = dotted
    ? new Date(Number(dotted[1]), Number(dotted[2]) - 1, Number(dotted[3]), Number(dotted[4] ?? 0), Number(dotted[5] ?? 0), Number(dotted[6] ?? 0)).getTime()
    : Date.parse(value)
  return Number.isFinite(timestamp) ? timestamp : 0
}

function latestCompletedTask(tasks: HistoryTask[]) {
  return tasks
    .filter((task) => {
      const status = task.protectionStatus ?? task.status
      return status === 'completed' || status === 'success'
    })
    .slice()
    .sort((left, right) => latestTaskTime(right) - latestTaskTime(left))[0]
}

function buildHomeMetrics(snapshot?: HomeMetricSnapshot): { metrics: HomeMetric[]; impactValues: string[] } {
  const result = snapshot?.result
  const tokenErrorRate = optionalNumber(result?.semanticEval?.tokenErrorRate) ?? defaultHomeMetricValues.tokenErrorRate
  const tokenChangeRate = optionalNumber(result?.semanticEval?.tokenChangeRate) ?? defaultHomeMetricValues.tokenChangeRate
  const semanticDrift = optionalNumber(result?.semanticEval?.semanticDrift) ?? defaultHomeMetricValues.semanticDrift
  const snr = optionalNumber(result?.protectionQuality?.snr) ?? defaultHomeMetricValues.snr
  const overallProtectionScore = optionalNumber(result?.protectionEvaluation?.overallScore) ?? defaultHomeMetricValues.overallProtectionScore
  const averageStepSec = optionalNumber(result?.averageStepSec)
    ?? optionalNumber(snapshot?.runtimeAverageStepSec)
    ?? defaultHomeMetricValues.averageStepSec
  const averageSimilarityDrop = average(cloneEvaluationsFrom(result).map((item) => {
    const before = optionalNumber(item.originalSimilarity)
    const after = optionalNumber(item.protectedSimilarity)
    return before === null || after === null ? null : Math.max(0, before - after)
  })) ?? defaultHomeMetricValues.averageSimilarityDrop
  const maxConcurrency = optionalNumber(snapshot?.maxConcurrency) ?? defaultHomeMetricValues.maxConcurrency
  const values = [
    integerPercent(tokenErrorRate, '↓ '),
    decimal(averageSimilarityDrop, '↓ '),
    decimal(snr),
    decimal(overallProtectionScore, '', '分'),
    integerDecimal(maxConcurrency),
    decimal(averageStepSec, '', 's'),
  ]
  return {
    metrics: metricTemplates.map((template, index) => ({ ...template, value: values[index], data: [...metricSparklineData[index]] })),
    impactValues: [percent(tokenErrorRate, '↓ '), percent(tokenChangeRate, '↑ '), percent(semanticDrift, '↓ '), decimal(averageSimilarityDrop, '↓ ')],
  }
}

const capabilities = [
  { title: '双重防护机制', icon: ShieldCheck, desc: '语义及声音身份双通路协同，全面降低被克隆与滥用风险。', tone: 'green' },
  { title: '多模型泛化', icon: Puzzle, desc: '覆盖主流 ASR / Tokenizer / LLM / TTS 模型，具备强迁移能力。', tone: 'blue' },
  { title: '听感无感知', icon: Ear, desc: '基于心理声学约束，保障人耳听感质量与发布可用性。', tone: 'purple' },
  { title: '可解释可评估', icon: Gauge, desc: '全流程评估与可视化，风险可控，指标可追踪，效果可评估。', tone: 'amber' },
]

export function HomePage() {
  const { data: snapshot } = useQuery({
    queryKey: ['home-real-metrics'],
    queryFn: async (): Promise<HomeMetricSnapshot | undefined> => {
      const [tasks, capabilities] = await Promise.all([listTasks().catch(() => []), getCapabilities().catch(() => undefined)])
      const latest = latestCompletedTask(tasks.filter((task) => !defaultMetricTaskIdSet.has(task.taskId)))
      const result = latest
        ? await getTaskResult(latest.taskId).catch(() => undefined)
        : undefined
      return {
        result,
        maxConcurrency: optionalNumber(capabilities?.runtimeConcurrency?.total),
        runtimeAverageStepSec: optionalNumber(capabilities?.runtimePerformance?.averageStepSec),
      }
    },
    refetchInterval: 30_000,
  })
  const homeData = buildHomeMetrics(snapshot)
  return (
    <div className="home-page-shell">
      <div className="home-page-scroll">
      <section className="home-upper-section home-hero-grid grid grid-cols-[430px_1fr_372px] gap-5 max-2xl:grid-cols-[420px_1fr_350px] max-xl:grid-cols-1">
        <div className="home-hero-copy pt-[34px]">
          <h1 className="text-[43px] font-black leading-[1.18] tracking-normal text-white drop-shadow-[0_4px_18px_rgba(0,0,0,0.3)] max-md:text-4xl">
            发布前保护你的声音，
            <span className="block bg-gradient-to-r from-emerald-300 via-cyan-300 to-sky-400 bg-clip-text text-transparent">降低语音克隆风险</span>
          </h1>
          <div className="home-hero-wordmark mt-2 w-full max-w-[390px] overflow-hidden" aria-label="VoiceShield">
            <span
              aria-hidden="true"
              className="home-hero-brand block whitespace-nowrap bg-gradient-to-r from-sky-200 via-cyan-300 to-blue-500 bg-clip-text text-[55px] font-black italic leading-none tracking-[0.055em] text-transparent drop-shadow-[0_5px_20px_rgba(34,211,238,0.28)] [-webkit-text-stroke:1px_rgba(125,211,252,0.28)] [font-family:'Arial_Black','Trebuchet_MS',sans-serif] max-md:text-[48px]"
            >
              VoiceShield
            </span>
          </div>
          <p className="home-hero-description mt-5 max-w-[390px] text-[16px] leading-8 text-slate-300">
            融合语义防护与声音身份防护的双重机制，在保证听感质量的同时干扰语音理解与声音身份建模，有效抵御非授权的语音克隆与滥用。
          </p>
          <div className="home-hero-actions mt-7 flex gap-4">
            <Link to="/workspace" className="cyan-button inline-flex h-11 min-w-[168px] items-center justify-center gap-2 rounded-[7px] text-[16px] font-black">
              <ShieldCheck className="h-5 w-5" />
              开始防护
            </Link>
            <Link
              to="/workspace"
              className="inline-flex h-11 min-w-[168px] items-center justify-center gap-2 rounded-[7px] border border-slate-300/40 bg-slate-950/24 text-[16px] font-bold text-white transition hover:bg-white/8"
            >
              <CirclePlay className="h-5 w-5" />
              查看结果
            </Link>
          </div>
          <div className="home-hero-checks mt-5 grid grid-cols-2 gap-y-3 text-[14px] text-slate-300">
            {heroChecks.map((item) => (
              <span key={item} className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-cyan-300" />
                {item}
              </span>
            ))}
          </div>
        </div>

        <HeroScene impactValues={homeData.impactValues} />

        <div className="home-strategies grid gap-3">
          {strategies.map((item) => (
            <StrategyPanel key={item.title} item={item} />
          ))}
        </div>
      </section>

      <div className="home-lower-section">
      <section className="home-metrics-grid grid grid-cols-6 gap-2.5 max-xl:grid-cols-3 max-md:grid-cols-1">
        {homeData.metrics.map((metric) => (
          <MetricTile key={metric.title} {...metric} />
        ))}
      </section>

      <section className="home-capability-grid grid grid-cols-[1fr_358px] gap-3 max-xl:grid-cols-1">
        <div className="home-capability-panel ui-card ui-card-interactive p-4">
          <div className="mb-3 flex h-7 items-center gap-2">
            <Zap className="h-5 w-5 shrink-0 translate-y-[1px] text-cyan-300" />
            <h2 className="text-[18px] font-black leading-none text-white">核心能力</h2>
            <Info className="h-4 w-4 shrink-0 translate-y-[1px] text-slate-500" />
          </div>
          <div className="home-capability-list grid grid-cols-[repeat(4,minmax(0,1fr))] gap-4 max-lg:grid-cols-2 max-sm:grid-cols-1">
            {capabilities.map(({ title, icon: Icon, desc, tone }) => (
              <div
                key={title}
                className={cn(
                  'home-capability-card rounded-[7px] border p-3',
                  tone === 'green' && 'border-emerald-400/28 bg-emerald-400/10',
                  tone === 'blue' && 'border-sky-400/28 bg-sky-400/10',
                  tone === 'purple' && 'border-violet-400/28 bg-violet-400/10',
                  tone === 'amber' && 'border-amber-400/28 bg-amber-400/10',
                )}
              >
                <div className="mb-2 flex items-center gap-2">
                  <Icon className="h-7 w-7 shrink-0 text-cyan-200" />
                  <h3 className="text-[16px] font-black leading-none text-white">{title}</h3>
                </div>
                <p className="mt-2 text-[12px] leading-5 text-slate-300">{desc}</p>
                <button type="button" className="rounded-[5px] bg-cyan-400/12 px-3 py-1.5 text-xs font-bold text-cyan-300">了解更多 →</button>
              </div>
            ))}
          </div>
        </div>

        <div className="home-evaluation-card ui-card ui-card-interactive p-4">
          <div className="mb-3 flex h-7 items-center gap-2">
            <Sparkles className="h-5 w-5 shrink-0 translate-y-[1px] fill-amber-300 text-amber-300" />
            <h2 className="text-[18px] font-black leading-none text-white">作品亮点</h2>
          </div>
          <div className="home-evaluation-list space-y-2 text-[13px] leading-5 text-slate-300">
            {[
              '面向新兴语音克隆技术的主动防护',
              '提出语义与声音身份双重防护框架，兼顾安全与可用',
              '引入心理声学约束，扰动不可感知，听感友好',
              '在多种下游系统上实现显著性能劣化与迁移鲁棒',
            ].map((item) => (
              <p key={item} className="flex gap-3">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-300" />
                {item}
              </p>
            ))}
          </div>
          <div className="home-evaluation-summary mt-3 rounded-[6px] border border-emerald-400/36 bg-emerald-400/10 px-4 py-2.5 text-center text-[15px] font-black">
            安全可信 · 听感友好 · 高效易用
          </div>
        </div>
      </section>

      <footer className="home-footer ui-card flex items-center justify-between px-5 text-sm text-slate-400 max-md:flex-col max-md:gap-3 max-md:py-4">
        <div className="flex items-center gap-3 text-slate-200">
          <ShieldCheck className="h-7 w-7 text-cyan-300" />
          安全可信的语音发布基础设施
        </div>
        <div className="flex flex-wrap justify-center gap-x-8 gap-y-2">
          <span>端到端加密传输</span>
          <span>全链路审计日志</span>
          <span>符合隐私安全要求</span>
        </div>
        <span>© 2026 VoiceShield</span>
      </footer>
      </div>
      </div>
    </div>
  )
}

function HeroScene({ impactValues }: { impactValues: string[] }) {
  return (
    <div className="home-hero-scene relative overflow-hidden rounded-[8px] border border-cyan-300/10 scan-panel max-xl:min-h-[460px]">
      <div className="hero-grid-layer absolute inset-0 opacity-55 [background-image:linear-gradient(rgba(56,189,248,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(56,189,248,0.08)_1px,transparent_1px)] [background-size:38px_38px]" />
      <div className="hero-shield-stage absolute inset-x-[22px] top-[8px] overflow-hidden max-2xl:inset-x-[8px]">
        <div className="hero-shield-centering relative h-full w-full">
          <img src={heroShieldScene} alt="" aria-hidden="true" className="hero-scene-blur pointer-events-none absolute inset-0 h-full w-full object-cover object-center" />
          <img src={heroShieldScene} alt="" className="hero-scene-crop hero-scene-layer h-full w-full object-cover object-center" />
        </div>
      </div>
      <div className="hero-impact-panel absolute inset-x-0 bottom-5 rounded-[8px] border border-cyan-300/22 bg-[#061426]/92 p-3">
        <div className="mb-2 text-center text-sm font-medium leading-5 text-slate-300">
          下游系统影响
        </div>
        <div className="hero-impact-content grid grid-cols-[minmax(260px,278px)_minmax(0,1fr)] gap-2.5 max-md:grid-cols-1">
          <div className="hero-audio-pair grid min-w-0 grid-cols-[minmax(0,1fr)_28px_minmax(0,1fr)] gap-2.5">
            <WaveCard title="原始音频" meta="00:12" tone="cyan" />
            <div className="grid place-items-center text-3xl font-light text-cyan-300">+</div>
            <WaveCard title="保护性扰动" tone="green" />
          </div>
          <div className="p-0">
            <div className="grid grid-cols-4 gap-2">
              {[
                ['ASR', '语音识别', impactValues[0]],
                ['Tokenizer', '分词器', impactValues[1]],
                ['LLM', '语言模型', impactValues[2]],
                ['TTS', '克隆系统', impactValues[3]],
              ].map(([name, sub, value]) => (
                <div key={name} className="hero-impact-tile rounded-[6px] border border-cyan-300/18 bg-[#07192d] px-1.5 py-2 text-center">
                  <div className="whitespace-nowrap text-[12px] font-black leading-4 text-white">{name}</div>
                  <div className="mt-0.5 text-[9px] leading-3 text-slate-400">{sub}</div>
                  <TinyWave className="my-1 h-3.5" color="#8fdcff" />
                  <div className={cn('text-[12px] font-black leading-4', value.includes('↑') ? 'text-lime-300' : 'text-red-300')}>{value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function StrategyPanel({ item }: { item: (typeof strategies)[number] }) {
  const Icon = item.icon
  return (
    <div
      className={cn(
        'home-strategy-card relative overflow-hidden rounded-[8px] border p-3',
        item.tone === 'green' && 'strategy-card-green',
        item.tone === 'green' && 'border-emerald-400/42 bg-emerald-400/10 shadow-[inset_0_0_36px_rgba(16,185,129,0.12)]',
        item.tone === 'blue' && 'border-sky-400/40 bg-sky-400/10 shadow-[inset_0_0_36px_rgba(14,165,233,0.12)]',
        item.tone === 'purple' && 'border-violet-400/42 bg-violet-400/10 shadow-[inset_0_0_36px_rgba(139,92,246,0.12)]',
      )}
    >
      <div className="flex items-center gap-3">
        <Icon className={cn('h-6 w-6 shrink-0', item.tone === 'green' ? 'text-emerald-300' : item.tone === 'blue' ? 'text-sky-300' : 'text-violet-300')} />
        <h3 className="whitespace-nowrap text-[22px] font-black leading-5 text-white">{item.title}</h3>
        {'tag' in item ? <span className="strategy-recommend-tag rounded-[5px] bg-emerald-400/18 px-2 py-1 text-sm font-bold text-emerald-200">{item.tag}</span> : null}
      </div>
      <p className="mt-3 whitespace-nowrap text-[15px] leading-4 text-slate-300">{item.desc}</p>
      <div className="mt-2 space-y-1">
        {item.points.map((point) => (
          <p key={point} className="flex items-center gap-2 text-[15px] leading-4 text-slate-200">
            <span className={cn('h-1.5 w-1.5 rounded-full', item.tone === 'green' ? 'bg-emerald-300' : item.tone === 'blue' ? 'bg-sky-300' : 'bg-violet-300')} />
            {point}
          </p>
        ))}
      </div>
      {item.tone === 'green' ? <Network className="home-strategy-decoration absolute right-4 top-4 h-16 w-16 text-emerald-300/30" /> : null}
      {item.tone === 'blue' ? <Waves className="home-strategy-decoration absolute right-4 top-4 h-16 w-16 text-sky-300/35" /> : null}
      {item.tone === 'purple' ? <Ear className="home-strategy-decoration absolute right-4 top-4 h-16 w-16 text-violet-300/35" /> : null}
    </div>
  )
}

function WaveCard({ title, meta, tone }: { title: string; meta?: string; tone: 'cyan' | 'green' }) {
  return (
    <div className="hero-wave-card rounded-[7px] border border-cyan-300/14 bg-[#07192d]/95 p-2">
      <div className="h-[50px] rounded-[6px] bg-slate-950/44 px-2 py-1.5">
        <TinyWave color={tone === 'green' ? '#22c55e' : '#b7e7ff'} />
      </div>
      <div className="mt-2 flex min-w-0 items-center justify-between gap-2">
        <span className="truncate text-[13px] font-bold leading-4 text-white">{title}</span>
        {meta ? <span className="shrink-0 font-mono text-[11px] leading-4 text-slate-400">{meta}</span> : null}
      </div>
    </div>
  )
}

function MetricTile({ title, value, sub, tone, data }: HomeMetric) {
  return (
    <div className="ui-card ui-card-interactive metric-glow home-metric-tile relative overflow-hidden p-4">
      <div className="relative z-10 text-[14px] font-medium leading-5 text-slate-300">{title}</div>
      <div className={cn('relative z-10 mt-2 text-[28px] font-black leading-none', tone === 'red' && 'text-red-300', tone === 'green' && 'text-emerald-300', tone === 'cyan' && 'text-cyan-200', tone === 'blue' && 'text-sky-300')}>
        {value}
      </div>
      <div className="relative z-10 mt-2 text-[12px] leading-4 text-slate-400">{sub}</div>
      <div className="absolute bottom-4 right-4 h-12 w-28 overflow-hidden">
        <SparkLine data={data} tone={tone} />
      </div>
    </div>
  )
}

function SparkLine({ data, tone }: { data: number[]; tone: string }) {
  const width = 112
  const height = 52
  const paddingTop = 8
  const paddingBottom = 8
  const min = Math.min(...data)
  const max = Math.max(...data)
  const span = Math.max(1, max - min)
  const points = data
    .map((value, index) => {
      const x = (index / (data.length - 1)) * width
      const normalized = (value - min) / span
      const y = height - paddingBottom - normalized * (height - paddingTop - paddingBottom)
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
  const color = tone === 'green' ? '#34d399' : tone === 'cyan' ? '#22d3ee' : tone === 'blue' ? '#38bdf8' : '#f87171'
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full overflow-hidden">
      <polyline points={`0,${height - paddingBottom} ${points} ${width},${height - paddingBottom}`} fill={`${color}22`} stroke="none" />
      <polyline className="chart-draw" points={points} fill="none" stroke={color} strokeWidth="1.8" />
      {data.map((value, index) => (
        <circle
          key={index}
          cx={(index / (data.length - 1)) * width}
          cy={height - paddingBottom - ((value - min) / span) * (height - paddingTop - paddingBottom)}
          r="1.8"
          fill={color}
        />
      ))}
    </svg>
  )
}

function TinyWave({ color, className }: { color: string; className?: string }) {
  return (
    <svg viewBox="0 0 180 52" className={cn('wave-bars h-full w-full', className)} preserveAspectRatio="none">
      {Array.from({ length: 34 }, (_, index) => {
        const height = 10 + Math.abs(Math.sin(index * 0.82) * 36)
        return (
          <rect
            key={index}
            x={index * 5.2}
            y={(52 - height) / 2}
            width="2.1"
            height={height}
            rx="1"
            fill={color}
            opacity={0.5 + (index % 3) * 0.16}
            style={{ animationDelay: `${index * 12}ms`, transformOrigin: 'center' }}
          />
        )
      })}
    </svg>
  )
}
