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
import { Link } from 'react-router-dom'
import heroShieldScene from '@/assets/reference-hero-shield-scene.png'
import { cn } from '@/lib/utils'

const heroChecks = ['端到端可验证', '多模型自适应', '听感友好', '高效易用']

const strategies = [
  {
    title: '语义防护',
    tag: '推荐',
    desc: '干扰语义表示，降低 ASR/LLM 理解准确率',
    points: ['语义表示扰动', '对抗迁移鲁棒', '多模型泛化'],
    icon: BrainCircuit,
    tone: 'green',
  },
  {
    title: 'Feature 特征防护',
    desc: '削弱声学特征，降低相似度与克隆质量',
    points: ['声学特征干扰', '抑制特征建模', '克隆攻击抵御'],
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

const metrics = [
  { title: 'ASR 干扰（平均）', value: '↓ 62.3%', sub: '识别准确率下降', tone: 'red', data: [40, 72, 52, 69, 50, 80, 61, 34] },
  { title: '声纹相似度下降', value: '↓ 81.4%', sub: '克隆相似度降低', tone: 'red', data: [28, 46, 66, 50, 74, 42, 56, 38] },
  { title: '听感保真（PESQ）', value: '4.38 / 5', sub: '高保真听感', tone: 'green', data: [35, 45, 38, 62, 54, 77, 69, 82] },
  { title: '对抗评估（攻击成功率）', value: '↓ 85.6%', sub: '多攻击平均下降', tone: 'red', data: [22, 38, 55, 31, 76, 42, 37, 25] },
  { title: '任务通过率', value: '98.7%', sub: '通过率（内部基准）', tone: 'cyan', data: [16, 22, 28, 45, 39, 56, 73, 88] },
  { title: '单步平均时长', value: '9.2s', sub: '平均耗时', tone: 'blue', data: [18, 47, 26, 62, 38, 70, 54, 76] },
]

const capabilities = [
  { title: '双重防护机制', icon: ShieldCheck, desc: '语义及 Feature 特征双通路协同，全面降低被克隆与滥用风险。', tone: 'green' },
  { title: '多模型泛化', icon: Puzzle, desc: '覆盖主流 ASR / Tokenizer / LLM / TTS 模型，具备强迁移能力。', tone: 'blue' },
  { title: '听感无感知', icon: Ear, desc: '基于心理声学约束，保障人耳听感质量与发布可用性。', tone: 'purple' },
  { title: '可解释可评估', icon: Gauge, desc: '全流程评估与可视化，风险可控、结果可追踪、报告可导出。', tone: 'amber' },
]

export function HomePage() {
  return (
    <div className="space-y-6 pb-2">
      <section className="grid min-h-[486px] grid-cols-[430px_1fr_372px] gap-5 max-2xl:grid-cols-[420px_1fr_350px] max-xl:grid-cols-1">
        <div className="pt-[34px]">
          <h1 className="text-[43px] font-black leading-[1.18] tracking-normal text-white drop-shadow-[0_4px_18px_rgba(0,0,0,0.3)] max-md:text-4xl">
            发布前保护你的声音，
            <span className="block bg-gradient-to-r from-emerald-300 via-cyan-300 to-sky-400 bg-clip-text text-transparent">降低语音克隆风险</span>
          </h1>
          <p className="mt-5 max-w-[390px] text-[16px] leading-8 text-slate-300">
            融合语义防护与 Feature 特征防护的双重机制，在保证听感质量的同时干扰语音理解与声学特征建模，有效抵御非授权的语音克隆与滥用。
          </p>
          <div className="mt-7 flex gap-4">
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
          <div className="mt-5 flex flex-wrap gap-x-6 gap-y-3 text-[14px] text-slate-300">
            {heroChecks.map((item) => (
              <span key={item} className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-cyan-300" />
                {item}
              </span>
            ))}
          </div>
        </div>

        <HeroScene />

        <div className="grid content-start gap-3 pt-[10px]">
          {strategies.map((item) => (
            <StrategyPanel key={item.title} item={item} />
          ))}
        </div>
      </section>

      <section className="mb-1 grid grid-cols-6 gap-2.5 max-xl:grid-cols-3 max-md:grid-cols-1">
        {metrics.map((metric) => (
          <MetricTile key={metric.title} {...metric} />
        ))}
      </section>

      <section className="mt-6 grid grid-cols-[1fr_358px] gap-3 max-xl:grid-cols-1">
        <div className="ui-card p-4">
          <div className="mb-3 flex h-7 items-center gap-2">
            <Zap className="h-5 w-5 shrink-0 translate-y-[1px] text-cyan-300" />
            <h2 className="text-[18px] font-black leading-none text-white">核心能力</h2>
            <Info className="h-4 w-4 shrink-0 translate-y-[1px] text-slate-500" />
          </div>
          <div className="grid grid-cols-[repeat(4,minmax(0,1fr))] gap-4 max-lg:grid-cols-2 max-sm:grid-cols-1">
            {capabilities.map(({ title, icon: Icon, desc, tone }) => (
              <div
                key={title}
                className={cn(
                  'min-h-[124px] rounded-[7px] border p-3',
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
                <button className="mt-2 rounded-[5px] bg-cyan-400/12 px-3 py-1.5 text-xs font-bold text-cyan-300">了解更多 →</button>
              </div>
            ))}
          </div>
        </div>

        <div className="ui-card p-4">
          <div className="mb-3 flex h-7 items-center gap-2">
            <Sparkles className="h-5 w-5 shrink-0 translate-y-[1px] fill-amber-300 text-amber-300" />
            <h2 className="text-[18px] font-black leading-none text-white">作品亮点</h2>
          </div>
          <div className="space-y-2 text-[13px] leading-5 text-slate-300">
            {[
              '提出语义与 Feature 特征双重防护框架，兼顾安全与可用',
              '引入心理声学约束，扰动不可感知，听感友好',
              '在多种下游系统上实现显著性能劣化与迁移鲁棒',
              '高效优化与工程实现，支持快速接入与大规模应用',
            ].map((item) => (
              <p key={item} className="flex gap-3">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-300" />
                {item}
              </p>
            ))}
          </div>
          <div className="mt-3 rounded-[6px] border border-emerald-400/36 bg-emerald-400/10 px-4 py-2.5 text-center text-[15px] font-black text-emerald-200">
            安全可信 · 听感友好 · 高效可用
          </div>
        </div>
      </section>

      <footer className="ui-card flex min-h-[54px] items-center justify-between px-5 text-sm text-slate-400 max-md:flex-col max-md:gap-3 max-md:py-4">
        <div className="flex items-center gap-3 text-slate-200">
          <ShieldCheck className="h-7 w-7 text-cyan-300" />
          安全可信的语音发布基础设施
        </div>
        <div className="flex flex-wrap justify-center gap-x-8 gap-y-2">
          <span>端到端加密传输</span>
          <span>全链路审计日志</span>
          <span>符合隐私安全要求</span>
        </div>
        <span>© 2026 语音克隆防护平台</span>
      </footer>
    </div>
  )
}

function HeroScene() {
  return (
    <div className="relative overflow-hidden rounded-[8px] border border-cyan-300/10 scan-panel max-xl:min-h-[460px]">
      <div className="absolute inset-0 opacity-55 [background-image:linear-gradient(rgba(56,189,248,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(56,189,248,0.08)_1px,transparent_1px)] [background-size:38px_38px]" />
      <div className="absolute inset-x-[22px] top-[8px] h-[304px] overflow-hidden max-2xl:inset-x-[8px]">
        <img src={heroShieldScene} alt="" className="hero-scene-crop hero-scene-layer h-full w-full object-cover object-center" />
      </div>
      <div className="absolute inset-x-0 bottom-5 rounded-[8px] border border-cyan-300/22 bg-[#061426]/92 p-3">
        <div className="mb-2 text-center text-sm font-medium leading-5 text-slate-300">
          下游系统影响（攻击面）
        </div>
        <div className="grid grid-cols-[120px_28px_130px_1fr] gap-2.5 max-md:grid-cols-1">
          <WaveCard title="原始音频" subtitle="00:12" tone="cyan" />
          <div className="grid place-items-center text-3xl font-light text-cyan-300 max-md:hidden">+</div>
          <WaveCard title="保护性扰动" subtitle="不可感知的微小扰动" tone="green" />
          <div className="p-0">
            <div className="grid grid-cols-4 gap-2">
              {[
                ['ASR', '语音识别', '↓ 62.3%'],
                ['Tokenizer', '分词器', '↑ 73.8%'],
                ['LLM', '语言模型', '↓ 58.7%'],
                ['TTS', '克隆系统', '↓ 81.4%'],
              ].map(([name, sub, value]) => (
                <div key={name} className="h-[96px] rounded-[6px] border border-cyan-300/18 bg-[#07192d] px-1.5 py-2 text-center">
                  <div className="whitespace-nowrap text-[12px] font-black leading-4 text-white">{name}</div>
                  <div className="mt-0.5 text-[9px] leading-3 text-slate-400">{sub}</div>
                  <TinyWave className="my-1 h-3.5" color="#8fdcff" />
                  <div className={cn('text-[13px] font-black leading-4', value.includes('↑') ? 'text-lime-300' : 'text-red-300')}>{value}</div>
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
        'relative h-[156px] overflow-hidden rounded-[8px] border p-3',
        item.tone === 'green' && 'border-emerald-400/42 bg-emerald-400/10 shadow-[inset_0_0_36px_rgba(16,185,129,0.12)]',
        item.tone === 'blue' && 'border-sky-400/40 bg-sky-400/10 shadow-[inset_0_0_36px_rgba(14,165,233,0.12)]',
        item.tone === 'purple' && 'border-violet-400/42 bg-violet-400/10 shadow-[inset_0_0_36px_rgba(139,92,246,0.12)]',
      )}
    >
      <div className="flex items-center gap-3">
        <Icon className={cn('h-6 w-6 shrink-0', item.tone === 'green' ? 'text-emerald-300' : item.tone === 'blue' ? 'text-sky-300' : 'text-violet-300')} />
        <h3 className="whitespace-nowrap text-[22px] font-black leading-5 text-white">{item.title}</h3>
        {'tag' in item ? <span className="rounded-[5px] bg-emerald-400/18 px-2 py-1 text-sm font-bold text-emerald-200">{item.tag}</span> : null}
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
      {item.tone === 'green' ? <Network className="absolute bottom-5 right-8 h-[72px] w-[72px] text-emerald-300/30" /> : null}
      {item.tone === 'blue' ? <Waves className="absolute bottom-4 right-8 h-[72px] w-[72px] text-sky-300/35" /> : null}
      {item.tone === 'purple' ? <Ear className="absolute bottom-4 right-10 h-[72px] w-[72px] text-violet-300/35" /> : null}
    </div>
  )
}

function WaveCard({ title, subtitle, tone }: { title: string; subtitle: string; tone: 'cyan' | 'green' }) {
  return (
    <div className="h-[118px] rounded-[7px] border border-cyan-300/14 bg-[#07192d]/95 p-2">
      <div className="h-[50px] rounded-[6px] bg-slate-950/44 px-2 py-1.5">
        <TinyWave color={tone === 'green' ? '#22c55e' : '#b7e7ff'} />
      </div>
      <div className="mt-2 text-[13px] font-bold leading-4 text-white">{title}</div>
      <div className="mt-0.5 text-[11px] leading-4 text-slate-400">{subtitle}</div>
    </div>
  )
}

function MetricTile({ title, value, sub, tone, data }: (typeof metrics)[number]) {
  return (
    <div className="ui-card metric-glow relative h-[128px] overflow-hidden p-5">
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
            style={{ animationDelay: `${index * -55}ms`, transformOrigin: 'center' }}
          />
        )
      })}
    </svg>
  )
}
