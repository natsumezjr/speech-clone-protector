import { ArrowRight, AudioLines, BrainCircuit, Ear, Fingerprint, Gauge, Network, ShieldCheck, Waves } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Badge } from '@/components/common/Badge'
import { Button } from '@/components/common/Button'
import { Panel } from '@/components/common/Panel'
import { MetricCard } from '@/components/cards/MetricCard'
import { StrategyCard } from '@/components/cards/StrategyCard'
import { FeatureCard } from '@/components/cards/FeatureCard'

export function HomePage() {
  return (
    <div className="space-y-8">
      <section className="grid min-h-[520px] items-center gap-8 lg:grid-cols-[1.05fr_0.95fr]">
        <div>
          <div className="mb-5 flex flex-wrap gap-2">
            {['端到端可验证', '多模型自适应', '听感友好', '高效易用'].map((tag) => (
              <Badge key={tag} tone="cyan">
                {tag}
              </Badge>
            ))}
          </div>
          <h1 className="max-w-4xl text-4xl font-bold leading-tight tracking-normal text-white md:text-6xl">
            发布前保护你的声音，降低语音克隆风险
          </h1>
          <p className="mt-6 max-w-3xl text-base leading-8 text-slate-300">
            融合语义防护与音色防护的双重机制，在保证听感质量的同时干扰语音理解与音色建模，有效抵御非授权的语音克隆与滥用。
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to="/workspace">
              <Button icon={<ShieldCheck className="h-4 w-4" />}>开始防护</Button>
            </Link>
            <Link to="/results/demo-task">
              <Button variant="secondary" icon={<Gauge className="h-4 w-4" />}>
                查看演示
              </Button>
            </Link>
          </div>
        </div>
        <Panel className="relative overflow-hidden p-6">
          <div className="absolute right-0 top-0 h-44 w-44 rounded-full bg-cyan-400/10 blur-3xl" />
          <div className="relative">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">System Loop</p>
                <h2 className="text-xl font-semibold text-white">主动式语音防护闭环</h2>
              </div>
              <Waves className="h-8 w-8 text-cyan-200" />
            </div>
            <div className="space-y-3">
              {[
                ['原始音频', 'x', '清晰语义与声纹特征'],
                ['保护性扰动', 'δ', '语义分支 + 音色分支 + 心理声学约束'],
                ['保护音频', "x'", '可发布、可下载、听感保真'],
                ['下游影响', '↓', 'ASR / Tokenizer / LLM / TTS 同时受干扰'],
              ].map(([title, symbol, desc], index) => (
                <div key={title} className="flex items-center gap-3">
                  <div className="grid h-12 w-12 shrink-0 place-items-center rounded-xl border border-cyan-300/25 bg-cyan-300/10 text-lg font-bold text-cyan-100">
                    {symbol}
                  </div>
                  <div className="min-w-0 flex-1 rounded-xl border border-white/10 bg-slate-950/45 p-3">
                    <p className="text-sm font-semibold text-white">{title}</p>
                    <p className="text-xs text-slate-400">{desc}</p>
                  </div>
                  {index < 3 ? <ArrowRight className="hidden h-5 w-5 text-cyan-200 xl:block" /> : null}
                </div>
              ))}
            </div>
          </div>
        </Panel>
      </section>

      <section className="grid gap-4 lg:grid-cols-4">
        {[
          { title: 'ASR 语音识别', desc: '识别准确率下降', Icon: BrainCircuit },
          { title: 'Tokenizer 分词器', desc: '表示偏移增大', Icon: Network },
          { title: 'LLM 语言模型', desc: '理解偏差增大', Icon: AudioLines },
          { title: '克隆系统 / TTS', desc: '声纹相似度下降', Icon: Fingerprint },
        ].map(({ title, desc, Icon }) => (
          <Panel key={title} className="p-4">
            <Icon className="mb-3 h-6 w-6 text-cyan-200" />
            <h3 className="text-sm font-semibold text-white">{title}</h3>
            <p className="mt-2 text-xs text-slate-400">{desc}</p>
          </Panel>
        ))}
      </section>

      <section className="grid gap-5 lg:grid-cols-3">
        <StrategyCard
          title="语义防护"
          icon={<BrainCircuit className="h-6 w-6" />}
          items={['干扰语义表示', '降低 ASR / LALM 理解准确率', '多模型语义编码器', 'token 变化与语义漂移']}
        />
        <StrategyCard
          title="音色防护"
          icon={<Fingerprint className="h-6 w-6" />}
          items={['削弱声纹特征', '降低说话人相似度', '抑制音色建模', '阻断克隆条件提取']}
        />
        <StrategyCard
          title="心理声学约束"
          icon={<Ear className="h-6 w-6" />}
          items={['控制扰动可感知性', '掩蔽阈值建模', '听感保真优化']}
        />
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
        <MetricCard label="ASR 干扰" value="WER 68.7%" icon={<BrainCircuit className="h-5 w-5" />} />
        <MetricCard label="声纹相似度下降" value="86.2%" tone="green" icon={<Fingerprint className="h-5 w-5" />} />
        <MetricCard label="听感保真" value="PESQ 3.67" tone="blue" icon={<Ear className="h-5 w-5" />} />
        <MetricCard label="对抗评估" value="85.6%" tone="purple" description="多模型平均成功率" />
        <MetricCard label="任务通过率" value="98.7%" tone="green" />
        <MetricCard label="平均处理时长" value="72s" tone="orange" />
      </section>

      <FeatureCard
        title="作品亮点"
        items={[
          '提出语义与音色双重防护框架，兼顾安全与可用。',
          '引入心理声学约束，扰动不可感知、听感友好。',
          '通过 ASR、Tokenizer、LLM、TTS 多层指标展示防护效果。',
          '前端支持 Mock / Backend 快速切换，便于竞赛演示和真实后端对接。',
        ]}
      />
    </div>
  )
}
