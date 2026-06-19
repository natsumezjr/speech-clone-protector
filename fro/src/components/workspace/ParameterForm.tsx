import { zodResolver } from '@hookform/resolvers/zod'
import { ChevronDown, Play, TestTube2 } from 'lucide-react'
import { useState } from 'react'
import { Controller, useForm } from 'react-hook-form'
import { z } from 'zod'
import { Button } from '@/components/common/Button'
import { Badge } from '@/components/common/Badge'
import { Panel } from '@/components/common/Panel'
import { isBackendMode, isMockMode } from '@/config/runtime'
import { useTaskStore } from '@/store/taskStore'
import { ProtectionModeSelector } from './ProtectionModeSelector'
import { ProtectionTargetSelector } from './ProtectionTargetSelector'
import type { ProtectionTaskRequest } from '@/types/task'

const formSchema = z.object({
  mode: z.enum(['standard', 'strong', 'high_fidelity', 'custom']),
  targets: z.array(z.enum(['semantic', 'timbre'])).min(1, '请至少选择一个防护目标'),
  epsilon: z.coerce.number().min(0.01).max(0.2),
  steps: z.coerce.number().int().min(1).max(500),
  lambdaPsy: z.coerce.number().min(0).max(1),
  asrModel: z.string().min(1),
  timbreMode: z.enum(['untargeted', 'targeted']),
})

type ProtectionFormInput = z.input<typeof formSchema>
export type ProtectionFormValues = z.output<typeof formSchema>

interface ParameterFormProps {
  onSubmitTask: (payload: ProtectionTaskRequest) => void
  running: boolean
}

export function ParameterForm({ onSubmitTask, running }: ParameterFormProps) {
  const [showAdvanced, setShowAdvanced] = useState(true)
  const uploadedFile = useTaskStore((state) => state.uploadedFile)
  const {
    control,
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<ProtectionFormInput, unknown, ProtectionFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      mode: 'standard',
      targets: ['semantic', 'timbre'],
      epsilon: 0.08,
      steps: 20,
      lambdaPsy: 0.15,
      asrModel: 'Whisper-large-v3',
      timbreMode: 'untargeted',
    },
  })

  const submit = (values: ProtectionFormValues) => {
    const payload: ProtectionTaskRequest = {
      fileId: uploadedFile?.fileId,
      mode: values.mode,
      targets: values.targets,
      semantic: {
        enabled: values.targets.includes('semantic'),
        asrModel: values.asrModel,
        encoders: ['s3-tokenizer', 'hubert', 'whisper', 'mfcc'],
        lambdaSemantic: 1,
      },
      timbre: {
        enabled: values.targets.includes('timbre'),
        mode: values.timbreMode,
        encoders: ['wavlm', 'ecapa-tdnn', 'cosyvoice', 'styletts2'],
        lambdaTimbre: 1,
      },
      psychoacoustic: {
        enabled: true,
        lambdaPsy: values.lambdaPsy,
      },
      optimization: {
        epsilon: values.epsilon,
        steps: values.steps,
      },
    }
    onSubmitTask(payload)
  }

  const backendBlocked = isBackendMode && !uploadedFile?.fileId

  return (
    <Panel>
      <div className="mb-5 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">防护策略配置</h2>
        <Badge tone={isMockMode ? 'cyan' : 'orange'}>{isMockMode ? 'Mock 模式' : 'Backend 模式'}</Badge>
      </div>
      <form className="space-y-6" onSubmit={handleSubmit(submit)}>
        <section>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">保护模式</h3>
          <Controller
            control={control}
            name="mode"
            render={({ field }) => <ProtectionModeSelector value={field.value} onChange={field.onChange} />}
          />
        </section>

        <section>
          <h3 className="mb-3 text-sm font-semibold text-slate-200">防护目标</h3>
          <Controller
            control={control}
            name="targets"
            render={({ field }) => <ProtectionTargetSelector value={field.value} onChange={field.onChange} />}
          />
          {errors.targets ? <p className="mt-2 text-xs text-red-200">{errors.targets.message}</p> : null}
        </section>

        <section className="grid gap-4 md:grid-cols-3">
          <label className="space-y-2 text-sm">
            <span className="text-slate-300">epsilon / 扰动强度</span>
            <input className="w-full rounded-lg border border-white/10 bg-slate-950/60 px-3 py-2 text-white" type="number" step="0.01" {...register('epsilon')} />
            {errors.epsilon ? <span className="text-xs text-red-200">范围 0.01 到 0.2</span> : null}
          </label>
          <label className="space-y-2 text-sm">
            <span className="text-slate-300">优化轮数 Steps</span>
            <input className="w-full rounded-lg border border-white/10 bg-slate-950/60 px-3 py-2 text-white" type="number" {...register('steps')} />
            {errors.steps ? <span className="text-xs text-red-200">范围 1 到 500</span> : null}
          </label>
          <label className="space-y-2 text-sm">
            <span className="text-slate-300">心理声学权重 lambdaPsy</span>
            <input className="w-full rounded-lg border border-white/10 bg-slate-950/60 px-3 py-2 text-white" type="number" step="0.01" {...register('lambdaPsy')} />
            {errors.lambdaPsy ? <span className="text-xs text-red-200">范围 0 到 1</span> : null}
          </label>
        </section>

        <button type="button" className="flex items-center gap-2 text-sm text-cyan-100" onClick={() => setShowAdvanced((value) => !value)}>
          <ChevronDown className={`h-4 w-4 transition ${showAdvanced ? 'rotate-180' : ''}`} />
          高级选项
        </button>
        {showAdvanced ? (
          <section className="rounded-2xl border border-white/10 bg-slate-950/35 p-4">
            <div className="grid gap-4 md:grid-cols-2">
              <label className="space-y-2 text-sm">
                <span className="text-slate-300">ASR 模型</span>
                <select className="w-full rounded-lg border border-white/10 bg-slate-950/80 px-3 py-2 text-white" {...register('asrModel')}>
                  <option>Whisper-large-v3</option>
                  <option>Paraformer-large</option>
                </select>
              </label>
              <label className="space-y-2 text-sm">
                <span className="text-slate-300">Timbre 模式</span>
                <select className="w-full rounded-lg border border-white/10 bg-slate-950/80 px-3 py-2 text-white" {...register('timbreMode')}>
                  <option value="untargeted">Untargeted</option>
                  <option value="targeted">Targeted</option>
                </select>
              </label>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {['S3 Tokenizer', 'HuBERT', 'Whisper', 'MFCC', 'WavLM', 'ECAPA-TDNN', 'CosyVoice Encoder', 'Style Encoder'].map((tag) => (
                <Badge key={tag} tone="slate">
                  {tag}
                </Badge>
              ))}
            </div>
          </section>
        ) : null}

        <div className="rounded-xl border border-cyan-300/20 bg-cyan-300/8 p-4 text-xs leading-5 text-cyan-50">
          {isMockMode
            ? 'Mock 模式：使用本地固定模拟数据，快速展示平台流程与评估结果，不调用后端。'
            : 'Backend 模式：调用后端服务执行真实防护流程，结果来自后端返回。两种模式互斥，不混合数据。'}
        </div>

        {backendBlocked ? <p className="text-sm text-amber-200">Backend 模式下必须先上传文件并获得 fileId 后才能创建任务。</p> : null}

        <div className="flex flex-col gap-3 sm:flex-row">
          <Button disabled={running || backendBlocked} type="submit" icon={<Play className="h-4 w-4" />}>
            开始生成保护音频
          </Button>
          <Button disabled={running || (!isMockMode && backendBlocked)} type="submit" variant="secondary" icon={<TestTube2 className="h-4 w-4" />}>
            使用 Mock 数据演示
          </Button>
        </div>
      </form>
    </Panel>
  )
}
