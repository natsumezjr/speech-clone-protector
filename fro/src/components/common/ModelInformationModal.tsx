import { useEffect } from 'react'
import { X } from 'lucide-react'

import type { RuntimeModelOption, RuntimeModelType } from '@/types/task'

type Props = {
  model: RuntimeModelOption | null
  modelTypes?: Record<string, RuntimeModelType[]>
  onClose: () => void
}

export function ModelInformationModal({ model, modelTypes, onClose }: Props) {
  useEffect(() => {
    if (model) window.dispatchEvent(new CustomEvent('voiceshield:overlay-open', { detail: 'model-information' }))
  }, [model])

  if (!model) return null

  const typeDefinitions = Object.values(modelTypes ?? {}).flat()
  const resolvedTypes = (model.type ?? []).map((value) => typeDefinitions.find((item) => item.value === value) ?? { value, name: value, information: '' })

  return (
    <div className="fixed inset-0 z-[220] grid place-items-center bg-slate-950/72 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label={`${model.name ?? model.label ?? model.value} 模型信息`}>
      <div className="ui-card max-h-[76vh] w-full max-w-[520px] overflow-y-auto p-5 shadow-[0_28px_80px_rgba(0,0,0,0.5)]">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-bold tracking-[0.12em] text-cyan-300">模型信息</p>
            <h3 className="mt-2 text-[21px] font-black text-white">{model.name ?? model.label ?? model.value}</h3>
          </div>
          <button type="button" onClick={onClose} className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-cyan-300/14 bg-white/[0.035] text-slate-300 hover:text-white" aria-label="关闭模型信息">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-5 space-y-4">
          <section>
            <p className="text-xs font-black text-slate-500">模型类型</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {resolvedTypes.length ? resolvedTypes.map((item) => (
                <span key={item.value} title={item.information || undefined} className="rounded-full border border-cyan-300/18 bg-cyan-400/8 px-2.5 py-1 text-xs font-bold text-cyan-100">
                  {item.name}
                </span>
              )) : <span className="text-sm text-slate-500">暂未声明类型</span>}
            </div>
          </section>
          <section>
            <p className="text-xs font-black text-slate-500">模型说明</p>
            <p className="mt-2 text-sm leading-6 text-slate-300">{model.information || '暂未提供模型说明。'}</p>
          </section>
          {model.status && model.status !== 'available' ? (
            <p className="rounded-[7px] border border-amber-300/16 bg-amber-300/8 px-3 py-2 text-xs leading-5 text-amber-100">
              当前状态：{model.status}{model.reason ? ` · ${model.reason}` : ''}
            </p>
          ) : null}
        </div>
      </div>
    </div>
  )
}
