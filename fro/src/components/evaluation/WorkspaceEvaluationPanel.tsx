import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronDown, Loader2, Search, Sparkles, TestTube2, X } from 'lucide-react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'

import { cloneVoice, createEvaluationBatch, getTaskStatus, listTasks, runAsrEval } from '@/services/apiClient'
import { useAppStore } from '@/store/appStore'
import { useTaskStore } from '@/store/taskStore'
import type { AsrEvalResponse, CapabilitiesResponse, CloneVoiceRequest, HistoryTask, ProtectionRuntimeConfig, RuntimeModelOption } from '@/types/task'
import { cn } from '@/lib/utils'
import { cloneModelRequiresReferenceText, normalizeCloneReferenceTextRequest } from '@/utils/cloneModelCapabilities'
import { findReusableAsrAnnotation } from '@/utils/asrAnnotationReuse'
import { defaultCloneTextForLanguage, translateDefaultCloneText } from '@/utils/cloneDefaultText'

type ModelOption = RuntimeModelOption & { label: string }
type CloneDialogMode = 'single' | 'all'
type CloneAnnotationSource = 'manual' | 'asr'

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

function evaluationBatchId(type: 'asr' | 'clone') {
  const suffix = typeof crypto.randomUUID === 'function' ? crypto.randomUUID().replaceAll('-', '').slice(0, 12) : `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
  return `${type}_batch_${suffix}`
}

function options(values?: Array<string | RuntimeModelOption>): ModelOption[] {
  return (values ?? []).map((item) =>
    typeof item === 'string'
      ? { value: item, label: item, name: item }
      : { ...item, label: item.label ?? item.name ?? item.value },
  )
}

function isAvailable(item: ModelOption) {
  return item.status === undefined || item.status === 'available'
}

function shortModelName(value?: string) {
  const model = String(value ?? '')
  if (/xtts[_:/-]?v?2/i.test(model)) return 'XTTS v2'
  if (/xtts[_:/-]?v?1[._-]?1/i.test(model)) return 'XTTS v1.1'
  if (/yourtts/i.test(model)) return 'YourTTS'
  if (/cosyvoice/i.test(model)) return 'CosyVoice2-0.5B'
  if (/gpt.?sovits/i.test(model)) return 'GPT-SoVITS'
  if (/whisper/i.test(model)) return model.replace(/^openai[-/:]?/i, '').replaceAll(':', ' ')
  if (/funasr|paraformer/i.test(model)) return 'Paraformer 中文'
  if (/wav2vec/i.test(model)) return 'Wav2Vec2 Base'
  return model.split('/').at(-1)?.replaceAll('_', ' ') || '未命名模型'
}

function compatibleAsrModels(values: ModelOption[], language: string) {
  return values.filter((item) => {
    if (!isAvailable(item)) return false
    if (item.languages?.length) return item.languages.some((value) => value.toLowerCase().startsWith(language === 'zh-cn' ? 'zh' : 'en'))
    if (language === 'zh-cn') return !/wav2vec|960h/i.test(item.value)
    return !/funasr|paraformer|chinese/i.test(`${item.value} ${(item.type ?? []).join(' ')}`)
  })
}

function compatibleCloneModels(values: ModelOption[], language: string) {
  const languagePrefix = language === 'zh-cn' ? 'zh' : 'en'
  return values.filter((item) => {
    if (!isAvailable(item)) return false
    if (item.languages?.length) {
      return item.languages.some((value) => value.toLowerCase().startsWith(languagePrefix))
    }
    if (language === 'zh-cn') return !/your[-_]?tts/i.test(item.value)
    return true
  })
}

function defaultAsrModel(values: ModelOption[], language: string) {
  const compatible = compatibleAsrModels(values, language)
  if (language === 'zh-cn') {
    return compatible.find((item) => /funasr|paraformer|chinese/i.test(`${item.value} ${(item.type ?? []).join(' ')}`)) ?? compatible[0]
  }
  return compatible.find((item) => /whisper[:/_-]?medium/i.test(item.value)) ?? compatible.find((item) => /whisper/i.test(item.value)) ?? compatible[0]
}

function parseTaskDate(value?: string | null) {
  if (!value) return null
  const dotted = value.trim().match(/^(\d{4})\.(\d{1,2})\.(\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?$/)
  if (dotted) {
    const date = new Date(Number(dotted[1]), Number(dotted[2]) - 1, Number(dotted[3]), Number(dotted[4] ?? 0), Number(dotted[5] ?? 0), Number(dotted[6] ?? 0))
    return Number.isNaN(date.getTime()) ? null : date
  }
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date
}

function taskTimestamp(task: HistoryTask) {
  return parseTaskDate(task.protectionCompletedAt ?? task.createdAt)?.getTime() ?? 0
}

function formatTaskTime(value?: string | null) {
  const date = parseTaskDate(value)
  if (!date) return value || '时间未记录'
  const pad = (input: number) => String(input).padStart(2, '0')
  return `${date.getFullYear()}.${date.getMonth() + 1}.${date.getDate()} ${date.getHours()}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

function completedProtectionTasks(tasks: HistoryTask[]) {
  return tasks
    .filter((task) => (task.protectionStatus ?? task.status) === 'completed' || (task.protectionStatus ?? task.status) === 'success')
    .sort((left, right) => taskTimestamp(right) - taskTimestamp(left))
}

async function waitForAsr(taskId: string, asrSubId: string) {
  for (let attempt = 0; attempt < 240; attempt += 1) {
    const status = await getTaskStatus(taskId)
    const task = status.asrTasks?.find((item) => item.asrSubId === asrSubId) ?? status.asrTask
    const result = task?.asrResult ?? status.asrResult
    if (result?.asr?.originalText && result.asr.protectedText) return result
    if (task?.status === 'failed' || task?.status === 'error' || task?.status === 'cancelled') {
      const message = typeof task.error === 'string' ? task.error : task.error?.message
      throw new Error(message || task.message || '自动标注未完成。')
    }
    await wait(1000)
  }
  throw new Error('自动标注仍在执行，请稍后从历史记录查看。')
}

function ProtectionTaskSelector({ tasks, value, onChange, view }: { tasks: HistoryTask[]; value: string; onChange: (value: string) => void; view: 'asr' | 'clone' }) {
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const selectedTask = tasks.find((task) => task.taskId === value)
  const selectedFilename = selectedTask?.filename && selectedTask.filename !== '-' ? selectedTask.filename : ''
  return (
    <div className="workspace-task-selector-wrap relative shrink-0">
      <button type="button" onClick={() => setOpen((current) => !current)} className="workspace-task-selector workspace-hover-control flex h-10 w-full items-center gap-2 rounded-[7px] border border-cyan-300/14 bg-slate-950/55 pl-3 pr-2 text-left text-xs text-slate-200 hover:border-cyan-300/28">
        <span className={cn('min-w-0 flex-1 truncate', !selectedFilename && 'text-slate-500')} title={selectedFilename || undefined}>
          {selectedFilename || (value ? '正在读取音频名称' : '选择已完成的保护任务')}
        </span>
        <ChevronDown className={cn('h-4 w-4 shrink-0 text-slate-500 transition', open && 'rotate-180')} />
      </button>
      <button
        type="button"
        disabled={!selectedFilename}
        onClick={() => navigate(`/history?view=${view}&search=${encodeURIComponent(selectedFilename)}`)}
        className="workspace-hover-control absolute right-9 top-1.5 grid h-7 w-7 place-items-center rounded-[5px] text-cyan-200 hover:bg-cyan-300/10 disabled:opacity-30"
        aria-label="在历史记录中查看"
      >
        <Search className="h-3.5 w-3.5" />
      </button>
      {open ? (
        <div className="ui-popover-surface absolute inset-x-0 top-11 z-40 max-h-48 overflow-y-auto rounded-[8px] border border-cyan-300/18 p-1.5 shadow-[0_18px_45px_rgba(0,0,0,0.48)]" role="listbox">
          {tasks.length ? tasks.map((task) => (
            <button key={task.taskId} type="button" onClick={() => { onChange(task.taskId); setOpen(false) }} className="workspace-hover-row block w-full rounded-[6px] px-3 py-2 text-left hover:bg-cyan-300/[0.08]">
              <span className="flex items-center justify-between gap-3">
                <span className="min-w-0 truncate text-[11px] font-black text-cyan-100" title={task.filename}>{task.filename || '未命名音频'}</span>
                <span className="shrink-0 font-mono text-[10px] text-slate-500">{formatTaskTime(task.protectionCompletedAt ?? task.createdAt)}</span>
              </span>
            </button>
          )) : <p className="px-3 py-4 text-center text-xs text-slate-500">暂无可测试的保护任务</p>}
        </div>
      ) : null}
    </div>
  )
}

function ModelGrid({ values, selected, onSelect }: { values: ModelOption[]; selected?: string; onSelect: (item: ModelOption) => void }) {
  return (
    <div className="workspace-model-grid grid grid-cols-2 pr-1">
      {values.map((item) => {
        const unavailable = !isAvailable(item)
        return (
          <div key={item.value} className={cn('flex min-h-0 items-center rounded-[7px] border', !unavailable && 'workspace-hover-row', selected === item.value ? 'border-cyan-300 bg-cyan-400/12' : 'border-cyan-300/12 bg-slate-950/38', unavailable && 'opacity-50')}>
            <button type="button" disabled={unavailable} onClick={() => onSelect(item)} className={cn('h-full min-w-0 flex-1 truncate px-3 text-center text-[11px] font-black', selected === item.value ? 'text-cyan-100' : 'text-slate-300')} title={unavailable ? '当前模型暂不可用' : shortModelName(item.value)}>
              {shortModelName(item.value)}
            </button>
          </div>
        )
      })}
    </div>
  )
}

function ModelSummary({ model, modelTypes }: { model?: ModelOption; modelTypes?: CapabilitiesResponse['modelTypes'] }) {
  const definitions = Object.values(modelTypes ?? {}).flat()
  const types = (model?.type ?? []).map((value) => definitions.find((item) => item.value === value)?.name ?? value)
  return (
    <div className="workspace-model-info workspace-hover-card h-full min-h-0 overflow-y-auto rounded-[7px] border border-cyan-300/10 bg-slate-950/28 px-4 py-3">
      {model ? (
        <>
          <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
            <span className="shrink-0 text-xs font-black text-cyan-300">模型信息</span>
            <span className="min-w-0 truncate text-sm font-black text-cyan-100">{shortModelName(model.value)}</span>
            {types.length ? <div className="flex min-w-0 flex-wrap gap-1">{types.map((type) => <span key={type} className="whitespace-nowrap rounded-full border border-cyan-300/16 bg-cyan-400/8 px-2 py-0.5 text-[10px] font-bold text-cyan-100">{type}</span>)}</div> : null}
          </div>
          <p className="mt-2 text-xs leading-6 text-slate-400">{model.information || '当前模型暂无补充说明。'}</p>
        </>
      ) : <p className="grid h-full place-items-center text-xs text-slate-500">暂无可用模型</p>}
    </div>
  )
}

export function WorkspaceEvaluationPanel({ runtimeConfig, modelTypes }: { runtimeConfig?: ProtectionRuntimeConfig; modelTypes?: CapabilitiesResponse['modelTypes'] }) {
  const pushToast = useAppStore((state) => state.pushToast)
  const currentTaskStatus = useTaskStore((state) => state.currentTaskStatus)
  const { data: tasks = [] } = useQuery({ queryKey: ['tasks'], queryFn: listTasks, refetchInterval: 5000 })
  const completedTasks = useMemo(() => completedProtectionTasks(tasks), [tasks])
  const asrModels = useMemo(() => options(runtimeConfig?.models.asr), [runtimeConfig?.models.asr])
  const cloneModels = useMemo(() => options(runtimeConfig?.models.tts), [runtimeConfig?.models.tts])
  const [asrTaskId, setAsrTaskId] = useState('')
  const [cloneTaskId, setCloneTaskId] = useState('')
  const [asrLanguage, setAsrLanguage] = useState('en')
  const [cloneLanguage, setCloneLanguage] = useState('en')
  const [selectedAsr, setSelectedAsr] = useState('')
  const [selectedClone, setSelectedClone] = useState('')
  const [asrRunning, setAsrRunning] = useState(false)
  const [cloneRunning, setCloneRunning] = useState(false)
  const [cloneDialogMode, setCloneDialogMode] = useState<CloneDialogMode | null>(null)
  const [cloneAnnotationSource, setCloneAnnotationSource] = useState<CloneAnnotationSource>('manual')
  const [cloneText, setCloneText] = useState(() => defaultCloneTextForLanguage('en'))
  const cloneTextUsesDefaultRef = useRef(true)
  const [manualAnnotation, setManualAnnotation] = useState('')
  const lastPreferredTaskRef = useRef('')

  useEffect(() => {
    const currentTaskCompleted = currentTaskStatus?.status === 'completed' || currentTaskStatus?.status === 'success'
    const preferredTaskId = currentTaskCompleted ? currentTaskStatus.taskId : completedTasks[0]?.taskId
    if (!preferredTaskId) return
    const preferredChanged = lastPreferredTaskRef.current !== preferredTaskId
    lastPreferredTaskRef.current = preferredTaskId
    const validTaskIds = new Set(completedTasks.map((task) => task.taskId))
    const selectPreferred = (current: string) => preferredChanged || !current || (!validTaskIds.has(current) && current !== preferredTaskId) ? preferredTaskId : current
    setAsrTaskId(selectPreferred)
    setCloneTaskId(selectPreferred)
  }, [completedTasks, currentTaskStatus?.status, currentTaskStatus?.taskId])

  const availableAsr = compatibleAsrModels(asrModels, asrLanguage)
  const effectiveAsr = availableAsr.find((item) => item.value === selectedAsr) ?? defaultAsrModel(asrModels, asrLanguage)
  const availableClone = compatibleCloneModels(cloneModels, cloneLanguage)
  const effectiveClone = availableClone.find((item) => item.value === selectedClone) ?? availableClone.find((item) => /cosyvoice/i.test(item.value)) ?? availableClone[0]
  const dialogCloneModels = cloneDialogMode === 'single' ? (effectiveClone ? [effectiveClone] : []) : availableClone
  const dialogReferenceTextModels = dialogCloneModels.filter(cloneModelRequiresReferenceText)
  const dialogNeedsReferenceText = dialogReferenceTextModels.length > 0

  const changeCloneLanguage = (language: string) => {
    setCloneLanguage(language)
    setSelectedClone('')
    if (cloneTextUsesDefaultRef.current) {
      setCloneText((current) => translateDefaultCloneText(current, language))
    }
  }

  const requireTask = (taskId: string) => {
    if (taskId) return true
    pushToast({ kind: 'error', title: '请先选择保护任务', description: '仅已完成保护的音频可以进行测试。' })
    return false
  }

  const queueAsr = async (models: ModelOption[], allModels = false) => {
    if (!requireTask(asrTaskId) || !models.length) return
    try {
      setAsrRunning(true)
      const batchId = allModels ? evaluationBatchId('asr') : undefined
      const batchItems = models.map((model, index) => ({
        batchItemId: `${batchId ?? 'single'}_${index + 1}`,
        model: model.value,
        modelName: shortModelName(model.value),
        modelType: (model.type ?? []).join(' / ') || undefined,
      }))
      if (batchId) {
        await createEvaluationBatch(asrTaskId, { batchId, type: 'asr', items: batchItems })
      }
      const settled = await Promise.allSettled(models.map((model, index) => runAsrEval(asrTaskId, {
        model: model.value,
        language: asrLanguage,
        batchId,
        batchItemId: batchId ? batchItems[index].batchItemId : undefined,
      })))
      const queuedCount = settled.filter((item) => item.status === 'fulfilled').length
      const failedCount = settled.length - queuedCount
      if (!queuedCount) throw settled.find((item): item is PromiseRejectedResult => item.status === 'rejected')?.reason ?? new Error('所有识别任务均提交失败。')
      pushToast({ kind: failedCount ? 'error' : 'success', title: allModels ? '全模型识别测试已开始' : '识别测试已开始', description: failedCount ? `已提交 ${queuedCount} 个，${failedCount} 个提交失败。` : `已并行提交 ${queuedCount} 个模型。` })
    } catch (error) {
      pushToast({ kind: 'error', title: '识别测试提交失败', description: error instanceof Error ? error.message : '请稍后重试。' })
    } finally {
      setAsrRunning(false)
    }
  }

  const openCloneDialog = (mode: CloneDialogMode) => {
    if (!requireTask(cloneTaskId)) return
    if (mode === 'single' && !effectiveClone) {
      pushToast({ kind: 'error', title: '没有可用模型', description: '请先检查模型部署状态。' })
      return
    }
    setCloneDialogMode(mode)
  }

  const queueCloneRequests = async () => {
    if (!cloneDialogMode || !requireTask(cloneTaskId)) return
    const dialogMode = cloneDialogMode
    if (!cloneText.trim()) {
      pushToast({ kind: 'error', title: '请填写目标测试文本', description: '目标测试文本不能为空。' })
      return
    }
    const models = dialogMode === 'single' && effectiveClone ? [effectiveClone] : availableClone
    const referenceTextModels = models.filter(cloneModelRequiresReferenceText)
    if (dialogMode === 'single' && referenceTextModels.length && cloneAnnotationSource === 'manual' && !manualAnnotation.trim()) {
      pushToast({ kind: 'error', title: '请填写人工标注', description: `${shortModelName(referenceTextModels[0].value)} 需要参考音频对应文本。` })
      return
    }

    try {
      setCloneRunning(true)
      setCloneDialogMode(null)
      let annotation: AsrEvalResponse | null = null
      let annotationError: string | null = null
      const useAsrAnnotation = referenceTextModels.length > 0 && (dialogMode === 'all' || cloneAnnotationSource === 'asr')
      if (useAsrAnnotation) {
        try {
          const annotationModel = defaultAsrModel(asrModels, cloneLanguage)
          if (!annotationModel) throw new Error(`${cloneLanguage === 'zh-cn' ? '中文' : '英文'}自动标注模型当前不可用。`)
          const currentStatus = await getTaskStatus(cloneTaskId).catch(() => null)
          annotation = currentStatus
            ? findReusableAsrAnnotation(currentStatus, annotationModel.value, cloneLanguage)
            : null
          if (!annotation) {
            const asrQueued = await runAsrEval(cloneTaskId, { model: annotationModel.value, language: cloneLanguage })
            if (!asrQueued.asrSubId) throw new Error('自动标注任务未返回有效编号。')
            annotation = await waitForAsr(cloneTaskId, asrQueued.asrSubId)
          }
          const evaluatedAsr = annotation.asr
          const originalText = evaluatedAsr?.originalText?.trim() ?? ''
          const protectedText = evaluatedAsr?.protectedText?.trim() ?? ''
          if (!evaluatedAsr || !originalText || !protectedText) throw new Error('自动标注未同时生成原始音频和保护音频文本。')
        } catch (error) {
          annotationError = error instanceof Error ? error.message : '自动标注未完成。'
          if (dialogMode === 'single') throw new Error(`ASR 自动标注失败：${annotationError}`, { cause: error })
        }
      }

      const requests: CloneVoiceRequest[] = []
      const skippedReferenceModels: ModelOption[] = []
      models.forEach((model) => {
        const base = { text: cloneText.trim(), model: model.value, language: cloneLanguage, speed: 1 }
        if (cloneModelRequiresReferenceText(model)) {
          let requestCount = 0
          const includeManual = dialogMode === 'all' || cloneAnnotationSource === 'manual'
          const includeAsr = dialogMode === 'all' || cloneAnnotationSource === 'asr'
          if (includeManual && manualAnnotation.trim()) {
            requests.push(normalizeCloneReferenceTextRequest({ ...base, annotationSource: 'manual', speakerPrompt: manualAnnotation.trim() }, model))
            requestCount += 1
          }
          const evaluatedAsr = annotation?.asr
          const originalText = evaluatedAsr?.originalText?.trim() ?? ''
          const protectedText = evaluatedAsr?.protectedText?.trim() ?? ''
          if (includeAsr && annotation?.asrSubId && evaluatedAsr && originalText && protectedText) {
            requests.push(normalizeCloneReferenceTextRequest({ ...base, annotationSource: 'asr', annotationAsrSubId: annotation.asrSubId, annotationAsrModel: evaluatedAsr.model, annotationCreatedAt: annotation.createdAt ?? undefined, speakerPrompt: originalText, originalSpeakerPrompt: originalText, protectedSpeakerPrompt: protectedText }, model))
            requestCount += 1
          }
          if (!requestCount) skippedReferenceModels.push(model)
        } else {
          requests.push(normalizeCloneReferenceTextRequest(base, model))
        }
      })
      if (!requests.length) {
        const names = skippedReferenceModels.map((model) => shortModelName(model.value)).join('、')
        throw new Error(`${names || '所选模型'}需要参考音频对应文本，请填写人工标注后重试。`)
      }
      const batchId = dialogMode === 'all' ? evaluationBatchId('clone') : undefined
      const batchItems = requests.map((request, index) => ({
        batchItemId: `${batchId ?? 'single'}_${index + 1}`,
        model: request.model,
        modelName: shortModelName(request.model),
        modelType: (models.find((model) => model.value === request.model)?.type ?? []).join(' / ') || undefined,
        ...(request.annotationSource ? { annotationSource: request.annotationSource } : {}),
      }))
      if (batchId) {
        await createEvaluationBatch(cloneTaskId, { batchId, type: 'clone', items: batchItems })
      }
      const settled = await Promise.allSettled(requests.map((request, index) => cloneVoice(cloneTaskId, {
        ...request,
        batchId,
        batchItemId: batchId ? batchItems[index].batchItemId : undefined,
      })))
      const queuedCount = settled.filter((item) => item.status === 'fulfilled').length
      const failedCount = settled.length - queuedCount
      if (!queuedCount) throw settled.find((item): item is PromiseRejectedResult => item.status === 'rejected')?.reason ?? new Error('所有克隆任务均提交失败。')
      const skippedText = skippedReferenceModels.length ? ` ${skippedReferenceModels.map((model) => shortModelName(model.value)).join('、')} 缺少参考文本，已跳过。` : ''
      const annotationText = annotationError && !skippedReferenceModels.length ? ` 自动标注未加入：${annotationError}` : ''
      const description = failedCount
        ? `已提交 ${queuedCount} 个，${failedCount} 个提交失败。${skippedText}${annotationText}`
        : `已并行提交 ${queuedCount} 个克隆任务。${skippedText}${annotationText}`
      pushToast({ kind: failedCount || skippedReferenceModels.length || Boolean(annotationError) ? 'error' : 'success', title: dialogMode === 'all' ? '全模型克隆已开始' : '克隆测试已开始', description })
    } catch (error) {
      pushToast({ kind: 'error', title: '克隆测试提交失败', description: error instanceof Error ? error.message : '请稍后重试。' })
    } finally {
      setCloneRunning(false)
    }
  }

  return (
    <section className="workspace-evaluation-panel workspace-hover-surface ui-card grid h-full min-h-0 grid-rows-2 gap-3 p-3">
      <div className="workspace-evaluation-section workspace-hover-card flex min-h-0 flex-col rounded-[9px] border border-violet-300/14 bg-slate-950/18 p-4">
        <div className="workspace-evaluation-header relative mb-3 flex min-h-8 items-center">
          <h2 className="pointer-events-none absolute inset-x-0 text-center text-[17px] font-black text-white">VoiceShield 的语音克隆测试</h2>
          <select value={cloneLanguage} onChange={(event) => changeCloneLanguage(event.target.value)} className="workspace-hover-control relative z-10 ml-auto h-8 rounded-[6px] border border-violet-300/14 bg-slate-950/70 px-2 text-xs font-bold text-slate-200">
            <option value="en">英文</option>
            <option value="zh-cn">中文</option>
          </select>
        </div>
        <ProtectionTaskSelector tasks={completedTasks} value={cloneTaskId} onChange={setCloneTaskId} view="clone" />
        <div className="workspace-evaluation-model-grid mt-3"><ModelGrid values={availableClone} selected={effectiveClone?.value} onSelect={(item) => setSelectedClone(item.value)} /></div>
        <div className="workspace-evaluation-model-summary mt-2 min-h-0 flex-1"><ModelSummary model={effectiveClone} modelTypes={modelTypes} /></div>
        <div className="workspace-evaluation-actions workspace-hover-group mt-auto flex justify-end gap-2 pt-3">
          <button type="button" disabled={cloneRunning || !effectiveClone} onClick={() => openCloneDialog('single')} className="workspace-hover-control h-9 rounded-[7px] border border-violet-300/18 px-3 text-xs font-black text-slate-200 disabled:opacity-45">测试所选模型</button>
          <button type="button" disabled={cloneRunning || !availableClone.length} onClick={() => openCloneDialog('all')} className="workspace-hover-control inline-flex h-9 items-center gap-2 rounded-[7px] bg-violet-400 px-3 text-xs font-black text-slate-950 shadow-[0_0_20px_rgba(167,139,250,0.2)] disabled:opacity-45">{cloneRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}一键克隆</button>
        </div>
      </div>

      <div className="workspace-evaluation-section workspace-hover-card flex min-h-0 flex-col rounded-[9px] border border-cyan-300/12 bg-slate-950/18 p-4">
        <div className="workspace-evaluation-header relative mb-3 flex min-h-8 items-center">
          <h2 className="pointer-events-none absolute inset-x-0 text-center text-[17px] font-black text-white">VoiceShield 的 ASR 自动标注测试</h2>
          <select value={asrLanguage} onChange={(event) => { setAsrLanguage(event.target.value); setSelectedAsr('') }} className="workspace-hover-control relative z-10 ml-auto h-8 rounded-[6px] border border-cyan-300/14 bg-slate-950/70 px-2 text-xs font-bold text-slate-200">
            <option value="en">英文</option>
            <option value="zh-cn">中文</option>
          </select>
        </div>
        <ProtectionTaskSelector tasks={completedTasks} value={asrTaskId} onChange={setAsrTaskId} view="asr" />
        <div className="workspace-evaluation-model-grid mt-3"><ModelGrid values={availableAsr} selected={effectiveAsr?.value} onSelect={(item) => setSelectedAsr(item.value)} /></div>
        <div className="workspace-evaluation-model-summary mt-2 min-h-0 flex-1"><ModelSummary model={effectiveAsr} modelTypes={modelTypes} /></div>
        <div className="workspace-evaluation-actions workspace-hover-group mt-auto flex justify-end gap-2 pt-3">
          <button type="button" disabled={asrRunning || !effectiveAsr} onClick={() => void queueAsr(effectiveAsr ? [effectiveAsr] : [], false)} className="workspace-hover-control h-9 rounded-[7px] border border-cyan-300/16 px-3 text-xs font-black text-slate-200 disabled:opacity-45">测试所选模型</button>
          <button type="button" disabled={asrRunning || !availableAsr.length} onClick={() => void queueAsr(availableAsr, true)} className="workspace-hover-control cyan-button inline-flex h-9 items-center gap-2 rounded-[7px] px-3 text-xs font-black disabled:opacity-45">{asrRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}一键测试</button>
        </div>
      </div>

      {cloneDialogMode ? createPortal((
        <div className="fixed inset-0 z-[95] grid place-items-center bg-slate-950/72 px-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="克隆测试设置">
          <div className="ui-card w-full max-w-[600px] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.48)]">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-[20px] font-black text-white">{cloneDialogMode === 'all' ? '全模型克隆设置' : shortModelName(effectiveClone?.value)}</h3>
                <p className="mt-1 text-xs leading-5 text-slate-500">使用{cloneLanguage === 'zh-cn' ? '中文' : '英文'}；{dialogNeedsReferenceText ? '仅需要参考文本的模型会使用标注。' : '所选模型无需人工标注。'}</p>
              </div>
              <button type="button" onClick={() => setCloneDialogMode(null)} className="grid h-9 w-9 place-items-center rounded-full border border-cyan-300/14 text-slate-300 hover:text-white" aria-label="关闭"><X className="h-4 w-4" /></button>
            </div>
            <label className="mt-5 block text-sm font-black text-slate-200">目标测试文本<textarea value={cloneText} onChange={(event) => { cloneTextUsesDefaultRef.current = false; setCloneText(event.target.value) }} className="mt-2 min-h-24 w-full resize-none rounded-[7px] border border-cyan-300/14 bg-slate-950/70 p-3 text-sm leading-6 text-slate-100 outline-none focus:border-cyan-300" /></label>
            {dialogNeedsReferenceText && cloneDialogMode === 'single' ? (
              <div className="mt-4">
                <p className="text-sm font-black text-slate-200">参考音频标注来源</p>
                <div className="mt-2 grid grid-cols-2 gap-2 rounded-[8px] border border-violet-300/12 bg-slate-950/38 p-1.5">
                  <button type="button" onClick={() => setCloneAnnotationSource('manual')} className={cn('h-10 rounded-[6px] text-sm font-black transition', cloneAnnotationSource === 'manual' ? 'bg-emerald-600 text-white shadow-[0_0_16px_rgba(5,150,105,0.18)]' : 'text-emerald-300 hover:bg-emerald-500/12 hover:text-emerald-200')}>人工标注</button>
                  <button type="button" onClick={() => setCloneAnnotationSource('asr')} className={cn('h-10 rounded-[6px] text-sm font-black transition', cloneAnnotationSource === 'asr' ? 'bg-violet-600 text-white shadow-[0_0_16px_rgba(124,58,237,0.2)]' : 'text-violet-300 hover:bg-violet-500/12 hover:text-violet-200')}>ASR 自动标注</button>
                </div>
              </div>
            ) : null}
            {dialogNeedsReferenceText && (cloneDialogMode === 'all' || cloneAnnotationSource === 'manual') ? <label className="mt-4 block text-sm font-black text-slate-200">人工标注{cloneDialogMode === 'single' ? '（必填）' : '（仅需要参考文本的模型使用）'}<textarea value={manualAnnotation} onChange={(event) => setManualAnnotation(event.target.value)} placeholder="输入人工核对后的原始音频文本" className="mt-2 min-h-20 w-full resize-none rounded-[7px] border border-violet-300/16 bg-slate-950/70 p-3 text-sm leading-6 text-slate-100 outline-none focus:border-violet-300" /></label> : null}
            {dialogNeedsReferenceText && cloneDialogMode === 'single' && cloneAnnotationSource === 'asr' ? (
              <div className="mt-4 rounded-[8px] border border-cyan-300/14 bg-cyan-400/[0.06] px-4 py-3 text-xs leading-6 text-slate-300">
                系统会优先复用当前保护任务最近的同语言 ASR 标注；若没有可用结果，将自动运行默认 ASR 模型，并分别使用原始音频和保护音频的转写结果进行克隆。
              </div>
            ) : null}
            <div className="mt-5 flex justify-end gap-3">
              <button type="button" onClick={() => setCloneDialogMode(null)} className="h-10 rounded-[7px] border border-cyan-300/14 px-4 text-sm font-bold text-slate-300">取消</button>
              <button type="button" onClick={() => void queueCloneRequests()} className="cyan-button inline-flex h-10 items-center gap-2 rounded-[7px] px-4 text-sm font-black"><TestTube2 className="h-4 w-4" />开始测试</button>
            </div>
          </div>
        </div>
      ), document.body) : null}

    </section>
  )
}
