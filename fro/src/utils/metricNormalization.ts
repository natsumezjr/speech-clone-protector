export type ErrorSharesInput = {
  substituteShare?: unknown
  insertShare?: unknown
  deleteShare?: unknown
} | null | undefined

export type EditCountsInput = {
  substitutions?: unknown
  insertions?: unknown
  deletions?: unknown
  totalErrors?: unknown
} | null | undefined

export type NormalizedErrorShares = {
  substituteShare: number | null
  insertShare: number | null
  deleteShare: number | null
}

function finiteNumber(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function normalizeShares(value: ErrorSharesInput): NormalizedErrorShares | null {
  if (!value) return null
  const shares = {
    substituteShare: finiteNumber(value.substituteShare),
    insertShare: finiteNumber(value.insertShare),
    deleteShare: finiteNumber(value.deleteShare),
  }
  return Object.values(shares).some((share) => share !== null) ? shares : null
}

function sharesFromCounts(value: EditCountsInput): NormalizedErrorShares | null {
  if (!value) return null
  const substitutions = finiteNumber(value.substitutions)
  const insertions = finiteNumber(value.insertions)
  const deletions = finiteNumber(value.deletions)
  const totalErrors = finiteNumber(value.totalErrors)
  if (substitutions === null || insertions === null || deletions === null || totalErrors === null) return null
  if (totalErrors <= 0) {
    return { substituteShare: 0, insertShare: 0, deleteShare: 0 }
  }
  return {
    substituteShare: substitutions / totalErrors,
    insertShare: insertions / totalErrors,
    deleteShare: deletions / totalErrors,
  }
}

export function resolveAsrErrorShares(
  direct: ErrorSharesInput,
  counts: EditCountsInput,
  fallback: ErrorSharesInput,
): NormalizedErrorShares | null {
  const directShares = normalizeShares(direct)
  const countShares = sharesFromCounts(counts)
  if (directShares) {
    return {
      substituteShare: directShares.substituteShare ?? countShares?.substituteShare ?? null,
      insertShare: directShares.insertShare ?? countShares?.insertShare ?? null,
      deleteShare: directShares.deleteShare ?? countShares?.deleteShare ?? null,
    }
  }
  return countShares ?? normalizeShares(fallback)
}

function hasOwnMetricKey(layer: Record<string, unknown>, keys: string[]) {
  return keys.some((key) => Object.prototype.hasOwnProperty.call(layer, key))
}

export function layeredMetricNumber(layers: Array<Record<string, unknown>>, keys: string[]) {
  for (const layer of layers) {
    if (!hasOwnMetricKey(layer, keys)) continue
    for (const key of keys) {
      const value = finiteNumber(layer[key])
      if (value !== null) return value
    }
    // An explicit null in the higher-priority layer means the metric is
    // unavailable. Do not revive a stale value from a legacy outer layer.
    return null
  }
  return null
}
