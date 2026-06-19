export function MiniSparkline({ values }: { values: number[] }) {
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(1, values.length - 1)) * 100
      const y = 40 - value * 34
      return `${x},${y}`
    })
    .join(' ')

  return (
    <svg viewBox="0 0 100 44" className="h-11 w-full overflow-visible">
      <polyline fill="none" stroke="#22d3ee" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" points={points} />
    </svg>
  )
}
