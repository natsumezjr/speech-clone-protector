import ReactECharts from 'echarts-for-react'
import { useAppStore } from '@/store/appStore'
import type { LossTrendPoint } from '@/types/task'

const lossSeries = [
  { key: 'Lid', legacyKey: 'Lfeat', tooltipName: 'L<sub>id</sub>', color: '#22d3ee' },
  { key: 'Lsem', tooltipName: 'L<sub>sem</sub>', color: '#22c55e' },
  { key: 'Lpsy', tooltipName: 'L<sub>psy</sub>', color: '#f59e0b' },
  { key: 'L2', tooltipName: 'L<sub>2</sub>', color: '#a78bfa' },
] as const

function lossValue(point: LossTrendPoint, series: (typeof lossSeries)[number]) {
  return point[series.key] ?? ('legacyKey' in series ? point[series.legacyKey] : null)
}

function formatLossValue(value: unknown) {
  const numberValue = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numberValue)) return '未生成'
  const abs = Math.abs(numberValue)
  if (abs > 0 && (abs < 0.001 || abs >= 10000)) return numberValue.toExponential(3)
  return numberValue.toFixed(6).replace(/\.?0+$/, '')
}

export function TrendChart({ data }: { data: LossTrendPoint[] }) {
  const themeMode = useAppStore((state) => state.themeMode)
  const chartTheme =
    themeMode === 'light'
      ? {
          axisColor: '#475569',
          gridColor: 'rgba(100,116,139,0.18)',
          axisLineColor: 'rgba(100,116,139,0.24)',
          tooltipBg: '#ffffff',
          tooltipBorder: 'rgba(14,116,144,0.22)',
          tooltipText: '#0f172a',
        }
      : {
          axisColor: '#94a3b8',
          gridColor: 'rgba(148,163,184,0.10)',
          axisLineColor: 'rgba(148,163,184,0.18)',
          tooltipBg: 'rgba(7, 16, 31, 0.96)',
          tooltipBorder: 'rgba(56, 189, 248, 0.22)',
          tooltipText: '#e2e8f0',
        }

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    tooltip: {
      trigger: 'axis',
      backgroundColor: chartTheme.tooltipBg,
      borderColor: chartTheme.tooltipBorder,
      textStyle: { color: chartTheme.tooltipText },
      formatter: (params: Array<{ axisValue: number | string; seriesName: string; marker: string; value: unknown }>) => {
        const step = params[0]?.axisValue ?? '-'
        const rows = params
          .map((item) => `${item.marker}${item.seriesName}: ${formatLossValue(item.value)}`)
          .join('<br/>')
        const point = data.find((item) => String(item.step) === String(step))
        const total = point?.total === null || point?.total === undefined ? '' : `<br/>total loss: ${formatLossValue(point.total)}`
        return `step: ${step}<br/>${rows}${total}`
      },
    },
    axisPointer: { link: [{ xAxisIndex: [0, 1, 2, 3] }] },
    legend: { show: false },
    grid: lossSeries.map((_, index) => ({
      left: 48,
      right: 16,
      top: `${5 + index * 24}%`,
      height: '16%',
      containLabel: false,
    })),
    xAxis: lossSeries.map((_, index) => ({
      type: 'category',
      gridIndex: index,
      data: data.map((item) => item.step),
      axisTick: { show: index === lossSeries.length - 1 },
      axisLabel: { show: index === lossSeries.length - 1, color: chartTheme.axisColor },
      axisLine: { lineStyle: { color: chartTheme.axisLineColor } },
    })),
    yAxis: lossSeries.map((_, index) => ({
      type: 'value',
      gridIndex: index,
      axisLabel: {
        color: chartTheme.axisColor,
        formatter: (value: number) => formatLossValue(value),
      },
      splitNumber: 2,
      splitLine: { lineStyle: { color: chartTheme.gridColor } },
    })),
    series: lossSeries.map((series, index) => ({
      name: series.tooltipName,
      type: 'line',
      xAxisIndex: index,
      yAxisIndex: index,
      smooth: true,
      showSymbol: true,
      symbolSize: 5,
      connectNulls: false,
      data: data.map((item) => lossValue(item, series)),
      color: series.color,
      lineStyle: { width: 2 },
    })),
  }

  return <ReactECharts option={option} className="h-full w-full" />
}
