import ReactECharts from 'echarts-for-react'
import { useAppStore } from '@/store/appStore'
import type { LossTrendPoint } from '@/types/task'
import { lossTrendSeries } from '@/utils/lossTrendSeries'

function lossValue(point: LossTrendPoint, series: (typeof lossTrendSeries)[number]) {
  return point[series.key] ?? ('legacyKey' in series ? point[series.legacyKey] : null)
}

function formatLossValue(value: unknown) {
  const numberValue = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numberValue)) return '未生成'
  return numberValue.toFixed(2)
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

  const gridHeight = 12
  const gridGap = (100 - gridHeight * lossTrendSeries.length) / (lossTrendSeries.length + 1)

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
        return `step: ${step}<br/>${rows}`
      },
    },
    axisPointer: { link: [{ xAxisIndex: lossTrendSeries.map((_, index) => index) }] },
    legend: { show: false },
    grid: lossTrendSeries.map((_, index) => ({
      left: 48,
      right: 16,
      top: `${gridGap + index * (gridHeight + gridGap)}%`,
      height: `${gridHeight}%`,
      containLabel: false,
    })),
    xAxis: lossTrendSeries.map((_, index) => ({
      type: 'category',
      gridIndex: index,
      data: data.map((item) => item.step),
      axisTick: { show: index === lossTrendSeries.length - 1 },
      axisLabel: { show: index === lossTrendSeries.length - 1, color: chartTheme.axisColor },
      axisLine: { lineStyle: { color: chartTheme.axisLineColor } },
    })),
    yAxis: lossTrendSeries.map((_, index) => ({
      type: 'value',
      gridIndex: index,
      axisLabel: {
        color: chartTheme.axisColor,
        formatter: (value: number) => formatLossValue(value),
      },
      splitNumber: 2,
      splitLine: { lineStyle: { color: chartTheme.gridColor } },
    })),
    series: lossTrendSeries.map((series, index) => ({
      name: series.name,
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

  return <ReactECharts option={option} className="h-full w-full" style={{ height: '100%', width: '100%' }} />
}
