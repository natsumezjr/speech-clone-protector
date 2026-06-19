import ReactECharts from 'echarts-for-react'
import type { TrendPoint } from '@/types/task'

export function TrendChart({ data }: { data: TrendPoint[] }) {
  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(7, 16, 31, 0.94)',
      borderColor: 'rgba(56, 189, 248, 0.22)',
      textStyle: { color: '#e2e8f0' },
    },
    legend: { textStyle: { color: '#cbd5e1' }, top: 0 },
    grid: { left: 36, right: 16, top: 42, bottom: 28 },
    xAxis: { type: 'category', data: data.map((item) => item.step), axisLabel: { color: '#94a3b8' } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: 'rgba(148,163,184,0.12)' } } },
    series: [
      { name: 'WER', type: 'line', smooth: true, data: data.map((item) => item.wer), color: '#22d3ee' },
      { name: 'Feature 相似度', type: 'line', smooth: true, data: data.map((item) => item.sim), color: '#22c55e' },
      { name: 'MOS-LQO', type: 'line', smooth: true, data: data.map((item) => item.mos / 5), color: '#f59e0b' },
      { name: 'PESQ', type: 'line', smooth: true, data: data.map((item) => item.pesq / 5), color: '#8b5cf6' },
    ],
  }

  return <ReactECharts option={option} className="h-72 w-full" />
}
