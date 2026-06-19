import ReactECharts from 'echarts-for-react'
import type { PsychoacousticPoint } from '@/types/task'

export function PsychoacousticChart({ data }: { data: PsychoacousticPoint[] }) {
  const option = {
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#cbd5e1' } },
    grid: { left: 42, right: 20, top: 36, bottom: 36 },
    xAxis: { type: 'category', name: '频率 Hz', data: data.map((item) => item.frequency), axisLabel: { color: '#94a3b8' } },
    yAxis: {
      type: 'value',
      name: '强度 dB',
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.12)' } },
    },
    series: [
      { name: '掩蔽阈值', type: 'line', smooth: true, data: data.map((item) => item.maskingThreshold), color: '#38bdf8' },
      { name: '保护干扰谱', type: 'line', smooth: true, data: data.map((item) => item.perturbation), color: '#22c55e', areaStyle: { opacity: 0.12 } },
    ],
  }

  return <ReactECharts option={option} className="h-72 w-full" />
}
