import ReactECharts from 'echarts-for-react'

interface RadarChartProps {
  before: number[]
  after: number[]
}

export function RadarChart({ before, after }: RadarChartProps) {
  const option = {
    tooltip: {},
    legend: { data: ['防护前', '防护后'], textStyle: { color: '#cbd5e1' }, top: 0 },
    radar: {
      radius: '64%',
      indicator: ['Feature 相似度', '基频特征', '共振峰特征', '韵律特征', '声道特征'].map((name) => ({ name, max: 1 })),
      axisName: { color: '#cbd5e1' },
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.18)' } },
      splitArea: { areaStyle: { color: ['rgba(15,23,42,0.4)', 'rgba(15,23,42,0.18)'] } },
      axisLine: { lineStyle: { color: 'rgba(148,163,184,0.18)' } },
    },
    series: [
      {
        type: 'radar',
        data: [
          { value: before, name: '防护前', areaStyle: { color: 'rgba(34,211,238,0.18)' }, lineStyle: { color: '#22d3ee' } },
          { value: after, name: '防护后', areaStyle: { color: 'rgba(34,197,94,0.16)' }, lineStyle: { color: '#22c55e' } },
        ],
      },
    ],
  }
  return <ReactECharts option={option} className="h-72 w-full" />
}
