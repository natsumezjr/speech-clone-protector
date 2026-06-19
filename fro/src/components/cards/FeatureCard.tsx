import { Panel } from '@/components/common/Panel'

interface FeatureCardProps {
  title: string
  items: string[]
}

export function FeatureCard({ title, items }: FeatureCardProps) {
  return (
    <Panel>
      <h3 className="mb-4 text-lg font-semibold text-white">{title}</h3>
      <ul className="space-y-3 text-sm leading-6 text-slate-300">
        {items.map((item) => (
          <li key={item} className="border-l border-cyan-300/35 pl-3">
            {item}
          </li>
        ))}
      </ul>
    </Panel>
  )
}
