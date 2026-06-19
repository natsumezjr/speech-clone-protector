import { Bell, Shield, Waves, UserRound } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { Badge } from '@/components/common/Badge'
import { StatusBadge } from './StatusBadge'
import { cn } from '@/lib/utils'

const navItems = [
  { label: '首页', href: '/' },
  { label: '防护工作台', href: '/workspace' },
  { label: '结果分析', href: '/results/mock-task-001' },
  { label: '历史任务', href: '/history' },
]

export function TopNav() {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-slate-950/76 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-[1500px] items-center justify-between gap-5 px-5">
        <NavLink to="/" className="flex min-w-0 items-center gap-3">
          <div className="relative grid h-10 w-10 place-items-center rounded-xl border border-cyan-300/30 bg-cyan-300/10">
            <Shield className="h-5 w-5 text-cyan-200" />
            <Waves className="absolute bottom-1.5 h-3.5 w-3.5 text-emerald-300" />
          </div>
          <div className="hidden min-w-0 sm:block">
            <div className="flex items-center gap-2">
              <span className="truncate text-sm font-bold text-white">语音克隆防护平台</span>
              <Badge tone="blue" className="px-2 py-0.5">
                V2.0
              </Badge>
            </div>
            <p className="text-xs text-slate-400">发布前源头防护</p>
          </div>
        </NavLink>

        <nav className="hidden items-center gap-1 lg:flex">
          {navItems.map((item) => (
            <NavLink
              key={item.href}
              to={item.href}
              className={({ isActive }) =>
                cn(
                  'rounded-lg px-3 py-2 text-sm font-medium text-slate-300 transition hover:bg-white/8 hover:text-white',
                  isActive && 'bg-cyan-400/10 text-cyan-100 ring-1 ring-cyan-300/20',
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <StatusBadge />
          <button className="rounded-lg border border-white/10 bg-white/5 p-2 text-slate-300 hover:bg-white/10" aria-label="通知">
            <Bell className="h-4 w-4" />
          </button>
          <div className="hidden items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200 md:flex">
            <UserRound className="h-4 w-4 text-cyan-200" />
            评委用户
          </div>
        </div>
      </div>
    </header>
  )
}
