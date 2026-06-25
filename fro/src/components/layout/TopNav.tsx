import { Bell, ChevronDown, ShieldCheck, UserRound } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import navShield from '@/assets/reference-nav-shield.png'
import { cn } from '@/lib/utils'

const navItems = [
  { label: '首页', href: '/' },
  { label: '防护工作台', href: '/workspace' },
  { label: '结果分析', href: '/results' },
  { label: '历史任务', href: '/history' },
]

export function TopNav() {
  return (
    <header className="sticky top-0 z-40 border-b border-cyan-300/10 bg-[#020a13]/95 shadow-[0_1px_0_rgba(56,189,248,0.06)] backdrop-blur-xl">
      <div className="mx-auto grid h-[66px] max-w-[1586px] grid-cols-[390px_1fr_390px] items-center px-[30px] max-xl:grid-cols-[1fr_auto] max-lg:h-[62px] max-lg:px-4">
        <NavLink to="/" className="flex min-w-0 items-center gap-3 whitespace-nowrap">
          <img src={navShield} alt="" className="h-[50px] w-[52px] shrink-0 object-contain drop-shadow-[0_0_14px_rgba(14,165,233,0.55)]" />
          <div className="flex min-w-0 items-center gap-3 whitespace-nowrap">
            <span className="truncate text-[27px] font-black leading-none tracking-normal text-white max-lg:text-xl">语音克隆防护平台</span>
            <span className="rounded-[5px] border border-cyan-400/60 px-1.5 py-0.5 font-mono text-xs font-bold text-cyan-300">V2.0</span>
          </div>
        </NavLink>

        <nav className="flex h-full justify-center max-lg:hidden">
          {navItems.map((item) => (
            <NavLink
              key={item.href}
              to={item.href}
              className={({ isActive }) =>
                cn(
                  'relative flex h-full min-w-[112px] items-center justify-center px-5 text-[17px] font-semibold text-slate-300 transition',
                  'hover:bg-cyan-400/[0.06] hover:text-white',
                  isActive &&
                    'bg-cyan-400/[0.08] text-white after:absolute after:inset-x-0 after:bottom-0 after:h-[4px] after:bg-cyan-400 after:shadow-[0_0_18px_rgba(34,211,238,0.95)]',
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center justify-end gap-5 max-lg:gap-2">
          <div className="hidden h-[36px] items-center gap-2 rounded-full border border-cyan-300/12 bg-cyan-400/[0.05] px-4 py-2 text-sm font-semibold text-slate-300 md:flex">
            <ShieldCheck className="h-4 w-4 text-emerald-300" />
            防护工作流
          </div>
          <button className="grid h-10 w-10 place-items-center rounded-full text-slate-300 transition hover:bg-white/5" aria-label="通知">
            <Bell className="h-5 w-5" />
          </button>
          <div className="hidden items-center gap-2 rounded-full border border-cyan-300/12 bg-white/[0.04] px-3.5 py-2 text-sm font-semibold text-slate-200 md:flex">
            <span className="grid h-8 w-8 place-items-center rounded-full bg-slate-700 text-slate-200">
              <UserRound className="h-5 w-5" />
            </span>
            评委用户
            <ChevronDown className="h-4 w-4 text-slate-400" />
          </div>
        </div>
      </div>
    </header>
  )
}
