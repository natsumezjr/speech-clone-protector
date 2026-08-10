import { useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Toaster } from '@/components/common/Toaster'
import { useAppStore } from '@/store/appStore'
import { TopNav } from './TopNav'

export function AppShell() {
  const themeMode = useAppStore((state) => state.themeMode)
  const location = useLocation()

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.dataset.theme = themeMode
    }
  }, [themeMode])

  return (
    <div data-theme={themeMode} className="app-shell min-h-screen overflow-x-hidden bg-[#031221] text-slate-100">
      <TopNav />
      <main className="app-main relative mx-auto max-w-[1586px] px-[30px] pb-[26px] pt-[14px] max-lg:px-4">
        <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_50%_16%,rgba(14,165,233,0.16),transparent_32%),linear-gradient(180deg,#020912_0%,#031221_42%,#02101d_100%)]" />
        <div className="pointer-events-none fixed inset-0 -z-10 bg-[linear-gradient(90deg,rgba(56,189,248,0.035)_1px,transparent_1px),linear-gradient(180deg,rgba(56,189,248,0.026)_1px,transparent_1px)] bg-[size:72px_72px]" />
        <div key={location.pathname} className="route-enter min-h-0">
          <Outlet />
        </div>
      </main>
      <Toaster />
    </div>
  )
}
