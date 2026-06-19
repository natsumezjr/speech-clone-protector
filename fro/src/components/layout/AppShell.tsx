import { Outlet } from 'react-router-dom'
import { Toaster } from '@/components/common/Toaster'
import { TopNav } from './TopNav'

export function AppShell() {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_20%_0%,rgba(6,182,212,0.14),transparent_28%),linear-gradient(135deg,#020617_0%,#07111f_45%,#0b1220_100%)] text-slate-100">
      <TopNav />
      <main className="mx-auto max-w-[1500px] px-5 py-7">
        <Outlet />
      </main>
      <Toaster />
    </div>
  )
}
