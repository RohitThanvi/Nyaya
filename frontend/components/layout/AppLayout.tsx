'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import {
  Scale, Search, MessageSquare, FileUp, FileText,
  PenTool, LayoutDashboard, LogOut, ChevronLeft, ChevronRight,
  Moon, Sun
} from 'lucide-react'
import { useTheme } from 'next-themes'
import { authApi } from '@/lib/api'
import { useAuthStore } from '@/lib/store'
import { toast } from 'sonner'

const NAV = [
  { href: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { href: '/search', icon: Search, label: 'Legal Search' },
  { href: '/chat', icon: MessageSquare, label: 'AI Chat' },
  { href: '/upload', icon: FileUp, label: 'Upload' },
  { href: '/judgments', icon: FileText, label: 'Judgments' },
  { href: '/drafting', icon: PenTool, label: 'Drafting' },
]

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const router = useRouter()
  const { user, clearUser, syncWithCookies } = useAuthStore()
  const { theme, setTheme } = useTheme()
  const [collapsed, setCollapsed] = useState(false)

  // On every mount, verify the Zustand persisted auth state still matches
  // the actual cookie. If the access_token cookie expired/was cleared while
  // the app was closed, isAuthenticated in localStorage would still be true
  // and the user would see a flash of authenticated UI before the 401 fires.
  useEffect(() => { syncWithCookies() }, [])

  const handleLogout = () => {
    authApi.logout()
    clearUser()
    router.push('/auth/login')
    toast.success('Signed out')
  }

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Sidebar */}
      <aside className={`flex flex-col shrink-0 border-r border-border bg-card/50 transition-all duration-200 ${collapsed ? 'w-16' : 'w-56'}`}>
        {/* Logo */}
        <div className="flex items-center justify-between px-3 py-4 border-b border-border">
          {!collapsed && (
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center">
                <Scale className="w-3.5 h-3.5 text-primary" />
              </div>
              <span className="font-semibold text-sm">NyayaAI</span>
            </div>
          )}
          {collapsed && (
            <div className="mx-auto w-7 h-7 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center">
              <Scale className="w-3.5 h-3.5 text-primary" />
            </div>
          )}
          {!collapsed && (
            <button onClick={() => setCollapsed(true)} className="text-muted-foreground hover:text-foreground">
              <ChevronLeft className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 py-3 space-y-0.5 px-2">
          {NAV.map(({ href, icon: Icon, label }) => {
            const active = pathname === href || pathname.startsWith(href + '/')
            return (
              <Link key={href} href={href}
                className={`flex items-center gap-3 px-2 py-2 rounded-lg text-sm transition-colors ${
                  active
                    ? 'bg-primary/10 text-primary border border-primary/20'
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                } ${collapsed ? 'justify-center' : ''}`}
                title={collapsed ? label : undefined}
              >
                <Icon className="w-4 h-4 shrink-0" />
                {!collapsed && <span>{label}</span>}
              </Link>
            )
          })}
        </nav>

        {/* Bottom actions */}
        <div className="p-2 border-t border-border space-y-1">
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className={`w-full flex items-center gap-3 px-2 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-accent transition-colors ${collapsed ? 'justify-center' : ''}`}
          >
            {theme === 'dark' ? <Sun className="w-4 h-4 shrink-0" /> : <Moon className="w-4 h-4 shrink-0" />}
            {!collapsed && <span>Toggle Theme</span>}
          </button>

          {user && (
            <div className={`flex items-center gap-2 px-2 py-2 ${collapsed ? 'justify-center' : ''}`}>
              {!collapsed && (
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate">{user.full_name}</p>
                  <p className="text-xs text-muted-foreground capitalize">{user.role}</p>
                </div>
              )}
              <button onClick={handleLogout}
                className="text-muted-foreground hover:text-foreground transition-colors"
                title="Sign out">
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          )}

          {collapsed && (
            <button onClick={() => setCollapsed(false)}
              className="w-full flex justify-center px-2 py-2 text-muted-foreground hover:text-foreground">
              <ChevronRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  )
}
