'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Scale, Search, MessageSquare, FileUp, FileText, PenTool, Activity, ChevronRight, BookOpen, Gavel, Shield } from 'lucide-react'
import { healthApi } from '@/lib/api'

const NAV_ITEMS = [
  { href: '/search', icon: Search, label: 'Legal Search', desc: 'BNS/BNSS/BSA + judgment retrieval', color: 'text-blue-400' },
  { href: '/chat', icon: MessageSquare, label: 'AI Chat', desc: 'Citation-backed legal Q&A', color: 'text-emerald-400' },
  { href: '/upload', icon: FileUp, label: 'Upload Document', desc: 'Analyze your own PDFs', color: 'text-violet-400' },
  { href: '/judgments', icon: FileText, label: 'Judgments', desc: 'Browse Supreme Court library', color: 'text-amber-400' },
  { href: '/drafting', icon: PenTool, label: 'Draft Documents', desc: 'Bail, notices, affidavits', color: 'text-rose-400' },
]

const LAW_STATS = [
  { label: 'BNS Sections', value: '358', icon: BookOpen },
  { label: 'BNSS Sections', value: '531', icon: Gavel },
  { label: 'BSA Sections', value: '170', icon: Shield },
]

export default function DashboardPage() {
  const [health, setHealth] = useState<Record<string, any> | null>(null)

  useEffect(() => {
    healthApi.check().then(setHealth).catch(() => {})
  }, [])

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center">
              <Scale className="w-4 h-4 text-primary" />
            </div>
            <span className="font-semibold text-lg tracking-tight">NyayaAI</span>
            <span className="text-xs text-muted-foreground border border-border rounded px-1.5 py-0.5">BETA</span>
          </div>
          <nav className="flex items-center gap-6">
            {NAV_ITEMS.slice(0, 3).map((item) => (
              <Link key={item.href} href={item.href}
                className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                {item.label}
              </Link>
            ))}
            <Link href="/auth/login"
              className="text-sm bg-primary text-primary-foreground px-3 py-1.5 rounded-md font-medium hover:bg-primary/90 transition-colors">
              Sign In
            </Link>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-16">
        {/* Hero */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 text-xs text-primary border border-primary/30 rounded-full px-3 py-1 mb-6 bg-primary/5">
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
            BNS · BNSS · BSA · Supreme Court Judgments
          </div>
          <h1 className="text-5xl font-bold tracking-tight mb-4">
            Indian Legal Research,{' '}
            <span className="text-primary">Reimagined</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
            Hybrid BM25 + semantic retrieval over Bharatiya codes and Supreme Court judgments.
            Every answer backed by verifiable citations.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Link href="/search"
              className="flex items-center gap-2 bg-primary text-primary-foreground px-6 py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors">
              <Search className="w-4 h-4" />
              Start Searching
            </Link>
            <Link href="/chat"
              className="flex items-center gap-2 border border-border px-6 py-3 rounded-lg font-medium hover:bg-card transition-colors">
              <MessageSquare className="w-4 h-4" />
              Ask a Question
            </Link>
          </div>
        </div>

        {/* Law Stats */}
        <div className="grid grid-cols-3 gap-4 mb-12">
          {LAW_STATS.map(({ label, value, icon: Icon }) => (
            <div key={label} className="bg-card border border-border rounded-xl p-6 text-center">
              <Icon className="w-5 h-5 text-primary mx-auto mb-2" />
              <div className="text-3xl font-bold text-foreground">{value}</div>
              <div className="text-sm text-muted-foreground">{label}</div>
            </div>
          ))}
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">
          {NAV_ITEMS.map(({ href, icon: Icon, label, desc, color }) => (
            <Link key={href} href={href}
              className="group bg-card border border-border rounded-xl p-6 hover:border-primary/40 hover:bg-card/80 transition-all">
              <div className="flex items-start justify-between mb-4">
                <div className={`w-10 h-10 rounded-lg bg-current/10 border border-current/20 flex items-center justify-center ${color}`}>
                  <Icon className="w-5 h-5" />
                </div>
                <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground group-hover:translate-x-0.5 transition-all" />
              </div>
              <h3 className="font-semibold mb-1">{label}</h3>
              <p className="text-sm text-muted-foreground">{desc}</p>
            </Link>
          ))}

          {/* System Status */}
          <div className="bg-card border border-border rounded-xl p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-emerald-400/10 border border-emerald-400/20 flex items-center justify-center">
                <Activity className="w-5 h-5 text-emerald-400" />
              </div>
            </div>
            <h3 className="font-semibold mb-3">System Status</h3>
            {health ? (
              <div className="space-y-1.5">
                {Object.entries(health.components || {}).map(([name, info]: [string, any]) => (
                  <div key={name} className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground capitalize">{name}</span>
                    <span className={info.status === 'ok' ? 'text-emerald-400' : 'text-amber-400'}>
                      {info.status === 'ok' ? `●  ${info.latency_ms}ms` : '○  degraded'}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">Checking...</p>
            )}
          </div>
        </div>

        {/* Legal disclaimer */}
        <div className="text-center text-xs text-muted-foreground border border-border/50 rounded-lg p-4 bg-card/30">
          <strong>Disclaimer:</strong> NyayaAI is a research tool and does not constitute legal advice.
          Always verify citations with official sources. Consult a qualified advocate for legal counsel.
        </div>
      </main>
    </div>
  )
}
