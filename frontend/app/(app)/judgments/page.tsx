'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import {
  FileText, Search, Loader2, Star, Calendar, Scale,
  ChevronRight, X, Hash, MessageSquare, BookOpen,
  FileUp, ChevronDown, ChevronUp, Gavel
} from 'lucide-react'
import { documentsApi, getErrorMessage } from '@/lib/api'
import type { Document } from '@/types/api'
import { toast } from 'sonner'

type Tab = 'judgments' | 'uploads'

// ── Full Case Modal ───────────────────────────────────────────────────────────
function CaseModal({ doc, onClose }: { doc: Document; onClose: () => void }) {
  const router = useRouter()
  const [summary, setSummary] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [expandedSection, setExpandedSection] = useState<string | null>('summary_brief')

  const title = typeof doc.parties === 'object' && doc.parties !== null
    ? (doc.parties as any).name || doc.topic
    : doc.topic || doc.citation || 'Untitled'

  useEffect(() => {
    const load = async () => {
      try {
        const data = await documentsApi.summarize(doc.document_id)
        setSummary(data)
      } catch (e: any) {
        toast.error(getErrorMessage(e, 'Could not load case details'))
      } finally {
        setLoading(false)
      }
    }
    load()
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const chatAboutCase = () => {
    const params = new URLSearchParams({ document_id: doc.document_id, document_title: title })
    router.push(`/chat?${params.toString()}`)
  }

  const SECTIONS = summary ? [
    { key: 'summary_brief', label: 'Executive Summary', content: summary.summary_brief, always: true },
    { key: 'facts', label: 'Facts', content: summary.facts },
    { key: 'issues', label: 'Legal Issues', content: Array.isArray(summary.issues) ? summary.issues.join('\n') : summary.issues },
    { key: 'findings', label: 'Findings', content: summary.findings },
    { key: 'ratio_decidendi', label: 'Ratio Decidendi', content: summary.ratio_decidendi },
    { key: 'final_order', label: 'Final Order', content: summary.final_order },
    { key: 'sections_discussed', label: 'Sections Discussed', content: Array.isArray(summary.sections_discussed) ? summary.sections_discussed.join(', ') : null },
  ].filter(s => s.content) : []

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="bg-card border border-border rounded-2xl w-full max-w-2xl max-h-[88vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 p-6 border-b border-border">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 mt-0.5">
              <Scale className="w-4 h-4 text-primary" />
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1 flex-wrap">
                {doc.is_landmark && <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />}
                {doc.citation && <span className="text-xs text-primary font-mono">{doc.citation}</span>}
                {doc.year && <span className="text-xs text-muted-foreground">{doc.year}</span>}
              </div>
              <h2 className="font-semibold text-base leading-snug">{title}</h2>
              {doc.court_name && <p className="text-xs text-muted-foreground mt-1">{doc.court_name}</p>}
            </div>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground shrink-0">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Meta pills */}
        <div className="flex items-center gap-2 px-6 py-3 border-b border-border text-xs flex-wrap">
          {doc.law && <span className="border border-primary/30 text-primary rounded-full px-2 py-0.5 bg-primary/5">{doc.law}</span>}
          <span className="flex items-center gap-1 text-muted-foreground"><Hash className="w-3 h-3" />{doc.total_chunks} chunks indexed</span>
          {doc.is_landmark && <span className="text-amber-400 font-medium flex items-center gap-1"><Star className="w-3 h-3 fill-amber-400" />Landmark</span>}
          {summary?.sections_discussed?.length > 0 && (
            <span className="text-muted-foreground">Sections: {summary.sections_discussed.slice(0, 3).join(', ')}{summary.sections_discussed.length > 3 ? '…' : ''}</span>
          )}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-3">
          {loading ? (
            <div className="flex items-center gap-3 text-muted-foreground py-8 justify-center">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Summarization Agent running…</span>
            </div>
          ) : SECTIONS.length > 0 ? (
            SECTIONS.map(({ key, label, content }) => (
              <div key={key} className="border border-border rounded-xl overflow-hidden">
                <button
                  onClick={() => setExpandedSection(expandedSection === key ? null : key)}
                  className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium hover:bg-accent/50 transition-colors text-left">
                  <span>{label}</span>
                  {expandedSection === key
                    ? <ChevronUp className="w-4 h-4 text-muted-foreground" />
                    : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
                </button>
                {expandedSection === key && (
                  <div className="px-4 pb-4 text-sm text-muted-foreground leading-relaxed border-t border-border pt-3">
                    {content}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-center py-8 text-muted-foreground text-sm">
              <p>Could not generate structured summary.</p>
              <p className="text-xs mt-2">Use AI Chat to ask questions about this case.</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border flex gap-2">
          <button onClick={chatAboutCase}
            className="flex-1 flex items-center justify-center gap-2 text-sm bg-primary text-primary-foreground py-2.5 rounded-lg font-medium hover:bg-primary/90 transition-colors">
            <MessageSquare className="w-4 h-4" />
            Chat about this case
          </button>
          <button onClick={onClose}
            className="px-4 text-sm border border-border rounded-lg hover:bg-accent transition-colors">
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Document Card ─────────────────────────────────────────────────────────────
function DocCard({ doc, onClick, isUpload = false }: { doc: Document; onClick: () => void; isUpload?: boolean }) {
  const router = useRouter()
  const title = isUpload
    ? (doc.topic || doc.document_id)
    : (typeof doc.parties === 'object' && doc.parties !== null
      ? (doc.parties as any).name || doc.citation || doc.topic
      : doc.citation || doc.topic || 'Untitled')

  const chatAbout = (e: React.MouseEvent) => {
    e.stopPropagation()
    const params = new URLSearchParams({ document_id: doc.document_id, document_title: title || doc.topic || '' })
    router.push(`/chat?${params.toString()}`)
  }

  return (
    <div onClick={onClick}
      className="group bg-card border border-border rounded-xl p-5 hover:border-primary/40 transition-all cursor-pointer">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 mt-0.5">
            {isUpload ? <FileUp className="w-3.5 h-3.5 text-primary" /> : <Scale className="w-3.5 h-3.5 text-primary" />}
          </div>
          <div>
            <p className="text-sm font-medium leading-snug group-hover:text-primary transition-colors line-clamp-2">{title}</p>
            {doc.court_name && <p className="text-xs text-muted-foreground mt-0.5">{doc.court_name}</p>}
            {isUpload && (
              <p className="text-xs text-muted-foreground mt-0.5">
                Uploaded · {doc.total_chunks} chunks
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {doc.is_landmark && <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />}
          <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary group-hover:translate-x-0.5 transition-all" />
        </div>
      </div>
      <div className="flex items-center justify-between mt-3">
        <div className="flex items-center gap-2 text-xs text-muted-foreground flex-wrap">
          {doc.law && <span className="border border-primary/30 text-primary rounded-full px-2 py-0.5 bg-primary/5">{doc.law}</span>}
          {doc.year && <span className="flex items-center gap-1"><Calendar className="w-3 h-3" />{doc.year}</span>}
          {!isUpload && <span>{doc.total_chunks} chunks</span>}
        </div>
        <button onClick={chatAbout}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-primary border border-border hover:border-primary/40 rounded-lg px-2 py-1 transition-colors">
          <MessageSquare className="w-3 h-3" />Chat
        </button>
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function JudgmentsPage() {
  const [tab, setTab] = useState<Tab>('judgments')
  const [docs, setDocs] = useState<Document[]>([])
  const [uploads, setUploads] = useState<Document[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [lawFilter, setLawFilter] = useState('')
  const [yearFilter, setYearFilter] = useState('')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<Document | null>(null)

  const fetch = async (p = 1, t: Tab = tab) => {
    setLoading(true)
    try {
      const [judgeRes, uploadRes] = await Promise.all([
        t === 'judgments' ? documentsApi.list({
          law: lawFilter || undefined,
          year: yearFilter ? parseInt(yearFilter) : undefined,
          document_type: 'judgment',
          page: p, page_size: 20,
        }) : Promise.resolve(null),
        t === 'uploads' ? documentsApi.list({ document_type: 'upload', page: p, page_size: 20 }) : Promise.resolve(null),
      ])
      if (judgeRes) { setDocs(judgeRes.documents); setTotal(judgeRes.total) }
      if (uploadRes) { setUploads(uploadRes.documents); setTotal(uploadRes.total) }
      setPage(p)
    } catch { toast.error('Failed to load documents') }
    finally { setLoading(false) }
  }

  useEffect(() => { fetch(1, tab) }, [tab, lawFilter, yearFilter])

  const currentDocs = tab === 'judgments' ? docs : uploads
  const filtered = currentDocs.filter(d => {
    if (!search) return true
    const s = search.toLowerCase()
    const t = typeof d.parties === 'object' ? (d.parties as any)?.name || '' : ''
    return d.citation?.toLowerCase().includes(s) || d.topic?.toLowerCase().includes(s) || t.toLowerCase().includes(s)
  })

  return (
    <div className="min-h-screen bg-background">
      {selected && <CaseModal doc={selected} onClose={() => setSelected(null)} />}
      <div className="max-w-5xl mx-auto px-6 py-10">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold mb-1">Document Library</h1>
          <p className="text-sm text-muted-foreground">Browse judgments and uploaded documents. Click to view AI summary and chat.</p>
          <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground bg-card border border-border rounded-lg px-3 py-2 w-fit">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Pipeline: Hybrid Retrieval → Cross-Encoder Reranker → Summarization Agent
          </div>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1 bg-card border border-border rounded-xl p-1 w-fit mb-6">
          {[
            { id: 'judgments', label: 'SC Judgments', icon: Gavel },
            { id: 'uploads', label: 'My Uploads', icon: FileUp },
          ].map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setTab(id as Tab)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${tab === id
                ? 'bg-primary text-primary-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'}`}>
              <Icon className="w-3.5 h-3.5" />{label}
            </button>
          ))}
        </div>

        {/* Filters */}
        <div className="flex items-center gap-3 mb-5 flex-wrap">
          <div className="relative flex-1 min-w-[180px] max-w-xs">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <input value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Search…"
              className="w-full pl-9 pr-3 py-2 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground" />
          </div>
          {tab === 'judgments' && (
            <>
              <select value={lawFilter} onChange={e => setLawFilter(e.target.value)}
                className="bg-card border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30">
                {['', 'BNS', 'BNSS', 'BSA', 'IPC', 'CrPC', 'Constitution'].map(l => (
                  <option key={l} value={l}>{l || 'All Laws'}</option>
                ))}
              </select>
              <input value={yearFilter} onChange={e => setYearFilter(e.target.value)}
                placeholder="Year" type="number" min="1950" max="2030"
                className="w-24 bg-card border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground" />
            </>
          )}
          <span className="text-xs text-muted-foreground ml-auto">{total} documents</span>
        </div>

        {/* Grid */}
        {loading ? (
          <div className="text-center py-24">
            <Loader2 className="w-6 h-6 mx-auto text-primary animate-spin mb-3" />
            <p className="text-sm text-muted-foreground">Loading…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-24 text-muted-foreground">
            <FileText className="w-10 h-10 mx-auto mb-4 opacity-30" />
            <p className="text-sm">{tab === 'uploads' ? 'No uploaded documents yet.' : 'No judgments found.'}</p>
            {tab === 'uploads' && (
              <a href="/upload" className="text-xs text-primary hover:underline mt-2 inline-block">Upload a document →</a>
            )}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-8">
              {filtered.map(doc => (
                <DocCard key={doc.document_id} doc={doc}
                  isUpload={tab === 'uploads'}
                  onClick={() => setSelected(doc)} />
              ))}
            </div>
            {Math.ceil(total / 20) > 1 && (
              <div className="flex items-center justify-center gap-2">
                <button onClick={() => fetch(page - 1)} disabled={page === 1}
                  className="px-3 py-1.5 text-sm border border-border rounded-lg disabled:opacity-40 hover:border-primary/40 transition-colors">Previous</button>
                <span className="text-sm text-muted-foreground">Page {page} of {Math.ceil(total / 20)}</span>
                <button onClick={() => fetch(page + 1)} disabled={page === Math.ceil(total / 20)}
                  className="px-3 py-1.5 text-sm border border-border rounded-lg disabled:opacity-40 hover:border-primary/40 transition-colors">Next</button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
