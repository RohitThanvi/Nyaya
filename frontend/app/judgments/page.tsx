'use client'

import { useState, useEffect } from 'react'
import { FileText, Search, Loader2, Star, Calendar, Scale, ChevronRight, X, BookOpen, Gavel, Hash } from 'lucide-react'
import { documentsApi } from '@/lib/api'
import type { Document, LawCategory } from '@/types/api'
import { toast } from 'sonner'

const LAW_OPTS: (LawCategory | '')[] = ['', 'BNS', 'BNSS', 'BSA', 'IPC', 'CrPC', 'Constitution']

// ── Judgment Detail Modal ─────────────────────────────────────────────────────
function JudgmentModal({ doc, onClose }: { doc: Document; onClose: () => void }) {
  const [summary, setSummary] = useState<string | null>(null)
  const [loadingSummary, setLoadingSummary] = useState(false)

  const loadSummary = async () => {
    setLoadingSummary(true)
    try {
      const data = await documentsApi.summarize(doc.document_id)
      setSummary(data.summary)
    } catch {
      toast.error('Could not load summary')
    } finally {
      setLoadingSummary(false)
    }
  }

  useEffect(() => {
    loadSummary()
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="bg-card border border-border rounded-2xl w-full max-w-2xl max-h-[85vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 p-6 border-b border-border">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 mt-0.5">
              <Scale className="w-4 h-4 text-primary" />
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                {doc.is_landmark && <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />}
                <span className="text-xs text-primary font-mono">{doc.citation}</span>
              </div>
              <h2 className="font-semibold text-base leading-snug">
                {typeof doc.parties === 'object' && doc.parties !== null
                  ? (doc.parties as any).name || doc.topic
                  : doc.topic}
              </h2>
              {doc.court_name && (
                <p className="text-xs text-muted-foreground mt-1">{doc.court_name}</p>
              )}
            </div>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground shrink-0">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Meta */}
        <div className="flex items-center gap-3 px-6 py-3 border-b border-border text-xs text-muted-foreground flex-wrap">
          {doc.law && (
            <span className="border border-primary/30 text-primary rounded-full px-2 py-0.5 bg-primary/5">{doc.law}</span>
          )}
          {doc.year && <span className="flex items-center gap-1"><Calendar className="w-3 h-3" />{doc.year}</span>}
          <span className="flex items-center gap-1"><Hash className="w-3 h-3" />{doc.total_chunks} chunks indexed</span>
          {doc.is_landmark && <span className="text-amber-400 font-medium">Landmark Judgment</span>}
        </div>

        {/* Summary */}
        <div className="flex-1 overflow-y-auto p-6">
          {loadingSummary ? (
            <div className="flex items-center gap-3 text-muted-foreground">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Generating AI summary via pipeline…</span>
            </div>
          ) : summary ? (
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-1 h-4 bg-primary rounded-full" />
                <h3 className="text-sm font-semibold text-foreground">AI Summary</h3>
                <span className="text-xs text-muted-foreground bg-primary/5 border border-primary/20 rounded px-1.5 py-0.5">
                  Summarization Agent
                </span>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">{summary}</p>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">
              <p>Summary unavailable. The document has <strong>{doc.total_chunks}</strong> indexed chunks available for search and Q&A.</p>
              <p className="mt-3 text-xs">Use <strong>AI Chat</strong> to ask questions about this case, or <strong>Legal Search</strong> to find specific provisions.</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border flex gap-2">
          <a href="/chat" className="flex-1 text-center text-sm bg-primary text-primary-foreground py-2 rounded-lg font-medium hover:bg-primary/90 transition-colors">
            Ask AI about this case
          </a>
          <button onClick={onClose} className="px-4 text-sm border border-border rounded-lg hover:bg-accent transition-colors">
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Document Card ─────────────────────────────────────────────────────────────
function DocumentCard({ doc, onClick }: { doc: Document; onClick: () => void }) {
  const title = typeof doc.parties === 'object' && doc.parties !== null
    ? (doc.parties as any).name || doc.citation || doc.topic
    : doc.citation || doc.topic || 'Untitled'

  return (
    <div onClick={onClick}
      className="group bg-card border border-border rounded-xl p-5 hover:border-primary/40 hover:bg-card/80 transition-all cursor-pointer">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 mt-0.5">
            <Scale className="w-3.5 h-3.5 text-primary" />
          </div>
          <div>
            <p className="text-sm font-medium leading-snug group-hover:text-primary transition-colors line-clamp-2">
              {title}
            </p>
            {doc.court_name && (
              <p className="text-xs text-muted-foreground mt-0.5">{doc.court_name}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {doc.is_landmark && <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />}
          <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary group-hover:translate-x-0.5 transition-all" />
        </div>
      </div>
      <div className="flex items-center gap-3 text-xs text-muted-foreground flex-wrap">
        {doc.law && (
          <span className="border border-primary/30 text-primary rounded-full px-2 py-0.5 bg-primary/5">{doc.law}</span>
        )}
        {doc.year && <span className="flex items-center gap-1"><Calendar className="w-3 h-3" />{doc.year}</span>}
        <span>{doc.total_chunks} chunks</span>
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function JudgmentsPage() {
  const [docs, setDocs] = useState<Document[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [isLoading, setIsLoading] = useState(true)
  const [lawFilter, setLawFilter] = useState<string>('')
  const [yearFilter, setYearFilter] = useState('')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<Document | null>(null)

  const fetchDocs = async (p = 1) => {
    setIsLoading(true)
    try {
      const res = await documentsApi.list({
        law: lawFilter || undefined,
        year: yearFilter ? parseInt(yearFilter) : undefined,
        document_type: 'judgment',
        page: p,
        page_size: 20,
      })
      setDocs(res.documents)
      setTotal(res.total)
      setPage(p)
    } catch {
      toast.error('Failed to load judgments')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => { fetchDocs(1) }, [lawFilter, yearFilter])

  const filtered = docs.filter(d => {
    if (!search) return true
    const s = search.toLowerCase()
    const title = typeof d.parties === 'object' && d.parties !== null
      ? (d.parties as any).name || '' : ''
    return (
      d.citation?.toLowerCase().includes(s) ||
      d.topic?.toLowerCase().includes(s) ||
      title.toLowerCase().includes(s)
    )
  })

  const totalPages = Math.ceil(total / 20)

  return (
    <div className="min-h-screen bg-background">
      {selected && <JudgmentModal doc={selected} onClose={() => setSelected(null)} />}

      <div className="max-w-5xl mx-auto px-6 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold mb-1">Judgments Library</h1>
          <p className="text-sm text-muted-foreground">
            {total} judgments indexed · Click any case to view AI summary
          </p>
          {/* Agent pipeline info */}
          <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground bg-card border border-border rounded-lg px-3 py-2 w-fit">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Pipeline: Query Understanding → Hybrid Retrieval → Cross-Encoder Reranking → Summarization Agent
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-3 mb-6 flex-wrap">
          <div className="relative flex-1 min-w-[200px] max-w-xs">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <input value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Search case name or citation…"
              className="w-full pl-9 pr-3 py-2 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground" />
          </div>
          <select value={lawFilter} onChange={e => setLawFilter(e.target.value)}
            className="bg-card border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 text-foreground">
            {LAW_OPTS.map(l => <option key={l} value={l}>{l || 'All Laws'}</option>)}
          </select>
          <input value={yearFilter} onChange={e => setYearFilter(e.target.value)}
            placeholder="Year" type="number" min="1950" max="2030"
            className="w-24 bg-card border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground" />
        </div>

        {/* Grid */}
        {isLoading ? (
          <div className="text-center py-24">
            <Loader2 className="w-6 h-6 mx-auto text-primary animate-spin mb-3" />
            <p className="text-sm text-muted-foreground">Loading judgments…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-24 text-muted-foreground">
            <FileText className="w-10 h-10 mx-auto mb-4 opacity-30" />
            <p className="text-sm">No judgments found.</p>
            <p className="text-xs mt-2 opacity-60">Run the seed_judgments script to populate landmark SC cases.</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-8">
              {filtered.map(doc => (
                <DocumentCard key={doc.document_id} doc={doc} onClick={() => setSelected(doc)} />
              ))}
            </div>
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2">
                <button onClick={() => fetchDocs(page - 1)} disabled={page === 1}
                  className="px-3 py-1.5 text-sm border border-border rounded-lg disabled:opacity-40 hover:border-primary/40 transition-colors">
                  Previous
                </button>
                <span className="text-sm text-muted-foreground">Page {page} of {totalPages}</span>
                <button onClick={() => fetchDocs(page + 1)} disabled={page === totalPages}
                  className="px-3 py-1.5 text-sm border border-border rounded-lg disabled:opacity-40 hover:border-primary/40 transition-colors">
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
