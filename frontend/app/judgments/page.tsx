'use client'

import { useState, useEffect } from 'react'
import { FileText, Search, Filter, Loader2, Star, Calendar, Scale, ChevronRight } from 'lucide-react'
import { documentsApi } from '@/lib/api'
import type { Document, LawCategory } from '@/types/api'
import Link from 'next/link'
import { toast } from 'sonner'

const LAW_OPTS: (LawCategory | '')[] = ['', 'BNS', 'BNSS', 'BSA', 'IPC', 'CrPC', 'Constitution']

function DocumentCard({ doc }: { doc: Document }) {
  return (
    <div className="group bg-card border border-border rounded-xl p-5 hover:border-primary/30 transition-all cursor-pointer">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 mt-0.5">
            <Scale className="w-3.5 h-3.5 text-primary" />
          </div>
          <div>
            <p className="text-sm font-medium leading-snug group-hover:text-primary transition-colors">
              {doc.citation || doc.topic || 'Untitled Document'}
            </p>
            {doc.court_name && (
              <p className="text-xs text-muted-foreground mt-0.5">{doc.court_name}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {doc.is_landmark && (
            <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" aria-label="Landmark judgment" />
          )}
          <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:translate-x-0.5 transition-transform" />
        </div>
      </div>
      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        {doc.law && (
          <span className="border border-primary/30 text-primary rounded-full px-2 py-0.5 bg-primary/5">{doc.law}</span>
        )}
        {doc.year && (
          <span className="flex items-center gap-1"><Calendar className="w-3 h-3" />{doc.year}</span>
        )}
        <span>{doc.total_chunks} chunks</span>
        <span className="capitalize text-muted-foreground/60">{doc.document_type}</span>
      </div>
    </div>
  )
}

export default function JudgmentsPage() {
  const [docs, setDocs] = useState<Document[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [isLoading, setIsLoading] = useState(true)
  const [lawFilter, setLawFilter] = useState<string>('')
  const [yearFilter, setYearFilter] = useState('')
  const [search, setSearch] = useState('')

  const fetchDocs = async (p = 1) => {
    setIsLoading(true)
    try {
      const res = await documentsApi.list({
        law: lawFilter || undefined,
        year: yearFilter ? parseInt(yearFilter) : undefined,
        page: p,
        page_size: 20,
      })
      setDocs(res.documents)
      setTotal(res.total)
      setPage(p)
    } catch {
      toast.error('Failed to load documents')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => { fetchDocs(1) }, [lawFilter, yearFilter])

  const totalPages = Math.ceil(total / 20)

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-5xl mx-auto px-6 py-10">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold mb-1">Legal Library</h1>
            <p className="text-sm text-muted-foreground">{total.toLocaleString()} documents indexed</p>
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-3 mb-6">
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search citations…"
              className="w-full pl-9 pr-3 py-2 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground"
            />
          </div>
          <select
            value={lawFilter}
            onChange={e => setLawFilter(e.target.value)}
            className="bg-card border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 text-foreground"
          >
            {LAW_OPTS.map(l => (
              <option key={l} value={l}>{l || 'All Laws'}</option>
            ))}
          </select>
          <input
            value={yearFilter}
            onChange={e => setYearFilter(e.target.value)}
            placeholder="Year"
            type="number"
            min="1950" max="2025"
            className="w-24 bg-card border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground"
          />
        </div>

        {/* Documents grid */}
        {isLoading ? (
          <div className="text-center py-24">
            <Loader2 className="w-6 h-6 mx-auto text-primary animate-spin mb-3" />
            <p className="text-sm text-muted-foreground">Loading library…</p>
          </div>
        ) : docs.length === 0 ? (
          <div className="text-center py-24 text-muted-foreground">
            <FileText className="w-10 h-10 mx-auto mb-4 opacity-30" />
            <p className="text-sm">No documents found. Try adjusting your filters.</p>
            <p className="text-xs mt-2 opacity-60">Use the ingestion pipeline to add BNS/BNSS/BSA statutes and judgments.</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-8">
              {docs.filter(d => !search || d.citation?.toLowerCase().includes(search.toLowerCase()) || d.topic?.toLowerCase().includes(search.toLowerCase())).map(doc => (
                <DocumentCard key={doc.document_id} doc={doc} />
              ))}
            </div>

            {/* Pagination */}
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
