'use client'

import { useState, useCallback, useRef } from 'react'
import { Search, Filter, X, BookOpen, Gavel, AlertTriangle, CheckCircle2, Clock, ChevronDown, ChevronUp } from 'lucide-react'
import { searchApi } from '@/lib/api'
import { useSearchStore } from '@/lib/store'
import type { LegalResponse, LawCategory, RelevantSection, Precedent } from '@/types/api'
import { toast } from 'sonner'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const LAW_OPTIONS: LawCategory[] = ['BNS', 'BNSS', 'BSA', 'IPC', 'CrPC', 'Constitution']

function ConfidenceBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const cls = score > 0.75 ? 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30'
    : score > 0.5 ? 'text-amber-400 bg-amber-400/10 border-amber-400/30'
    : 'text-red-400 bg-red-400/10 border-red-400/30'
  return (
    <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border ${cls}`}>
      {pct}% confidence
    </span>
  )
}

function SectionCard({ section }: { section: RelevantSection }) {
  const [expanded, setExpanded] = useState(false)
  const confColor = section.confidence === 'HIGH' ? 'text-emerald-400' : section.confidence === 'MEDIUM' ? 'text-amber-400' : 'text-red-400'
  return (
    <div className="border border-border rounded-lg bg-card/50 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start justify-between p-4 text-left hover:bg-card transition-colors"
      >
        <div className="flex items-start gap-3">
          <div className="mt-0.5 w-8 h-8 rounded-md bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
            <BookOpen className="w-3.5 h-3.5 text-primary" />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span className="font-mono text-sm font-semibold text-primary">{section.law} §{section.section_number}</span>
              <span className={`text-xs font-medium ${confColor}`}>{section.confidence}</span>
            </div>
            <p className="text-sm font-medium text-foreground">{section.title}</p>
          </div>
        </div>
        {expanded ? <ChevronUp className="w-4 h-4 text-muted-foreground shrink-0 mt-1" /> : <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0 mt-1" />}
      </button>
      {expanded && (
        <div className="px-4 pb-4 border-t border-border/50 pt-3 space-y-3">
          <p className="text-sm text-foreground/80">{section.relevance}</p>
          {section.elements_to_prove?.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground mb-1.5 font-medium uppercase tracking-wide">Elements to prove</p>
              <ul className="space-y-1">
                {section.elements_to_prove.map((el, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-foreground/70">
                    <span className="text-primary mt-0.5">·</span> {el}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {section.punishment && (
            <div className="flex items-center gap-2 text-xs text-amber-400 bg-amber-400/5 border border-amber-400/20 rounded px-3 py-2">
              <Gavel className="w-3 h-3 shrink-0" />
              {section.punishment}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function PrecedentCard({ precedent }: { precedent: Precedent }) {
  return (
    <div className="border border-border/60 rounded-lg p-3 bg-card/30 hover:border-border transition-colors">
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <span className="font-mono text-xs text-primary font-medium">{precedent.citation}</span>
        <span className="text-xs text-muted-foreground shrink-0">{precedent.court} {precedent.year && `· ${precedent.year}`}</span>
      </div>
      <p className="text-xs text-foreground/70 line-clamp-2">{precedent.relevance}</p>
    </div>
  )
}

function WarningBanner({ warnings }: { warnings: string[] }) {
  if (!warnings?.length) return null
  return (
    <div className="flex items-start gap-3 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
      <div className="space-y-1">
        {warnings.map((w, i) => (
          <p key={i} className="text-xs text-amber-200/80">{w}</p>
        ))}
      </div>
    </div>
  )
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [lawFilter, setLawFilter] = useState<LawCategory[]>([])
  const [yearFrom, setYearFrom] = useState('')
  const [yearTo, setYearTo] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const { lastResult, setResult, setSearching } = useSearchStore()
  const inputRef = useRef<HTMLInputElement>(null)

  const toggleLaw = (law: LawCategory) => {
    setLawFilter(prev => prev.includes(law) ? prev.filter(l => l !== law) : [...prev, law])
  }

  const handleSearch = useCallback(async () => {
    if (!query.trim() || isLoading) return
    setIsLoading(true)
    setSearching(true)
    try {
      const result = await searchApi.search({
        query: query.trim(),
        law_filter: lawFilter.length ? lawFilter : undefined,
        year_from: yearFrom ? parseInt(yearFrom) : undefined,
        year_to: yearTo ? parseInt(yearTo) : undefined,
        top_k: 10,
      })
      setResult(query.trim(), result)
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Search failed. Please try again.')
    } finally {
      setIsLoading(false)
      setSearching(false)
    }
  }, [query, lawFilter, yearFrom, yearTo, isLoading, setResult, setSearching])

  const result = lastResult

  return (
    <div className="min-h-screen bg-background">
      {/* Search bar */}
      <div className="sticky top-0 z-40 bg-background/95 backdrop-blur border-b border-border">
        <div className="max-w-5xl mx-auto px-6 py-4">
          <div className="flex gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSearch()}
                placeholder="e.g. 'anticipatory bail in financial fraud' or 'BNS Section 318'"
                className="w-full pl-10 pr-4 py-2.5 bg-card border border-border rounded-lg text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50"
              />
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`flex items-center gap-2 px-3 py-2.5 rounded-lg border text-sm transition-colors ${showFilters ? 'border-primary/50 bg-primary/5 text-primary' : 'border-border hover:border-primary/30 text-muted-foreground hover:text-foreground'}`}
            >
              <Filter className="w-4 h-4" />
              Filters {lawFilter.length > 0 && <span className="bg-primary text-primary-foreground text-xs rounded-full w-4 h-4 flex items-center justify-center">{lawFilter.length}</span>}
            </button>
            <button
              onClick={handleSearch}
              disabled={isLoading || !query.trim()}
              className="px-5 py-2.5 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isLoading ? (
                <span className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              Search
            </button>
          </div>

          {/* Filters */}
          {showFilters && (
            <div className="mt-3 p-4 bg-card border border-border rounded-lg space-y-3">
              <div>
                <p className="text-xs text-muted-foreground mb-2 font-medium uppercase tracking-wide">Law Category</p>
                <div className="flex flex-wrap gap-2">
                  {LAW_OPTIONS.map(law => (
                    <button
                      key={law}
                      onClick={() => toggleLaw(law)}
                      className={`text-xs px-3 py-1 rounded-full border transition-colors ${lawFilter.includes(law) ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:border-primary/40'}`}
                    >
                      {law}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Year from</p>
                  <input value={yearFrom} onChange={e => setYearFrom(e.target.value)}
                    placeholder="1950" type="number" min="1950" max="2025"
                    className="w-24 px-2 py-1 bg-background border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary/50" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Year to</p>
                  <input value={yearTo} onChange={e => setYearTo(e.target.value)}
                    placeholder="2025" type="number" min="1950" max="2025"
                    className="w-24 px-2 py-1 bg-background border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary/50" />
                </div>
                {(lawFilter.length > 0 || yearFrom || yearTo) && (
                  <button onClick={() => { setLawFilter([]); setYearFrom(''); setYearTo('') }}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground mt-4">
                    <X className="w-3 h-3" /> Clear
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      <div className="max-w-5xl mx-auto px-6 py-8">
        {!result && !isLoading && (
          <div className="text-center py-24 text-muted-foreground">
            <Search className="w-10 h-10 mx-auto mb-4 opacity-30" />
            <p className="text-sm">Search BNS provisions, BNSS procedures, BSA evidence rules, or Supreme Court judgments</p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {['Section 318 BNS cheating', 'Anticipatory bail BNSS 482', 'Electronic evidence BSA', 'Quashing FIR Supreme Court'].map(eg => (
                <button key={eg} onClick={() => { setQuery(eg); setTimeout(handleSearch, 0) }}
                  className="text-xs border border-border rounded-full px-3 py-1 hover:border-primary/40 hover:text-primary transition-colors">
                  {eg}
                </button>
              ))}
            </div>
          </div>
        )}

        {isLoading && (
          <div className="text-center py-24">
            <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm text-muted-foreground">Searching BNS, BNSS, BSA + judgments...</p>
          </div>
        )}

        {result && !isLoading && (
          <div className="space-y-6">
            {/* Meta row */}
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <span>Query: <span className="text-foreground">"{result.query}"</span></span>
              {result.intent && <span>Intent: <span className="text-primary capitalize">{result.intent.replace('_', ' ')}</span></span>}
              {result.latency_ms && <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{result.latency_ms}ms</span>}
              <ConfidenceBadge score={result.confidence} />
            </div>

            <WarningBanner warnings={result.warnings} />

            {/* AI Answer */}
            <div className="bg-card border border-border rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-medium">Analysis</span>
              </div>
              <div className="prose prose-sm prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.answer}</ReactMarkdown>
              </div>
            </div>

            {/* Relevant Sections */}
            {result.relevant_sections?.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-primary" />
                  Applicable Provisions ({result.relevant_sections.length})
                </h3>
                <div className="space-y-2">
                  {result.relevant_sections.map((sec, i) => (
                    <SectionCard key={i} section={sec} />
                  ))}
                </div>
              </div>
            )}

            {/* Procedural Requirements */}
            {result.procedural_requirements?.length > 0 && (
              <div className="bg-card/50 border border-border/60 rounded-lg p-4">
                <h3 className="text-sm font-semibold mb-2 text-amber-400">Procedural Requirements (BNSS)</h3>
                <ul className="space-y-1">
                  {result.procedural_requirements.map((req, i) => (
                    <li key={i} className="text-sm text-foreground/80 flex items-start gap-2">
                      <span className="text-amber-400 mt-1">·</span> {req}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Precedents */}
            {result.precedents?.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <Gavel className="w-4 h-4 text-primary" />
                  Relevant Judgments ({result.precedents.length})
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {result.precedents.map((prec, i) => (
                    <PrecedentCard key={i} precedent={prec} />
                  ))}
                </div>
              </div>
            )}

            {/* Citations */}
            {result.citations?.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold mb-2 text-muted-foreground uppercase tracking-wide text-xs">Verified Citations</h3>
                <div className="flex flex-wrap gap-2">
                  {result.citations.map((cit, i) => (
                    <span key={i} className="citation-card flex items-center gap-1.5">
                      {cit.verified && <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />}
                      {cit.citation_text}
                      {cit.section && <span className="text-muted-foreground">§{cit.section}</span>}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
