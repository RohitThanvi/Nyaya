'use client'

import { useState } from 'react'
import { PenTool, Loader2, Copy, Download, AlertTriangle, CheckCircle2, ChevronDown } from 'lucide-react'
import { draftingApi } from '@/lib/api'
import type { DraftType, DraftResponse } from '@/types/api'
import { toast } from 'sonner'

const DRAFT_TYPES: { value: DraftType; label: string; desc: string }[] = [
  { value: 'bail_application', label: 'Bail Application', desc: 'Regular bail under BNSS §480/481' },
  { value: 'anticipatory_bail', label: 'Anticipatory Bail', desc: 'Pre-arrest bail under BNSS §482' },
  { value: 'legal_notice', label: 'Legal Notice', desc: 'Formal demand/legal notice' },
  { value: 'affidavit', label: 'Affidavit', desc: 'Sworn statement of facts' },
  { value: 'complaint', label: 'Complaint', desc: 'Criminal complaint under BNSS §173' },
  { value: 'fir_quashing_petition', label: 'FIR Quashing Petition', desc: 'Petition under Article 226' },
  { value: 'written_statement', label: 'Written Statement', desc: 'Defence in civil suit' },
  { value: 'vakalatnama', label: 'Vakalatnama', desc: 'Authority to represent' },
]

function ConfidenceBadge({ score, notes }: { score: number; notes?: string[] }) {
  const pct = Math.round(score * 100)
  const [open, setOpen] = useState(false)

  const reasons: string[] = []
  if (pct < 100) {
    if (pct < 50) reasons.push('Insufficient facts provided — add more case-specific details')
    if (pct < 70) reasons.push('Some placeholders may need manual completion')
    if (notes?.some(n => n.includes('VERIFY'))) reasons.push('Some citations need advocate verification')
    if (notes?.some(n => n.includes('fill'))) reasons.push('Template fields may need manual input')
    if (reasons.length === 0) reasons.push('Score reflects completeness of provided facts and available legal context')
  }

  return (
    <div className="relative">
      <button onClick={() => setOpen(!open)}
        className={`flex items-center gap-1 text-xs font-medium ${pct >= 70 ? 'text-emerald-400' : pct >= 50 ? 'text-amber-400' : 'text-red-400'}`}>
        {pct >= 70
          ? <CheckCircle2 className="w-3 h-3" />
          : <AlertTriangle className="w-3 h-3" />}
        {pct}% quality
        {reasons.length > 0 && <ChevronDown className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} />}
      </button>
      {open && reasons.length > 0 && (
        <div className="absolute top-6 left-0 z-10 bg-popover border border-border rounded-lg p-3 w-72 shadow-xl">
          <p className="text-xs font-semibold mb-2 text-foreground">Why not 100%?</p>
          <ul className="space-y-1.5">
            {reasons.map((r, i) => (
              <li key={i} className="text-xs text-muted-foreground flex items-start gap-1.5">
                <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5 text-amber-400" />{r}
              </li>
            ))}
          </ul>
          <p className="text-xs text-muted-foreground/60 mt-2 pt-2 border-t border-border">
            Increase score by providing more detailed facts, specific section numbers, and complete party names.
          </p>
        </div>
      )}
    </div>
  )
}

export default function DraftingPage() {
  const [draftType, setDraftType] = useState<DraftType>('bail_application')
  const [facts, setFacts] = useState('')
  const [court, setCourt] = useState('')
  const [parties, setParties] = useState({ accused: '', state: '', advocate: '' })
  const [sections, setSections] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [result, setResult] = useState<DraftResponse | null>(null)
  const [showTypeDropdown, setShowTypeDropdown] = useState(false)

  const selectedType = DRAFT_TYPES.find(t => t.value === draftType)!

  const handleGenerate = async () => {
    if (!facts.trim() || isGenerating) return
    if (facts.trim().length < 50) {
      toast.error('Please provide at least 50 characters of facts for a quality draft')
      return
    }
    setIsGenerating(true)
    try {
      const res = await draftingApi.draft({
        draft_type: draftType,
        facts: facts.trim(),
        parties: Object.fromEntries(Object.entries(parties).filter(([, v]) => v.trim())),
        court: court.trim() || undefined,
        sections_involved: sections.split(',').map(s => s.trim()).filter(Boolean),
      })
      setResult(res)
      toast.success('Draft generated. Please review before use.')
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Generation failed. Please try again.')
    } finally {
      setIsGenerating(false)
    }
  }

  const copyDraft = () => {
    if (!result?.content) return
    navigator.clipboard.writeText(result.content)
    toast.success('Draft copied to clipboard')
  }

  const downloadDraft = () => {
    if (!result?.content) return
    const blob = new Blob([result.content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${draftType}_draft.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="mb-8">
          <h1 className="text-2xl font-bold mb-2">Legal Drafting Workspace</h1>
          <p className="text-sm text-muted-foreground">
            AI-assisted drafting using templates + retrieved BNS/BNSS/BSA provisions. Always review before filing.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input panel */}
          <div className="space-y-5">
            {/* Document type selector */}
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">Document Type</label>
              <div className="relative">
                <button
                  onClick={() => setShowTypeDropdown(!showTypeDropdown)}
                  className="w-full flex items-center justify-between bg-card border border-border rounded-lg px-4 py-3 text-sm hover:border-primary/40 transition-colors"
                >
                  <div className="text-left">
                    <p className="font-medium">{selectedType.label}</p>
                    <p className="text-xs text-muted-foreground">{selectedType.desc}</p>
                  </div>
                  <ChevronDown className={`w-4 h-4 text-muted-foreground transition-transform ${showTypeDropdown ? 'rotate-180' : ''}`} />
                </button>
                {showTypeDropdown && (
                  <div className="absolute top-full left-0 right-0 mt-1 bg-card border border-border rounded-lg shadow-xl z-50 overflow-hidden">
                    {DRAFT_TYPES.map(t => (
                      <button
                        key={t.value}
                        onClick={() => { setDraftType(t.value); setShowTypeDropdown(false) }}
                        className={`w-full text-left px-4 py-3 text-sm hover:bg-accent transition-colors ${t.value === draftType ? 'bg-primary/5 text-primary' : ''}`}
                      >
                        <p className="font-medium">{t.label}</p>
                        <p className="text-xs text-muted-foreground">{t.desc}</p>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Facts */}
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
                Facts of the Case <span className="text-red-400">*</span>
              </label>
              <textarea
                value={facts}
                onChange={e => setFacts(e.target.value)}
                placeholder="Describe the facts in detail: What happened, when, where, who was involved, nature of offence, FIR details, prior court orders, etc."
                rows={8}
                className="w-full bg-card border border-border rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 placeholder:text-muted-foreground"
              />
              <p className="text-xs text-muted-foreground mt-1">{facts.length} chars — minimum 50 required</p>
            </div>

            {/* Parties */}
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">Parties</label>
              <div className="space-y-2">
                {[
                  { key: 'accused', placeholder: 'Accused / Applicant name' },
                  { key: 'state', placeholder: 'State / Respondent' },
                  { key: 'advocate', placeholder: 'Advocate name (optional)' },
                ].map(({ key, placeholder }) => (
                  <input
                    key={key}
                    value={parties[key as keyof typeof parties]}
                    onChange={e => setParties(p => ({ ...p, [key]: e.target.value }))}
                    placeholder={placeholder}
                    className="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 placeholder:text-muted-foreground"
                  />
                ))}
              </div>
            </div>

            {/* Court & Sections */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">Court</label>
                <input
                  value={court}
                  onChange={e => setCourt(e.target.value)}
                  placeholder="e.g. Sessions Court, Jaipur"
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">Sections (comma-separated)</label>
                <input
                  value={sections}
                  onChange={e => setSections(e.target.value)}
                  placeholder="e.g. BNS 318, BNSS 482"
                  className="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground"
                />
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={isGenerating || facts.trim().length < 50}
              className="w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground py-3 rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGenerating ? (
                <><Loader2 className="w-4 h-4 animate-spin" /> Generating Draft…</>
              ) : (
                <><PenTool className="w-4 h-4" /> Generate Draft</>
              )}
            </button>
          </div>

          {/* Output panel */}
          <div className="space-y-4">
            {!result && !isGenerating && (
              <div className="h-full flex flex-col items-center justify-center text-center bg-card/30 border border-dashed border-border rounded-xl p-12">
                <PenTool className="w-10 h-10 text-muted-foreground/40 mb-4" />
                <p className="text-sm text-muted-foreground">Draft will appear here</p>
                <p className="text-xs text-muted-foreground/60 mt-1 max-w-xs">
                  Template + AI filling + retrieved legal provisions = structured, citation-grounded document
                </p>
              </div>
            )}

            {isGenerating && (
              <div className="h-full flex flex-col items-center justify-center bg-card border border-border rounded-xl p-12">
                <Loader2 className="w-8 h-8 text-primary animate-spin mb-4" />
                <p className="text-sm text-muted-foreground">Retrieving applicable provisions…</p>
                <p className="text-xs text-muted-foreground/60 mt-1">Filling template with facts and citations</p>
              </div>
            )}

            {result && !isGenerating && (
              <div className="flex flex-col h-full bg-card border border-border rounded-xl overflow-hidden">
                {/* Draft header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-border">
                  <div className="flex items-center gap-3">
                    <PenTool className="w-4 h-4 text-primary" />
                    <span className="text-sm font-medium">{selectedType.label}</span>
                    <ConfidenceBadge score={result.confidence} notes={result.drafting_notes} />
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={copyDraft}
                      className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground px-2 py-1.5 rounded border border-transparent hover:border-border transition-all">
                      <Copy className="w-3 h-3" /> Copy
                    </button>
                    <button onClick={downloadDraft}
                      className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground px-2 py-1.5 rounded border border-transparent hover:border-border transition-all">
                      <Download className="w-3 h-3" /> Download
                    </button>
                  </div>
                </div>

                {/* Draft content */}
                <div className="flex-1 overflow-y-auto p-4">
                  <pre className="text-xs legal-prose whitespace-pre-wrap leading-relaxed">{result.content}</pre>
                </div>

                {/* Metadata footer */}
                <div className="border-t border-border px-4 py-3 space-y-2">
                  {result.key_arguments?.length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground mb-1.5">Key Arguments Used</p>
                      <ul className="space-y-1">
                        {result.key_arguments.map((arg: string, i: number) => (
                          <li key={i} className="text-xs text-muted-foreground flex items-start gap-1.5">
                            <CheckCircle2 className="w-3 h-3 shrink-0 mt-0.5 text-emerald-400" />{arg}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {result.sections_cited?.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {result.sections_cited.map((s, i) => (
                        <span key={i} className="citation-card text-xs">{s}</span>
                      ))}
                    </div>
                  )}
                  {result.drafting_notes?.length > 0 && (
                    <div className="flex items-start gap-2 text-xs text-amber-400">
                      <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" />
                      <ul className="space-y-0.5">
                        {result.drafting_notes.map((note, i) => (
                          <li key={i}>{note}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <p className="text-xs text-muted-foreground/60">
                    This is an AI-generated draft for reference only. Review and verify all citations before filing.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
