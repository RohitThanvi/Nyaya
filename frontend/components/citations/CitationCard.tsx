/**
 * CitationCard — renders a verified Citation with deep-link to source document.
 *
 * Shows:
 * - Citation text (e.g. "AIR 2025 SC 111" or "Section 318 BNS")
 * - Verification badge (green tick / amber warning)
 * - Snippet from the matching chunk
 * - "Open source" button linking to source_url#page=N
 * - Hallucination warning if not verified
 */
'use client'

import { ExternalLink, CheckCircle, AlertTriangle } from 'lucide-react'
import type { Citation } from '@/types/api'

interface CitationCardProps {
  citation: Citation
  index?: number
}

export function CitationCard({ citation, index }: CitationCardProps) {
  const deepLink = citation.deep_link ||
    (citation.source_url
      ? `${citation.source_url}${citation.page_number ? `#page=${citation.page_number}` : ''}`
      : null)

  const typeLabel = citation.citation_type === 'statute' ? 'Statute' : 'Judgment'
  const typeColor = citation.citation_type === 'statute'
    ? 'bg-blue-50 text-blue-700 border-blue-200'
    : 'bg-purple-50 text-purple-700 border-purple-200'

  return (
    <div className={`rounded-lg border p-3 text-sm ${
      citation.verified
        ? 'border-green-200 bg-green-50'
        : 'border-amber-200 bg-amber-50'
    }`}>
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          {index !== undefined && (
            <span className="text-xs text-gray-400 font-mono">[{index + 1}]</span>
          )}
          <span className={`text-xs px-2 py-0.5 rounded border font-medium ${typeColor}`}>
            {typeLabel}
          </span>
          {citation.verified ? (
            <span className="flex items-center gap-1 text-xs text-green-700 font-medium">
              <CheckCircle className="w-3 h-3" />
              Verified
            </span>
          ) : (
            <span className="flex items-center gap-1 text-xs text-amber-700 font-medium">
              <AlertTriangle className="w-3 h-3" />
              Unverified
            </span>
          )}
        </div>

        {deepLink && (
          <a
            href={deepLink}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 hover:underline whitespace-nowrap shrink-0"
            title={`Open source document${citation.page_number ? ` at page ${citation.page_number}` : ''}`}
          >
            <ExternalLink className="w-3 h-3" />
            {citation.page_number ? `p. ${citation.page_number}` : 'Source'}
          </a>
        )}
      </div>

      {/* Citation text */}
      <p className="mt-1.5 font-semibold text-gray-800 leading-snug">
        {citation.citation_text}
      </p>

      {/* Metadata */}
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-gray-500">
        {citation.court && <span>{citation.court}</span>}
        {citation.year && <span>{citation.year}</span>}
        {citation.section && (
          <span>§{citation.section}{citation.subsection ? `(${citation.subsection})` : ''}</span>
        )}
      </div>

      {/* Snippet */}
      {citation.snippet && (
        <p className="mt-1.5 text-gray-600 leading-relaxed line-clamp-2 italic text-xs">
          &ldquo;{citation.snippet}&rdquo;
        </p>
      )}

      {/* Unverified warning */}
      {!citation.verified && (
        <p className="mt-1.5 text-amber-700 text-xs">
          This citation could not be verified in the knowledge base.
          Please confirm independently before relying on it.
        </p>
      )}
    </div>
  )
}

export function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations || citations.length === 0) return null
  return (
    <div className="mt-3 space-y-2">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Citations ({citations.length})
      </p>
      {citations.map((cit, i) => (
        <CitationCard key={cit.citation_id || i} citation={cit} index={i} />
      ))}
    </div>
  )
}
