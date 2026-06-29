/**
 * NyayaAI API type definitions v2.
 *
 * Changes:
 * - Citation: source_url, page_number, snippet, deep_link, verified
 * - Document: parties typed as Record<string,string>, source_url
 * - LegalResponse: hallucination_flags
 * - UploadResponse: source_url
 */

export type LawCategory = 'BNS' | 'BNSS' | 'BSA' | 'IPC' | 'CrPC' | 'Constitution' | 'Other'
export type CourtType = 'Supreme Court' | 'High Court' | 'District Court' | 'Tribunal' | 'Other'
export type DocumentType = 'judgment' | 'statute' | 'notification' | 'circular' | 'upload'
export type UserRole = 'admin' | 'advocate' | 'researcher' | 'student' | 'guest'
export type DraftType =
  | 'bail_application' | 'anticipatory_bail' | 'legal_notice'
  | 'affidavit' | 'complaint' | 'fir_quashing_petition'
  | 'written_statement' | 'vakalatnama'

export interface Citation {
  citation_id: string
  document_id: string
  chunk_id: string
  section?: string
  subsection?: string
  paragraph?: number
  page_number?: number        // physical page in source PDF
  citation_text: string
  citation_type: string       // 'statute' | 'judgment'
  court?: string
  year?: number
  source_url?: string         // direct link to source document
  snippet?: string            // first 150 chars of matching chunk
  relevance_note?: string
  verified: boolean
  deep_link?: string          // source_url#page=N
}

export interface RelevantSection {
  section?: string
  subsection?: string
  citation_text: string
  snippet?: string
  source_url?: string
  page_number?: number
  deep_link?: string
  verified: boolean
  relevance_score?: number
}

export interface Precedent {
  citation: string
  court?: string
  year?: number
  snippet?: string
  source_url?: string
  page_number?: number
  deep_link?: string
  verified: boolean
  relevance_score?: number
}

export interface LegalResponse {
  query: string
  session_id: string
  intent?: string
  answer: string
  relevant_sections: RelevantSection[]
  precedents: Precedent[]
  procedural_requirements: string[]
  citations: Citation[]
  confidence: number
  warnings: string[]
  hallucination_flags: string[]
  latency_ms?: number
  timestamp: string
}

export interface SearchRequest {
  query: string
  law_filter?: LawCategory[]
  court_filter?: CourtType[]
  year_from?: number
  year_to?: number
  document_type?: DocumentType
  top_k?: number
  include_statutes?: boolean
  include_judgments?: boolean
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
}

export interface ChatRequest {
  session_id?: string
  message: string
  history?: ChatMessage[]
  law_filter?: LawCategory[]
  document_id?: string        // scoped retrieval within a single document
  stream?: boolean
}

export interface Document {
  document_id: string
  document_type: DocumentType
  law?: LawCategory
  court?: CourtType
  court_name?: string
  case_number?: string
  citation?: string
  year?: number
  date_decided?: string
  bench?: string[]
  parties?: Record<string, string>   // FIX: typed as object, not string
  topic?: string
  keywords?: string[]
  source_url?: string                // canonical public URL
  is_landmark: boolean
  language: string
  pages: number
  created_at: string
}

export interface JudgmentSummary {
  document_id: string
  case_name?: string
  citation?: string
  court?: string
  date_decided?: string
  facts: string
  issues: string[]
  arguments: Record<string, string>
  findings: string
  ratio_decidendi?: string
  final_order: string
  sections_discussed: string[]
  is_landmark: boolean
  summary_brief: string
}

export interface UploadResponse {
  document_id: string
  filename: string
  pages: number
  chunks_created: number
  status: string
  message: string
}

export interface DraftRequest {
  draft_type: DraftType
  facts: string
  parties: Record<string, string>
  court?: string
  sections_involved?: string[]
  additional_context?: string
}

export interface DraftResponse {
  query: string
  session_id: string
  answer: string
  warnings: string[]
  confidence: number
}

export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface User {
  user_id: string
  email: string
  full_name?: string
  role: UserRole
}

export interface ChatSession {
  session_id: string
  title?: string
  created_at: string
  updated_at: string
}
