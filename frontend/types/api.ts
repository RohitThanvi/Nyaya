// NyayaAI TypeScript domain types — mirrors Python domain models

export type DocumentType = 'judgment' | 'statute' | 'notification' | 'circular' | 'upload'
export type LawCategory = 'BNS' | 'BNSS' | 'BSA' | 'IPC' | 'CrPC' | 'Constitution' | 'Other'
export type CourtType = 'Supreme Court' | 'High Court' | 'District Court' | 'Tribunal' | 'Other'
export type ChunkType = 'facts' | 'issues' | 'arguments' | 'findings' | 'ratio' | 'final_order' | 'chapter' | 'section' | 'subsection' | 'explanation' | 'punishment' | 'passage'
export type LegalIntentType = 'provision_lookup' | 'case_search' | 'procedure_query' | 'drafting_request' | 'summarization' | 'general_query'
export type DraftType = 'bail_application' | 'anticipatory_bail' | 'legal_notice' | 'affidavit' | 'complaint' | 'fir_quashing_petition' | 'written_statement' | 'vakalatnama'
export type UserRole = 'admin' | 'advocate' | 'researcher' | 'student' | 'guest'

export interface Citation {
  citation_id: string
  document_id: string
  chunk_id: string
  section?: string
  subsection?: string
  paragraph?: number
  citation_text: string
  citation_type: string
  court?: string
  year?: number
  relevance_note?: string
  verified: boolean
}

export interface RelevantSection {
  section_number: string
  law: LawCategory
  title: string
  relevance: string
  elements_to_prove: string[]
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
  punishment?: string
  citation_chunk_id?: string
}

export interface Precedent {
  citation: string
  court: string
  year?: number
  relevance: string
  score: number
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
  timestamp: string
}

export interface ChatRequest {
  session_id?: string
  message: string
  history: ChatMessage[]
  law_filter?: LawCategory[]
  stream?: boolean
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
  draft_type: string
  content: string
  sections_cited: string[]
  key_arguments: string[]
  drafting_notes: string[]
  confidence: number
  template_used: string
  session_id: string
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

export interface Document {
  document_id: string
  document_type: DocumentType
  law?: LawCategory
  court_name?: string
  citation?: string
  year?: number
  topic?: string
  total_chunks: number
  is_landmark: boolean
  created_at: string
}

export interface UploadResponse {
  document_id: string
  filename: string
  pages: number
  chunks_created: number
  status: string
  message: string
}

export interface ChatSession {
  session_id: string
  title?: string
  created_at: string
  updated_at: string
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
  full_name: string
  role: UserRole
}
