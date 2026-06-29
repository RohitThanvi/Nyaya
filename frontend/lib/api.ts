/**
 * NyayaAI API client v2.
 *
 * Fixes:
 * - refresh_token sent as JSON body (not query param) — matches auth route v2
 * - deleteSession added
 * - uploadChunked added for large files
 * - streamChat parses new {token, done} SSE format
 * - chatWithDocument routes to /chat with document_id (not /search)
 */
import axios, { AxiosInstance } from 'axios'
import Cookies from 'js-cookie'
import type {
  ChatRequest, DocumentListParams, DraftRequest, DraftResponse, JudgmentSummary,
  LegalResponse, PaginatedDocuments, SearchRequest, TokenResponse, UploadResponse, User
} from '@/types/api'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const API_PREFIX = '/api/v1'
const CHUNK_SIZE = 10 * 1024 * 1024   // 10 MB per upload chunk

function createApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: `${BASE_URL}${API_PREFIX}`,
    timeout: 120000,
    headers: { 'Content-Type': 'application/json' },
  })

  client.interceptors.request.use((config) => {
    const token = Cookies.get('access_token')
    if (token) config.headers.Authorization = `Bearer ${token}`
    return config
  })

  client.interceptors.response.use(
    (res) => res,
    async (error) => {
      const original = error.config
      if (error.response?.status === 401 && !original._retry) {
        original._retry = true
        try {
          const refresh = Cookies.get('refresh_token')
          if (refresh) {
            // FIX: send as JSON body, not query param
            const res = await axios.post<TokenResponse>(
              `${BASE_URL}${API_PREFIX}/auth/refresh`,
              { refresh_token: refresh },
              { headers: { 'Content-Type': 'application/json' } }
            )
            const { access_token, refresh_token } = res.data
            Cookies.set('access_token', access_token, { secure: true, sameSite: 'strict' })
            if (refresh_token) {
              Cookies.set('refresh_token', refresh_token, { secure: true, sameSite: 'strict' })
            }
            original.headers.Authorization = `Bearer ${access_token}`
            return client(original)
          }
        } catch {
          Cookies.remove('access_token')
          Cookies.remove('refresh_token')
          if (typeof window !== 'undefined') window.location.href = '/auth/login'
        }
      }
      return Promise.reject(error)
    }
  )

  return client
}

export const api = createApiClient()

// ── Auth ──────────────────────────────────────────────────────────────────

export const authApi = {
  async login(email: string, password: string): Promise<TokenResponse> {
    const res = await api.post<TokenResponse>('/auth/login', { email, password })
    Cookies.set('access_token', res.data.access_token, { secure: true, sameSite: 'strict' })
    Cookies.set('refresh_token', res.data.refresh_token, { secure: true, sameSite: 'strict' })
    return res.data
  },

  async register(data: {
    email: string; password: string; full_name: string; role?: string
  }): Promise<TokenResponse> {
    const res = await api.post<TokenResponse>('/auth/register', data)
    Cookies.set('access_token', res.data.access_token, { secure: true, sameSite: 'strict' })
    Cookies.set('refresh_token', res.data.refresh_token, { secure: true, sameSite: 'strict' })
    return res.data
  },

  async me(): Promise<User> {
    const res = await api.get<User>('/auth/me')
    return res.data
  },

  isAuthenticated(): boolean {
    return !!Cookies.get('access_token')
  },

  logout() {
    Cookies.remove('access_token')
    Cookies.remove('refresh_token')
    if (typeof window !== 'undefined') window.location.href = '/auth/login'
  },
}

// ── Search ────────────────────────────────────────────────────────────────

export const searchApi = {
  async search(request: SearchRequest): Promise<LegalResponse> {
    const res = await api.post<LegalResponse>('/search', request)
    return res.data
  },
}

// ── Chat ──────────────────────────────────────────────────────────────────

export const chatApi = {
  async chat(request: ChatRequest): Promise<LegalResponse> {
    const res = await api.post<LegalResponse>('/chat', { ...request, stream: false })
    return res.data
  },

  // FIX: document-scoped queries route to /chat with document_id,
  // NOT to /search (which queries the global corpus)
  async chatWithDocument(
    documentId: string,
    message: string,
    history: ChatRequest['history'] = [],
  ): Promise<LegalResponse> {
    const res = await api.post<LegalResponse>('/chat', {
      message,
      document_id: documentId,
      history,
      stream: false,
    })
    return res.data
  },

  async streamChat(
    request: ChatRequest,
    onToken: (token: string) => void,
    onDone: (citations?: unknown[]) => void,
    onError: (err: string) => void,
  ): Promise<void> {
    const token = Cookies.get('access_token')
    let response: Response
    try {
      response = await fetch(`${BASE_URL}${API_PREFIX}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ ...request, stream: true }),
      })
    } catch (e) {
      onError(String(e))
      return
    }

    if (!response.ok) {
      onError(`HTTP ${response.status}`)
      return
    }

    const reader = response.body?.getReader()
    if (!reader) { onError('No response body'); return }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const raw = line.slice(6).trim()
        try {
          const parsed = JSON.parse(raw)
          if (parsed.done) {
            onDone(parsed.citations)
            return
          }
          if (parsed.error) {
            onError(parsed.error)
            return
          }
          if (parsed.token) {
            onToken(parsed.token)
          }
        } catch {
          // Non-JSON line — treat as raw token
          if (raw) onToken(raw)
        }
      }
    }
    onDone()
  },

  async getSessions() {
    const res = await api.get('/chat/sessions')
    return res.data
  },

  async getSessionMessages(sessionId: string) {
    const res = await api.get(`/chat/sessions/${sessionId}/messages`)
    return res.data
  },

  async deleteSession(sessionId: string): Promise<void> {
    await api.delete(`/chat/sessions/${sessionId}`)
  },
}

// ── Documents ──────────────────────────────────────────────────────────────

export const documentsApi = {
  // FIX: matches the actual backend contract — page/page_size in,
  // {documents, total, page, page_size} out. Previously called with
  // page/page_size but backend expected limit/offset and returned a bare
  // array, causing judgeRes.documents to be undefined and crashing the
  // judgments page on .filter().
  async list(params?: DocumentListParams): Promise<PaginatedDocuments> {
    const res = await api.get<PaginatedDocuments>('/documents', { params })
    return res.data
  },

  async get(documentId: string) {
    const res = await api.get(`/documents/${documentId}`)
    return res.data
  },

  async getChunks(documentId: string, chunkType?: string, limit = 100) {
    const res = await api.get(`/documents/${documentId}/chunks`, {
      params: { chunk_type: chunkType, limit }
    })
    return res.data
  },

  async summarize(documentId: string): Promise<JudgmentSummary> {
    const res = await api.post<JudgmentSummary>(`/documents/${documentId}/summarize`)
    return res.data
  },

  async delete(documentId: string): Promise<void> {
    await api.delete(`/documents/${documentId}`)
  },
}

// ── Upload ─────────────────────────────────────────────────────────────────

export const uploadApi = {
  async upload(
    file: File,
    sourceUrl?: string,
    onProgress?: (pct: number) => void,
  ): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    if (sourceUrl) formData.append('source_url', sourceUrl)
    const res = await api.post<UploadResponse>('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
      },
    })
    return res.data
  },

  /**
   * Chunked upload for large files (> MAX_UPLOAD_SIZE_MB).
   * Splits file into CHUNK_SIZE chunks, uploads sequentially with Content-Range.
   * Calls /upload/chunked/init → /upload/chunked/{id} × N → /upload/chunked/{id}/finalise
   */
  async uploadChunked(
    file: File,
    sourceUrl?: string,
    onProgress?: (pct: number) => void,
  ): Promise<UploadResponse> {
    // Init session
    const initForm = new FormData()
    initForm.append('filename', file.name)
    initForm.append('total_size', String(file.size))
    if (sourceUrl) initForm.append('source_url', sourceUrl)

    const initRes = await api.post('/upload/chunked/init', initForm, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    const { upload_id } = initRes.data

    // Upload chunks
    let offset = 0
    let chunkNum = 0
    while (offset < file.size) {
      const end = Math.min(offset + CHUNK_SIZE, file.size)
      const blob = file.slice(offset, end)
      const chunkForm = new FormData()
      chunkForm.append('chunk', blob, `chunk_${chunkNum}`)

      await api.post(`/upload/chunked/${upload_id}`, chunkForm, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Content-Range': `bytes ${offset}-${end - 1}/${file.size}`,
        },
      })

      offset = end
      chunkNum++
      if (onProgress) onProgress(Math.round((offset / file.size) * 90))
    }

    // Finalise
    const finalRes = await api.post<UploadResponse>(`/upload/chunked/${upload_id}/finalise`)
    if (onProgress) onProgress(100)
    return finalRes.data
  },
}

// ── Drafting ───────────────────────────────────────────────────────────────

export const draftingApi = {
  async draft(request: DraftRequest): Promise<DraftResponse> {
    const res = await api.post<DraftResponse>('/draft', request)
    return res.data
  },
}

// ── Health ─────────────────────────────────────────────────────────────────

export const healthApi = {
  async check() {
    const res = await api.get('/health')
    return res.data
  },
}
