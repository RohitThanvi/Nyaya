/**
 * NyayaAI API client.
 * Handles auth tokens, retries, and streaming.
 */
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios'
import Cookies from 'js-cookie'
import type {
  ChatRequest, DraftRequest, DraftResponse, JudgmentSummary,
  LegalResponse, SearchRequest, TokenResponse, UploadResponse, User
} from '@/types/api'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const API_PREFIX = '/api/v1'

function createApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: `${BASE_URL}${API_PREFIX}`,
    timeout: 60000,
    headers: { 'Content-Type': 'application/json' },
  })

  // Attach JWT from cookie
  client.interceptors.request.use((config) => {
    const token = Cookies.get('access_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  })

  // Auto-refresh on 401
  client.interceptors.response.use(
    (res) => res,
    async (error) => {
      const original = error.config
      if (error.response?.status === 401 && !original._retry) {
        original._retry = true
        try {
          const refresh = Cookies.get('refresh_token')
          if (refresh) {
            const res = await axios.post<TokenResponse>(
              `${BASE_URL}${API_PREFIX}/auth/refresh`,
              null,
              { params: { refresh_token: refresh } }
            )
            Cookies.set('access_token', res.data.access_token, { secure: true, sameSite: 'strict' })
            original.headers.Authorization = `Bearer ${res.data.access_token}`
            return client(original)
          }
        } catch {
          Cookies.remove('access_token')
          Cookies.remove('refresh_token')
          window.location.href = '/auth/login'
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

  async register(data: { email: string; password: string; full_name: string; role?: string }): Promise<TokenResponse> {
    const res = await api.post<TokenResponse>('/auth/register', data)
    Cookies.set('access_token', res.data.access_token, { secure: true, sameSite: 'strict' })
    Cookies.set('refresh_token', res.data.refresh_token, { secure: true, sameSite: 'strict' })
    return res.data
  },

  async me(): Promise<User> {
    const res = await api.get<User>('/auth/me')
    return res.data
  },

  logout() {
    Cookies.remove('access_token')
    Cookies.remove('refresh_token')
  },
}

// ── Search ────────────────────────────────────────────────────────────────

export const searchApi = {
  async search(request: SearchRequest): Promise<LegalResponse> {
    const res = await api.post<LegalResponse>('/search', request)
    return res.data
  },

  async lookupSection(law: string, section: string): Promise<LegalResponse> {
    const res = await api.get<LegalResponse>(`/search/sections/${law}/${section}`)
    return res.data
  },
}

// ── Chat ──────────────────────────────────────────────────────────────────

export const chatApi = {
  async chat(request: ChatRequest): Promise<LegalResponse> {
    const res = await api.post<LegalResponse>('/chat', { ...request, stream: false })
    return res.data
  },

  async streamChat(
    request: ChatRequest,
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (err: string) => void,
  ): Promise<void> {
    const token = Cookies.get('access_token')
    const response = await fetch(`${BASE_URL}${API_PREFIX}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ ...request, stream: true }),
    })

    if (!response.ok) {
      onError(`HTTP ${response.status}`)
      return
    }

    const reader = response.body?.getReader()
    if (!reader) return

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          if (data === '[DONE]') {
            onDone()
            return
          } else if (data.startsWith('[ERROR]')) {
            onError(data.slice(7))
            return
          } else {
            // Unescape newlines
            onToken(data.replace(/\\n/g, '\n'))
          }
        }
      }
    }
    onDone()
  },

  async getSessions() {
    const res = await api.get('/chat/sessions')
    return res.data
  },

  async getSession(sessionId: string) {
    const res = await api.get(`/chat/sessions/${sessionId}`)
    return res.data
  },
}

// ── Documents ──────────────────────────────────────────────────────────────

export const documentsApi = {
  async list(params?: { law?: string; court?: string; year?: number; page?: number; page_size?: number }) {
    const res = await api.get('/documents', { params })
    return res.data
  },

  async get(documentId: string) {
    const res = await api.get(`/documents/${documentId}`)
    return res.data
  },

  async summarize(documentId: string): Promise<JudgmentSummary> {
    const res = await api.post<JudgmentSummary>(`/documents/${documentId}/summarize`)
    return res.data
  },

  async summarizeText(text: string): Promise<JudgmentSummary> {
    const res = await api.post<JudgmentSummary>('/summarize', { text })
    return res.data
  },
}

// ── Upload ─────────────────────────────────────────────────────────────────

export const uploadApi = {
  async upload(file: File, onProgress?: (pct: number) => void): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    const res = await api.post<UploadResponse>('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      },
    })
    return res.data
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
    const res = await api.get('/health/detailed')
    return res.data
  },
}
