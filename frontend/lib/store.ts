/**
 * Zustand global state store.
 * Covers: auth, chat, search results.
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { chatApi } from '@/lib/api'
import type { ChatMessage, ChatSessionSummary, LegalResponse, User } from '@/types/api'

// ── Auth Store ────────────────────────────────────────────────────────────

interface AuthStore {
  user: User | null
  isAuthenticated: boolean
  setUser: (user: User) => void
  clearUser: () => void
  /** Call on app mount: wipes persisted auth state if the access_token
   *  cookie is gone so the store never shows isAuthenticated:true while
   *  all API calls 401. */
  syncWithCookies: () => void
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      setUser: (user) => set({ user, isAuthenticated: true }),
      clearUser: () => set({ user: null, isAuthenticated: false }),
      syncWithCookies: () => {
        // Avoid importing Cookies at module level — js-cookie is browser-only
        if (typeof document === 'undefined') return
        const hasCookie = document.cookie.split(';').some(
          (c) => c.trim().startsWith('access_token=') && c.trim().length > 'access_token='.length
        )
        if (!hasCookie) set({ user: null, isAuthenticated: false })
      },
    }),
    { name: 'nyaya-auth' }
  )
)

// ── Chat Store ────────────────────────────────────────────────────────────

interface ChatStore {
  sessionId: string | null
  messages: ChatMessage[]
  isStreaming: boolean
  streamingContent: string
  // Chat history sidebar state
  sessions: ChatSessionSummary[]
  sessionsLoading: boolean
  activeSessionLoading: boolean

  addMessage: (msg: ChatMessage) => void
  setStreaming: (streaming: boolean) => void
  appendStreamToken: (token: string) => void
  // FIX: previously threw away the backend's final response and just used
  // streamingContent for the assistant message, and never captured
  // session_id — so setSessionId existed but was never called, meaning
  // every message created a brand-new backend session instead of
  // continuing one. Now takes the full final response (from the fixed
  // streamChat onDone) and captures session_id + citations/confidence.
  commitStreamedMessage: (response?: LegalResponse) => void
  clearSession: () => void
  setSessionId: (id: string) => void

  // History sidebar actions
  loadSessions: () => Promise<void>
  loadSession: (sessionId: string) => Promise<void>
  deleteSession: (sessionId: string) => Promise<void>
  startNewChat: () => void
}

export const useChatStore = create<ChatStore>((set, get) => ({
  sessionId: null,
  messages: [],
  isStreaming: false,
  streamingContent: '',
  sessions: [],
  sessionsLoading: false,
  activeSessionLoading: false,

  addMessage: (msg) =>
    set((state) => ({ messages: [...state.messages, msg] })),

  setStreaming: (streaming) =>
    set({ isStreaming: streaming, streamingContent: streaming ? '' : '' }),

  appendStreamToken: (token) =>
    set((state) => ({ streamingContent: state.streamingContent + token })),

  commitStreamedMessage: (response) =>
    set((state) => {
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: response?.answer ?? state.streamingContent,
        timestamp: response?.timestamp ?? new Date().toISOString(),
        citations: response?.citations,
        confidence: response?.confidence,
        warnings: response?.warnings,
        hallucination_flags: response?.hallucination_flags,
      }
      return {
        messages: [...state.messages, assistantMsg],
        isStreaming: false,
        streamingContent: '',
        // Backend always returns the canonical session_id (mints one if the
        // request didn't send one) — capture it so subsequent messages in
        // this conversation continue the same session instead of forking.
        sessionId: response?.session_id ?? state.sessionId,
      }
    }),

  clearSession: () =>
    set({ sessionId: null, messages: [], isStreaming: false, streamingContent: '' }),

  setSessionId: (id) => set({ sessionId: id }),

  loadSessions: async () => {
    set({ sessionsLoading: true })
    try {
      const sessions = await chatApi.getSessions()
      set({ sessions, sessionsLoading: false })
    } catch {
      set({ sessionsLoading: false })
    }
  },

  loadSession: async (sessionId) => {
    set({ activeSessionLoading: true })
    try {
      const rows = await chatApi.getSessionMessages(sessionId)
      const messages: ChatMessage[] = rows.map((m) => ({
        role: m.role,
        content: m.content,
        timestamp: m.created_at ?? undefined,
        citations: m.citations,
        confidence: m.confidence ?? undefined,
        hallucination_flags: m.hallucination_flags,
      }))
      set({
        sessionId,
        messages,
        isStreaming: false,
        streamingContent: '',
        activeSessionLoading: false,
      })
    } catch {
      set({ activeSessionLoading: false })
    }
  },

  deleteSession: async (sessionId) => {
    await chatApi.deleteSession(sessionId)
    set((state) => ({ sessions: state.sessions.filter((s) => s.session_id !== sessionId) }))
    // If the deleted session was the active one, reset to a blank chat
    if (get().sessionId === sessionId) {
      set({ sessionId: null, messages: [], isStreaming: false, streamingContent: '' })
    }
  },

  startNewChat: () =>
    set({ sessionId: null, messages: [], isStreaming: false, streamingContent: '' }),
}))

// ── Search Store ──────────────────────────────────────────────────────────

interface SearchStore {
  lastQuery: string
  lastResult: LegalResponse | null
  isSearching: boolean
  history: Array<{ query: string; result: LegalResponse; timestamp: string }>
  setSearching: (v: boolean) => void
  setResult: (query: string, result: LegalResponse) => void
  clearResult: () => void
}

export const useSearchStore = create<SearchStore>((set) => ({
  lastQuery: '',
  lastResult: null,
  isSearching: false,
  history: [],

  setSearching: (v) => set({ isSearching: v }),

  setResult: (query, result) =>
    set((state) => ({
      lastQuery: query,
      lastResult: result,
      isSearching: false,
      history: [
        { query, result, timestamp: new Date().toISOString() },
        ...state.history.slice(0, 19),
      ],
    })),

  clearResult: () => set({ lastQuery: '', lastResult: null }),
}))
