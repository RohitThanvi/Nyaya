/**
 * Zustand global state store.
 * Covers: auth, chat, search results.
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ChatMessage, LegalResponse, User } from '@/types/api'

// ── Auth Store ────────────────────────────────────────────────────────────

interface AuthStore {
  user: User | null
  isAuthenticated: boolean
  setUser: (user: User) => void
  clearUser: () => void
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      setUser: (user) => set({ user, isAuthenticated: true }),
      clearUser: () => set({ user: null, isAuthenticated: false }),
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
  addMessage: (msg: ChatMessage) => void
  setStreaming: (streaming: boolean) => void
  appendStreamToken: (token: string) => void
  commitStreamedMessage: () => void
  clearSession: () => void
  setSessionId: (id: string) => void
}

export const useChatStore = create<ChatStore>((set) => ({
  sessionId: null,
  messages: [],
  isStreaming: false,
  streamingContent: '',

  addMessage: (msg) =>
    set((state) => ({ messages: [...state.messages, msg] })),

  setStreaming: (streaming) =>
    set({ isStreaming: streaming, streamingContent: streaming ? '' : '' }),

  appendStreamToken: (token) =>
    set((state) => ({ streamingContent: state.streamingContent + token })),

  commitStreamedMessage: () =>
    set((state) => {
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: state.streamingContent,
        timestamp: new Date().toISOString(),
      }
      return {
        messages: [...state.messages, assistantMsg],
        isStreaming: false,
        streamingContent: '',
      }
    }),

  clearSession: () =>
    set({ sessionId: null, messages: [], isStreaming: false, streamingContent: '' }),

  setSessionId: (id) => set({ sessionId: id }),
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
