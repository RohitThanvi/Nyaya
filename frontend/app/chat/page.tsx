'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, MessageSquare, Scale, AlertTriangle, CheckCircle2, ChevronDown, Trash2, BookOpen } from 'lucide-react'
import { chatApi } from '@/lib/api'
import { useChatStore } from '@/lib/store'
import type { LawCategory } from '@/types/api'
import { toast } from 'sonner'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const LAW_OPTIONS: LawCategory[] = ['BNS', 'BNSS', 'BSA', 'IPC', 'CrPC']

function MessageBubble({
  role,
  content,
  citations,
  confidence,
  warnings,
  isStreaming = false,
}: {
  role: 'user' | 'assistant'
  content: string
  citations?: any[]
  confidence?: number
  warnings?: string[]
  isStreaming?: boolean
}) {
  const isUser = role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} gap-3`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center shrink-0 mt-1">
          <Scale className="w-3.5 h-3.5 text-primary" />
        </div>
      )}

      <div className={`max-w-[80%] space-y-2 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div className={`rounded-xl px-4 py-3 text-sm ${
          isUser
            ? 'bg-primary text-primary-foreground rounded-tr-sm'
            : 'bg-card border border-border rounded-tl-sm'
        } ${isStreaming ? 'streaming-cursor' : ''}`}>
          {isUser ? (
            <p>{content}</p>
          ) : (
            <div className="prose prose-sm prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Metadata row */}
        {!isUser && (citations?.length || confidence !== undefined || warnings?.length) ? (
          <div className="flex flex-wrap items-center gap-2 px-1">
            {confidence !== undefined && (
              <span className={`text-xs ${
                confidence > 0.75 ? 'text-emerald-400' : confidence > 0.5 ? 'text-amber-400' : 'text-red-400'
              }`}>
                {Math.round(confidence * 100)}% verified
              </span>
            )}
            {citations?.map((cit, i) => (
              <span key={i} className="citation-card flex items-center gap-1">
                <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" />
                {cit.citation_text}
              </span>
            ))}
            {warnings?.map((w, i) => (
              <span key={i} className="flex items-center gap-1 text-xs text-amber-400">
                <AlertTriangle className="w-2.5 h-2.5" />
                {w.slice(0, 60)}{w.length > 60 ? '…' : ''}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}

function LawFilterPill({ law, active, onClick }: { law: LawCategory; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
        active ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:border-primary/30'
      }`}
    >
      {law}
    </button>
  )
}

export default function ChatPage() {
  const {
    messages, sessionId, isStreaming, streamingContent,
    addMessage, setStreaming, appendStreamToken, commitStreamedMessage,
    clearSession, setSessionId,
  } = useChatStore()

  const [input, setInput] = useState('')
  const [lawFilter, setLawFilter] = useState<LawCategory[]>([])
  const [showScrollBtn, setShowScrollBtn] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    if (isStreaming || messages.length > 0) {
      scrollToBottom()
    }
  }, [messages.length, isStreaming, streamingContent, scrollToBottom])

  const handleScroll = () => {
    if (!scrollRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
    setShowScrollBtn(scrollHeight - scrollTop - clientHeight > 100)
  }

  const toggleLaw = (law: LawCategory) => {
    setLawFilter(prev => prev.includes(law) ? prev.filter(l => l !== law) : [...prev, law])
  }

  const handleSend = useCallback(async () => {
    const msg = input.trim()
    if (!msg || isStreaming) return

    const userMessage = { role: 'user' as const, content: msg, timestamp: new Date().toISOString() }
    addMessage(userMessage)
    setInput('')

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    setStreaming(true)

    await chatApi.streamChat(
      {
        message: msg,
        history: messages.slice(-10),
        session_id: sessionId || undefined,
        law_filter: lawFilter.length ? lawFilter : undefined,
        stream: true,
      },
      (token) => appendStreamToken(token),
      () => commitStreamedMessage(),
      (err) => {
        toast.error(`Chat error: ${err}`)
        commitStreamedMessage()
      }
    )
  }, [
    input, isStreaming, messages, sessionId, lawFilter,
    addMessage, setStreaming, appendStreamToken, commitStreamedMessage
  ])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    // Auto-resize
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`
  }

  const SUGGESTIONS = [
    'What is Section 318 of BNS? When does it apply?',
    'Explain anticipatory bail under BNSS Section 482',
    'How is electronic evidence admitted under BSA?',
    'What are the grounds for quashing an FIR?',
    'Difference between BNS and old IPC for murder',
  ]

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="shrink-0 border-b border-border bg-card/50 backdrop-blur-sm px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center">
            <MessageSquare className="w-3.5 h-3.5 text-primary" />
          </div>
          <span className="font-medium text-sm">Legal Assistant</span>
          {messages.length > 0 && (
            <span className="text-xs text-muted-foreground">{messages.length} messages</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Law filters */}
          <div className="flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-muted-foreground" />
            {LAW_OPTIONS.map(law => (
              <LawFilterPill key={law} law={law} active={lawFilter.includes(law)} onClick={() => toggleLaw(law)} />
            ))}
          </div>
          {messages.length > 0 && (
            <button
              onClick={() => clearSession()}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear
            </button>
          )}
        </div>
      </header>

      {/* Messages */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-6 py-6 space-y-4"
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
            <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
              <Scale className="w-7 h-7 text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-semibold mb-2">NyayaAI Legal Assistant</h2>
              <p className="text-sm text-muted-foreground max-w-md">
                Ask about BNS provisions, BNSS procedures, BSA evidence rules, or Supreme Court judgments.
                Every answer is backed by verified citations.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => { setInput(s); textareaRef.current?.focus() }}
                  className="text-left text-xs border border-border rounded-lg px-3 py-2.5 hover:border-primary/40 hover:bg-card transition-colors text-muted-foreground hover:text-foreground"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble
                key={i}
                role={msg.role as 'user' | 'assistant'}
                content={msg.content}
              />
            ))}
            {isStreaming && (
              <MessageBubble
                role="assistant"
                content={streamingContent || ''}
                isStreaming={!streamingContent}
              />
            )}
          </>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Scroll to bottom button */}
      {showScrollBtn && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-24 right-8 w-8 h-8 rounded-full bg-card border border-border flex items-center justify-center shadow-lg hover:border-primary/40 transition-colors"
        >
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        </button>
      )}

      {/* Input */}
      <div className="shrink-0 border-t border-border bg-card/50 backdrop-blur-sm p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end gap-3 bg-background border border-border rounded-xl px-4 py-3 focus-within:ring-2 focus-within:ring-primary/30 focus-within:border-primary/50 transition-all">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask about Indian law… (Enter to send, Shift+Enter for new line)"
              rows={1}
              disabled={isStreaming}
              className="flex-1 bg-transparent text-sm resize-none focus:outline-none placeholder:text-muted-foreground disabled:opacity-50 min-h-[24px] max-h-40"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="w-8 h-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center hover:bg-primary/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
            >
              {isStreaming
                ? <span className="w-3.5 h-3.5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                : <Send className="w-3.5 h-3.5" />
              }
            </button>
          </div>
          <p className="text-xs text-muted-foreground mt-2 text-center">
            Answers are grounded in retrieved legal text. Not legal advice — consult an advocate.
          </p>
        </div>
      </div>
    </div>
  )
}
