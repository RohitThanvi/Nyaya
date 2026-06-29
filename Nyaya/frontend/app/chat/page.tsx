'use client'

import { useState, useRef, useEffect, useCallback, Suspense } from 'react'
import { useSearchParams } from 'next/navigation'
import { Send, MessageSquare, Scale, AlertTriangle, CheckCircle2, ChevronDown, Trash2, BookOpen, FileText, X } from 'lucide-react'
import { chatApi } from '@/lib/api'
import { useChatStore } from '@/lib/store'
import type { LawCategory } from '@/types/api'
import { toast } from 'sonner'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const LAW_OPTIONS: LawCategory[] = ['BNS', 'BNSS', 'BSA', 'IPC', 'CrPC']

function MessageBubble({ role, content, citations, confidence, warnings, isStreaming = false }:
  { role: 'user' | 'assistant'; content: string; citations?: any[]; confidence?: number; warnings?: string[]; isStreaming?: boolean }) {
  const isUser = role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} gap-3`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center shrink-0 mt-1">
          <Scale className="w-3.5 h-3.5 text-primary" />
        </div>
      )}
      <div className={`max-w-[80%] space-y-2 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div className={`rounded-xl px-4 py-3 text-sm ${isUser
          ? 'bg-primary text-primary-foreground rounded-tr-sm'
          : 'bg-card border border-border rounded-tl-sm'
        } ${isStreaming && !content ? 'animate-pulse' : ''}`}>
          {isUser ? <p>{content}</p> : (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content || '…'}</ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && (citations?.length || confidence !== undefined || warnings?.length) ? (
          <div className="flex flex-wrap items-center gap-2 px-1">
            {confidence !== undefined && (
              <span className={`text-xs font-medium ${confidence > 0.75 ? 'text-emerald-400' : confidence > 0.5 ? 'text-amber-400' : 'text-red-400'}`}>
                {Math.round(confidence * 100)}% verified
              </span>
            )}
            {citations?.map((cit, i) => (
              <span key={i} className="text-xs flex items-center gap-1 bg-emerald-400/5 border border-emerald-400/20 text-emerald-300 rounded px-2 py-0.5">
                <CheckCircle2 className="w-2.5 h-2.5" />{cit.citation_text}
              </span>
            ))}
            {warnings?.map((w, i) => (
              <span key={i} className="flex items-center gap-1 text-xs text-amber-400">
                <AlertTriangle className="w-2.5 h-2.5" />{w.slice(0, 80)}{w.length > 80 ? '…' : ''}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}

function ChatContent() {
  const searchParams = useSearchParams()
  const documentId = searchParams.get('document_id')
  const documentTitle = searchParams.get('document_title')

  const { messages, sessionId, isStreaming, streamingContent,
    addMessage, setStreaming, appendStreamToken, commitStreamedMessage,
    clearSession } = useChatStore()

  const [input, setInput] = useState('')
  const [lawFilter, setLawFilter] = useState<LawCategory[]>([])
  const [showScrollBtn, setShowScrollBtn] = useState(false)
  const [contextDismissed, setContextDismissed] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => { scrollToBottom() }, [messages.length, isStreaming, streamingContent])

  // Pre-fill input when coming from a specific document
  useEffect(() => {
    if (documentTitle && messages.length === 0) {
      setInput(`Summarize the key legal principles from the case: ${documentTitle}`)
    }
  }, [documentTitle])

  const handleSend = useCallback(async () => {
    const msg = input.trim()
    if (!msg || isStreaming) return

    addMessage({ role: 'user', content: msg, timestamp: new Date().toISOString() })
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    setStreaming(true)

    // If chatting about a specific document, inject document_id as filter
    // Strip extra fields from history — backend only accepts role/content/timestamp
    const cleanHistory = messages.slice(-10).map(m => ({
      role: m.role,
      content: m.content,
      timestamp: (m as any).timestamp || new Date().toISOString(),
    }))

    await chatApi.streamChat(
      {
        message: msg,
        history: cleanHistory,
        session_id: sessionId || undefined,
        law_filter: lawFilter.length ? lawFilter : undefined,
        document_id: documentId || undefined,
        stream: true,
      },
      (token) => appendStreamToken(token),
      () => commitStreamedMessage(),
      (err) => { toast.error(`Chat error: ${err}`); commitStreamedMessage() }
    )
  }, [input, isStreaming, messages, sessionId, lawFilter, documentId,
    addMessage, setStreaming, appendStreamToken, commitStreamedMessage])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`
  }

  const SUGGESTIONS = documentTitle ? [
    `What is the ratio decidendi in ${documentTitle}?`,
    `What constitutional provisions were discussed?`,
    `What was the final order in this case?`,
    `How does this judgment affect future cases?`,
  ] : [
    'What is Section 103 of BNS? When does murder become culpable homicide?',
    'Explain anticipatory bail under BNSS Section 482',
    'How is electronic evidence admitted under BSA?',
    'What are the grounds for quashing an FIR under BNSS 528?',
    'Difference between BNS and old IPC for theft provisions',
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
          {documentTitle && (
            <span className="flex items-center gap-1.5 text-xs bg-primary/10 border border-primary/20 text-primary rounded-full px-2.5 py-1">
              <FileText className="w-3 h-3" />
              {documentTitle.length > 40 ? documentTitle.slice(0, 40) + '…' : documentTitle}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-muted-foreground" />
            {LAW_OPTIONS.map(law => (
              <button key={law} onClick={() => setLawFilter(p => p.includes(law) ? p.filter(l => l !== law) : [...p, law])}
                className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${lawFilter.includes(law)
                  ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:border-primary/30'}`}>
                {law}
              </button>
            ))}
          </div>
          {messages.length > 0 && (
            <button onClick={clearSession}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
              <Trash2 className="w-3.5 h-3.5" />Clear
            </button>
          )}
        </div>
      </header>

      {/* Document context banner */}
      {documentTitle && !contextDismissed && messages.length === 0 && (
        <div className="shrink-0 bg-primary/5 border-b border-primary/20 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-4 h-4 text-primary shrink-0" />
            <div>
              <p className="text-sm font-medium text-primary">Chatting about: {documentTitle}</p>
              <p className="text-xs text-muted-foreground">
                Retrieval will prioritise chunks from this document. Ask anything about the case.
              </p>
            </div>
          </div>
          <button onClick={() => setContextDismissed(true)} className="text-muted-foreground hover:text-foreground">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Messages */}
      <div ref={scrollRef} onScroll={() => {
        if (!scrollRef.current) return
        const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
        setShowScrollBtn(scrollHeight - scrollTop - clientHeight > 100)
      }} className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
            <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
              <Scale className="w-7 h-7 text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-semibold mb-2">
                {documentTitle ? `Ask about ${documentTitle}` : 'NyayaAI Legal Assistant'}
              </h2>
              <p className="text-sm text-muted-foreground max-w-md">
                {documentTitle
                  ? 'Ask anything about this case — facts, issues, ratio decidendi, final order, or implications.'
                  : 'Ask about BNS provisions, BNSS procedures, BSA evidence rules, or Supreme Court judgments. Every answer is backed by verified citations.'}
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => { setInput(s); textareaRef.current?.focus() }}
                  className="text-left text-xs border border-border rounded-lg px-3 py-2.5 hover:border-primary/40 hover:bg-card transition-colors text-muted-foreground hover:text-foreground">
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble key={i} role={msg.role as 'user' | 'assistant'}
                content={msg.content} citations={(msg as any).citations}
                confidence={(msg as any).confidence} warnings={(msg as any).warnings} />
            ))}
            {isStreaming && (
              <MessageBubble role="assistant" content={streamingContent || ''} isStreaming={!streamingContent} />
            )}
          </>
        )}
        <div ref={bottomRef} />
      </div>

      {showScrollBtn && (
        <button onClick={scrollToBottom}
          className="absolute bottom-24 right-8 w-8 h-8 rounded-full bg-card border border-border flex items-center justify-center shadow-lg hover:border-primary/40 transition-colors">
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        </button>
      )}

      {/* Input */}
      <div className="shrink-0 border-t border-border bg-card/50 backdrop-blur-sm p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end gap-3 bg-background border border-border rounded-xl px-4 py-3 focus-within:ring-2 focus-within:ring-primary/30 focus-within:border-primary/50 transition-all">
            <textarea ref={textareaRef} value={input} onChange={handleChange} onKeyDown={handleKeyDown}
              placeholder={documentTitle ? `Ask about ${documentTitle}…` : 'Ask about Indian law… (Enter to send)'}
              rows={1} disabled={isStreaming}
              className="flex-1 bg-transparent text-sm resize-none focus:outline-none placeholder:text-muted-foreground disabled:opacity-50 min-h-[24px] max-h-40" />
            <button onClick={handleSend} disabled={!input.trim() || isStreaming}
              className="w-8 h-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center hover:bg-primary/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0">
              {isStreaming
                ? <span className="w-3.5 h-3.5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                : <Send className="w-3.5 h-3.5" />}
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

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-screen"><div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full" /></div>}>
      <ChatContent />
    </Suspense>
  )
}
