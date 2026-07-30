'use client'

import { useEffect, useState } from 'react'
import { MessageSquarePlus, Trash2, ChevronLeft, ChevronRight, MessageSquare } from 'lucide-react'
import { useChatStore } from '@/lib/store'
import { toast } from 'sonner'

function formatRelativeTime(iso: string | null): string {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''

  const diffMs = Date.now() - date.getTime()
  const diffSec = Math.round(diffMs / 1000)
  const diffMin = Math.round(diffSec / 60)
  const diffHr = Math.round(diffMin / 60)
  const diffDay = Math.round(diffHr / 24)

  if (diffSec < 60) return 'Just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHr < 24) return `${diffHr}h ago`
  if (diffDay < 7) return `${diffDay}d ago`
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function ChatHistorySidebar() {
  const {
    sessions, sessionsLoading, sessionId,
    loadSessions, loadSession, deleteSession, startNewChat,
  } = useChatStore()

  const [collapsed, setCollapsed] = useState(false)
  const [confirmingId, setConfirmingId] = useState<string | null>(null)

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  const handleSelect = async (id: string) => {
    if (id === sessionId) return
    try {
      await loadSession(id)
    } catch {
      toast.error('Could not load that conversation')
    }
  }

  const handleDeleteClick = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    if (confirmingId !== id) {
      setConfirmingId(id)
      return
    }
    handleConfirmDelete(id)
  }

  const handleConfirmDelete = async (id: string) => {
    setConfirmingId(null)
    try {
      await deleteSession(id)
    } catch {
      toast.error('Failed to delete conversation')
    }
  }

  if (collapsed) {
    return (
      <div className="flex flex-col items-center shrink-0 w-12 border-r border-border bg-card/30 py-3 gap-2">
        <button
          onClick={() => setCollapsed(false)}
          title="Show chat history"
          className="text-muted-foreground hover:text-foreground p-1.5 rounded-lg hover:bg-accent transition-colors"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
        <button
          onClick={startNewChat}
          title="New chat"
          className="text-muted-foreground hover:text-foreground p-1.5 rounded-lg hover:bg-accent transition-colors"
        >
          <MessageSquarePlus className="w-4 h-4" />
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col shrink-0 w-64 border-r border-border bg-card/30">
      <div className="flex items-center justify-between px-3 py-3 border-b border-border">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Chats</span>
        <button
          onClick={() => setCollapsed(true)}
          title="Hide chat history"
          className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-accent transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
      </div>

      <div className="px-2 pt-2">
        <button
          onClick={startNewChat}
          className="w-full flex items-center gap-2 text-sm px-2.5 py-2 rounded-lg border border-border hover:border-primary/40 hover:bg-accent transition-colors text-foreground"
        >
          <MessageSquarePlus className="w-4 h-4 shrink-0" />
          New chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-0.5">
        {sessionsLoading && sessions.length === 0 ? (
          <div className="px-2.5 py-2 text-xs text-muted-foreground">Loading…</div>
        ) : sessions.length === 0 ? (
          <div className="px-2.5 py-4 text-xs text-muted-foreground text-center">
            No conversations yet
          </div>
        ) : (
          sessions.map((s) => {
            const active = s.session_id === sessionId
            const confirming = confirmingId === s.session_id
            return (
              <div
                key={s.session_id}
                onClick={() => handleSelect(s.session_id)}
                className={`group flex items-center gap-2 px-2.5 py-2 rounded-lg text-sm cursor-pointer transition-colors ${
                  active
                    ? 'bg-primary/10 text-primary border border-primary/20'
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent border border-transparent'
                }`}
              >
                <MessageSquare className="w-3.5 h-3.5 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="truncate">{s.title || 'New conversation'}</p>
                  <p className="text-[11px] text-muted-foreground/70">
                    {formatRelativeTime(s.updated_at)}
                  </p>
                </div>
                <button
                  onClick={(e) => handleDeleteClick(e, s.session_id)}
                  onBlur={() => setConfirmingId(null)}
                  title={confirming ? 'Click again to confirm delete' : 'Delete conversation'}
                  className={`shrink-0 p-1 rounded transition-colors ${
                    confirming
                      ? 'text-red-400 opacity-100'
                      : 'opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-red-400'
                  }`}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
