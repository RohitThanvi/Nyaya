'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Upload, X, CheckCircle, AlertCircle, Clock, FileText,
  Loader2, FolderOpen, ChevronDown, ChevronUp, RotateCcw,
} from 'lucide-react'
import { toast } from 'sonner'
import axios from 'axios'
import Cookies from 'js-cookie'
import { getErrorMessage } from '@/lib/api'
import { resumableUpload } from '@/lib/resumableUpload'
import type { UploadProgressInfo } from '@/lib/resumableUpload'

// ── Constants ─────────────────────────────────────────────────────────────────
const BASE_URL   = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const API        = `${BASE_URL}/api/v1`
const CONCURRENT = 4                               // parallel uploads at once
const CHUNK_MB   = 50                              // MB per HTTP chunk
const MAX_GB     = 50                              // max file size
const ALLOWED    = ['.pdf', '.txt', '.docx', '.doc']

// ── Types ─────────────────────────────────────────────────────────────────────
type FileStatus = 'queued' | 'uploading' | 'processing' | 'done' | 'error' | 'cancelled'

interface FileEntry {
  id:        string
  file:      File
  status:    FileStatus
  progress:  UploadProgressInfo | null
  result:    { document_id: string; pages: number; chunks: number } | null
  error:     string | null
  uploadId:  string | null   // server-assigned resumable upload ID
}

function fmt(bytes: number): string {
  if (bytes < 1024)        return `${bytes} B`
  if (bytes < 1024 ** 2)  return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3)  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

function authHeaders() {
  const t = Cookies.get('access_token')
  return t ? { Authorization: `Bearer ${t}` } : {}
}

// ── Bulk initiate: register all files server-side in one request ──────────────
async function bulkInitiate(files: File[]): Promise<Map<string, string>> {
  const items = files.map(f => ({
    filename:     f.name,
    total_size:   f.size,
    content_type: f.type || 'application/pdf',
  }))
  const res = await axios.post(`${API}/upload/bulk/initiate`, items, {
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
  })
  const map = new Map<string, string>()
  for (const s of res.data.sessions) {
    if (s.upload_id) map.set(s.filename, s.upload_id)
  }
  return map
}

// ── Status badge ──────────────────────────────────────────────────────────────
function StatusBadge({ status }: { status: FileStatus }) {
  const cfg: Record<FileStatus, { icon: React.ReactNode; label: string; cls: string }> = {
    queued:     { icon: <Clock size={12} />,   label: 'Queued',     cls: 'text-muted-foreground' },
    uploading:  { icon: <Loader2 size={12} className="animate-spin" />, label: 'Uploading', cls: 'text-primary' },
    processing: { icon: <Loader2 size={12} className="animate-spin" />, label: 'Processing', cls: 'text-amber-400' },
    done:       { icon: <CheckCircle size={12} />, label: 'Done',   cls: 'text-emerald-400' },
    error:      { icon: <AlertCircle size={12} />, label: 'Failed', cls: 'text-destructive' },
    cancelled:  { icon: <X size={12} />,       label: 'Cancelled',  cls: 'text-muted-foreground' },
  }
  const { icon, label, cls } = cfg[status]
  return (
    <span className={`flex items-center gap-1 text-xs font-medium ${cls}`}>
      {icon}{label}
    </span>
  )
}

// ── Per-file progress bar ─────────────────────────────────────────────────────
function FileRow({
  entry, onCancel, onRetry,
}: {
  entry: FileEntry
  onCancel: (id: string) => void
  onRetry:  (id: string) => void
}) {
  const pct = entry.progress?.percent ?? 0
  const isActive = entry.status === 'uploading' || entry.status === 'processing'

  return (
    <div className="border border-border rounded-lg p-3 bg-card/30 gap-2 flex flex-col">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <FileText size={14} className="text-muted-foreground shrink-0" />
          <span className="text-sm truncate font-medium" title={entry.file.name}>
            {entry.file.name}
          </span>
          <span className="text-xs text-muted-foreground shrink-0">{fmt(entry.file.size)}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <StatusBadge status={entry.status} />
          {entry.status === 'error' && (
            <button onClick={() => onRetry(entry.id)}
              className="text-xs text-primary hover:underline flex items-center gap-1">
              <RotateCcw size={11} /> Retry
            </button>
          )}
          {(entry.status === 'queued' || isActive) && (
            <button onClick={() => onCancel(entry.id)}
              className="text-muted-foreground hover:text-foreground">
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      {isActive && (
        <div className="space-y-1">
          <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>{entry.progress?.message || '…'}</span>
            <span>{pct.toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Results */}
      {entry.status === 'done' && entry.result && (
        <div className="flex gap-3 text-xs text-muted-foreground">
          {entry.result.pages > 0 && <span>{entry.result.pages} pages</span>}
          {entry.result.chunks > 0 && <span>· {entry.result.chunks} chunks indexed</span>}
        </div>
      )}
      {entry.status === 'error' && entry.error && (
        <p className="text-xs text-destructive">{entry.error}</p>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function UploadPage() {
  const [entries, setEntries] = useState<FileEntry[]>([])
  const [dragging, setDragging] = useState(false)
  const [showDone, setShowDone] = useState(true)
  const [initiating, setInitiating] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Active upload slots — keys are entry IDs currently being uploaded
  const active = useRef<Set<string>>(new Set())
  // Abort map for in-progress uploads
  const abortMap = useRef<Map<string, () => void>>(new Map())
  // Entries ref so callbacks always have fresh state
  const entriesRef = useRef<FileEntry[]>([])
  entriesRef.current = entries

  // ── Stats ──────────────────────────────────────────────────────────────────
  const total     = entries.length
  const done      = entries.filter(e => e.status === 'done').length
  const failed    = entries.filter(e => e.status === 'error').length
  const cancelled = entries.filter(e => e.status === 'cancelled').length
  const inFlight  = entries.filter(e => e.status === 'uploading' || e.status === 'processing').length
  const queued    = entries.filter(e => e.status === 'queued').length
  const overallPct = total === 0 ? 0 : Math.round(((done + failed + cancelled) / total) * 100)
  const allFinished = total > 0 && queued === 0 && inFlight === 0

  // ── Update a single entry ──────────────────────────────────────────────────
  const update = useCallback((id: string, patch: Partial<FileEntry>) => {
    setEntries(prev => prev.map(e => e.id === id ? { ...e, ...patch } : e))
  }, [])

  // ── Upload one file ────────────────────────────────────────────────────────
  const uploadOne = useCallback(async (entry: FileEntry) => {
    if (entry.status === 'cancelled') return
    active.current.add(entry.id)
    update(entry.id, { status: 'uploading' })

    let aborted = false
    abortMap.current.set(entry.id, () => { aborted = true })

    try {
      const result = await resumableUpload(
        entry.file,
        (info) => {
          if (aborted) return
          const status: FileStatus = info.phase === 'ready' ? 'done'
            : info.phase === 'indexing' || info.phase === 'finalising' ? 'processing'
            : 'uploading'
          update(entry.id, { progress: info, status })
        },
        entry.uploadId ?? undefined,   // pre-registered upload_id
      )

      if (aborted) return

      update(entry.id, {
        status: 'done',
        result: {
          document_id: result.document_id,
          pages:  result.pages,
          chunks: result.chunks_created,
        },
        progress: { phase: 'ready', percent: 100, bytesUploaded: entry.file.size,
                    bytesTotal: entry.file.size, message: 'Indexed' },
      })
    } catch (err: any) {
      if (aborted) return
      update(entry.id, {
        status: 'error',
        error: getErrorMessage(err, 'Upload failed'),
      })
    } finally {
      active.current.delete(entry.id)
      abortMap.current.delete(entry.id)
      startNext()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [update])

  // ── Drain queue: start up to CONCURRENT uploads ────────────────────────────
  const startNext = useCallback(() => {
    const slots = CONCURRENT - active.current.size
    if (slots <= 0) return
    const waiting = entriesRef.current.filter(e => e.status === 'queued')
    waiting.slice(0, slots).forEach(e => uploadOne(e))
  }, [uploadOne])

  // ── Add files to queue ─────────────────────────────────────────────────────
  const enqueue = useCallback(async (files: File[]) => {
    const valid = files.filter(f => {
      const ext = '.' + f.name.split('.').pop()!.toLowerCase()
      if (!ALLOWED.includes(ext)) {
        toast.error(`${f.name}: unsupported type (${ext})`)
        return false
      }
      if (f.size > MAX_GB * 1024 ** 3) {
        toast.error(`${f.name}: exceeds ${MAX_GB}GB limit`)
        return false
      }
      return true
    })
    if (!valid.length) return

    setInitiating(true)
    let uploadIdMap = new Map<string, string>()
    try {
      uploadIdMap = await bulkInitiate(valid)
    } catch (err) {
      toast.error('Failed to register uploads with server — will retry on upload start')
    } finally {
      setInitiating(false)
    }

    const newEntries: FileEntry[] = valid.map(f => ({
      id:       crypto.randomUUID(),
      file:     f,
      status:   'queued',
      progress: null,
      result:   null,
      error:    null,
      uploadId: uploadIdMap.get(f.name) ?? null,
    }))

    setEntries(prev => [...prev, ...newEntries])
    // startNext will fire via useEffect watching entries
  }, [])

  // Start uploads whenever entries changes and there are free slots
  useEffect(() => {
    startNext()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entries.length])

  // ── Cancel ─────────────────────────────────────────────────────────────────
  const cancel = useCallback((id: string) => {
    const abort = abortMap.current.get(id)
    if (abort) abort()
    update(id, { status: 'cancelled', error: null })
    active.current.delete(id)
    setTimeout(startNext, 50)
  }, [update, startNext])

  // ── Retry ──────────────────────────────────────────────────────────────────
  const retry = useCallback((id: string) => {
    update(id, { status: 'queued', error: null, progress: null })
    setTimeout(startNext, 50)
  }, [update, startNext])

  const cancelAll = () => {
    entries.filter(e => e.status === 'queued' || e.status === 'uploading' || e.status === 'processing')
      .forEach(e => cancel(e.id))
  }

  const retryAll = () => {
    entries.filter(e => e.status === 'error').forEach(e => retry(e.id))
  }

  const clearAll = () => {
    cancelAll()
    setEntries([])
    active.current.clear()
  }

  // ── Drop handling ──────────────────────────────────────────────────────────
  const collectFiles = async (items: DataTransferItemList | FileList): Promise<File[]> => {
    const files: File[] = []

    const walk = async (entry: FileSystemEntry | null) => {
      if (!entry) return
      if (entry.isFile) {
        await new Promise<void>(res => {
          (entry as FileSystemFileEntry).file(f => { files.push(f); res() }, () => res())
        })
      } else if (entry.isDirectory) {
        await new Promise<void>(res => {
          const reader = (entry as FileSystemDirectoryEntry).createReader()
          const readAll = () => reader.readEntries(async entries => {
            if (!entries.length) return res()
            for (const e of entries) await walk(e)
            readAll()
          }, () => res())
          readAll()
        })
      }
    }

    if (items instanceof FileList) {
      for (const f of Array.from(items)) files.push(f)
    } else {
      for (const item of Array.from(items)) {
        if (item.kind !== 'file') continue
        const entry = item.webkitGetAsEntry?.()
        if (entry) await walk(entry)
        else { const f = item.getAsFile(); if (f) files.push(f) }
      }
    }
    return files
  }

  const onDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const files = await collectFiles(e.dataTransfer.items)
    if (files.length) enqueue(files)
  }, [enqueue])

  const onFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      enqueue(Array.from(e.target.files))
      e.target.value = ''
    }
  }, [enqueue])

  const visibleDone = showDone ? entries : entries.filter(e => e.status !== 'done')

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Bulk Document Upload</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Drop thousands of files or entire folders — {CONCURRENT} upload in parallel, all indexed automatically
          </p>
        </div>
        {total > 0 && (
          <div className="flex items-center gap-2">
            {failed > 0 && (
              <button onClick={retryAll}
                className="text-xs px-3 py-1.5 rounded border border-border hover:bg-muted flex items-center gap-1">
                <RotateCcw size={12} /> Retry {failed} failed
              </button>
            )}
            <button onClick={clearAll}
              className="text-xs px-3 py-1.5 rounded border border-destructive/50 text-destructive hover:bg-destructive/10">
              Clear all
            </button>
          </div>
        )}
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all
          ${dragging
            ? 'border-primary bg-primary/10 scale-[1.01]'
            : 'border-border hover:border-primary/50 hover:bg-muted/20'
          }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.docx,.doc"
          className="hidden"
          onChange={onFileInput}
        />
        {initiating ? (
          <div className="flex flex-col items-center gap-2">
            <Loader2 size={36} className="animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">Registering files…</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className={`p-4 rounded-full transition-colors ${dragging ? 'bg-primary/20' : 'bg-muted'}`}>
              {dragging ? <FolderOpen size={32} className="text-primary" /> : <Upload size={32} className="text-muted-foreground" />}
            </div>
            <div>
              <p className="font-medium">Drop files or folders here</p>
              <p className="text-sm text-muted-foreground mt-1">
                PDF, DOCX, TXT · Up to {MAX_GB}GB per file · Drag entire folders for recursive import
              </p>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xs px-3 py-1.5 bg-primary text-primary-foreground rounded-full font-medium">
                Browse Files
              </span>
              <span className="text-xs text-muted-foreground">or drag here</span>
            </div>
          </div>
        )}
      </div>

      {/* Overall progress bar */}
      {total > 0 && (
        <div className="space-y-2 p-4 bg-card border border-border rounded-xl">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-4 text-sm">
              <span className="font-medium">{total} file{total !== 1 ? 's' : ''}</span>
              {done > 0       && <span className="text-emerald-400">✓ {done} done</span>}
              {inFlight > 0   && <span className="text-primary">⟳ {inFlight} active</span>}
              {queued > 0     && <span className="text-muted-foreground">· {queued} queued</span>}
              {failed > 0     && <span className="text-destructive">✕ {failed} failed</span>}
              {cancelled > 0  && <span className="text-muted-foreground">✕ {cancelled} cancelled</span>}
            </div>
            <span className="text-sm font-mono text-muted-foreground">{overallPct}%</span>
          </div>
          <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
            <div className="h-full flex rounded-full overflow-hidden">
              <div className="bg-emerald-500 transition-all duration-500"
                style={{ width: `${(done / total) * 100}%` }} />
              <div className="bg-destructive transition-all duration-500"
                style={{ width: `${(failed / total) * 100}%` }} />
              <div className="bg-muted-foreground/30 transition-all duration-500"
                style={{ width: `${(cancelled / total) * 100}%` }} />
            </div>
          </div>
          {allFinished && (
            <p className="text-xs text-muted-foreground">
              {done > 0 && `${done} document${done !== 1 ? 's' : ''} indexed and ready for search. `}
              {failed > 0 && `${failed} failed — click Retry above.`}
            </p>
          )}
        </div>
      )}

      {/* File list */}
      {total > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-muted-foreground">
              {visibleDone.length} of {total} shown
            </span>
            {done > 0 && (
              <button onClick={() => setShowDone(p => !p)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
                {showDone ? <><ChevronUp size={12}/> Hide {done} done</> : <><ChevronDown size={12}/> Show {done} done</>}
              </button>
            )}
          </div>
          <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
            {visibleDone.map(e => (
              <FileRow key={e.id} entry={e} onCancel={cancel} onRetry={retry} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
