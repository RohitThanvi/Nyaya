'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { FileUp, File, CheckCircle2, AlertCircle, X, Search, Loader2, RotateCcw, WifiOff } from 'lucide-react'
import { chatApi, getErrorMessage } from '@/lib/api'
import { uploadResumable, cancelResumableUpload, type UploadProgressInfo } from '@/lib/resumableUpload'
import type { UploadResponse, LegalResponse } from '@/types/api'
import { toast } from 'sonner'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { CitationList } from '@/components/citations/CitationCard'

type UploadState = 'idle' | 'uploading' | 'resuming' | 'finalising' | 'indexing' | 'ready' | 'error'

const MAX_FILE_GB = 10   // matches backend INGEST_MAX_FILE_SIZE_GB

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`
  return `${(bytes / 1e3).toFixed(0)} KB`
}

export default function UploadPage() {
  const [uploadState, setUploadState] = useState<UploadState>('idle')
  const [progressInfo, setProgressInfo] = useState<UploadProgressInfo | null>(null)
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [query, setQuery] = useState('')
  const [isQuerying, setIsQuerying] = useState(false)
  const [queryResult, setQueryResult] = useState<LegalResponse | null>(null)
  const [isOffline, setIsOffline] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const currentFileRef = useRef<File | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Detect network status — surfaces clearly when a retry is network-related
  useEffect(() => {
    const goOffline = () => setIsOffline(true)
    const goOnline = () => setIsOffline(false)
    window.addEventListener('offline', goOffline)
    window.addEventListener('online', goOnline)
    setIsOffline(!navigator.onLine)
    return () => {
      window.removeEventListener('offline', goOffline)
      window.removeEventListener('online', goOnline)
    }
  }, [])

  const handleFile = useCallback(async (file: File) => {
    const allowed = ['application/pdf', 'text/plain']
    const maxBytes = MAX_FILE_GB * 1024 * 1024 * 1024
    if (!allowed.includes(file.type) && !file.name.toLowerCase().endsWith('.pdf') && !file.name.toLowerCase().endsWith('.txt')) {
      toast.error('Only PDF and TXT files are supported')
      return
    }
    if (file.size > maxBytes) {
      toast.error(`File too large. Maximum ${MAX_FILE_GB}GB`)
      return
    }

    currentFileRef.current = file
    abortControllerRef.current = new AbortController()
    setUploadState('uploading')
    setProgressInfo(null)
    setError(null)
    setUploadResult(null)
    setQueryResult(null)

    try {
      const result = await uploadResumable(file, {
        signal: abortControllerRef.current.signal,
        onProgress: (info) => {
          setProgressInfo(info)
          if (info.phase === 'resuming') setUploadState('resuming')
          else if (info.phase === 'finalising') setUploadState('finalising')
          else if (info.phase === 'indexing') setUploadState('indexing')
          else if (info.phase === 'uploading') setUploadState('uploading')
        },
      })
      setUploadResult(result)
      setUploadState('ready')
      if (result.status === 'partial') {
        toast.warning(
          `Indexed ${result.chunks_created} chunks, but ${result.failed_chunk_ids.length} ` +
          `failed embedding and won't appear in semantic search. Text search still works for them.`
        )
      } else if (result.status === 'processing') {
        toast.info('Document uploaded. Indexing is running in the background — it will be ready for search shortly.')
      } else {
        toast.success(`Indexed ${result.chunks_created} chunks from ${result.pages} pages`)
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        setUploadState('idle')
        return
      }
      const msg = getErrorMessage(err, 'Upload failed. Please try again.')
      setError(msg)
      setUploadState('error')
      toast.error(msg)
    }
  }, [])

  const handleRetry = useCallback(() => {
    // Re-dropping the same file triggers automatic resume — the engine
    // detects the matching fingerprint in localStorage and only uploads
    // whatever bytes never made it to the server.
    if (currentFileRef.current) {
      handleFile(currentFileRef.current)
    }
  }, [handleFile])

  const handleCancel = useCallback(async () => {
    abortControllerRef.current?.abort()
    if (currentFileRef.current) {
      await cancelResumableUpload(currentFileRef.current)
    }
    setUploadState('idle')
    setProgressInfo(null)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const handleQuery = async () => {
    if (!query.trim() || !uploadResult || isQuerying) return
    setIsQuerying(true)
    try {
      const result = await chatApi.chatWithDocument(uploadResult.document_id, query.trim(), [])
      setQueryResult(result)
    } catch {
      toast.error('Query failed. Please try again.')
    } finally {
      setIsQuerying(false)
    }
  }

  const reset = () => {
    currentFileRef.current = null
    setUploadState('idle')
    setProgressInfo(null)
    setUploadResult(null)
    setError(null)
    setQuery('')
    setQueryResult(null)
  }

  const phaseLabel: Record<string, string> = {
    uploading: 'Uploading document…',
    resuming: 'Resuming interrupted upload…',
    finalising: 'Verifying upload…',
    indexing: 'Building semantic index…',
  }

  const phaseSubLabel: Record<string, string> = {
    uploading: 'Transferring your file to the server',
    resuming: 'Picking up exactly where the connection dropped — no data lost',
    finalising: 'Checking all bytes arrived correctly',
    indexing: 'Chunking, embedding, and indexing for search',
  }

  const isBusy = ['uploading', 'resuming', 'finalising', 'indexing'].includes(uploadState)

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="mb-10">
          <h1 className="text-2xl font-bold mb-2">Upload & Analyze Document</h1>
          <p className="text-sm text-muted-foreground">
            Upload a PDF judgment, statute, or any legal document to ask questions against it.
            Large files upload in resumable chunks — network drops won&apos;t lose your progress.
          </p>
        </div>

        {isOffline && isBusy && (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-300">
            <WifiOff className="w-4 h-4 shrink-0" />
            You&apos;re offline. The upload will automatically resume once your connection returns.
          </div>
        )}

        {uploadState === 'idle' && (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all ${
              dragActive
                ? 'border-primary bg-primary/5 scale-[1.01]'
                : 'border-border hover:border-primary/40 hover:bg-card/50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            <FileUp className={`w-12 h-12 mx-auto mb-4 transition-colors ${dragActive ? 'text-primary' : 'text-muted-foreground'}`} />
            <p className="text-base font-medium mb-1">Drop your document here</p>
            <p className="text-sm text-muted-foreground mb-4">or click to browse</p>
            <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
              <span className="flex items-center gap-1.5 border border-border rounded-full px-3 py-1">
                <File className="w-3 h-3" /> PDF
              </span>
              <span className="flex items-center gap-1.5 border border-border rounded-full px-3 py-1">
                <File className="w-3 h-3" /> TXT
              </span>
              <span className="border border-border rounded-full px-3 py-1">Up to {MAX_FILE_GB}GB · resumable</span>
            </div>
          </div>
        )}

        {isBusy && (
          <div className="bg-card border border-border rounded-2xl p-8 text-center">
            <Loader2 className="w-10 h-10 mx-auto mb-4 text-primary animate-spin" />
            <p className="font-medium mb-2">{phaseLabel[uploadState] || 'Processing…'}</p>
            <p className="text-sm text-muted-foreground mb-4">
              {progressInfo?.message || phaseSubLabel[uploadState] || ''}
            </p>
            <div className="w-full bg-background rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-300"
                style={{ width: `${progressInfo?.percent ?? (uploadState === 'indexing' ? 95 : 0)}%` }}
              />
            </div>
            <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
              <span>
                {progressInfo ? `${formatBytes(progressInfo.bytesUploaded)} / ${formatBytes(progressInfo.bytesTotal)}` : ''}
              </span>
              <span>{progressInfo?.percent ?? 0}%</span>
            </div>
            {progressInfo && progressInfo.currentRetry > 0 && (
              <p className="text-xs text-amber-400 mt-2">
                Retrying chunk (attempt {progressInfo.currentRetry}) — your upload is not lost
              </p>
            )}
            <button
              onClick={handleCancel}
              className="mt-4 text-xs text-muted-foreground hover:text-foreground underline transition-colors"
            >
              Cancel upload
            </button>
          </div>
        )}

        {uploadState === 'error' && (
          <div className="bg-card border border-red-500/30 rounded-2xl p-8 text-center">
            <AlertCircle className="w-10 h-10 mx-auto mb-4 text-red-400" />
            <p className="font-medium mb-1 text-red-400">Upload interrupted</p>
            <p className="text-sm text-muted-foreground mb-6">{error}</p>
            <div className="flex items-center justify-center gap-3">
              <button
                onClick={handleRetry}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Resume upload
              </button>
              <button onClick={reset}
                className="px-4 py-2 bg-card border border-border rounded-lg text-sm hover:border-primary/40 transition-colors">
                Start over
              </button>
            </div>
            <p className="text-xs text-muted-foreground mt-4">
              Resume will continue from where it stopped — already-uploaded data isn&apos;t re-sent.
            </p>
          </div>
        )}

        {uploadState === 'ready' && uploadResult && (
          <div className="space-y-6">
            <div className={`bg-card border rounded-xl p-5 flex items-start justify-between ${
              uploadResult.status === 'partial' ? 'border-amber-500/30' : 'border-emerald-500/30'
            }`}>
              <div className="flex items-start gap-4">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${
                  uploadResult.status === 'partial'
                    ? 'bg-amber-400/10 border border-amber-400/20'
                    : 'bg-emerald-400/10 border border-emerald-400/20'
                }`}>
                  {uploadResult.status === 'partial'
                    ? <AlertCircle className="w-5 h-5 text-amber-400" />
                    : <CheckCircle2 className="w-5 h-5 text-emerald-400" />}
                </div>
                <div>
                  <p className="font-medium text-sm mb-1">{uploadResult.filename}</p>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground flex-wrap">
                    {uploadResult.pages > 0 && <span>{uploadResult.pages} pages</span>}
                    {uploadResult.pages > 0 && <span>·</span>}
                    {uploadResult.chunks_created > 0
                      ? <span>{uploadResult.chunks_created} searchable chunks</span>
                      : <span className="text-primary">Indexing in background…</span>
                    }
                    {uploadResult.status === 'partial' ? (
                      <>
                        <span>·</span>
                        <span className="text-amber-400">{uploadResult.failed_chunk_ids.length} chunk(s) not embedded</span>
                      </>
                    ) : uploadResult.status === 'processing' ? (
                      <>
                        <span>·</span>
                        <span className="text-primary">Will be searchable shortly</span>
                      </>
                    ) : (
                      <>
                        <span>·</span>
                        <span className="text-emerald-400">Ready for Q&amp;A</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
              <button onClick={reset} className="text-muted-foreground hover:text-foreground transition-colors">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="bg-card border border-border rounded-xl p-5">
              <h2 className="font-medium text-sm mb-3">Ask questions about this document</h2>
              <div className="flex gap-3">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
                  placeholder="e.g. 'What are the key findings?' or 'Which sections are discussed?'"
                  className="flex-1 bg-background border border-border rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 placeholder:text-muted-foreground"
                />
                <button
                  onClick={handleQuery}
                  disabled={isQuerying || !query.trim()}
                  className="flex items-center gap-2 px-4 py-2.5 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isQuerying ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                  Ask
                </button>
              </div>
              <div className="flex flex-wrap gap-2 mt-3">
                {['Summarize the facts', 'What is the final order?', 'List all sections discussed', 'What are the key legal principles?'].map(s => (
                  <button key={s} onClick={() => setQuery(s)}
                    className="text-xs border border-border rounded-full px-2.5 py-1 text-muted-foreground hover:border-primary/40 hover:text-primary transition-colors">
                    {s}
                  </button>
                ))}
              </div>
            </div>

            {isQuerying && (
              <div className="bg-card border border-border rounded-xl p-6 text-center">
                <Loader2 className="w-6 h-6 mx-auto mb-2 text-primary animate-spin" />
                <p className="text-sm text-muted-foreground">Searching document…</p>
              </div>
            )}
            {queryResult && !isQuerying && (
              <div className="bg-card border border-border rounded-xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Answer</span>
                  <span className={`text-xs ${queryResult.confidence > 0.7 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {Math.round(queryResult.confidence * 100)}% confidence
                  </span>
                </div>
                <div className="prose prose-sm prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{queryResult.answer}</ReactMarkdown>
                </div>
                {queryResult.hallucination_flags?.length > 0 && (
                  <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
                    <p className="text-xs font-semibold text-amber-400 mb-1">⚠ Unverified citations</p>
                    {queryResult.hallucination_flags.map((flag, i) => (
                      <p key={i} className="text-xs text-amber-300/80">{flag}</p>
                    ))}
                  </div>
                )}
                <CitationList citations={queryResult.citations ?? []} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
