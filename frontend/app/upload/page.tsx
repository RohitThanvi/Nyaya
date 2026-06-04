'use client'

import { useState, useCallback, useRef } from 'react'
import { FileUp, File, CheckCircle2, AlertCircle, X, Search, Loader2 } from 'lucide-react'
import { uploadApi, searchApi } from '@/lib/api'
import type { UploadResponse, LegalResponse } from '@/types/api'
import { toast } from 'sonner'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type UploadState = 'idle' | 'uploading' | 'indexing' | 'ready' | 'error'

export default function UploadPage() {
  const [uploadState, setUploadState] = useState<UploadState>('idle')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [query, setQuery] = useState('')
  const [isQuerying, setIsQuerying] = useState(false)
  const [queryResult, setQueryResult] = useState<LegalResponse | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(async (file: File) => {
    const allowed = ['application/pdf', 'text/plain']
    if (!allowed.includes(file.type)) {
      toast.error('Only PDF and TXT files are supported')
      return
    }
    if (file.size > 50 * 1024 * 1024) {
      toast.error('File too large. Maximum 50MB')
      return
    }

    setUploadState('uploading')
    setUploadProgress(0)
    setError(null)
    setUploadResult(null)
    setQueryResult(null)

    try {
      const result = await uploadApi.upload(file, (pct) => {
        setUploadProgress(pct)
        if (pct === 100) setUploadState('indexing')
      })
      setUploadResult(result)
      setUploadState('ready')
      toast.success(`Indexed ${result.chunks_created} chunks from ${result.pages} pages`)
    } catch (err: any) {
      const msg = err?.response?.data?.detail || 'Upload failed. Please try again.'
      setError(msg)
      setUploadState('error')
      toast.error(msg)
    }
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
      const result = await searchApi.search({
        query: query.trim(),
        top_k: 8,
      })
      setQueryResult(result)
    } catch (err: any) {
      toast.error('Query failed. Please try again.')
    } finally {
      setIsQuerying(false)
    }
  }

  const reset = () => {
    setUploadState('idle')
    setUploadProgress(0)
    setUploadResult(null)
    setError(null)
    setQuery('')
    setQueryResult(null)
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="mb-10">
          <h1 className="text-2xl font-bold mb-2">Upload & Analyze Document</h1>
          <p className="text-sm text-muted-foreground">
            Upload a PDF judgment, statute, or any legal document to ask questions against it.
          </p>
        </div>

        {/* Upload zone */}
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
              <span className="border border-border rounded-full px-3 py-1">Max 50MB</span>
            </div>
          </div>
        )}

        {/* Upload progress */}
        {(uploadState === 'uploading' || uploadState === 'indexing') && (
          <div className="bg-card border border-border rounded-2xl p-8 text-center">
            <Loader2 className="w-10 h-10 mx-auto mb-4 text-primary animate-spin" />
            <p className="font-medium mb-2">
              {uploadState === 'uploading' ? 'Uploading document…' : 'Building semantic index…'}
            </p>
            <p className="text-sm text-muted-foreground mb-4">
              {uploadState === 'uploading'
                ? 'Transferring your file to the server'
                : 'Chunking, embedding, and indexing for search'}
            </p>
            <div className="w-full bg-background rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-300"
                style={{ width: `${uploadState === 'indexing' ? 100 : uploadProgress}%` }}
              />
            </div>
            {uploadState === 'uploading' && (
              <p className="text-xs text-muted-foreground mt-2">{uploadProgress}%</p>
            )}
          </div>
        )}

        {/* Error state */}
        {uploadState === 'error' && (
          <div className="bg-card border border-red-500/30 rounded-2xl p-8 text-center">
            <AlertCircle className="w-10 h-10 mx-auto mb-4 text-red-400" />
            <p className="font-medium mb-1 text-red-400">Upload failed</p>
            <p className="text-sm text-muted-foreground mb-6">{error}</p>
            <button onClick={reset}
              className="px-4 py-2 bg-card border border-border rounded-lg text-sm hover:border-primary/40 transition-colors">
              Try again
            </button>
          </div>
        )}

        {/* Ready state — document indexed */}
        {uploadState === 'ready' && uploadResult && (
          <div className="space-y-6">
            {/* Document info card */}
            <div className="bg-card border border-emerald-500/30 rounded-xl p-5 flex items-start justify-between">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-emerald-400/10 border border-emerald-400/20 flex items-center justify-center shrink-0">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <p className="font-medium text-sm mb-1">{uploadResult.filename}</p>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <span>{uploadResult.pages} pages</span>
                    <span>·</span>
                    <span>{uploadResult.chunks_created} searchable chunks</span>
                    <span>·</span>
                    <span className="text-emerald-400">Ready for Q&A</span>
                  </div>
                </div>
              </div>
              <button onClick={reset} className="text-muted-foreground hover:text-foreground transition-colors">
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Query input */}
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
                  {isQuerying
                    ? <Loader2 className="w-4 h-4 animate-spin" />
                    : <Search className="w-4 h-4" />}
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

            {/* Query result */}
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
                {queryResult.citations?.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-2 border-t border-border">
                    {queryResult.citations.map((cit, i) => (
                      <span key={i} className="citation-card">{cit.citation_text}</span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
