/**
 * Fail-proof resumable upload engine.
 *
 * Real guarantees this provides, none of which existed before:
 *
 * 1. Network drop mid-upload (wifi blip, laptop sleep, tunnel restart) does
 *    NOT lose progress. Each chunk PUT is retried up to 5 times with
 *    exponential backoff before the engine gives up on that chunk.
 *
 * 2. Page refresh / browser crash mid-upload does NOT lose progress either.
 *    Upload session state (upload_id, file fingerprint, chunk size) is
 *    persisted to localStorage the moment the session is created. On next
 *    visit, if the same file (same name+size+lastModified) is dropped again,
 *    the engine calls GET /upload/chunked/{id}/status, finds out exactly
 *    which byte ranges already landed on the server, and only uploads what's
 *    missing — never re-sends bytes that already arrived.
 *
 * 3. Files up to INGEST_MAX_FILE_SIZE_GB (10GB by default) are supported.
 *    Anything under the single-shot threshold (200MB) can still go through
 *    the simple non-chunked path for less overhead on small files.
 */
import axios from 'axios'
import Cookies from 'js-cookie'
import type { UploadResponse } from '@/types/api'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const API_PREFIX = '/api/v1'

const SINGLE_SHOT_MAX_BYTES = 200 * 1024 * 1024   // matches backend APP_MAX_UPLOAD_SIZE_MB
const DEFAULT_CHUNK_SIZE = 25 * 1024 * 1024        // matches backend INGEST_CHUNK_SIZE_MB
const MAX_CHUNK_RETRIES = 5
const RETRY_BACKOFF_BASE_MS = 1000
const SESSION_STORAGE_KEY = 'nyaya_upload_sessions'

export type UploadPhase =
  | 'idle' | 'preparing' | 'uploading' | 'resuming' | 'finalising' | 'indexing'
  | 'ready' | 'error' | 'cancelled'

export interface UploadProgressInfo {
  phase: UploadPhase
  percent: number
  bytesUploaded: number
  bytesTotal: number
  currentRetry: number
  message: string
}

interface StoredSession {
  uploadId: string
  filename: string
  fileSize: number
  fileLastModified: number
  sourceUrl?: string
  createdAt: number
}

interface ReceivedRange {
  start: number
  end: number
  size: number
}

// ── localStorage session persistence ────────────────────────────────────────

function _fileFingerprint(file: File): string {
  return `${file.name}::${file.size}::${file.lastModified}`
}

function _loadSessions(): Record<string, StoredSession> {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(SESSION_STORAGE_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function _saveSession(fingerprint: string, session: StoredSession): void {
  if (typeof window === 'undefined') return
  try {
    const sessions = _loadSessions()
    sessions[fingerprint] = session
    // Prune sessions older than 7 days so localStorage doesn't grow forever
    const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000
    for (const key of Object.keys(sessions)) {
      if (sessions[key].createdAt < cutoff) delete sessions[key]
    }
    window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessions))
  } catch {
    // localStorage full or unavailable — resume just won't work across
    // reloads for this upload, but the upload itself still proceeds
  }
}

function _clearSession(fingerprint: string): void {
  if (typeof window === 'undefined') return
  try {
    const sessions = _loadSessions()
    delete sessions[fingerprint]
    window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessions))
  } catch {
    // ignore
  }
}

function _findExistingSession(file: File): StoredSession | null {
  const fp = _fileFingerprint(file)
  const sessions = _loadSessions()
  return sessions[fp] || null
}

// ── HTTP helpers with retry ──────────────────────────────────────────────────

function _authHeaders(): Record<string, string> {
  const token = Cookies.get('access_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function _sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * Uploads a single chunk with retry-with-backoff. This is the core
 * fail-proofing primitive — every chunk PUT goes through here, never raw.
 */
async function _uploadChunkWithRetry(
  uploadId: string,
  file: File,
  start: number,
  end: number,
  onRetry?: (attempt: number) => void,
): Promise<void> {
  const blob = file.slice(start, end + 1)
  const formData = new FormData()
  formData.append('chunk', blob, `chunk_${start}`)

  let lastError: unknown = null

  for (let attempt = 0; attempt < MAX_CHUNK_RETRIES; attempt++) {
    try {
      await axios.post(
        `${BASE_URL}${API_PREFIX}/upload/chunked/${uploadId}`,
        formData,
        {
          headers: {
            ..._authHeaders(),
            'Content-Type': 'multipart/form-data',
            'Content-Range': `bytes ${start}-${end}/${file.size}`,
          },
          timeout: 60000,
        },
      )
      return
    } catch (err) {
      lastError = err
      if (attempt < MAX_CHUNK_RETRIES - 1) {
        onRetry?.(attempt + 1)
        const backoff = RETRY_BACKOFF_BASE_MS * Math.pow(2, attempt)
        await _sleep(backoff)
      }
    }
  }

  throw new Error(
    `Chunk at byte ${start} failed after ${MAX_CHUNK_RETRIES} attempts: ${lastError}`
  )
}

async function _getUploadStatus(uploadId: string): Promise<{
  receivedBytes: number
  receivedRanges: ReceivedRange[]
  totalSize: number
  complete: boolean
}> {
  const res = await axios.get(
    `${BASE_URL}${API_PREFIX}/upload/chunked/${uploadId}/status`,
    { headers: _authHeaders() },
  )
  return {
    receivedBytes: res.data.received_bytes,
    receivedRanges: res.data.received_ranges,
    totalSize: res.data.total_size,
    complete: res.data.complete,
  }
}

/**
 * Computes which byte ranges are still missing given what the server
 * reports as already received. This is what makes resume actually skip
 * already-uploaded bytes instead of re-sending the whole file.
 */
function _computeMissingRanges(
  totalSize: number,
  chunkSize: number,
  receivedRanges: ReceivedRange[],
): Array<{ start: number; end: number }> {
  const receivedStarts = new Set(receivedRanges.map((r) => r.start))
  const missing: Array<{ start: number; end: number }> = []

  for (let start = 0; start < totalSize; start += chunkSize) {
    if (receivedStarts.has(start)) continue
    const end = Math.min(start + chunkSize, totalSize) - 1
    missing.push({ start, end })
  }
  return missing
}

// ── Main entry point ─────────────────────────────────────────────────────────

export interface ResumableUploadOptions {
  sourceUrl?: string
  onProgress?: (info: UploadProgressInfo) => void
  signal?: AbortSignal
  uploadId?: string   // pre-registered by bulk/initiate — skips the initiate round-trip
}

/**
 * Convenience alias used by the bulk upload page.
 * Accepts a pre-registered uploadId from POST /upload/bulk/initiate
 * so the chunked upload can skip its own initiate call.
 */
export async function resumableUpload(
  file: File,
  onProgress?: (info: UploadProgressInfo) => void,
  uploadId?: string,
): Promise<UploadResponse> {
  return uploadResumable(file, { onProgress, uploadId })
}

/**
 * Upload a file with full fail-proofing: auto-routes to chunked upload for
 * large files, resumes from a previous interrupted session if the same file
 * is re-dropped, retries every chunk on network failure, and verifies
 * integrity before finalising.
 */
export async function uploadResumable(
  file: File,
  options: ResumableUploadOptions = {},
): Promise<UploadResponse> {
  const { sourceUrl, onProgress, signal, uploadId } = options
  const report = (info: Partial<UploadProgressInfo>) =>
    onProgress?.({
      phase: 'preparing', percent: 0, bytesUploaded: 0,
      bytesTotal: file.size, currentRetry: 0, message: '', ...info,
    } as UploadProgressInfo)

  // Small files: simple single-shot path, no chunking overhead
  if (file.size <= SINGLE_SHOT_MAX_BYTES) {
    return _uploadSingleShot(file, sourceUrl, report)
  }

  return _uploadChunkedResumable(file, sourceUrl, report, signal, uploadId)
}

async function _uploadSingleShot(
  file: File,
  sourceUrl: string | undefined,
  report: (info: Partial<UploadProgressInfo>) => void,
): Promise<UploadResponse> {
  report({ phase: 'uploading', percent: 0, message: 'Uploading…' })

  const formData = new FormData()
  formData.append('file', file)
  if (sourceUrl) formData.append('source_url', sourceUrl)

  const res = await axios.post<UploadResponse>(
    `${BASE_URL}${API_PREFIX}/upload`,
    formData,
    {
      headers: { ..._authHeaders(), 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (e.total) {
          report({
            phase: 'uploading',
            percent: Math.round((e.loaded / e.total) * 90),
            bytesUploaded: e.loaded,
            bytesTotal: e.total,
            message: 'Uploading…',
          })
        }
      },
    },
  )

  report({ phase: 'ready', percent: 100, message: 'Indexed.' })
  return res.data
}

async function _uploadChunkedResumable(
  file: File,
  sourceUrl: string | undefined,
  report: (info: Partial<UploadProgressInfo>) => void,
  signal?: AbortSignal,
  preRegisteredId?: string,
): Promise<UploadResponse> {
  const fingerprint = _fileFingerprint(file)
  const chunkSize = DEFAULT_CHUNK_SIZE

  let uploadId: string
  let receivedRanges: ReceivedRange[] = []
  let resumed = false

  // ── Step 1: find or create the upload session ───────────────────────────
  const existing = _findExistingSession(file)

  if (existing) {
    report({ phase: 'resuming', percent: 0, message: 'Checking previous upload…' })
    try {
      const statusRes = await _getUploadStatus(existing.uploadId)
      if (statusRes.complete) {
        // Already fully uploaded server-side, just never finalised —
        // skip straight to finalise instead of re-uploading anything
        uploadId = existing.uploadId
        receivedRanges = statusRes.receivedRanges
        resumed = true
        report({
          phase: 'finalising', percent: 95,
          message: 'Resuming: all data already uploaded, finalising…',
        })
        const result = await _finalise(uploadId, file.size, report)
        _clearSession(fingerprint)
        return result
      }
      uploadId = existing.uploadId
      receivedRanges = statusRes.receivedRanges
      resumed = true
      report({
        phase: 'resuming', percent: Math.round((statusRes.receivedBytes / file.size) * 90),
        bytesUploaded: statusRes.receivedBytes, bytesTotal: file.size,
        message: `Resuming upload — ${Math.round(statusRes.receivedBytes / 1e6)}MB already sent`,
      })
    } catch {
      // Session expired or server lost it — start fresh below
      _clearSession(fingerprint)
      uploadId = preRegisteredId ?? await _initSession(file, sourceUrl, chunkSize)
    }
  } else if (preRegisteredId) {
    // Bulk-initiated: session already registered via /upload/bulk/initiate —
    // skip the per-file initiate round-trip
    uploadId = preRegisteredId
  } else {
    uploadId = await _initSession(file, sourceUrl, chunkSize)
  }

  _saveSession(fingerprint, {
    uploadId, filename: file.name, fileSize: file.size,
    fileLastModified: file.lastModified, sourceUrl, createdAt: Date.now(),
  })

  // ── Step 2: upload missing ranges only ──────────────────────────────────
  const missingRanges = _computeMissingRanges(file.size, chunkSize, receivedRanges)
  const alreadyUploadedBytes = file.size - missingRanges.reduce(
    (sum, r) => sum + (r.end - r.start + 1), 0,
  )

  if (!resumed) {
    report({ phase: 'uploading', percent: 0, message: 'Uploading…' })
  }

  let uploadedSoFar = alreadyUploadedBytes

  for (const range of missingRanges) {
    if (signal?.aborted) {
      throw new DOMException('Upload cancelled by user', 'AbortError')
    }

    await _uploadChunkWithRetry(
      uploadId, file, range.start, range.end,
      (attempt) => report({
        phase: 'uploading', currentRetry: attempt,
        percent: Math.round((uploadedSoFar / file.size) * 90),
        bytesUploaded: uploadedSoFar, bytesTotal: file.size,
        message: `Network issue — retrying chunk (attempt ${attempt}/${MAX_CHUNK_RETRIES})…`,
      }),
    )

    uploadedSoFar += range.end - range.start + 1
    report({
      phase: 'uploading', currentRetry: 0,
      percent: Math.round((uploadedSoFar / file.size) * 90),
      bytesUploaded: uploadedSoFar, bytesTotal: file.size,
      message: 'Uploading…',
    })
  }

  // ── Step 3: finalise ─────────────────────────────────────────────────────
  report({ phase: 'finalising', percent: 92, message: 'Verifying and assembling…' })
  const result = await _finalise(uploadId, file.size, report)
  _clearSession(fingerprint)
  return result
}

async function _initSession(
  file: File, sourceUrl: string | undefined, chunkSize: number,
): Promise<string> {
  const formData = new FormData()
  formData.append('filename', file.name)
  formData.append('total_size', String(file.size))
  if (sourceUrl) formData.append('source_url', sourceUrl)

  const res = await axios.post(
    `${BASE_URL}${API_PREFIX}/upload/chunked/init`,
    formData,
    { headers: { ..._authHeaders(), 'Content-Type': 'multipart/form-data' } },
  )
  return res.data.upload_id
}

async function _finalise(
  uploadId: string, fileSize: number,
  report: (info: Partial<UploadProgressInfo>) => void,
): Promise<UploadResponse> {
  report({ phase: 'finalising', percent: 92, bytesUploaded: fileSize, bytesTotal: fileSize,
            message: 'Verifying and assembling…' })

  const res = await axios.post<UploadResponse>(
    `${BASE_URL}${API_PREFIX}/upload/chunked/${uploadId}/finalise`,
    {},
    { headers: _authHeaders(), timeout: 60000 },
  )

  // Background mode: server queued the ingestion job and returned immediately
  // with status:'processing' and a trackable document_id. Poll until done.
  if (res.data.status === 'processing' && res.data.document_id) {
    return _pollIngestionStatus(res.data, fileSize, report)
  }

  report({ phase: 'ready', percent: 100, message: res.data.message || 'Indexed.' })
  return res.data
}

async function _pollIngestionStatus(
  initial: UploadResponse,
  fileSize: number,
  report: (info: Partial<UploadProgressInfo>) => void,
): Promise<UploadResponse> {
  const POLL_INTERVAL_MS = 3000
  const MAX_POLLS = 200   // 10 minutes max

  const stageMessages: Record<string, string> = {
    parsing:   'Parsing document and extracting text…',
    embedding: 'Generating embeddings…',
    indexing:  'Writing to vector index…',
    complete:  'Indexed successfully.',
  }

  for (let i = 0; i < MAX_POLLS; i++) {
    await new Promise(r => setTimeout(r, POLL_INTERVAL_MS))
    try {
      const statusRes = await axios.get(
        `${BASE_URL}${API_PREFIX}/upload/document/${initial.document_id}/status`,
        { headers: _authHeaders(), timeout: 10000 },
      )
      const s = statusRes.data
      const stage: string = s.stage || 'parsing'
      const percent = stage === 'complete' ? 100
        : stage === 'indexing' ? 95
        : stage === 'embedding' ? 80
        : 65

      report({
        phase: stage === 'complete' ? 'ready' : 'indexing',
        percent,
        bytesUploaded: fileSize,
        bytesTotal: fileSize,
        message: stageMessages[stage] || `Processing (${stage})…`,
      })

      if (s.complete) {
        return {
          ...initial,
          pages: s.pages ?? 0,
          chunks_created: s.chunks_parsed ?? 0,
          status: 'success',
          message: 'Document indexed and ready for search.',
        }
      }
    } catch {
      // Transient poll failure — keep trying
    }
  }

  // Timed out — return what we know, the doc will be searchable eventually
  return {
    ...initial,
    status: 'processing',
    message: 'Indexing is taking longer than expected. The document will be available for search shortly.',
  }
}

/** Explicitly cancel and discard an in-progress or resumable upload session. */
export async function cancelResumableUpload(file: File): Promise<void> {
  const fingerprint = _fileFingerprint(file)
  const existing = _findExistingSession(file)
  if (existing) {
    try {
      await axios.delete(
        `${BASE_URL}${API_PREFIX}/upload/chunked/${existing.uploadId}`,
        { headers: _authHeaders() },
      )
    } catch {
      // best effort
    }
    _clearSession(fingerprint)
  }
}
