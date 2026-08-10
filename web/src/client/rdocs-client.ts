/**
 * RussianDocs OCR — a thin TypeScript client for the recognition service.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * COPY THIS FILE INTO YOUR SITE. It is the whole integration.
 * ────────────────────────────────────────────────────────────────────────────
 *
 * No dependencies: `fetch`, `FormData` and `URL` only, all of them standard in
 * every browser since 2017 and in Node 18+. It therefore drops into a Vue, React,
 * Svelte or plain-HTML page unchanged, and into a Node backend that wants to call
 * the service server-side.
 *
 * It talks to ANY of the four service implementations — Python, Go, .NET or
 * Kotlin — because all four publish the same REST contract
 * (`conformance/spec/viewmodel.md`). They differ only in the port they listen on.
 *
 * ## The whole flow is three calls
 *
 * ```ts
 * const rdocs = createClient({ baseUrl: 'http://localhost:8002', apiKey: 'rdk_…' })
 * const row = await rdocs.upload(file)              // 202 + the full list row
 * const doc = await rdocs.waitForResult(row.id)     // poll until done|failed
 * console.log(doc.doc_type, doc.ocr)                // the recognised fields
 * ```
 *
 * ## Five things that are easy to get wrong
 *
 * 1. **Recognition is ASYNCHRONOUS.** `upload` answers `202 Accepted` as soon as
 *    the bytes are stored; the document is then `queued`. Poll — do not expect
 *    fields in the upload response.
 * 2. **`/progress` answers `200` with a JSON `null`**, not `404`, when there is
 *    nothing to report. A client that treats `null` as an error shows a finished
 *    document as missing.
 * 3. **Images need a fetch, not an `<img src>`.** They are behind the same
 *    credential as everything else, and `<img>` cannot send a header. Putting the
 *    token in the query string would leak it into server logs and browser history,
 *    so it is not offered. Use {@link RdocsClient.imageObjectUrl}.
 * 4. **Box coordinates are in CANVAS space, never in the uploaded image's space.**
 *    The library does not retain the deskew angle, so they cannot be mapped back
 *    onto the original photo. Draw them over `canvas`, and if you show the
 *    original, hide the overlay. The response says so in `coord_space`.
 * 5. **CORS.** A page served from a different origin than the service must be
 *    listed in the service's `CORS_ALLOWED_ORIGINS`, or the browser blocks the
 *    response before your code sees it. Same-origin needs nothing.
 *
 * ## Credentials
 *
 * Two kinds, and the choice is not cosmetic:
 *
 * - `apiKey` → the `X-API-Key` header. **This is what an integration uses.**
 *   Keys are managed on the service's own API-keys page.
 * - `token` → `Authorization: Bearer`, a short-lived JWT from the PIN login. Only
 *   for a browser session on the service's own UI.
 *
 * Working endpoints accept either. Operator endpoints (keys, settings, logs,
 * status) accept ONLY the session token — an integration has no business there.
 */

// ─── the wire types ────────────────────────────────────────────────────────────
// Hand-written from conformance/spec/viewmodel.md rather than generated, so that
// what a caller sees is exactly what the service documents.

export type DocStatus = 'queued' | 'processing' | 'done' | 'failed'

/** One row of the document log — what `upload` returns and what `list` pages. */
export interface DocumentRow {
    id: number
    filename: string
    size_bytes: number
    status: DocStatus
    doc_type: string | null
    doc_type_base: string | null
    doc_type_era: string | null
    recognised: boolean
    doc_conf: number | null
    /**
     * Quality verdicts, denormalised.
     *
     * **The vocabularies differ between checks and a client must not assume one:**
     * `Glare` and `Blur` are `good`/`bad`, while `PrintSpoofing` and `LCDSpoofing`
     * are `REAL`/`FAKE`. That inconsistency is in the library, and the wire
     * carries it rather than papering over it.
     */
    quality: Record<string, string | number | null>
    field_count: number
    device: string | null
    processing_ms: number | null
    error: string | null
    /** A stable token to branch on. `error` is prose and may be in Russian. */
    error_code: string | null
    retry_count: number
    has_canvas: boolean
    created_at: string | null
    started_at: string | null
    finished_at: string | null
    /** Present on the upload response only: how many documents are ahead of this one. */
    queue_position?: number | null
}

export interface Box {
    id: string
    label: string
    display: string
    kind: 'text' | 'visual'
    x1: number
    y1: number
    x2: number
    y2: number
    /** Quantised to three decimals upstream, and nullable. */
    conf: number | null
    /** The detector's numeric class. Present for callers that key off it rather than the label. */
    cls: number | null
    text: string | null
    /**
     * True when another box shares this label and owns the recognised text.
     *
     * Not a defect to filter out: `Licence_number` is detected twice on an internal passport, and
     * the pipeline deduplicates the FIELD, not the boxes. Draw both, attribute the text to one.
     */
    ambiguous: boolean
}

export interface Field {
    name: string
    display: string
    value: string
    script: 'ru' | 'en'
    conf: number | null
    /**
     * Which boxes this field came from.
     *
     * A LIST, because the mapping is genuinely one-to-many: a split field such as
     * a place of birth spans several boxes, and `Licence_number` is detected twice
     * on an internal passport. Matching a field to a box by comparing labels does
     * not work, which is why these ids exist.
     */
    box_ids: string[]
}

export interface OrientedBox {
    cx: number
    cy: number
    w: number
    h: number
    /** RADIANS, not degrees — SVG's `rotate()` wants degrees, so convert at the call site. */
    angle_rad: number
    conf: number | null
    label: string | null
}

export interface AddressLine {
    kind: 'printed' | 'handwritten'
    text: string | null
    p_handwritten: number | null
    obbox: OrientedBox | null
}

export interface AddressBlock {
    aligned: boolean
    lines: AddressLine[]
}

/** The full result. `GET /documents/{id}` on a finished document. */
export interface DocumentDetail extends DocumentRow {
    canvas: {
        url: string
        width: number | null
        height: number | null
        /** True when recognition short-circuited and the canvas is the upload itself. */
        is_fallback: boolean
    }
    original: {
        url: string
        width: number | null
        height: number | null
        content_type: string
    }
    /** Always `"canvas"`. See trap 4 in the file header. */
    coord_space: string | null
    coord_space_note: string | null
    boxes: Box[]
    fields: Field[]
    /** The same values as `fields`, keyed by name, for callers that want a lookup. */
    ocr: Record<string, string>
    timings: Record<string, number>
    /** Only for the internal passport's registration page. */
    address: AddressBlock | null
}

export interface Progress {
    step: string
    label: string
    pct: number
    eta_sec: number | null
    queue_position: number | null
}

export interface HealthInfo {
    status: string
    /** `initializing` | `ready` | `error`, when the implementation reports it. */
    runtime?: string
    version?: string
}

/**
 * A failed call.
 *
 * `status` is the HTTP code and `detail` is the service's own message — the body
 * is `{"detail": "<string>"}` everywhere except a rejected query parameter, where
 * the service reproduces FastAPI's pydantic shape and `detail` is a list. Both are
 * flattened into a readable string here, and `body` keeps the original.
 */
export class RdocsError extends Error {
    readonly status: number
    readonly detail: string
    readonly body: unknown

    constructor(status: number, detail: string, body: unknown) {
        super(`${status}: ${detail}`)
        this.name = 'RdocsError'
        this.status = status
        this.detail = detail
        this.body = body
    }

    /**
     * Whether retrying the same request could plausibly succeed.
     *
     * 503 means the pipeline is busy or still loading models — worth another go.
     * 422 means the bytes did not decode, and retrying identical bytes cannot help.
     */
    get transient(): boolean {
        return this.status === 503 || this.status === 429 || this.status === 0
    }
}

export interface ClientOptions {
    /** Service origin, e.g. `http://localhost:8002`. Empty string means same-origin. */
    baseUrl: string
    /** `X-API-Key`. What an integration uses. */
    apiKey?: string
    /** A session JWT from the PIN login. Only for the service's own UI. */
    token?: string
    /** Per-request timeout, ms. Recognition itself is not bounded by this — polling is. */
    timeoutMs?: number
}

/** Optional hooks, so a UI can show what is happening without wrapping every call. */
export interface WaitOptions {
    onProgress?: (progress: Progress | null, doc: DocumentRow) => void
    /** Poll interval, ms. Default 500: recognition takes ~0.5-1 s on CPU. */
    intervalMs?: number
    /** Give up after this long. Default 120 000, matching the service's job timeout. */
    timeoutMs?: number
    signal?: AbortSignal
}

const API_PREFIX = '/api/v1'

export class RdocsClient {
    private readonly baseUrl: string
    private readonly apiKey?: string
    private readonly token?: string
    private readonly timeoutMs: number

    constructor(options: ClientOptions) {
        // Trailing slashes are stripped so `${baseUrl}${API_PREFIX}` never doubles one:
        // `//api/v1` is a different path on a strict router and answers 404.
        this.baseUrl = options.baseUrl.replace(/\/+$/, '')
        this.apiKey = options.apiKey
        this.token = options.token
        this.timeoutMs = options.timeoutMs ?? 30_000
    }

    /** The absolute URL of an API path — useful for logging what was called. */
    url(path: string): string {
        return `${this.baseUrl}${API_PREFIX}${path}`
    }

    private headers(extra?: Record<string, string>): Record<string, string> {
        const headers: Record<string, string> = { ...extra }
        // The API key first: an integration sends that, and if both are present the
        // service checks the session token first anyway, so the order here is only
        // about which one this client considers primary.
        if (this.apiKey) headers['X-API-Key'] = this.apiKey
        if (this.token) headers.Authorization = `Bearer ${this.token}`
        return headers
    }

    private async request<T>(
        method: string,
        path: string,
        init: { body?: BodyInit; json?: unknown; signal?: AbortSignal; raw?: boolean } = {},
    ): Promise<T> {
        const controller = new AbortController()
        const timer = setTimeout(() => controller.abort(), this.timeoutMs)
        // A caller's own signal must also abort us; forwarding it rather than
        // replacing ours keeps the timeout in force for a caller who passed one.
        init.signal?.addEventListener('abort', () => controller.abort(), { once: true })

        let response: Response
        try {
            response = await fetch(this.url(path), {
                method,
                headers: this.headers(
                    init.json !== undefined ? { 'Content-Type': 'application/json' } : undefined,
                ),
                body: init.json !== undefined ? JSON.stringify(init.json) : init.body,
                signal: controller.signal,
            })
        } catch (error) {
            // A network-level failure. **CORS lands here**, and the browser
            // deliberately does not say so — the console does, the JS does not. Hence
            // the hint: it is the single most common first-run problem.
            const reason = error instanceof Error ? error.message : String(error)
            throw new RdocsError(
                0,
                `${reason} — the service may be down, or this origin is not in its `
                + 'CORS_ALLOWED_ORIGINS',
                error,
            )
        } finally {
            clearTimeout(timer)
        }

        if (response.status === 204) {
            return undefined as T
        }
        if (init.raw) {
            if (!response.ok) throw await this.error(response)
            return (await response.blob()) as unknown as T
        }

        const text = await response.text()
        let body: unknown = null
        if (text.length > 0) {
            try {
                body = JSON.parse(text)
            } catch {
                // Not JSON: an HTML error page from a proxy, most likely. Reported as-is
                // rather than as a parse error, because "'<' is an invalid start of a
                // value" tells a reader nothing about what actually happened.
                if (!response.ok) {
                    throw new RdocsError(response.status, text.slice(0, 200), text)
                }
                throw new RdocsError(response.status, 'response was not JSON', text)
            }
        }
        if (!response.ok) {
            throw new RdocsError(response.status, detailOf(body), body)
        }
        return body as T
    }

    private async error(response: Response): Promise<RdocsError> {
        let body: unknown = null
        try {
            body = JSON.parse(await response.text())
        } catch {
            /* left null: an unparsable error body must not mask the status code */
        }
        return new RdocsError(response.status, detailOf(body), body)
    }

    // ─── the calls an integration needs ───────────────────────────────────────

    /**
     * Liveness. Needs NO credential, which makes it the right probe for "is this
     * service up and are its models loaded".
     */
    async health(): Promise<HealthInfo> {
        // Note the path: /health sits OUTSIDE /api/v1, so the container's healthcheck
        // does not depend on the API version.
        const response = await fetch(`${this.baseUrl}/health`)
        if (!response.ok) throw new RdocsError(response.status, 'health check failed', null)
        return (await response.json()) as HealthInfo
    }

    /**
     * Queues one image. Returns the full list row — **202, not 200**.
     *
     * The part name must be `file`. Validation is immediate: a PDF is 415, an
     * undecodable image is 422, an oversized one 413 — so a bad upload fails here
     * with something actionable instead of becoming a mysterious failed job.
     */
    async upload(file: Blob, filename?: string): Promise<DocumentRow> {
        const form = new FormData()
        form.append('file', file, filename ?? (file instanceof File ? file.name : 'upload.jpg'))
        // Content-Type is NOT set: the browser must add it with the multipart boundary,
        // and setting it by hand produces a body the server cannot parse.
        return this.request<DocumentRow>('POST', '/documents', { body: form })
    }

    /** The full result, including boxes and fields once the status is `done`. */
    async get(id: number, options?: { includeDebug?: boolean }): Promise<DocumentDetail> {
        const query = options?.includeDebug ? '?include=debug' : ''
        return this.request<DocumentDetail>('GET', `/documents/${id}${query}`)
    }

    /** Live progress, or `null` when there is nothing to report. See trap 2. */
    async progress(id: number): Promise<Progress | null> {
        return this.request<Progress | null>('GET', `/documents/${id}/progress`)
    }

    async list(params: Record<string, string | number> = {}): Promise<{
        items: DocumentRow[]
        total: number
        page: number
        page_size: number
        stats: Record<string, number | null>
    }> {
        const query = new URLSearchParams()
        for (const [key, value] of Object.entries(params)) query.set(key, String(value))
        const suffix = query.toString() ? `?${query}` : ''
        return this.request('GET', `/documents${suffix}`)
    }

    async reprocess(id: number): Promise<DocumentRow> {
        return this.request<DocumentRow>('POST', `/documents/${id}/reprocess`)
    }

    /** 204 with an empty body. */
    async remove(id: number): Promise<void> {
        await this.request<void>('DELETE', `/documents/${id}`)
    }

    /**
     * Fetches an image and returns an object URL suitable for `<img src>`.
     *
     * **The caller owns the URL and must `URL.revokeObjectURL` it**, or every
     * document viewed leaks a full-size bitmap for the lifetime of the page.
     */
    async imageObjectUrl(id: number, kind: 'canvas' | 'original' | 'thumb' = 'canvas',
                         width?: number): Promise<string> {
        const query = kind === 'thumb' && width ? `?w=${width}` : ''
        const blob = await this.request<Blob>('GET', `/documents/${id}/image/${kind}${query}`,
            { raw: true })
        return URL.createObjectURL(blob)
    }

    /**
     * Polls until the document reaches a terminal state, then returns the full result.
     *
     * A fixed interval rather than exponential backoff, deliberately: recognition
     * takes about half a second, so backing off would add more latency than it saves
     * requests. Rejects with an `RdocsError` when the document fails, so a caller can
     * use one `try`/`catch` for both transport and recognition failures.
     */
    async waitForResult(id: number, options: WaitOptions = {}): Promise<DocumentDetail> {
        const interval = options.intervalMs ?? 500
        const deadline = Date.now() + (options.timeoutMs ?? 120_000)

        for (;;) {
            const doc = await this.get(id)
            if (options.onProgress) {
                // Progress is fetched only while the document is actually in flight:
                // asking for a finished one answers 200 with null, which is correct but
                // a wasted round trip on every poll.
                const live = doc.status === 'queued' || doc.status === 'processing'
                    ? await this.progress(id).catch(() => null)
                    : null
                options.onProgress(live, doc)
            }
            if (doc.status === 'done') return doc
            if (doc.status === 'failed') {
                throw new RdocsError(422, doc.error ?? doc.error_code ?? 'recognition failed', doc)
            }
            if (Date.now() > deadline) {
                throw new RdocsError(0, `document ${id} did not finish in time`, doc)
            }
            if (options.signal?.aborted) {
                throw new RdocsError(0, 'cancelled', null)
            }
            await sleep(interval)
        }
    }

    /**
     * Upload and wait, as one call. What most integrations actually want.
     */
    async recognise(file: Blob, options: WaitOptions & { filename?: string } = {}):
        Promise<{ row: DocumentRow; result: DocumentDetail }> {
        const row = await this.upload(file, options.filename)
        const result = await this.waitForResult(row.id, options)
        return { row, result }
    }

    /**
     * Exchanges the site PIN for a session token.
     *
     * Present for completeness — the service's own UI uses it. An integration should
     * use an API key instead: a four-digit PIN is a human affordance and a poor
     * service credential.
     */
    async pinLogin(pin: string): Promise<{ access_token: string; token_type: string }> {
        return this.request('POST', '/auth/pin-login', { json: { pin } })
    }
}

export function createClient(options: ClientOptions): RdocsClient {
    return new RdocsClient(options)
}

/** Flattens either error shape into one readable line. See {@link RdocsError}. */
function detailOf(body: unknown): string {
    if (body && typeof body === 'object' && 'detail' in body) {
        const detail = (body as { detail: unknown }).detail
        if (typeof detail === 'string') return detail
        if (Array.isArray(detail)) {
            // The pydantic shape: [{loc: ['query','page_size'], msg: '…'}, …]
            return detail
                .map((item) => {
                    if (item && typeof item === 'object') {
                        const entry = item as { loc?: unknown[]; msg?: string }
                        const where = Array.isArray(entry.loc) ? entry.loc.join('.') : ''
                        return where ? `${where}: ${entry.msg ?? ''}` : String(entry.msg ?? '')
                    }
                    return String(item)
                })
                .join('; ')
        }
    }
    return 'request failed'
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms))
}
