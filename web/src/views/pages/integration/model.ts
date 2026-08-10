import { computed, ref } from 'vue'
import type { DocumentDetail, DocumentRow, Progress } from '@/client/rdocs-client'

/**
 * State for the integration demo.
 *
 * Module scope, like every other page here: the discovered target and the last
 * result survive navigating away and back, which matters because discovery costs
 * four HTTP probes and a recognition costs a second of somebody's GPU.
 */

/**
 * Where each implementation listens.
 *
 * **The page does not ask which one to use — it FINDS one.** These four ports are the project's own
 * convention (each port's `docker/docker-compose.yaml`), and there are exactly FOUR rows: one per
 * implementation, never a fifth "this service" row.
 *
 * That fifth row was the first cut and it was actively confusing: the page served from :8005 listed
 * both "This service" and "Kotlin/JVM", which are the same process, and a reader could not tell
 * WHICH of the four they were talking to. Now the row whose port matches the page's own is simply
 * MARKED as such — and it is probed through a relative URL, so no CORS is involved for it at all.
 */
export interface Candidate {
    port: number
    label: string
}

export const CANDIDATES: Candidate[] = [
    { port: 8002, label: 'Python' },
    { port: 8003, label: 'Go' },
    { port: 8004, label: '.NET' },
    { port: 8005, label: 'Kotlin/JVM' },
]

/**
 * Whether a port is the one serving this page.
 *
 * The port alone, not the host: the page is reachable as both `localhost` and `127.0.0.1`, which are
 * different ORIGINS to a browser but the same service, and treating them as different is what put the
 * same process in two rows.
 */
export function isSelf(port: number): boolean {
    const own = window.location.port || (window.location.protocol === 'https:' ? '443' : '80')
    return own === String(port)
}

export type ProbeState = 'unknown' | 'probing' | 'ready' | 'loading' | 'down' | 'blocked'

export interface Probe extends Candidate {
    /** Empty for the page's own service, so the request is relative and CORS never applies. */
    baseUrl: string
    self: boolean
    state: ProbeState
    /** `initializing` | `ready` | `error`, when the implementation reports it. */
    runtime: string | null
    /** Shown for the CHOSEN row only: a duration for something that did not answer means nothing. */
    ms: number | null
    note: string | null
}

export const probes = ref<Probe[]>(CANDIDATES.map((c) => ({
    ...c,
    baseUrl: isSelf(c.port) ? '' : `http://localhost:${c.port}`,
    self: isSelf(c.port),
    state: 'unknown',
    runtime: null,
    ms: null,
    note: null,
})))

/** True when this page is not served by any of the four — a dev server, most likely. */
export const servedElsewhere = computed(() => !probes.value.some((p) => p.self))

/** The chosen target. Derived by discovery, never by a control on the page. */
export const target = ref<Probe | null>(null)
export const discovering = ref(false)

/**
 * The credential.
 *
 * Same-origin reuses the session token the operator already has, so the demo runs
 * with nothing typed. A cross-origin target needs an API key, because a session
 * token belongs to the origin that issued it.
 */
export const apiKey = ref('')

export const file = ref<File | null>(null)
export const filePreview = ref<string | null>(null)
export const dragging = ref(false)

/** One REST call, as the page reports it. This IS the demonstration. */
export interface Step {
    method: string
    url: string
    status: number | null
    ms: number | null
    note: string | null
    /** The client call that produced it, shown verbatim beside the result. */
    code: string
    state: 'pending' | 'ok' | 'fail'
}

export const steps = ref<Step[]>([])
export const running = ref(false)
export const failure = ref<string | null>(null)

export const row = ref<DocumentRow | null>(null)
export const progress = ref<Progress | null>(null)
export const result = ref<DocumentDetail | null>(null)

/** The canvas as an object URL — fetched with the credential, never via `<img src>`. */
export const canvasUrl = ref<string | null>(null)
export const canvasFallback = ref(false)

/** Overlay interaction, the same two-way highlight the document page has. */
export const hovered = ref<string | null>(null)
export const pinned = ref<string | null>(null)
export const showLabels = ref(true)
export const showRaw = ref(false)

export const activeKey = computed(() => pinned.value ?? hovered.value)

/** Which boxes belong to the field under the cursor, and the reverse. */
export const activeBoxIds = computed<Set<string>>(() => {
    const key = activeKey.value
    if (!key || !result.value) return new Set()
    if (key.startsWith('box:')) return new Set([key.slice(4)])
    const field = result.value.fields.find((f) => f.name === key.slice(6))
    return new Set(field?.box_ids ?? [])
})

export const activeFieldName = computed<string | null>(() => {
    const key = activeKey.value
    if (!key || !result.value) return null
    if (key.startsWith('field:')) return key.slice(6)
    const boxId = key.slice(4)
    return result.value.fields.find((f) => f.box_ids.includes(boxId))?.name ?? null
})

export function resetRun(): void {
    steps.value = []
    failure.value = null
    row.value = null
    progress.value = null
    result.value = null
    hovered.value = null
    pinned.value = null
    if (canvasUrl.value) {
        // Revoked, not just dropped: the client hands out an object URL and the caller
        // owns it — a page that forgets leaks a full bitmap per document viewed.
        URL.revokeObjectURL(canvasUrl.value)
        canvasUrl.value = null
    }
    canvasFallback.value = false
}
