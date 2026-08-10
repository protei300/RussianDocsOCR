import { onMounted } from 'vue'
import { AUTH_KEY } from '@/common/fetch'
import { createClient, RdocsError, type RdocsClient } from '@/client/rdocs-client'
import {
    apiKey, canvasFallback, canvasUrl, discovering, dragging, file, filePreview, failure,
    progress, probes, resetRun, result, row, running, steps, target, type Probe, type Step,
} from './model'

/**
 * The demo's logic.
 *
 * Everything here goes through `@/client/rdocs-client`, the same file an integrator
 * copies — nothing on this page reaches for the app's own axios instance. That is
 * the point: if the client is missing something, the demo cannot fake it.
 */
export const useHooks = () => {

    /** The session token the operator already has, for a same-origin target. */
    function sessionToken(): string | undefined {
        const raw = localStorage.getItem(AUTH_KEY)
        if (!raw) return undefined
        try {
            return JSON.parse(raw).token || undefined
        } catch {
            return undefined
        }
    }

    function clientFor(probe: Probe): RdocsClient {
        return createClient({
            baseUrl: probe.baseUrl,
            // Same origin: reuse the browser session, so the demo runs with nothing typed.
            // Cross origin: an API key, because a session token belongs to the origin that
            // issued it.
            token: probe.baseUrl === '' ? sessionToken() : undefined,
            apiKey: apiKey.value.trim() || undefined,
        })
    }

    /**
     * Probes every candidate and picks one. **No control on the page selects it.**
     *
     * Preference order is the order in [CANDIDATES]: the service that served this page
     * first, then the four ports. A `ready` target beats one still loading models,
     * because a demo that queues a document behind a cold start looks broken.
     */
    async function discover(): Promise<void> {
        discovering.value = true
        target.value = null
        for (const probe of probes.value) {
            probe.state = 'probing'
            probe.runtime = null
            probe.ms = null
            probe.note = null
        }

        await Promise.all(probes.value.map(async (probe) => {
            const started = performance.now()
            try {
                const info = await createClient({ baseUrl: probe.baseUrl, timeoutMs: 2500 })
                    .health()
                probe.ms = Math.round(performance.now() - started)
                probe.runtime = info.runtime ?? null
                // A service whose models are still loading is UP but not usable yet: it
                // accepts the upload and queues it. Reported separately so the page can
                // say "loading" rather than "down".
                probe.state = info.runtime && info.runtime !== 'ready' ? 'loading' : 'ready'
            } catch (error) {
                probe.ms = Math.round(performance.now() - started)
                // A cross-origin failure is indistinguishable from "down" in JavaScript —
                // the browser withholds the reason on purpose. So the two are reported as
                // one state with a hint, rather than guessing.
                probe.state = probe.baseUrl === '' ? 'down' : 'blocked'
                probe.note = error instanceof RdocsError && error.status === 0
                    ? 'no response, or this origin is not in its CORS_ALLOWED_ORIGINS'
                    : (error as Error).message
            }
        }))

        // The page's own service first — it needs no credential and no CORS — then any other that is
        // ready, then one still loading its models. A demo that queues a document behind a cold start
        // looks broken, which is why `loading` is the last resort rather than an equal.
        target.value = probes.value.find((p) => p.self && p.state === 'ready')
            ?? probes.value.find((p) => p.state === 'ready')
            ?? probes.value.find((p) => p.self && p.state === 'loading')
            ?? probes.value.find((p) => p.state === 'loading')
            ?? null

        // **A duration is kept only for the chosen row.** "no response · 2371 ms" reads as a
        // measurement of something, and it is nothing but the probe timeout — noise on four rows out
        // of five, and it made the one number that matters harder to find.
        for (const probe of probes.value) {
            if (probe.label !== target.value?.label) probe.ms = null
        }
        discovering.value = false
    }

    function pick(files: FileList | null): void {
        const chosen = files?.[0] ?? null
        if (!chosen) return
        if (filePreview.value) URL.revokeObjectURL(filePreview.value)
        file.value = chosen
        filePreview.value = URL.createObjectURL(chosen)
        resetRun()
    }

    function onDrop(event: DragEvent): void {
        dragging.value = false
        pick(event.dataTransfer?.files ?? null)
    }

    /** Records one call and times it, so the page can show the REST exchange itself. */
    async function step<T>(entry: Omit<Step, 'status' | 'ms' | 'state' | 'note'>,
                           run: () => Promise<T>): Promise<T> {
        const item: Step = { ...entry, status: null, ms: null, note: null, state: 'pending' }
        steps.value = [...steps.value, item]
        const started = performance.now()
        try {
            const value = await run()
            item.ms = Math.round(performance.now() - started)
            item.status = entry.method === 'POST' && entry.url.endsWith('/documents') ? 202 : 200
            item.state = 'ok'
            return value
        } catch (error) {
            item.ms = Math.round(performance.now() - started)
            item.state = 'fail'
            if (error instanceof RdocsError) {
                item.status = error.status || null
                item.note = error.detail
            } else {
                item.note = (error as Error).message
            }
            throw error
        }
    }

    /**
     * The whole integration, in the order an integrator writes it.
     *
     * Deliberately NOT `client.recognise()`, which does all of this in one call: the
     * page exists to show the three steps and what each returns.
     */
    async function run(): Promise<void> {
        if (!file.value || !target.value || running.value) return
        resetRun()
        running.value = true
        const client = clientFor(target.value)

        try {
            const uploaded = await step({
                method: 'POST',
                url: client.url('/documents'),
                code: 'const row = await rdocs.upload(file)',
            }, () => client.upload(file.value as File))
            row.value = uploaded

            const finished = await step({
                method: 'GET',
                url: client.url(`/documents/${uploaded.id}`) + '  (polled)',
                code: 'const doc = await rdocs.waitForResult(row.id, {\n'
                    + '    onProgress: (live) => { bar.value = live?.pct ?? 0 },\n'
                    + '})',
            }, () => client.waitForResult(uploaded.id, {
                onProgress: (live, doc) => {
                    progress.value = live
                    row.value = doc
                },
            }))
            result.value = finished
            progress.value = null

            // The canvas, and only if there is one: an unrecognised document
            // short-circuits before the warp and has no corrected canvas at all.
            if (finished.canvas.width) {
                canvasFallback.value = finished.canvas.is_fallback
                canvasUrl.value = await step({
                    method: 'GET',
                    url: client.url(`/documents/${uploaded.id}/image/canvas`),
                    code: "const src = await rdocs.imageObjectUrl(row.id, 'canvas')\n"
                        + '// an object URL, because <img src> cannot send the credential',
                }, () => client.imageObjectUrl(uploaded.id, 'canvas'))
            }
        } catch (error) {
            failure.value = error instanceof RdocsError
                ? `${error.status || 'network'} — ${error.detail}`
                : (error as Error).message
        } finally {
            running.value = false
        }
    }

    /** Removes the demo document from the service, so the log does not fill up. */
    async function cleanUp(): Promise<void> {
        if (!row.value || !target.value) return
        const client = clientFor(target.value)
        const id = row.value.id
        await step({
            method: 'DELETE',
            url: client.url(`/documents/${id}`),
            code: 'await rdocs.remove(row.id)   // 204, empty body',
        }, () => client.remove(id)).catch(() => {})
        resetRun()
    }

    onMounted(() => {
        if (!target.value) void discover()
    })

    return { discover, pick, onDrop, run, cleanUp }
}
