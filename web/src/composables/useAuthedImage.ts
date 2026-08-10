import { onUnmounted, ref, watch, type Ref } from 'vue'
import Api from '@/common/fetch'

/**
 * Load an image that sits behind bearer auth.
 *
 * `<img src>` cannot send an Authorization header, and putting the token in a
 * query string would leak it into server logs and browser history. So the
 * bytes are fetched with the normal axios instance (which already attaches the
 * header) and handed to the `<img>` as an object URL.
 *
 * The object URLs are cached and deliberately *not* revoked on unmount: the log
 * page mounts and unmounts thumbnails on every 3-second poll, and revoking
 * would re-download each one every time. The cache is bounded instead.
 */
const cache = new Map<string, string>()
const MAX_CACHE = 60

function remember(url: string, objectUrl: string): void {
    cache.set(url, objectUrl)
    while (cache.size > MAX_CACHE) {
        const oldest = cache.keys().next().value as string | undefined
        if (!oldest) break
        const stale = cache.get(oldest)
        cache.delete(oldest)
        if (stale) URL.revokeObjectURL(stale)
    }
}

export function useAuthedImage(url: Ref<string | null>) {
    const src = ref<string | null>(null)
    const loading = ref(false)
    const failed = ref(false)
    let requestId = 0

    async function load(target: string | null): Promise<void> {
        const id = ++requestId
        failed.value = false
        src.value = null
        if (!target) return

        const cached = cache.get(target)
        if (cached) { src.value = cached; return }

        loading.value = true
        try {
            const blob = (await Api.get(target, { responseType: 'blob' })) as unknown as Blob
            if (id !== requestId) return          // a newer request superseded this one
            const objectUrl = URL.createObjectURL(blob)
            remember(target, objectUrl)
            src.value = objectUrl
        } catch {
            if (id === requestId) failed.value = true
        } finally {
            if (id === requestId) loading.value = false
        }
    }

    watch(url, load, { immediate: true })
    onUnmounted(() => { requestId++ })            // ignore any in-flight response

    return { src, loading, failed }
}
