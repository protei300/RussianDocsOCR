import { onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useStore } from 'vuex'
import $api from '@/api'
import { boxToField, detail, loading, notFound, pinned, reset } from './model'

let pollTimer: ReturnType<typeof setInterval> | null = null

export function useHooks() {
    const route = useRoute()
    const router = useRouter()
    const store = useStore()
    const id = Number(route.params.id)

    function load(): void {
        loading.value = true
        $api.documents.get(id)
            .then((d) => {
                detail.value = d
                // Keep polling while the document is still being worked on.
                if (d.status === 'queued' || d.status === 'processing') startPoll()
                else stopPoll()
            })
            .catch((err) => {
                stopPoll()
                // A deleted document (or an id from a wiped ephemeral store) is
                // a normal thing to navigate to — say so rather than rendering
                // an empty shell with placeholder dashes.
                if (err?.message?.includes('not found') || err?.response?.status === 404) {
                    notFound.value = true
                }
            })
            .finally(() => { loading.value = false })
    }

    function startPoll(): void {
        if (pollTimer) return
        pollTimer = setInterval(load, 2000)
    }

    function stopPoll(): void {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
    }

    function pickBox(key: string): void {
        pinned.value = pinned.value === key ? null : key
        const field = boxToField.value.get(key)
        if (!field) return
        const el = document.getElementById('fld-' + field)
        el?.scrollIntoView({ block: 'center', behavior: 'smooth' })
        el?.classList.remove('fld-flash')
        void el?.offsetWidth          // force reflow so the animation restarts
        el?.classList.add('fld-flash')
    }

    function pinField(name: string): void {
        pinned.value = pinned.value === name ? null : name
    }

    function reprocess(): void {
        $api.documents.reprocess(id).then(() => { load(); startPoll() })
    }

    function remove(): void {
        $api.documents.remove(id).then(() => {
            store.dispatch('ui/toast', { kind: 'success', title: 'Deleted', message: '' })
            router.push('/documents')
        })
    }

    function copy(value: string | null): void {
        if (!value) return
        void navigator.clipboard.writeText(value)
        store.dispatch('ui/toast', { kind: 'success', title: 'Copied', message: value.slice(0, 40) })
    }

    function onEsc(e: KeyboardEvent): void {
        if (e.key === 'Escape') pinned.value = null
    }

    onMounted(() => {
        reset()
        load()
        document.addEventListener('keydown', onEsc)
    })
    onUnmounted(() => {
        stopPoll()
        document.removeEventListener('keydown', onEsc)
    })

    return { id, load, pickBox, pinField, reprocess, remove, copy }
}
