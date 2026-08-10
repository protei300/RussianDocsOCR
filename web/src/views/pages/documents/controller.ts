import { onMounted, onUnmounted } from 'vue'
import { useStore } from 'vuex'
import $api from '@/api'
import { documents, form, loading, stats, total, uploadOpen } from './model'
import type { DocStatus, DocumentRow } from '@/types'

const ACTIVE: DocStatus[] = ['queued', 'processing']
let requestId = 0
let controller: AbortController | null = null
let pollTimer: ReturnType<typeof setInterval> | null = null

export function getData(): void {
    const id = ++requestId
    controller?.abort()
    controller = new AbortController()
    loading.value = true
    $api.documents.list({ ...form }, controller.signal)
        .then((res) => {
            // Guard against an out-of-order response overwriting a newer one.
            if (id !== requestId) return
            documents.value = res.items
            total.value = res.total
            stats.value = res.stats ?? {}
            if (res.items.some((d) => ACTIVE.includes(d.status))) startPoll()
            else stopPoll()
        })
        .catch(() => { /* cancels are silent; real errors already toasted */ })
        .finally(() => { if (id === requestId) loading.value = false })
}

export function getDataSearch(): void {
    form.page = 1
    getData()
}

function startPoll(): void {
    if (pollTimer) return
    pollTimer = setInterval(() => {
        if (documents.value.some((d) => ACTIVE.includes(d.status))) getData()
        else stopPoll()
    }, 3000)
}

function stopPoll(): void {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

export function useHooks() {
    const store = useStore()

    function setSort(column: string): void {
        // Three-state cycle: asc -> desc -> back to the default ordering.
        if (form.sort_by !== column) { form.sort_by = column; form.sort_dir = 'asc' }
        else if (form.sort_dir === 'asc') { form.sort_dir = 'desc' }
        else { form.sort_by = 'created_at'; form.sort_dir = 'desc' }
        form.page = 1
        getData()
    }

    function sortIcon(column: string): string {
        if (form.sort_by !== column) return '⇅'
        return form.sort_dir === 'asc' ? '↑' : '↓'
    }

    function goPage(page: number): void { form.page = page; getData() }

    function reprocess(row: DocumentRow): void {
        $api.documents.reprocess(row.id).then(() => {
            // Optimistic: an immediate refetch can race the backend commit.
            row.status = 'queued'
            startPoll()
        })
    }

    function remove(row: DocumentRow): void {
        $api.documents.remove(row.id).then(() => {
            documents.value = documents.value.filter((d) => d.id !== row.id)
            total.value = Math.max(0, total.value - 1)
            store.dispatch('ui/toast', { kind: 'success', title: 'Deleted', message: row.filename })
        })
    }

    function onUploaded(): void {
        uploadOpen.value = false
        getData()
        startPoll()
    }

    onMounted(getData)
    onUnmounted(stopPoll)   // model.ts is module-scope; the timer must not leak

    return { setSort, sortIcon, goPage, reprocess, remove, onUploaded }
}
