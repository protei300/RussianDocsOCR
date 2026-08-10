import { computed, ref } from 'vue'
import type { DocumentDetail } from '@/types'

export const detail = ref<DocumentDetail | null>(null)
export const loading = ref(false)
/** Distinct from `loading`: the document is genuinely gone, not slow. */
export const notFound = ref(false)

export const hover = ref<string | null>(null)   // a box id OR a field name
export const pinned = ref<string | null>(null)
export const showLabels = ref(localStorage.getItem('rd_labels') !== '0')
export const hiddenLabels = ref<Set<string>>(new Set())
export const imageMode = ref<'canvas' | 'original'>('canvas')

/** box id -> the field name it belongs to */
export const boxToField = computed(() => {
    const map = new Map<string, string>()
    for (const b of detail.value?.boxes ?? []) map.set(b.id, b.label)
    for (const l of detail.value?.address?.lines ?? []) map.set(l.id, l.id)
    return map
})

/** field name -> every box that carries it (split fields have several) */
export const fieldToBoxes = computed(() => {
    const map = new Map<string, string[]>()
    for (const b of detail.value?.boxes ?? []) {
        const list = map.get(b.label) ?? []
        list.push(b.id)
        map.set(b.label, list)
    }
    return map
})

/**
 * One source of truth for "what is lit right now".
 *
 * Both directions derive from the same computed, so hovering a field or one of
 * its boxes produces the identical highlight without a two-way watcher and
 * without the feedback loop one would create.
 */
export const activeField = computed<string | null>(() => {
    const key = pinned.value ?? hover.value
    if (!key) return null
    return boxToField.value.get(key) ?? key
})

export const activeKeys = computed<Set<string>>(() => {
    const field = activeField.value
    if (!field) return new Set()
    const boxes = fieldToBoxes.value.get(field)
    return new Set(boxes ?? [field])
})

export const anyActive = computed(() => activeKeys.value.size > 0)

export function toggleLabels(): void {
    showLabels.value = !showLabels.value
    localStorage.setItem('rd_labels', showLabels.value ? '1' : '0')
}

export function toggleLabelVisibility(label: string): void {
    const next = new Set(hiddenLabels.value)
    if (next.has(label)) next.delete(label)
    else next.add(label)
    hiddenLabels.value = next
}

export function reset(): void {
    detail.value = null
    notFound.value = false
    hover.value = null
    pinned.value = null
    hiddenLabels.value = new Set()
    imageMode.value = 'canvas'
}
