import { reactive, ref } from 'vue'
import type { DocumentFilter, DocumentRow, ProcessingMode, RecognitionRow } from '@/types'

// Module-scope on purpose: filters survive navigating to a document and back.
export const loading = ref(false)
export const documents = ref<DocumentRow[]>([])
export const total = ref(0)
export const stats = ref<Record<string, number | null>>({})
export const uploadOpen = ref(false)

// In-memory recognition. `processingMode` stays null until /auth/config answers, so
// the page shows neither layout instead of flashing the stored one first. An older
// backend that does not report the mode is a stored one.
export const processingMode = ref<ProcessingMode | null>(null)
export const recognizeOpen = ref(false)
export const recognitions = ref<RecognitionRow[]>([])
export const recognitionsTotal = ref(0)
export const recognitionsLoading = ref(false)

export const form = reactive<DocumentFilter>({
    page: 1,
    page_size: 20,
    search: '',
    status: '',
    doc_type: '',
    date_from: '',
    date_to: '',
    sort_by: 'created_at',
    sort_dir: 'desc',
})

export const DOC_TYPES = [
    { value: '', label: 'All types' },
    { value: 'INTPASSPORT', label: 'Internal passport' },
    { value: 'INTPASSPORTADDR', label: 'Registration page' },
    { value: 'EXTPASSPORT', label: 'International passport' },
    { value: 'DL', label: "Driver's licence" },
    { value: 'SNILS', label: 'SNILS' },
    { value: '__none__', label: 'Not recognised' },
]

export function resetFilters(): void {
    form.search = ''
    form.status = ''
    form.doc_type = ''
    form.date_from = ''
    form.date_to = ''
    form.page = 1
}
