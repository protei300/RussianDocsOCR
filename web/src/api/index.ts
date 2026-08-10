import Api from '@/common/fetch'
import type {
    ApiKeyRow, DocumentDetail, DocumentFilter, DocumentListResponse, Progress, SettingDef,
} from '@/types'

const $api = {
    auth: {
        pinLogin(pin: string): Promise<{ access_token: string; user: { name: string; role: string } }> {
            return Api.post('/auth/pin-login', { pin })
        },
    },
    documents: {
        list(filter: Partial<DocumentFilter>, signal?: AbortSignal): Promise<DocumentListResponse> {
            return Api.get('/documents', { params: filter, signal })
        },
        get(id: number): Promise<DocumentDetail> {
            return Api.get(`/documents/${id}`)
        },
        progress(id: number): Promise<Progress | null> {
            return Api.get(`/documents/${id}/progress`)
        },
        upload(file: File, onProgress?: (pct: number) => void): Promise<any> {
            const form = new FormData()
            form.append('file', file)
            return Api.post('/documents', form, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (e) => {
                    if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
                },
            })
        },
        reprocess(id: number): Promise<any> { return Api.post(`/documents/${id}/reprocess`) },
        remove(id: number): Promise<void> { return Api.delete(`/documents/${id}`) },
        purge(): Promise<{ deleted: number }> { return Api.post('/documents/purge') },
        /** Images are behind auth, so <img src> cannot fetch them directly. */
        imageUrl(id: number, kind: 'canvas' | 'original' | 'thumb'): string {
            return `/documents/${id}/image/${kind}`
        },
    },
    apiKeys: {
        list(): Promise<{ items: ApiKeyRow[]; note: string }> { return Api.get('/api-keys') },
        create(label: string): Promise<ApiKeyRow & { key: string; warning: string }> {
            return Api.post('/api-keys', { label })
        },
        remove(id: number): Promise<void> { return Api.delete(`/api-keys/${id}`) },
    },
    settings: {
        get(): Promise<{ values: Record<string, string>; schema: SettingDef[] }> {
            return Api.get('/settings')
        },
        update(values: Record<string, unknown>): Promise<{
            values: Record<string, string>; schema: SettingDef[]; restart_required: string[]
        }> {
            return Api.put('/settings', { values })
        },
    },
    status: { get(): Promise<any> { return Api.get('/status') } },
    logs: {
        get(params: { n?: number; level?: string; search?: string }): Promise<any> {
            return Api.get('/logs', { params })
        },
    },
}
export default $api
