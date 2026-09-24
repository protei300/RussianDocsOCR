import Api from '@/common/fetch'
import type {
    ApiKeyRow, AuditRow, AuthConfig, AuthUserPayload, DocumentDetail, DocumentFilter,
    DocumentListResponse, LoginResponse, PasswordRule, Progress, SettingDef, UserRow,
} from '@/types'

const $api = {
    auth: {
        /** What the login page needs BEFORE anyone has signed in: which mode is live.
         *  This is what lets one frontend serve both, and lets the Go/.NET/Kotlin
         *  services — which only know about the PIN — keep using the same bundle. */
        config(): Promise<AuthConfig> {
            return Api.get('/auth/config')
        },
        pinLogin(pin: string): Promise<{ access_token: string; user: AuthUserPayload }> {
            return Api.post('/auth/pin-login', { pin })
        },
        login(username: string, password: string): Promise<LoginResponse> {
            return Api.post('/auth/login', { username, password })
        },
        me(): Promise<{ mode: string; user: AuthUserPayload }> {
            return Api.get('/auth/me')
        },
        changePassword(current_password: string, new_password: string):
            Promise<{ status: string; reauthenticate: boolean }> {
            return Api.post('/auth/change-password', { current_password, new_password })
        },
    },
    users: {
        list(): Promise<{ items: UserRow[]; roles: string[]; password_rules: PasswordRule[] }> {
            return Api.get('/users')
        },
        create(body: { username: string; password: string; role: string; display_name?: string }):
            Promise<UserRow> {
            return Api.post('/users', body)
        },
        update(id: number, body: { role?: string; display_name?: string; is_active?: boolean }):
            Promise<UserRow> {
            return Api.patch(`/users/${id}`, body)
        },
        resetPassword(id: number, new_password: string): Promise<UserRow> {
            return Api.post(`/users/${id}/password`, { new_password })
        },
        remove(id: number): Promise<void> {
            return Api.delete(`/users/${id}`)
        },
        audit(params: { limit?: number; action?: string; actor?: string } = {}):
            Promise<{ items: AuditRow[]; count: number }> {
            return Api.get('/users/audit/entries', { params })
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
