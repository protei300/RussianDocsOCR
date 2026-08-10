import axios from 'axios'
import store from '@/store'

export const AUTH_KEY = 'rd_auth'

const service = axios.create({
    baseURL: '/api/v1',
    headers: { 'Content-Type': 'application/json' },
})

service.interceptors.request.use((config) => {
    const raw = localStorage.getItem(AUTH_KEY)
    if (raw) {
        try {
            const token = JSON.parse(raw).token
            if (token) config.headers.Authorization = `Bearer ${token}`
        } catch {
            localStorage.removeItem(AUTH_KEY)
        }
    }
    return config
})

service.interceptors.response.use(
    // Unwrapped here so every api module returns T, not AxiosResponse<T>.
    (response) => response.data,
    (error) => {
        if (axios.isCancel(error) || error.code === 'ERR_CANCELED') {
            return Promise.reject(error)
        }
        if (error.response?.status === 401) {
            store.dispatch('auth/logout')
            return Promise.reject(error)
        }
        // Global toast means page controllers rarely need their own .catch().
        const message = error.response?.data?.detail
            || error.response?.data?.message
            || error.message
            || 'Request failed'
        store.dispatch('ui/toast', { kind: 'error', title: 'Error', message })
        return Promise.reject(new Error(message))
    },
)

export default service
