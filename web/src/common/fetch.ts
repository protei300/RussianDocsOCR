import axios from 'axios'
import router from '@/router'
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
            // The server's own words, not axios's. The first version rejected with
            // the raw error, so a wrong password read "Request failed with status
            // code 401" on the login page instead of "Wrong username or password".
            const message = error.response?.data?.detail || 'Sign in required'
            store.dispatch('auth/logout')
            // Route, do not just clear the token. Dropping the session without
            // moving leaves the user on a page that renders and then reports
            // "Request failed with status code 401" at every action — which is
            // what happens when a token dies mid-session, and a password change
            // kills one deliberately.
            if (router.currentRoute.value.name !== 'Login') {
                void router.push({ name: 'Login' })
            }
            return Promise.reject(new Error(message))
        }
        // The server refuses a restricted session with a machine-readable reason
        // rather than prose, so this test survives the message being reworded.
        // Without it the user sees a generic error toast on a page they are not
        // allowed to use yet, instead of being taken to the form that unblocks them.
        if (error.response?.status === 403
            && error.response?.data?.detail === 'password_change_required') {
            if (router.currentRoute.value.name !== 'ChangePassword') {
                void router.push({ name: 'ChangePassword' })
            }
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
