import type { Module } from 'vuex'
import { AUTH_KEY } from '@/common/fetch'

export interface AuthUser { name: string; role: string }
interface AuthState { token: string | null; user: AuthUser | null }

/** Read the JWT payload to recover the user and expiry after a page reload. */
function parseToken(token: string): { user: AuthUser | null; exp: number | null } {
    try {
        const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')))
        return {
            user: { name: payload.name ?? 'Operator', role: payload.role ?? 'admin' },
            exp: typeof payload.exp === 'number' ? payload.exp : null,
        }
    } catch {
        return { user: null, exp: null }
    }
}

function restore(): AuthState {
    const raw = localStorage.getItem(AUTH_KEY)
    if (!raw) return { token: null, user: null }
    try {
        const saved = JSON.parse(raw)
        const { user, exp } = parseToken(saved.token)
        // An expired token is the same as being logged out; without this check
        // the UI renders, then every request 401s.
        if (exp && exp * 1000 < Date.now()) {
            localStorage.removeItem(AUTH_KEY)
            return { token: null, user: null }
        }
        return { token: saved.token, user: user ?? saved.user ?? null }
    } catch {
        localStorage.removeItem(AUTH_KEY)
        return { token: null, user: null }
    }
}

const auth: Module<AuthState, unknown> = {
    namespaced: true,
    state: restore,
    getters: {
        isAuthenticated: (s) => Boolean(s.token),
        user: (s) => s.user,
        initials: (s) => (s.user?.name ?? 'OP').split(' ').map((p) => p[0]).join('').slice(0, 2).toUpperCase(),
    },
    mutations: {
        SET_SESSION(state, { token, user }: { token: string; user: AuthUser | null }) {
            const parsed = parseToken(token).user ?? user ?? { name: 'Operator', role: 'admin' }
            state.token = token
            state.user = parsed
            localStorage.setItem(AUTH_KEY, JSON.stringify({ token, user: parsed }))
        },
        CLEAR(state) {
            state.token = null
            state.user = null
            localStorage.removeItem(AUTH_KEY)
        },
    },
    actions: {
        signIn({ commit }, payload: { token: string; user: AuthUser | null }) {
            commit('SET_SESSION', payload)
        },
        logout({ commit }) { commit('CLEAR') },
    },
}
export default auth
