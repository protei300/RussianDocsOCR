import type { Module } from 'vuex'
import { AUTH_KEY } from '@/common/fetch'

export interface AuthUser {
    name: string
    role: string
    username?: string
    userId?: number
}

interface AuthState {
    token: string | null
    user: AuthUser | null
    /** Set at sign-in when the account still owes a password change. The token is
     *  real but restricted — the server refuses everything except the change — so
     *  the router sends the user straight to the form instead of letting them
     *  bounce off a 403 on whatever page they aimed at. */
    mustChangePassword: boolean
    /** Which authentication the server is running. Read once from /auth/config so
     *  the sidebar and the router can hide user management in PIN mode rather
     *  than showing a page that answers 404. */
    mode: 'pin' | 'users'
}

/** Read the JWT payload to recover the user and expiry after a page reload. */
function parseToken(token: string): { user: AuthUser | null; exp: number | null } {
    try {
        const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')))
        return {
            user: {
                name: payload.name ?? 'Operator',
                role: payload.role ?? 'admin',
                username: payload.sub,
                userId: typeof payload.uid === 'number' ? payload.uid : undefined,
            },
            exp: typeof payload.exp === 'number' ? payload.exp : null,
        }
    } catch {
        return { user: null, exp: null }
    }
}

function restore(): AuthState {
    const base: AuthState = { token: null, user: null, mustChangePassword: false, mode: 'pin' }
    const raw = localStorage.getItem(AUTH_KEY)
    if (!raw) return base
    try {
        const saved = JSON.parse(raw)
        const { user, exp } = parseToken(saved.token)
        // An expired token is the same as being logged out; without this check
        // the UI renders, then every request 401s.
        if (exp && exp * 1000 < Date.now()) {
            localStorage.removeItem(AUTH_KEY)
            return base
        }
        return {
            ...base,
            token: saved.token,
            user: user ?? saved.user ?? null,
            mustChangePassword: Boolean(saved.mustChangePassword),
            mode: saved.mode === 'users' ? 'users' : 'pin',
        }
    } catch {
        localStorage.removeItem(AUTH_KEY)
        return base
    }
}

function persist(state: AuthState): void {
    localStorage.setItem(AUTH_KEY, JSON.stringify({
        token: state.token,
        user: state.user,
        mustChangePassword: state.mustChangePassword,
        mode: state.mode,
    }))
}

const auth: Module<AuthState, unknown> = {
    namespaced: true,
    state: restore,
    getters: {
        isAuthenticated: (s) => Boolean(s.token),
        user: (s) => s.user,
        mode: (s) => s.mode,
        usersEnabled: (s) => s.mode === 'users',
        mustChangePassword: (s) => s.mustChangePassword,
        // Only an administrator sees user management — and only when accounts
        // exist at all. In PIN mode every session carries the admin role, which
        // is why the mode has to be part of this and not just the role.
        canManageUsers: (s) => s.mode === 'users' && s.user?.role === 'admin',
        role: (s) => s.user?.role ?? 'viewer',
        // Mirrors models.ROLES on the server. PIN sessions carry the admin role,
        // so in PIN mode everything below is true and nothing disappears — the
        // interface looks exactly as it did before named accounts.
        canWrite: (s) => ['operator', 'admin'].includes(s.user?.role ?? ''),
        isAdmin: (s) => s.user?.role === 'admin',
        initials: (s) => (s.user?.name ?? 'OP').split(' ')
            .map((p) => p[0]).join('').slice(0, 2).toUpperCase(),
    },
    mutations: {
        SET_SESSION(state, payload: { token: string; user: AuthUser | null;
            mustChangePassword?: boolean }) {
            const parsed = parseToken(payload.token).user ?? payload.user
                ?? { name: 'Operator', role: 'admin' }
            state.token = payload.token
            state.user = parsed
            state.mustChangePassword = Boolean(payload.mustChangePassword)
            persist(state)
        },
        SET_MODE(state, mode: 'pin' | 'users') {
            state.mode = mode
            if (state.token) persist(state)
        },
        PASSWORD_CHANGED(state) {
            // The server has just invalidated this token by bumping the account's
            // version, so keeping it would only produce 401s on the next click.
            state.token = null
            state.user = null
            state.mustChangePassword = false
            localStorage.removeItem(AUTH_KEY)
        },
        CLEAR(state) {
            state.token = null
            state.user = null
            state.mustChangePassword = false
            localStorage.removeItem(AUTH_KEY)
        },
    },
    actions: {
        signIn({ commit }, payload: { token: string; user: AuthUser | null;
            mustChangePassword?: boolean }) {
            commit('SET_SESSION', payload)
        },
        setMode({ commit }, mode: 'pin' | 'users') { commit('SET_MODE', mode) },
        passwordChanged({ commit }) { commit('PASSWORD_CHANGED') },
        logout({ commit }) { commit('CLEAR') },
    },
}
export default auth
