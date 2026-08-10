import type { Module } from 'vuex'

export interface Toast { id: number; kind: 'success' | 'error' | 'info'; title: string; message: string }
interface UiState { dark: boolean; sidebarCollapsed: boolean; toasts: Toast[] }

let toastId = 0

const ui: Module<UiState, unknown> = {
    namespaced: true,
    state: () => ({
        dark: document.documentElement.classList.contains('dark'),
        sidebarCollapsed: localStorage.getItem('sidebar_collapsed') === '1',
        toasts: [],
    }),
    mutations: {
        SET_DARK(state, value: boolean) {
            state.dark = value
            document.documentElement.classList.toggle('dark', value)
            localStorage.setItem('dark', value ? '1' : '0')
        },
        SET_SIDEBAR(state, collapsed: boolean) {
            state.sidebarCollapsed = collapsed
            localStorage.setItem('sidebar_collapsed', collapsed ? '1' : '0')
        },
        PUSH_TOAST(state, toast: Toast) { state.toasts.push(toast) },
        DROP_TOAST(state, id: number) { state.toasts = state.toasts.filter((t) => t.id !== id) },
    },
    actions: {
        toggleDark({ commit, state }) { commit('SET_DARK', !state.dark) },
        toggleSidebar({ commit, state }) { commit('SET_SIDEBAR', !state.sidebarCollapsed) },
        toast({ commit }, payload: Omit<Toast, 'id'>) {
            const id = ++toastId
            commit('PUSH_TOAST', { id, ...payload })
            setTimeout(() => commit('DROP_TOAST', id), 4000)
        },
        dismiss({ commit }, id: number) { commit('DROP_TOAST', id) },
    },
}
export default ui
