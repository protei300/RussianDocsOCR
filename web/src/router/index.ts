import { createRouter, createWebHistory } from 'vue-router'
import store from '@/store'

const SUFFIX = ' · RussianDocs OCR'

const router = createRouter({
    history: createWebHistory('/'),
    scrollBehavior: () => ({ top: 0 }),
    routes: [
        {
            path: '/login', name: 'Login',
            component: () => import('@/views/pages/login/Index.vue'),
            meta: { public: true, title: 'Sign in' + SUFFIX },
        },
        {
            // Outside the layout on purpose: a session that still owes a password
            // change may not see the application shell, because it may not use
            // anything inside it.
            path: '/change-password', name: 'ChangePassword',
            component: () => import('@/views/pages/change-password/Index.vue'),
            meta: { title: 'Change password' + SUFFIX },
        },
        {
            path: '/', component: () => import('@/views/layout/Layout.vue'),
            children: [
                { path: '', redirect: { name: 'Documents' } },
                {
                    path: 'documents', name: 'Documents',
                    component: () => import('@/views/pages/documents/Index.vue'),
                    meta: { title: 'Documents' + SUFFIX, crumb: ['Recognition', 'Documents'] },
                },
                {
                    // Right under Documents in the sidebar: it is the same subject — a
                    // document going through recognition — seen from the outside, by a site
                    // that is integrating rather than operating.
                    path: 'integration', name: 'Integration',
                    component: () => import('@/views/pages/integration/Index.vue'),
                    meta: { title: 'Integration demo' + SUFFIX,
                        crumb: ['Recognition', 'Integration demo'] },
                },
                {
                    path: 'documents/:id', name: 'DocumentDetail',
                    component: () => import('@/views/pages/document-detail/Index.vue'),
                    meta: { title: 'Document' + SUFFIX, crumb: ['Recognition', 'Document'] },
                },
                {
                    path: 'status', name: 'Status',
                    component: () => import('@/views/pages/status/Index.vue'),
                    meta: { title: 'Status' + SUFFIX, crumb: ['System', 'Status'] },
                },
                {
                    path: 'users', name: 'Users',
                    component: () => import('@/views/pages/users/Index.vue'),
                    meta: { title: 'Users' + SUFFIX, crumb: ['System', 'Users'],
                        requiresUsers: true },
                },
                {
                    path: 'api-keys', name: 'ApiKeys',
                    component: () => import('@/views/pages/api-keys/Index.vue'),
                    meta: { title: 'API keys' + SUFFIX, crumb: ['System', 'API keys'],
                        requiresAdmin: true },
                },
                {
                    path: 'settings', name: 'Settings',
                    component: () => import('@/views/pages/settings/Index.vue'),
                    meta: { title: 'Settings' + SUFFIX, crumb: ['System', 'Settings'],
                        requiresAdmin: true },
                },
                {
                    path: 'logs', name: 'Logs',
                    component: () => import('@/views/pages/logs/Index.vue'),
                    meta: { title: 'Logs' + SUFFIX, crumb: ['System', 'Logs'],
                        requiresAdmin: true },
                },
            ],
        },
        { path: '/:pathMatch(.*)*', redirect: '/' },
    ],
})

router.beforeEach((to) => {
    document.title = (to.meta.title as string) ?? 'RussianDocs OCR'
    const authed = store.getters['auth/isAuthenticated']
    if (!to.meta.public && !authed) return { name: 'Login', query: { redirect: to.fullPath } }
    if (to.name === 'Login' && authed) return { name: 'Documents' }

    // A restricted session may only reach the password form. The server enforces
    // this too — it is the authority — but without the guard the user would land
    // on a page that renders and then 403s on every request it makes.
    if (authed && store.getters['auth/mustChangePassword'] && to.name !== 'ChangePassword') {
        return { name: 'ChangePassword' }
    }
    // User management does not exist in PIN mode; the API answers 404 there, and
    // a route that renders an error page is worse than one that is not offered.
    if (to.meta.requiresUsers && !store.getters['auth/canManageUsers']) {
        return { name: 'Documents' }
    }
    if (to.meta.requiresAdmin && !store.getters['auth/isAdmin']) {
        return { name: 'Documents' }
    }
    return true
})

export default router
