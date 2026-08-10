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
                    path: 'api-keys', name: 'ApiKeys',
                    component: () => import('@/views/pages/api-keys/Index.vue'),
                    meta: { title: 'API keys' + SUFFIX, crumb: ['System', 'API keys'] },
                },
                {
                    path: 'settings', name: 'Settings',
                    component: () => import('@/views/pages/settings/Index.vue'),
                    meta: { title: 'Settings' + SUFFIX, crumb: ['System', 'Settings'] },
                },
                {
                    path: 'logs', name: 'Logs',
                    component: () => import('@/views/pages/logs/Index.vue'),
                    meta: { title: 'Logs' + SUFFIX, crumb: ['System', 'Logs'] },
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
    return true
})

export default router
