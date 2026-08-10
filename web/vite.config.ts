import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

/** Stamp a version onto `/_shared.css`.
 *
 * Vite fingerprints what it bundles, but `public/` is copied verbatim — so the
 * design system was the one file a browser could serve stale after a deploy.
 * That fails in the worst way: the app is new, the stylesheet is old, and the
 * layout looks broken for reasons nothing in the code explains.
 */
const stampSharedCss = (stamp: string) => ({
    name: 'stamp-shared-css',
    transformIndexHtml: (html: string) =>
        html.replace('/_shared.css', `/_shared.css?v=${stamp}`),
})

export default defineConfig({
    plugins: [vue(), stampSharedCss(Date.now().toString(36))],
    resolve: {
        alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
    },
    server: {
        port: 8000,
        // The API runs as a separate process in development. In production the
        // built SPA is served by FastAPI itself, so there is no proxy and no
        // cross-origin request at all.
        proxy: { '/api': { target: 'http://127.0.0.1:8002', changeOrigin: true } },
    },
    build: { outDir: 'dist', emptyOutDir: true },
})
