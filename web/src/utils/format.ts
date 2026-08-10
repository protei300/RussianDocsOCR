/** Day-first numeric dates: `03/08/2026`.
 *
 * The interface is English but the operators are in Russia, where day-first is
 * the convention — and `03/08` read as August 3rd by one person and March 8th
 * by another is a real, silent misreading. `en-GB` gives day-first with slashes
 * and zero-padding without pulling in a locale bundle. Time stays 24-hour for
 * the same reason.
 */
export function formatDate(iso: string | null): string {
    if (!iso) return '—'
    return new Date(iso).toLocaleDateString('en-GB', {
        day: '2-digit', month: '2-digit', year: 'numeric',
    })
}

export function formatTime(iso: string | null): string {
    if (!iso) return ''
    return new Date(iso).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
}

export function formatBytes(bytes: number | null): string {
    if (!bytes) return '—'
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function formatMs(ms: number | null): string {
    if (ms == null) return '—'
    return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`
}

export function formatUptime(seconds: number): string {
    const d = Math.floor(seconds / 86400)
    const h = Math.floor((seconds % 86400) / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`
}
