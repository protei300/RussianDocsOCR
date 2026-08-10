import type { Box, OrientedBox } from '@/types'

/**
 * Sign of the oriented-box rotation.
 *
 * Image coordinates are y-down, so a mathematically counter-clockwise rotation
 * renders clockwise on screen. Which convention the detector uses is not
 * documented, so this constant is the single place to flip it after checking
 * one real INTPASSPORTADDR sample: for a visibly tilted address line the
 * polygon must lean the same way as the text.
 */
export const ANGLE_SIGN = 1

/**
 * Corner points for an oriented box, as an SVG `points` string.
 *
 * A polygon rather than `<rect transform="rotate(...)">` because the same four
 * corners are needed for the label anchor, and because `angle_rad` is radians
 * while SVG `rotate()` wants degrees — one fewer conversion to get wrong.
 */
export function obbPoints(b: OrientedBox): string {
    const a = ANGLE_SIGN * b.angle_rad
    const cos = Math.cos(a)
    const sin = Math.sin(a)
    const hw = b.w / 2
    const hh = b.h / 2
    return ([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]] as const)
        .map(([dx, dy]) =>
            `${(b.cx + dx * cos - dy * sin).toFixed(1)},${(b.cy + dx * sin + dy * cos).toFixed(1)}`)
        .join(' ')
}

/** Top-most corner — where the label chip is anchored. */
export function obbAnchor(b: OrientedBox): { x: number; y: number } {
    const points = obbPoints(b).split(' ').map((p) => p.split(',').map(Number))
    return points.reduce(
        (best, [x, y]) => (y < best.y ? { x, y } : best),
        { x: points[0][0], y: points[0][1] },
    )
}

/**
 * Eight hues, all legible on both the light card (#FFFFFF) and the dark one
 * (#1A2535), so dark mode needs no separate palette.
 */
const PALETTE = ['#0067F5', '#00980F', '#F27405', '#8B2FD6',
    '#00A3B4', '#CB0B00', '#B58900', '#3D5AFE']

/** Labels the detector finds but never sends to OCR — drawn differently. */
export const NON_TEXT = new Set(['Face', 'Signature'])

/**
 * Colour for a label, assigned by its position in the document's own sorted
 * label list. Stable per document (the same passport always looks the same)
 * and adjacent fields never collide.
 */
export function colourForLabel(label: string, order: string[]): string {
    if (NON_TEXT.has(label)) return 'var(--color-text-muted)'
    const index = order.indexOf(label)
    return PALETTE[(index < 0 ? 0 : index) % PALETTE.length]
}

/** Distinct labels present, sorted — the ordering that drives colour assignment. */
export function labelOrder(boxes: Box[]): string[] {
    return [...new Set(boxes.map((b) => b.label))].sort()
}

export function pct(value: number, total: number | null): string {
    if (!total) return '0%'
    return `${(value / total) * 100}%`
}
