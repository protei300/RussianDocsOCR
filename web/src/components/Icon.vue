<script setup lang="ts">
/**
 * Inline stroke icons.
 *
 * Emoji were the first cut and looked wrong: the glyph comes from whatever
 * emoji font the OS ships, so it arrives at its own weight, its own colour
 * (a red bin, a blue eye) and its own optical size — blurry next to crisp
 * 1px table borders, and immune to the button's `currentColor`. Inline SVG
 * renders sharp at any DPI, inherits the button's colour on hover, and looks
 * the same on every machine.
 *
 * Paths use a 24×24 grid with round caps — the same convention as the icons
 * already inlined in AppSidebar, so the two sets sit together without a
 * visible weight difference. Size and colour come from CSS
 * (`.act-btn svg{width:14px}`), never from props.
 */
const props = defineProps<{ name: keyof typeof PATHS; strokeWidth?: number }>()

/** Each entry is the body of the icon; the wrapper supplies the shared attrs. */
const PATHS = {
    eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    refresh: '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 3v6h-6"/>',
    download: '<path d="M12 3v12"/><path d="M7 11l5 5 5-5"/><path d="M4 20h16"/>',
    trash: '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M6 6l1 14h10l1-14"/>'
        + '<path d="M10 11v5"/><path d="M14 11v5"/>',
} as const
</script>

<template>
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
       :stroke-width="props.strokeWidth ?? 1.8"
       stroke-linecap="round" stroke-linejoin="round"
       aria-hidden="true" focusable="false" v-html="PATHS[props.name]" />
</template>
