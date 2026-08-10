<script setup lang="ts">
/**
 * Detected regions drawn over the canvas image.
 *
 * Technique: an SVG layered on the `<img>`, with `viewBox` set to the canvas's
 * pixel dimensions. That makes one SVG user unit equal one canvas pixel, so
 * backend coordinates go straight into the markup — no scaling maths, no
 * resize listener, and it stays correct through window resizes, zoom and print.
 *
 * Three details that look cosmetic but are not:
 *
 * - `vector-effect="non-scaling-stroke"`. Without it `stroke-width:2` means two
 *   *canvas* pixels, so on a 2000px-wide document shown in a 700px column the
 *   outlines render at 0.7px (invisible) and become fat when zoomed in.
 * - Labels are HTML positioned in percentages, not SVG `<text>`. SVG text
 *   scales with the viewBox and would be unreadable when zoomed out.
 * - `line-height:0` on the wrapper (in _shared.css). An inline `<img>` sits on
 *   the text baseline and leaves ~4px of phantom space below it, which would
 *   offset the absolutely-positioned overlay by those pixels.
 */
import { computed, nextTick, ref } from 'vue'
import AuthedImage from '@/components/AuthedImage.vue'
import { colourForLabel, labelOrder, obbAnchor, obbPoints, pct } from '@/utils/boxes'
import type { AddressBlock, Box } from '@/types'

const props = defineProps<{
    imageUrl: string | null
    width: number | null
    height: number | null
    boxes: Box[]
    address: AddressBlock | null
    activeKeys: Set<string>
    anyActive: boolean
    showLabels: boolean
    hiddenLabels: Set<string>
    /** Replaces the region count in the header. Set when the viewer is showing
     *  something the overlay does not apply to, such as the original upload —
     *  "0 regions" there would read as a detection failure rather than as
     *  "boxes are not drawn on this view". */
    note?: string | null
}>()

const emit = defineEmits<{ HOVER: [key: string | null]; PICK: [key: string] }>()

/**
 * Two view modes rather than a zoom ladder.
 *
 * A passport is portrait and the detail column is landscape, so scaling to
 * container *width* pushes most of the document below the fold — you end up
 * scrolling to see what was detected, which defeats the point of the overlay.
 * `fit` therefore bounds the image on both axes so the whole document is
 * always visible, and is the default. `actual` shows natural pixel size for
 * reading small print, with the container scrolling.
 *
 * Both modes work with the same SVG because the overlay is `inset:0` over a
 * wrapper that shrinks to the rendered image — no coordinate maths changes.
 */
type ViewMode = 'fit' | 'actual'
const mode = ref<ViewMode>('fit')
const scrollEl = ref<HTMLElement | null>(null)

const order = computed(() => labelOrder(props.boxes))
const visibleBoxes = computed(() => props.boxes.filter((b) => !props.hiddenLabels.has(b.label)))

/** Oriented address-line boxes. Suppressed when the backend says the geometry
 *  and the text lists desynchronised — drawing them would caption boxes with
 *  another line's text. */
const orientedLines = computed(() => {
    if (!props.address?.aligned) return []
    return props.address.lines.filter((l) => l.obbox)
})

function colour(label: string): string {
    return colourForLabel(label, order.value)
}

function isActive(key: string): boolean {
    return props.activeKeys.has(key)
}

function setMode(next: ViewMode): void {
    mode.value = next
    if (next !== 'actual') return
    // Centre the (now larger) image so switching to 1:1 does not dump the
    // viewport in the top-left corner of a passport.
    void nextTick(() => {
        const el = scrollEl.value
        if (!el) return
        el.scrollLeft = (el.scrollWidth - el.clientWidth) / 2
        el.scrollTop = 0
    })
}

defineExpose({ mode, setMode })
</script>

<template>
  <div class="overlay-head">
    <div class="seg">
      <button :class="{ active: mode === 'fit' }" title="Fit the whole document"
              @click="setMode('fit')">Fit</button>
      <button :class="{ active: mode === 'actual' }" title="Actual pixel size"
              @click="setMode('actual')">1:1</button>
    </div>
    <span class="overlay-count">
      <template v-if="note">{{ note }}</template>
      <template v-else>
        {{ visibleBoxes.length }} region{{ visibleBoxes.length === 1 ? '' : 's' }}
      </template>
      <!-- Non-breaking space: Vue collapses the whitespace between sibling
           <template> blocks, so a plain one disappears and the separator ends
           up glued to the preceding word. -->
      <template v-if="width">&nbsp;· {{ width }}×{{ height }} px</template>
    </span>
  </div>

  <div ref="scrollEl" :class="['doc-view', mode]">
    <div class="doc-wrap">
      <AuthedImage :url="imageUrl" alt="Recognised document canvas" img-class="doc-img" />

      <svg v-if="width && height" class="doc-svg" :viewBox="`0 0 ${width} ${height}`"
           preserveAspectRatio="none" role="img" aria-label="Detected regions">
        <rect v-for="b in visibleBoxes" :key="b.id"
              :x="b.x1" :y="b.y1" :width="b.x2 - b.x1" :height="b.y2 - b.y1"
              :class="['bx', { active: isActive(b.id), dim: anyActive && !isActive(b.id),
                               visual: b.kind === 'visual' }]"
              :style="{ '--bx': colour(b.label) }"
              vector-effect="non-scaling-stroke"
              @mouseenter="emit('HOVER', b.id)" @mouseleave="emit('HOVER', null)"
              @click="emit('PICK', b.id)">
          <title>{{ b.display }}{{ b.text ? ' — ' + b.text : '' }}</title>
        </rect>

        <polygon v-for="line in orientedLines" :key="line.id"
                 :points="obbPoints(line.obbox!)"
                 :class="['bx', { active: isActive(line.id), dim: anyActive && !isActive(line.id) }]"
                 :style="{ '--bx': line.kind === 'handwritten' ? '#F27405' : '#0067F5' }"
                 vector-effect="non-scaling-stroke"
                 @mouseenter="emit('HOVER', line.id)" @mouseleave="emit('HOVER', null)"
                 @click="emit('PICK', line.id)">
          <title>{{ line.kind }}{{ line.text ? ' — ' + line.text : ' (not recognised)' }}</title>
        </polygon>
      </svg>

      <div v-if="showLabels && width && height" class="doc-labels">
        <span v-for="b in visibleBoxes" :key="b.id"
              class="bx-label" :class="{ active: isActive(b.id) }"
              :style="{ left: pct(b.x1, width), top: pct(b.y1, height), '--bx': colour(b.label) }">
          {{ b.display }}
        </span>
        <span v-for="line in orientedLines" :key="line.id"
              class="bx-label" :class="{ active: isActive(line.id) }"
              :style="{ left: pct(obbAnchor(line.obbox!).x, width),
                        top: pct(obbAnchor(line.obbox!).y, height),
                        '--bx': line.kind === 'handwritten' ? '#F27405' : '#0067F5' }">
          {{ line.kind }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.overlay-head{display:flex;align-items:center;justify-content:space-between;
  padding:12px 16px;border-bottom:1px solid var(--color-border-light);}
.overlay-count{font-size:12px;color:var(--color-text-muted);font-weight:600;}
</style>
