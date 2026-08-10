<script setup lang="ts">
import { computed, ref } from 'vue'
import { useStore } from 'vuex'
import AuthedImage from '@/components/AuthedImage.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'
import BoxOverlay from './components/BoxOverlay.vue'
import { formatBytes, formatDate, formatMs, formatTime } from '@/utils/format'
import { colourForLabel, labelOrder } from '@/utils/boxes'
import { useHooks } from './controller'
import {
    activeField, activeKeys, anyActive, detail, fieldToBoxes, hiddenLabels, hover,
    imageMode, loading, notFound, pinned, showLabels, toggleLabelVisibility, toggleLabels,
} from './model'

const { pickBox, pinField, reprocess, remove, copy } = useHooks()
const store = useStore()
const confirmDelete = ref(false)

const d = detail
const order = computed(() => labelOrder(d.value?.boxes ?? []))

/** Shared frozen blank, so the original view does not allocate a Set per render. */
const EMPTY_KEYS: Set<string> = new Set()

const imageUrl = computed(() => {
    if (!d.value) return null
    return imageMode.value === 'canvas'
        ? `/documents/${d.value.id}/image/canvas`
        : `/documents/${d.value.id}/image/original`
})

/** Distinct labels with counts, for the legend. */
const legend = computed(() => {
    const counts = new Map<string, number>()
    for (const b of d.value?.boxes ?? []) counts.set(b.label, (counts.get(b.label) ?? 0) + 1)
    return [...counts.entries()].map(([label, count]) => ({
        label,
        display: d.value?.boxes.find((b) => b.label === label)?.display ?? label,
        count,
        colour: colourForLabel(label, order.value),
        hidden: hiddenLabels.value.has(label),
    }))
})

const GOOD = new Set(['good', 'REAL', 'real', 'ok'])
function qualityClass(v: unknown): string {
    if (v == null) return 'q-unknown'
    return GOOD.has(String(v)) ? 'q-good' : 'q-bad'
}

const qualityFlags = computed(() =>
    Object.entries(d.value?.quality ?? {}).filter(([k]) => k !== 'DocConf'))

const timingRows = computed(() => {
    const t = d.value?.timings ?? {}
    return Object.entries(t)
        .filter(([k]) => k !== 'total')
        .sort((a, b) => b[1] - a[1])
})
const totalTime = computed(() => d.value?.timings?.total ?? null)

const prettyJson = computed(() => JSON.stringify(d.value, null, 2))
const jsonBytes = computed(() => `${(prettyJson.value.length / 1024).toFixed(1)} KB`)

function fieldColour(name: string): string {
    return colourForLabel(name, order.value)
}
function hasBox(name: string): boolean {
    return (fieldToBoxes.value.get(name)?.length ?? 0) > 0
}
function copyJson(): void {
    void navigator.clipboard.writeText(prettyJson.value)
    store.dispatch('ui/toast', { kind: 'success', title: 'Copied', message: 'Raw JSON' })
}
</script>

<template>
  <header class="topbar detail-topbar">
    <div class="detail-head">
      <router-link to="/documents" class="btn btn-ghost btn-sm">← Documents</router-link>
      <div class="divider"></div>
      <div>
        <h1 class="page-title">{{ d?.filename ?? '…' }}</h1>
        <div class="detail-meta">
          <span v-if="d?.recognised">{{ d?.doc_type }}</span>
          <span v-else class="u-dash">not recognised</span>
          <span>·</span><span>{{ formatDate(d?.created_at ?? null) }} {{ formatTime(d?.created_at ?? null) }}</span>
          <span>·</span><span class="u-mono">{{ formatMs(d?.processing_ms ?? null) }}</span>
          <span v-if="d?.device">· {{ d.device.toUpperCase() }}</span>
        </div>
      </div>
    </div>
    <div class="topbar-right">
      <span v-if="d" :class="['badge', 'badge-' + d.status]"><span class="bdot"></span>{{ d.status }}</span>
      <button v-if="d && (d.status === 'done' || d.status === 'failed')"
              class="btn btn-outline btn-sm" @click="reprocess()">Reprocess</button>
      <button class="btn btn-danger-outline btn-sm" @click="confirmDelete = true">Delete</button>
      <button class="icon-btn" @click="store.dispatch('ui/toggleDark')">
        {{ store.state.ui.dark ? '☀' : '☾' }}
      </button>
    </div>
  </header>

  <div class="content">
    <!-- A deleted document, or an id left over from a wiped ephemeral store,
         is a normal thing to navigate to. Say so plainly instead of rendering
         an empty shell full of placeholder dashes. -->
    <div v-if="notFound" class="card card-pad" style="text-align:center;padding:48px 24px">
      <div style="font-size:15px;font-weight:600;margin-bottom:6px">Document not found</div>
      <div class="field-help" style="margin-bottom:16px">
        It was deleted, or the service restarted and cleared its temporary storage.
      </div>
      <router-link to="/documents" class="btn btn-primary">Back to documents</router-link>
    </div>

    <div v-else-if="loading && !d" class="card card-pad">Loading…</div>

    <div v-else-if="d" class="detail-grid">
      <!-- Left: the image and its overlay -->
      <div class="detail-left">
        <div class="card u-overflow-hidden">
          <div class="card-head">
            <span>Detected regions</span>
            <div class="head-actions">
              <div class="seg">
                <button :class="{ active: imageMode === 'canvas' }" @click="imageMode = 'canvas'">Canvas</button>
                <button :class="{ active: imageMode === 'original' }" @click="imageMode = 'original'">Original</button>
              </div>
              <button class="btn btn-ghost btn-sm" @click="toggleLabels()">
                {{ showLabels ? 'Hide labels' : 'Show labels' }}
              </button>
            </div>
          </div>

          <!-- Boxes live in canvas coordinates. Drawing them over the original
               would be wrong by a perspective transform, so the overlay is
               simply not shown there — with an explanation rather than silence. -->
          <template v-if="imageMode === 'canvas'">
            <BoxOverlay
              :image-url="imageUrl" :width="d.canvas.width" :height="d.canvas.height"
              :boxes="d.boxes" :address="d.address" :active-keys="activeKeys"
              :any-active="anyActive" :show-labels="showLabels" :hidden-labels="hiddenLabels"
              @HOVER="hover = $event" @PICK="pickBox($event)" />

            <div v-if="legend.length" class="legend">
              <span v-for="item in legend" :key="item.label"
                    :class="['legend-item', { off: item.hidden }]"
                    :style="{ '--bx': item.colour }"
                    @click="toggleLabelVisibility(item.label)"
                    @mouseenter="hover = fieldToBoxes.get(item.label)?.[0] ?? null"
                    @mouseleave="hover = null">
                <span class="swatch"></span>{{ item.display }}
                <span v-if="item.count > 1" class="cnt">×{{ item.count }}</span>
              </span>
            </div>
          </template>

          <!-- Same viewer, no boxes. Rendering the original in a bare <img>
               instead made it jump when switching views: it lost the centring,
               the fit bounding and the Fit/1:1 control that the canvas view
               has. Passing empty geometry keeps the chrome identical and the
               overlay silent. -->
          <template v-else>
            <BoxOverlay
              :image-url="imageUrl" :width="d.original.width" :height="d.original.height"
              :boxes="[]" :address="null" :active-keys="EMPTY_KEYS"
              :any-active="false" :show-labels="false" :hidden-labels="EMPTY_KEYS"
              note="as uploaded — no regions drawn" />
            <div class="ephemeral-note" style="margin:12px">
              Boxes are in canvas coordinates — switch to Canvas to see them. The library does
              not retain the deskew angle, so they cannot be mapped onto the original.
            </div>
          </template>
        </div>
      </div>

      <!-- Right: the recognised data -->
      <div class="detail-right">
        <div v-if="d.status === 'failed'" class="danger-card">
          <strong>Recognition failed</strong>
          <div style="margin-top:6px">{{ d.error }}</div>
          <div v-if="d.error_code" class="u-mono" style="margin-top:4px;font-size:11px">{{ d.error_code }}</div>
        </div>

        <div v-if="d.fields.length" class="card u-overflow-hidden">
          <div class="card-head">Recognised fields <span class="u-muted">{{ d.fields.length }}</span></div>
          <table class="table">
            <thead>
              <tr><th style="width:180px">Field</th><th>Value</th><th style="width:64px">Conf</th><th style="width:36px"></th></tr>
            </thead>
            <tbody>
              <tr v-for="f in d.fields" :key="f.name" :id="'fld-' + f.name"
                  :class="['fld-row', { active: activeField === f.name, 'no-box': !hasBox(f.name) }]"
                  :style="{ '--bx': fieldColour(f.name) }"
                  @mouseenter="hover = f.name" @mouseleave="hover = null"
                  @click="hasBox(f.name) && pinField(f.name)">
                <td>
                  <span v-if="hasBox(f.name)" class="fld-dot" :style="{ background: fieldColour(f.name) }"></span>
                  <span class="fld-name">{{ f.display }}</span>
                </td>
                <td>
                  <span v-if="f.value" class="fld-val" :class="{ 'u-mono': f.script === 'num' }"
                        :lang="f.script === 'ru' ? 'ru' : 'en'">{{ f.value }}</span>
                  <span v-else class="u-dash">—</span>
                </td>
                <td class="u-mono">{{ f.conf != null ? f.conf.toFixed(2) : '—' }}</td>
                <td>
                  <button class="act-btn" title="Copy" @click.stop="copy(f.value)">⧉</button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-if="d.address" class="card u-overflow-hidden">
          <div class="card-head">
            Address lines
            <span v-if="!d.address.aligned" class="badge badge-failed" title="Geometry could not be matched to text">
              geometry unavailable
            </span>
          </div>
          <table class="table">
            <tbody>
              <tr v-for="line in d.address.lines" :key="line.id"
                  :class="['fld-row', { active: activeKeys.has(line.id) }]"
                  @mouseenter="hover = line.id" @mouseleave="hover = null"
                  @click="pickBox(line.id)">
                <td style="width:110px">
                  <span :class="['badge', line.kind === 'handwritten' ? 'badge-processing' : 'badge-done']">
                    {{ line.kind }}
                  </span>
                </td>
                <td>
                  <span v-if="line.text" class="fld-val" lang="ru">{{ line.text }}</span>
                  <span v-else class="u-dash">not recognised (handwritten)</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="card card-pad">
          <div class="card-head" style="padding:0 0 12px">Quality</div>
          <div class="kv-list">
            <div class="kv-row">
              <span class="kv-key">Document confidence</span>
              <span class="kv-val u-mono">{{ d.doc_conf != null ? d.doc_conf.toFixed(3) : '—' }}</span>
            </div>
            <div v-for="[key, value] in qualityFlags" :key="key" class="kv-row">
              <span class="kv-key">{{ key }}</span>
              <span :class="['q-pill', qualityClass(value)]">{{ value }}</span>
            </div>
          </div>
        </div>

        <div v-if="timingRows.length" class="card card-pad">
          <div class="card-head" style="padding:0 0 12px">
            Timings <span class="u-muted u-mono">{{ totalTime?.toFixed(3) }}s total</span>
          </div>
          <div class="kv-list">
            <div v-for="[stage, seconds] in timingRows" :key="stage" class="kv-row">
              <span class="kv-key u-mono">{{ stage }}</span>
              <span class="kv-val u-mono">{{ (seconds * 1000).toFixed(0) }} ms</span>
            </div>
          </div>
        </div>

        <details class="raw-json">
          <summary>
            Raw JSON <span class="u-muted">· {{ jsonBytes }}</span>
            <button class="btn btn-sm btn-ghost" @click.prevent.stop="copyJson()">Copy</button>
          </summary>
          <!-- Serialising the parsed object, not the raw response text, so
               Cyrillic shows as characters even though FastAPI escapes it. -->
          <pre class="u-mono">{{ prettyJson }}</pre>
        </details>
      </div>
    </div>
  </div>

  <ConfirmDialog :open="confirmDelete" title="Delete document" danger confirm-label="Delete"
                 message="This removes the document and all its images. This cannot be undone."
                 @CONFIRM="remove(); confirmDelete = false" @CANCEL="confirmDelete = false" />
</template>

<style scoped>
.detail-topbar{align-items:flex-start;}
.detail-head{display:flex;align-items:center;gap:14px;}
.divider{width:1px;height:30px;background:var(--color-border);}
.detail-meta{display:flex;gap:6px;flex-wrap:wrap;font-size:12px;color:var(--color-text-muted);margin-top:3px;}
.detail-right{display:flex;flex-direction:column;gap:16px;}
.head-actions{display:flex;gap:8px;align-items:center;}
.original-pane{padding:0 0 4px;}
.act-btn{width:26px;height:26px;border:1px solid var(--color-border);border-radius:6px;
  background:var(--color-card);cursor:pointer;color:var(--color-text-sub);font-size:12px;}
.act-btn:hover{border-color:var(--color-primary);color:var(--color-primary);}
.kv-list{display:flex;flex-direction:column;gap:8px;}
.kv-row{display:flex;justify-content:space-between;align-items:center;font-size:13px;}
.kv-key{color:var(--color-text-sub);}
.kv-val{font-weight:600;color:var(--color-text);}
</style>
