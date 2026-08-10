<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import AppTopbar from '@/components/AppTopbar.vue'
import AuthedImage from '@/components/AuthedImage.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'
import Icon from '@/components/Icon.vue'
import UploadModal from '@/components/UploadModal.vue'
import $api from '@/api'
import { formatBytes, formatDate, formatMs, formatTime } from '@/utils/format'
import type { DocumentRow } from '@/types'
import { getData, getDataSearch, useHooks } from './controller'
import { DOC_TYPES, documents, form, loading, stats, total, uploadOpen } from './model'

const { setSort, sortIcon, goPage, reprocess, remove, onUploaded } = useHooks()

const confirmRow = ref<DocumentRow | null>(null)
const pageDragging = ref(false)

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / form.page_size)))
const pages = computed(() => {
    const tp = totalPages.value
    const cur = form.page
    const wanted = new Set([1, 2, cur - 1, cur, cur + 1, tp - 1, tp].filter((p) => p >= 1 && p <= tp))
    const sorted = [...wanted].sort((a, b) => a - b)
    const out: (number | '…')[] = []
    let prev = 0
    for (const p of sorted) {
        if (prev && p - prev > 1) out.push('…')
        out.push(p)
        prev = p
    }
    return out
})

const topbarMeta = computed(() => {
    const q = stats.value.queued ?? 0
    const p = stats.value.processing ?? 0
    return q || p ? `${q} queued · ${p} processing` : undefined
})

const PRESETS = [
    { label: 'Today', days: 0 },
    { label: '2 days', days: 1 },
    { label: '7 days', days: 6 },
    { label: 'All', days: null as number | null },
]

function applyPreset(days: number | null): void {
    if (days === null) { form.date_from = ''; form.date_to = '' } else {
        const to = new Date()
        const from = new Date(Date.now() - days * 86400000)
        form.date_from = from.toISOString().slice(0, 10)
        form.date_to = to.toISOString().slice(0, 10)
    }
    getDataSearch()
}

function presetActive(days: number | null): boolean {
    if (days === null) return !form.date_from && !form.date_to
    const expected = new Date(Date.now() - days * 86400000).toISOString().slice(0, 10)
    return form.date_from === expected
}

/** Quality flags come back as strings; 'good'/'REAL' both mean healthy. */
const GOOD = new Set(['good', 'REAL', 'real', 'ok'])
function qualityClass(value: unknown): string {
    if (value == null) return 'q-unknown'
    return GOOD.has(String(value)) ? 'q-good' : 'q-bad'
}

const QUALITY_KEYS: [string, string][] = [
    ['Glare', 'G'], ['Blur', 'B'], ['PrintSpoofing', 'P'], ['LCDSpoofing', 'L'],
]

// Drag anywhere on the page opens the upload modal, so the affordance is
// discoverable without a permanent dropzone breaking the layout rhythm.
function onPageDragOver(e: DragEvent): void {
    if (e.dataTransfer?.types?.includes('Files')) { pageDragging.value = true; uploadOpen.value = true }
}
onMounted(() => document.addEventListener('dragover', onPageDragOver))
onUnmounted(() => document.removeEventListener('dragover', onPageDragOver))

function download(row: DocumentRow): void {
    void $api.documents.get(row.id) // no-op guard so the row exists
    window.open(`/api/v1/documents/${row.id}/image/original`, '_blank')
}
</script>

<template>
  <AppTopbar :meta="topbarMeta">
    <template #actions>
      <button class="btn btn-primary" @click="uploadOpen = true">Upload document</button>
      <button class="btn btn-outline" :disabled="loading" @click="getData()">Refresh</button>
    </template>
  </AppTopbar>

  <div class="content">
    <div class="filterbar">
      <div class="search-wrap">
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" />
        </svg>
        <input v-model="form.search" class="input search-input"
               placeholder="Search by filename or recognised text…" @input="getDataSearch()">
      </div>

      <select v-model="form.status" class="select" @change="getDataSearch()">
        <option value="">All statuses</option>
        <option value="queued">Queued</option>
        <option value="processing">Processing</option>
        <option value="done">Done</option>
        <option value="failed">Failed</option>
      </select>

      <select v-model="form.doc_type" class="select" @change="getDataSearch()">
        <option v-for="t in DOC_TYPES" :key="t.value" :value="t.value">{{ t.label }}</option>
      </select>

      <div class="date-range">
        <button v-for="p in PRESETS" :key="p.label"
                :class="['preset-btn', { active: presetActive(p.days) }]"
                @click="applyPreset(p.days)">{{ p.label }}</button>
        <input v-model="form.date_from" type="date" lang="en-GB" class="input date-input" @change="getDataSearch()">
        <span class="date-sep">—</span>
        <input v-model="form.date_to" type="date" lang="en-GB" class="input date-input" @change="getDataSearch()">
      </div>

      <span class="total-count">{{ total }} document{{ total === 1 ? '' : 's' }}</span>
    </div>

    <div class="card u-overflow-hidden">
      <table class="table">
        <thead>
          <tr>
            <th style="width:64px"></th>
            <th class="th-sort" style="width:180px" @click="setSort('created_at')">
              Uploaded <span class="si">{{ sortIcon('created_at') }}</span>
            </th>
            <th class="th-sort" style="width:160px" @click="setSort('doc_type')">
              Type <span class="si">{{ sortIcon('doc_type') }}</span>
            </th>
            <th class="th-sort" style="width:150px" @click="setSort('status')">
              Status <span class="si">{{ sortIcon('status') }}</span>
            </th>
            <th style="width:180px">Quality</th>
            <th style="width:70px">Fields</th>
            <th class="th-sort" style="width:90px" @click="setSort('processing_ms')">
              Time <span class="si">{{ sortIcon('processing_ms') }}</span>
            </th>
            <th style="width:120px"></th>
          </tr>
        </thead>

        <tbody v-if="loading && !documents.length">
          <tr v-for="i in 3" :key="i">
            <td><div class="skel" style="width:56px;height:38px;border-radius:4px"></div></td>
            <td colspan="6"><div class="skel skel-text"></div></td>
            <td><div class="skel skel-pill"></div></td>
          </tr>
        </tbody>

        <tbody v-else-if="!documents.length">
          <tr>
            <td colspan="8" style="padding:0">
              <div class="dz" style="margin:20px" @click="uploadOpen = true">
                <div class="dz-main">No documents yet</div>
                <div class="dz-sub">Drop an image anywhere on this page, or click to browse</div>
              </div>
            </td>
          </tr>
        </tbody>

        <tbody v-else :class="{ 'u-opacity-50': loading }">
          <tr v-for="row in documents" :key="row.id">
            <td>
              <router-link :to="`/documents/${row.id}`">
                <AuthedImage v-if="row.has_canvas"
                             :url="`/documents/${row.id}/image/thumb`"
                             :alt="row.filename" img-class="thumb" />
                <div v-else class="thumb thumb-fallback">{{ row.status === 'failed' ? '!' : '…' }}</div>
              </router-link>
            </td>
            <td>
              <div class="dval">{{ formatDate(row.created_at) }}</div>
              <div class="dtime">{{ formatTime(row.created_at) }}</div>
              <div class="row-sub" :title="row.filename">{{ row.filename }}</div>
            </td>
            <td>
              <template v-if="row.recognised">
                <span class="row-title">{{ row.doc_type_base }}</span>
                <span v-if="row.doc_type_era" class="era-chip">{{ row.doc_type_era }}</span>
              </template>
              <span v-else class="u-dash">not recognised</span>
              <div class="row-sub">{{ formatBytes(row.size_bytes) }}</div>
            </td>
            <td>
              <span :class="['badge', 'badge-' + row.status]" :title="row.error ?? ''">
                <span class="bdot"></span>{{ row.status }}
              </span>
              <div v-if="row.retry_count" class="row-sub">retries: {{ row.retry_count }}</div>
            </td>
            <td>
              <!-- Explicit two-row flex, and the confidence line is always
                   present. As loose inline content the pills sat on the text
                   baseline, so the number drifted up or down depending on how
                   tall the rest of the row happened to be. -->
              <div v-if="row.status === 'done'" class="q-cell">
                <div class="q-pills">
                  <span v-for="[key, letter] in QUALITY_KEYS" :key="key"
                        :class="['q-pill', qualityClass(row.quality?.[key] ?? null)]"
                        :title="`${key}: ${row.quality?.[key] ?? 'unknown'}`">{{ letter }}</span>
                </div>
                <div class="row-sub u-mono" title="Document type confidence">
                  {{ row.doc_conf != null ? row.doc_conf.toFixed(2) : '—' }}
                </div>
              </div>
              <span v-else class="u-dash">—</span>
            </td>
            <td class="u-mono">{{ row.field_count || '—' }}</td>
            <td class="u-mono">{{ formatMs(row.processing_ms) }}</td>
            <td>
              <div class="act-row">
                <router-link :to="`/documents/${row.id}`" class="act-btn" title="View">
                  <Icon name="eye" />
                </router-link>
                <button v-if="row.status === 'done' || row.status === 'failed'" class="act-btn"
                        title="Reprocess" @click="reprocess(row)"><Icon name="refresh" /></button>
                <button class="act-btn" title="Download original" @click="download(row)">
                  <Icon name="download" />
                </button>
                <button class="act-btn" title="Delete" @click="confirmRow = row">
                  <Icon name="trash" />
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>

      <div v-if="totalPages > 1" class="pagination">
        <button class="pg-btn" :disabled="form.page <= 1" @click="goPage(form.page - 1)">‹</button>
        <template v-for="(p, i) in pages" :key="i">
          <span v-if="p === '…'" class="pg-ellipsis">…</span>
          <button v-else :class="['pg-btn', { active: p === form.page }]" @click="goPage(p as number)">
            {{ p }}
          </button>
        </template>
        <button class="pg-btn" :disabled="form.page >= totalPages" @click="goPage(form.page + 1)">›</button>
      </div>
    </div>

    <div class="ephemeral-note" style="margin-top:12px">
      Documents are stored in ephemeral storage and are cleared when the service restarts.
    </div>
  </div>

  <UploadModal :open="uploadOpen" @CLOSE="uploadOpen = false" @UPLOADED="onUploaded()" />
  <ConfirmDialog :open="!!confirmRow" title="Delete document" danger confirm-label="Delete"
                 :message="`Delete ${confirmRow?.filename ?? ''}? This cannot be undone.`"
                 @CONFIRM="remove(confirmRow!); confirmRow = null" @CANCEL="confirmRow = null" />
</template>

<style scoped>
.filterbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:14px;}
.search-wrap{position:relative;flex:1;min-width:240px;}
.search-icon{position:absolute;left:11px;top:50%;transform:translateY(-50%);
  width:16px;height:16px;color:var(--color-text-muted);pointer-events:none;}
.search-input{width:100%;padding-left:34px;}
.date-range{display:flex;align-items:center;gap:6px;}
.date-input{width:140px;}
.date-sep{color:var(--color-text-muted);}
.preset-btn{padding:0 10px;height:32px;border:1px solid var(--color-border);border-radius:var(--radius-btn);
  background:var(--color-card);font-family:inherit;font-size:12px;font-weight:600;
  color:var(--color-text-sub);cursor:pointer;}
.preset-btn.active{background:var(--color-primary);color:#fff;border-color:var(--color-primary);}
.total-count{margin-left:auto;font-size:12px;color:var(--color-text-muted);font-weight:600;}
.dval{font-size:13px;font-weight:600;color:var(--color-text);}
.dtime{font-size:11px;color:var(--color-text-muted);font-variant-numeric:tabular-nums;}
.row-title{font-size:13px;font-weight:600;color:var(--color-text);}
.row-sub{font-size:11px;color:var(--color-text-muted);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;max-width:170px;}
.th-sort{cursor:pointer;user-select:none;}
.si{color:var(--color-text-muted);font-size:11px;}
.act-row{display:flex;gap:4px;}
.act-btn{width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;
  border:1px solid var(--color-border);border-radius:6px;background:var(--color-card);
  cursor:pointer;font-size:13px;text-decoration:none;color:var(--color-text-sub);}
.act-btn:hover{border-color:var(--color-primary);color:var(--color-primary);
  background:var(--color-primary-light);}
.pg-ellipsis{padding:0 6px;color:var(--color-text-muted);}
</style>
