<script setup lang="ts">
/**
 * Recognise one image without storing it (POST /recognize).
 *
 * The answer is shown once, here, and nowhere else: the server kept nothing, so
 * there is no page to come back to. Everything this component holds — the file,
 * the preview, the decoded canvas, the recognised text — is dropped and its object
 * URLs revoked when the dialog closes or another image is chosen. Nothing goes to
 * the store, to localStorage or to the URL.
 */
import { computed, onBeforeUnmount, ref, shallowRef } from 'vue'
import { useStore } from 'vuex'
import $api from '@/api'
import BoxOverlay from '@/views/pages/document-detail/components/BoxOverlay.vue'
import { formatMs } from '@/utils/format'
import type { RecognizeResult } from '@/types'

const MAX_MB = 20
const ACCEPT = 'image/jpeg,image/png,image/webp,image/bmp,image/tiff'
const QUALITY_KEYS = ['Glare', 'Blur', 'PrintSpoofing', 'LCDSpoofing']
const GOOD = new Set(['good', 'REAL', 'real', 'ok'])

defineProps<{ open: boolean }>()
const emit = defineEmits<{ CLOSE: []; DONE: [] }>()

const store = useStore()
const file = shallowRef<File | null>(null)
const previewUrl = ref<string | null>(null)
const canvasUrl = ref<string | null>(null)
const result = shallowRef<RecognizeResult | null>(null)
const dragging = ref(false)
const running = ref(false)
const progress = ref(0)
const input = ref<HTMLInputElement | null>(null)

const EMPTY = new Set<string>()
const quality = computed(() => QUALITY_KEYS
    .filter((k) => result.value?.quality?.[k] != null)
    .map((k) => ({ key: k, value: String(result.value!.quality[k]),
                   good: GOOD.has(String(result.value!.quality[k])) })))

function revoke(): void {
    if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
    if (canvasUrl.value) URL.revokeObjectURL(canvasUrl.value)
    previewUrl.value = null
    canvasUrl.value = null
}

/** Drop everything this dialog knows about the document. */
function reset(): void {
    revoke()
    file.value = null
    result.value = null
    progress.value = 0
    running.value = false
    if (input.value) input.value.value = ''
}

function pick(selected: FileList | null): void {
    if (!selected || selected.length === 0) return
    const chosen = selected[0]
    if (!chosen.type.startsWith('image/')) {
        store.dispatch('ui/toast', { kind: 'error', title: 'Unsupported file',
                                     message: 'Choose a JPEG, PNG, WEBP, BMP or TIFF image.' })
        return
    }
    if (chosen.size > MAX_MB * 1024 * 1024) {
        store.dispatch('ui/toast', { kind: 'error', title: 'File too large', message: `The limit is ${MAX_MB} MB.` })
        return
    }
    reset()
    file.value = chosen
    previewUrl.value = URL.createObjectURL(chosen)
}

function onDrop(event: DragEvent): void {
    dragging.value = false
    pick(event.dataTransfer?.files ?? null)
}

function canvasBlob(b64: string): Blob {
    const raw = atob(b64)
    const bytes = new Uint8Array(raw.length)
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i)
    return new Blob([bytes], { type: 'image/png' })
}

async function send(): Promise<void> {
    if (!file.value) return
    running.value = true
    try {
        const res = await $api.recognize.run(file.value, true, (p) => { progress.value = p })
        const b64 = res.canvas?.image_png_base64
        if (b64) {
            canvasUrl.value = URL.createObjectURL(canvasBlob(b64))
            delete res.canvas.image_png_base64      // keep one copy, the blob
        }
        // The original is no longer needed once there is an answer.
        if (previewUrl.value) { URL.revokeObjectURL(previewUrl.value); previewUrl.value = null }
        file.value = null
        result.value = res
        emit('DONE')
    } catch {
        emit('DONE')                                 // a refusal is journalled too
    } finally {
        running.value = false
    }
}

function close(): void { reset(); emit('CLOSE') }

onBeforeUnmount(reset)
</script>

<template>
  <div v-if="open" class="modal-overlay" @click.self="!running && close()">
    <div class="modal" :style="{ maxWidth: result ? '1100px' : '520px' }">
      <div class="modal-head">Recognise without storing</div>

      <div class="modal-body">
        <div class="once-note">
          Nothing is stored on the server: the image is recognised in memory and the
          answer is shown <strong>once</strong>, here. Closing this dialog discards it.
        </div>

        <template v-if="!result">
          <div v-if="!file" :class="['dz', { drag: dragging }]"
               @click="input?.click()"
               @dragover.prevent="dragging = true"
               @dragleave.prevent="dragging = false"
               @drop.prevent="onDrop">
            <div class="dz-main">Drop an image here, or click to browse</div>
            <div class="dz-sub">JPEG, PNG, WEBP, BMP or TIFF · up to {{ MAX_MB }} MB · one file</div>
          </div>

          <div v-else class="upload-preview">
            <img v-if="previewUrl" :src="previewUrl" alt="Selected document">
            <div class="upload-meta">
              <div class="upload-size">{{ (file.size / 1024).toFixed(0) }} KB</div>
              <button v-if="!running" class="btn btn-ghost btn-sm" @click="reset()">Choose another</button>
              <template v-else>
                <div class="pbar"><span :style="{ width: progress + '%' }"></span></div>
                <div class="upload-size">{{ progress < 100 ? 'Sending…' : 'Recognising…' }}</div>
              </template>
            </div>
          </div>
        </template>

        <div v-else class="res">
          <div class="res-view">
            <BoxOverlay v-if="canvasUrl" :src="canvasUrl" :image-url="null"
                        :width="result.canvas.width" :height="result.canvas.height"
                        :boxes="result.boxes" :address="result.address"
                        :active-keys="EMPTY" :any-active="false" :show-labels="false"
                        :hidden-labels="EMPTY" />
            <div v-else class="u-muted">No canvas was produced for this image.</div>
          </div>

          <div class="res-side">
            <div class="res-head">
              <template v-if="result.recognised">
                <span class="res-type">{{ result.doc_type_base }}</span>
                <span v-if="result.doc_type_era" class="era-chip">{{ result.doc_type_era }}</span>
              </template>
              <span v-else class="res-type u-dash">not recognised</span>
            </div>
            <div class="res-meta u-mono">
              conf {{ result.doc_conf != null ? result.doc_conf.toFixed(2) : '—' }}
              · {{ formatMs(result.processing_ms) }} · {{ result.device ?? '—' }}
            </div>
            <div v-if="quality.length" class="q-pills" style="margin:8px 0 12px">
              <span v-for="q in quality" :key="q.key" :class="['q-pill', q.good ? 'q-good' : 'q-bad']"
                    :title="`${q.key}: ${q.value}`">{{ q.key[0] }}</span>
            </div>

            <table v-if="result.fields.length" class="table">
              <thead><tr><th style="width:150px">Field</th><th>Value</th><th style="width:56px">Conf</th></tr></thead>
              <tbody>
                <tr v-for="f in result.fields" :key="f.name">
                  <td class="fld-name">{{ f.display }}</td>
                  <td>
                    <span v-if="f.value" :class="{ 'u-mono': f.script === 'num' }"
                          :lang="f.script === 'ru' ? 'ru' : 'en'">{{ f.value }}</span>
                    <span v-else class="u-dash">—</span>
                    <div v-if="f.normalized" class="fld-norm u-mono">{{ f.normalized }}</div>
                  </td>
                  <td class="u-mono">{{ f.conf != null ? f.conf.toFixed(2) : '—' }}</td>
                </tr>
              </tbody>
            </table>

            <table v-if="result.address?.lines.length" class="table" style="margin-top:12px">
              <thead><tr><th>Address line</th><th style="width:100px">Kind</th></tr></thead>
              <tbody>
                <tr v-for="line in result.address.lines" :key="line.id">
                  <td lang="ru">{{ line.text || '—' }}</td>
                  <td class="u-muted">{{ line.kind }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <input ref="input" type="file" :accept="ACCEPT" hidden
               @change="pick(($event.target as HTMLInputElement).files)">
      </div>

      <div class="modal-foot">
        <template v-if="result">
          <button class="btn btn-ghost" @click="reset()">Recognise another</button>
          <button class="btn btn-primary" @click="close()">Close and discard</button>
        </template>
        <template v-else>
          <button class="btn btn-ghost" :disabled="running" @click="close()">Cancel</button>
          <button class="btn btn-primary" :disabled="!file || running" @click="send()">
            {{ running ? 'Recognising…' : 'Recognise' }}
          </button>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.once-note{font-size:12px;color:var(--color-text-sub);background:var(--color-primary-light);
  border-radius:6px;padding:8px 12px;margin-bottom:14px;}
.upload-preview{display:flex;gap:14px;align-items:center;}
.upload-preview img{max-width:180px;max-height:140px;border-radius:6px;border:1px solid var(--color-border);}
.upload-meta{display:flex;flex-direction:column;gap:8px;flex:1;}
.upload-size{font-size:12px;color:var(--color-text-muted);}
.res{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px;}
.res-view{border:1px solid var(--color-border-light);border-radius:8px;overflow:hidden;}
.res-side{max-height:70vh;overflow:auto;}
.res-head{display:flex;align-items:center;gap:8px;}
.res-type{font-size:15px;font-weight:700;color:var(--color-text);}
.res-meta{font-size:11px;color:var(--color-text-muted);margin-top:4px;}
.fld-name{font-size:12px;font-weight:600;color:var(--color-text-sub);}
.fld-norm{font-size:11px;color:var(--color-text-muted);}
@media (max-width: 800px){ .res{grid-template-columns:1fr;} }
</style>
