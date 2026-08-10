<script setup lang="ts">
import { ref } from 'vue'
import { useStore } from 'vuex'
import $api from '@/api'

const MAX_MB = 20
const ACCEPT = 'image/jpeg,image/png,image/webp,image/bmp,image/tiff'

defineProps<{ open: boolean }>()
const emit = defineEmits<{ CLOSE: []; UPLOADED: [] }>()

const store = useStore()
const file = ref<File | null>(null)
const previewUrl = ref<string | null>(null)
const dragging = ref(false)
const uploading = ref(false)
const progress = ref(0)
const input = ref<HTMLInputElement | null>(null)

function pick(selected: FileList | null): void {
    if (!selected || selected.length === 0) return
    if (selected.length > 1) {
        store.dispatch('ui/toast', {
            kind: 'info', title: 'One file at a time',
            message: `Using ${selected[0].name}, ignored ${selected.length - 1} other file(s).`,
        })
    }
    const chosen = selected[0]
    if (!chosen.type.startsWith('image/')) {
        store.dispatch('ui/toast', {
            kind: 'error', title: 'Unsupported file',
            message: 'Upload a JPEG, PNG, WEBP, BMP or TIFF image.',
        })
        return
    }
    if (chosen.size > MAX_MB * 1024 * 1024) {
        store.dispatch('ui/toast', {
            kind: 'error', title: 'File too large', message: `The limit is ${MAX_MB} MB.`,
        })
        return
    }
    if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
    file.value = chosen
    // Local file, no auth needed — unlike the served artifacts.
    previewUrl.value = URL.createObjectURL(chosen)
}

function onDrop(event: DragEvent): void {
    dragging.value = false
    pick(event.dataTransfer?.files ?? null)
}

function reset(): void {
    if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
    file.value = null
    progress.value = 0
    uploading.value = false
}

function close(): void { reset(); emit('CLOSE') }

async function send(): Promise<void> {
    if (!file.value) return
    uploading.value = true
    try {
        await $api.documents.upload(file.value, (p) => { progress.value = p })
        store.dispatch('ui/toast', {
            kind: 'success', title: 'Queued', message: file.value.name,
        })
        reset()
        emit('UPLOADED')
    } catch {
        uploading.value = false   // the interceptor already surfaced the error
    }
}
</script>

<template>
  <div v-if="open" class="modal-overlay" @click.self="close()">
    <div class="modal" style="max-width:520px">
      <div class="modal-head">Upload document</div>
      <div class="modal-body">
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
            <div class="upload-name">{{ file.name }}</div>
            <div class="upload-size">{{ (file.size / 1024).toFixed(0) }} KB</div>
            <button v-if="!uploading" class="btn btn-ghost btn-sm" @click="reset()">Choose another</button>
            <div v-else class="pbar"><span :style="{ width: progress + '%' }"></span></div>
          </div>
        </div>

        <input ref="input" type="file" :accept="ACCEPT" hidden
               @change="pick(($event.target as HTMLInputElement).files)">
      </div>
      <div class="modal-foot">
        <button class="btn btn-ghost" :disabled="uploading" @click="close()">Cancel</button>
        <button class="btn btn-primary" :disabled="!file || uploading" @click="send()">
          {{ uploading ? 'Uploading…' : 'Upload' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.upload-preview{display:flex;gap:14px;align-items:center;}
.upload-preview img{width:120px;height:88px;object-fit:cover;border-radius:8px;
  border:1px solid var(--color-border);}
.upload-meta{flex:1;min-width:0;}
.upload-name{font-size:13px;font-weight:600;color:var(--color-text);word-break:break-all;}
.upload-size{font-size:12px;color:var(--color-text-muted);margin:2px 0 8px;}
.pbar{height:6px;background:var(--color-row-alt);border-radius:3px;overflow:hidden;}
.pbar span{display:block;height:100%;background:var(--color-primary);transition:width 120ms ease;}
</style>
