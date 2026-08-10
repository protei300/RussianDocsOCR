<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import AppTopbar from '@/components/AppTopbar.vue'
import $api from '@/api'

interface Entry { ts: number; level: string; logger: string; message: string; exc: string | null }

const entries = ref<Entry[]>([])
const level = ref('')
const search = ref('')
const paused = ref(false)
let timer: ReturnType<typeof setInterval> | null = null

function load(): void {
    if (paused.value) return
    // Skip polling in a hidden tab — the ring buffer is still there when we return.
    if (document.visibilityState !== 'visible') return
    $api.logs.get({ n: 400, level: level.value || undefined, search: search.value || undefined })
        .then((res) => { entries.value = res.entries })
        .catch(() => {})
}

function stamp(ts: number): string {
    return new Date(ts * 1000).toLocaleTimeString('en-US', { hour12: false })
}

onMounted(() => { load(); timer = setInterval(load, 5000) })
onUnmounted(() => { if (timer) clearInterval(timer) })
</script>

<template>
  <AppTopbar :meta="`${entries.length} lines`">
    <template #actions>
      <button class="btn btn-outline btn-sm" @click="paused = !paused">
        {{ paused ? 'Resume' : 'Pause' }}
      </button>
      <button class="btn btn-outline btn-sm" @click="load()">Refresh</button>
    </template>
  </AppTopbar>

  <div class="content">
    <div class="filterbar">
      <select v-model="level" class="select" @change="load()">
        <option value="">All levels</option>
        <option value="DEBUG">Debug+</option>
        <option value="INFO">Info+</option>
        <option value="WARNING">Warning+</option>
        <option value="ERROR">Error</option>
      </select>
      <input v-model="search" class="input" style="flex:1;min-width:220px"
             placeholder="Filter messages…" @input="load()">
    </div>

    <div class="card u-overflow-hidden">
      <div class="log-pane">
        <div v-for="(e, i) in entries" :key="i" class="log-line" :class="'lv-' + e.level">
          <span class="log-ts u-mono">{{ stamp(e.ts) }}</span>
          <span class="log-lv">{{ e.level }}</span>
          <span class="log-msg">{{ e.message }}</span>
        </div>
        <div v-if="!entries.length" class="u-dash" style="padding:20px;text-align:center">
          No log entries match.
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.filterbar{display:flex;gap:10px;margin-bottom:14px;}
.log-pane{max-height:calc(100vh - 240px);overflow:auto;padding:10px 0;}
.log-line{display:flex;gap:10px;padding:3px 16px;font-size:12px;line-height:1.5;
  font-family:'JetBrains Mono',monospace;}
.log-line:hover{background:var(--color-row-alt);}
.log-ts{color:var(--color-text-muted);flex-shrink:0;}
.log-lv{width:64px;flex-shrink:0;font-weight:600;color:var(--color-text-muted);}
.log-msg{color:var(--color-text-sub);word-break:break-word;}
.lv-WARNING .log-lv,.lv-WARNING .log-msg{color:var(--color-accent);}
.lv-ERROR .log-lv,.lv-ERROR .log-msg,.lv-CRITICAL .log-lv{color:var(--color-red);}
</style>
