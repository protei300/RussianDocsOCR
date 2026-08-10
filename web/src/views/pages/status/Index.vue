<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import AppTopbar from '@/components/AppTopbar.vue'
import $api from '@/api'
import { formatMs, formatUptime } from '@/utils/format'

const data = ref<any>(null)
let timer: ReturnType<typeof setInterval> | null = null

function load(): void { $api.status.get().then((d) => { data.value = d }).catch(() => {}) }

const gpu = computed(() => data.value?.gpu ?? null)
const compute = computed(() => data.value?.compute ?? {})
const server = computed(() => data.value?.server ?? {})
const service = computed(() => data.value?.service ?? {})

const ramPct = computed(() => server.value.ram_total_gb
    ? (server.value.ram_used_gb / server.value.ram_total_gb) * 100 : 0)
const diskPct = computed(() => server.value.disk_total_gb
    ? (server.value.disk_used_gb / server.value.disk_total_gb) * 100 : 0)
const vramPct = computed(() => gpu.value?.vram_total_gb
    ? (gpu.value.vram_used_gb / gpu.value.vram_total_gb) * 100 : 0)

const CIRCUMFERENCE = 314.16
const gaugeOffset = computed(() =>
    CIRCUMFERENCE * (1 - (gpu.value?.utilization_pct ?? 0) / 100))
const gaugeColour = computed(() => {
    const v = gpu.value?.utilization_pct ?? 0
    return v < 60 ? 'var(--color-green)' : v < 85 ? 'var(--color-accent)' : 'var(--color-red)'
})

/** The recognition device and the OCR device genuinely differ — see below. */
const computeSummary = computed(() => {
    const c = compute.value
    if (c.state !== 'ready') return c.state === 'error' ? `Error: ${c.error}` : 'Loading models…'
    const head = c.device === 'gpu' ? `GPU (${c.providers?.[0] ?? 'CUDA'})` : 'CPU only'
    return `${head} · OCR on ${String(c.ocr_device ?? '').toUpperCase()}`
})

onMounted(() => { load(); timer = setInterval(load, 5000) })
onUnmounted(() => { if (timer) clearInterval(timer) })
</script>

<template>
  <AppTopbar meta="Live · refreshed every 5s">
    <template #actions><button class="btn btn-outline btn-sm" @click="load()">Refresh</button></template>
  </AppTopbar>

  <div class="content">
    <div class="metric-grid">
      <div class="card card-pad metric-card">
        <div class="label">CPU</div>
        <div class="value">{{ server.cpu_pct ?? '—' }}<span class="unit">%</span></div>
        <div class="sub">{{ server.cpu_name ?? '' }}</div>
        <div class="sub">{{ server.cpu_cores ?? '?' }} cores · {{ server.cpu_threads ?? '?' }} threads</div>
        <div class="pbar"><span :style="{ width: (server.cpu_pct ?? 0) + '%' }"></span></div>
      </div>

      <div class="card card-pad metric-card">
        <div class="label">RAM</div>
        <div class="value">{{ server.ram_used_gb ?? '—' }}<span class="unit">/ {{ server.ram_total_gb ?? '—' }} GB</span></div>
        <div class="sub">{{ ramPct.toFixed(0) }}% used</div>
        <div class="pbar"><span class="green" :style="{ width: ramPct + '%' }"></span></div>
      </div>

      <div class="card card-pad metric-card">
        <div class="label">Disk</div>
        <div class="value">{{ diskPct.toFixed(0) }}<span class="unit">%</span></div>
        <div class="sub">{{ server.disk_used_gb ?? '—' }} / {{ server.disk_total_gb ?? '—' }} GB</div>
        <div class="pbar"><span :style="{ width: diskPct + '%' }"></span></div>
      </div>

      <div class="card card-pad metric-card">
        <div class="label">Uptime</div>
        <div class="value">{{ formatUptime(service.uptime_sec ?? 0) }}</div>
        <div class="sub">version {{ service.version ?? '—' }}</div>
        <div class="sub">
          <span class="live-dot"></span>{{ compute.state === 'ready' ? 'worker running' : compute.state }}
        </div>
      </div>
    </div>

    <div class="row-3">
      <div class="card card-pad">
        <div class="card-head" style="padding:0 0 12px">GPU utilisation</div>
        <template v-if="gpu">
          <svg class="gauge" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="50" fill="none" stroke="var(--color-border)" stroke-width="10" />
            <circle cx="60" cy="60" r="50" fill="none" :stroke="gaugeColour" stroke-width="10"
                    stroke-linecap="round" :stroke-dasharray="CIRCUMFERENCE"
                    :stroke-dashoffset="gaugeOffset" transform="rotate(-90 60 60)" />
            <text x="60" y="66" text-anchor="middle" class="gauge-text">{{ gpu.utilization_pct }}%</text>
          </svg>
          <div class="sub" style="text-align:center">{{ gpu.name }}</div>
          <div class="sub" style="text-align:center">{{ gpu.temperature_c }}°C</div>
        </template>
        <div v-else class="u-dash">No GPU detected</div>
      </div>

      <div class="card card-pad metric-card">
        <div class="label">VRAM</div>
        <template v-if="gpu">
          <div class="value">{{ gpu.vram_used_gb }}<span class="unit">/ {{ gpu.vram_total_gb }} GB</span></div>
          <div class="pbar"><span :style="{ width: vramPct + '%' }"></span></div>
        </template>
        <div v-else class="u-dash">—</div>
      </div>

      <div class="card card-pad">
        <div class="card-head" style="padding:0 0 12px">Throughput</div>
        <div class="kv-list">
          <div class="kv-row"><span class="kv-key">Documents</span><span class="kv-val u-mono">{{ service.documents_total ?? 0 }}</span></div>
          <div class="kv-row"><span class="kv-key">Recognised</span><span class="kv-val u-mono">{{ service.recognised ?? 0 }}</span></div>
          <div class="kv-row"><span class="kv-key">Failed</span><span class="kv-val u-mono">{{ service.documents_failed ?? 0 }}</span></div>
          <div class="kv-row"><span class="kv-key">Average time</span><span class="kv-val u-mono">{{ formatMs(service.avg_processing_ms ?? null) }}</span></div>
        </div>
      </div>
    </div>

    <div class="card card-pad" style="margin-top:16px">
      <div class="card-head" style="padding:0 0 12px">Recognition runtime</div>
      <div class="kv-list">
        <div class="kv-row"><span class="kv-key">State</span><span class="kv-val">{{ computeSummary }}</span></div>
        <!-- Reported separately on purpose: with GPU detectors the OCR engines
             still run on CPU, because dynamic-width per-word calls are slower
             on CUDA. Saying just "GPU" here would be misleading. -->
        <div class="kv-row"><span class="kv-key">Detectors</span><span class="kv-val u-mono">{{ compute.device ?? '—' }}</span></div>
        <div class="kv-row"><span class="kv-key">OCR engines</span><span class="kv-val u-mono">{{ compute.ocr_device ?? '—' }}</span></div>
        <div class="kv-row"><span class="kv-key">Providers</span><span class="kv-val u-mono">{{ (compute.providers ?? []).join(', ') || '—' }}</span></div>
        <div class="kv-row"><span class="kv-key">Model format / OCR tier</span><span class="kv-val u-mono">{{ compute.model_format }} / {{ compute.ocr_mode }}</span></div>
        <div class="kv-row"><span class="kv-key">Library version</span><span class="kv-val u-mono">{{ compute.library_version ?? '—' }}</span></div>
        <div class="kv-row"><span class="kv-key">Model load / warmup</span><span class="kv-val u-mono">{{ compute.load_ms ?? '—' }} ms / {{ compute.warmup_ms ?? '—' }} ms</span></div>
        <div v-if="compute.fell_back" class="kv-row">
          <span class="kv-key">Fallback</span>
          <span class="badge badge-failed">GPU requested but unavailable</span>
        </div>
      </div>
    </div>

    <div class="card card-pad" style="margin-top:16px">
      <div class="card-head" style="padding:0 0 12px">Queue &amp; storage</div>
      <div class="kv-list">
        <div class="kv-row"><span class="kv-key">Queued</span><span class="kv-val u-mono">{{ service.documents_queued ?? 0 }}</span></div>
        <div class="kv-row"><span class="kv-key">Processing</span><span class="kv-val u-mono">{{ service.documents_processing ?? 0 }}</span></div>
        <div class="kv-row"><span class="kv-key">Data directory</span><span class="kv-val u-mono">{{ service.data_dir_mb ?? 0 }} MB</span></div>
        <div class="kv-row">
          <span class="kv-key">Persistence</span>
          <span class="kv-val">{{ service.data_is_ephemeral ? 'Cleared on restart' : 'Retained' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;}
.row-3{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin-top:16px;}
.metric-card .label{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:var(--color-text-muted);}
.metric-card .value{font-size:30px;font-weight:700;color:var(--color-text);
  font-variant-numeric:tabular-nums;margin:4px 0;}
.metric-card .unit{font-size:13px;font-weight:600;color:var(--color-text-muted);margin-left:5px;}
.sub{font-size:12px;color:var(--color-text-muted);}
.pbar{height:6px;background:var(--color-row-alt);border-radius:3px;overflow:hidden;margin-top:10px;}
.pbar span{display:block;height:100%;background:var(--color-primary);}
.pbar span.green{background:var(--color-green);}
.gauge{width:130px;height:130px;display:block;margin:0 auto 8px;}
.gauge-text{font-size:22px;font-weight:700;fill:var(--color-text);font-family:inherit;}
.live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--color-green);
  margin-right:6px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.kv-list{display:flex;flex-direction:column;gap:8px;}
.kv-row{display:flex;justify-content:space-between;align-items:center;font-size:13px;gap:12px;}
.kv-key{color:var(--color-text-sub);}
.kv-val{font-weight:600;color:var(--color-text);text-align:right;}
</style>
