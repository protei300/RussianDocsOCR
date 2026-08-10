<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useStore } from 'vuex'
import AppTopbar from '@/components/AppTopbar.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'
import Icon from '@/components/Icon.vue'
import $api from '@/api'
import { formatDate, formatTime } from '@/utils/format'
import type { ApiKeyRow } from '@/types'

const store = useStore()
const keys = ref<ApiKeyRow[]>([])
const note = ref('')
const loading = ref(false)

const createOpen = ref(false)
const label = ref('')
const creating = ref(false)
/** The plaintext key, shown once. Only its hash is stored server-side. */
const created = ref<string | null>(null)
const confirmRow = ref<ApiKeyRow | null>(null)
const generatedDefault = computed(() => keys.value.some((k) => k.is_default && k.is_generated))

function load(): void {
    loading.value = true
    $api.apiKeys.list()
        .then((res) => { keys.value = res.items; note.value = res.note })
        .finally(() => { loading.value = false })
}

function create(): void {
    creating.value = true
    $api.apiKeys.create(label.value)
        .then((res) => { created.value = res.key; label.value = ''; load() })
        .finally(() => { creating.value = false })
}

function copyValue(value: string): void {
    void navigator.clipboard.writeText(value)
    store.dispatch('ui/toast', { kind: 'success', title: 'Copied', message: 'API key copied' })
}

function copyKey(): void {
    if (!created.value) return
    void navigator.clipboard.writeText(created.value)
    store.dispatch('ui/toast', { kind: 'success', title: 'Copied', message: 'API key copied' })
}

function closeCreate(): void {
    createOpen.value = false
    created.value = null
    label.value = ''
}

function remove(row: ApiKeyRow): void {
    $api.apiKeys.remove(row.id).then(() => {
        store.dispatch('ui/toast', { kind: 'success', title: 'Revoked', message: row.label })
        load()
    })
}

onMounted(load)
</script>

<template>
  <AppTopbar>
    <template #actions>
      <button class="btn btn-primary" @click="createOpen = true">New key</button>
    </template>
  </AppTopbar>

  <div class="content">
    <div class="ephemeral-note" style="margin-bottom:14px">{{ note }}</div>

    <div v-if="generatedDefault" class="danger-card" style="margin-bottom:14px">
      <strong>DEFAULT_API_KEY is not configured.</strong>
      The default key below was generated for this run and changes on every restart,
      so any integration using it will break. Set <code>DEFAULT_API_KEY</code> in the
      environment for a stable key.
    </div>

    <div class="card u-overflow-hidden">
      <table class="table">
        <thead>
          <tr>
            <th style="width:220px">Label</th>
            <th style="width:200px">Key</th>
            <th style="width:150px">Created</th>
            <th style="width:150px">Last used</th>
            <th style="width:70px"></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="k in keys" :key="k.id">
            <td>
              <span class="fld-name" style="font-size:13px">{{ k.label }}</span>
              <span v-if="k.is_default" class="era-chip">
                {{ k.is_generated ? 'generated' : 'environment' }}
              </span>
            </td>
            <td>
              <!-- A generated default is shown in full: it lives only in the
                   running process, so masking it would leave no way to learn
                   a key the service invented. -->
              <template v-if="k.key">
                <span class="key-value" style="font-size:11px;padding:4px 6px">{{ k.key }}</span>
                <button class="act-btn" style="margin-left:6px" title="Copy"
                        @click="copyValue(k.key!)">⧉</button>
              </template>
              <span v-else class="u-mono">{{ k.masked }}</span>
            </td>
            <td>{{ formatDate(k.created_at) }} <span class="u-muted">{{ formatTime(k.created_at) }}</span></td>
            <td>
              <template v-if="k.last_used_at">{{ formatDate(k.last_used_at) }}</template>
              <span v-else class="u-dash">never</span>
            </td>
            <td>
              <button v-if="!k.is_default" class="act-btn" title="Revoke" @click="confirmRow = k"><Icon name="trash" /></button>
              <span v-else class="u-dash" title="Defined by DEFAULT_API_KEY; change it and restart to rotate">—</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Create flow: the key is revealed exactly once. -->
  <div v-if="createOpen" class="modal-overlay" @click.self="closeCreate()">
    <div class="modal" style="max-width:520px">
      <div class="modal-head">{{ created ? 'Key created' : 'New API key' }}</div>
      <div class="modal-body">
        <template v-if="!created">
          <label class="field-label">Label</label>
          <input v-model="label" class="input" style="width:100%"
                 placeholder="e.g. CI pipeline" @keyup.enter="create()">
          <div class="field-help">A name to recognise this key by later.</div>
        </template>
        <template v-else>
          <div class="field-help" style="margin-bottom:8px">
            Copy this now — only a hash is stored, so it cannot be shown again.
          </div>
          <div class="key-value">{{ created }}</div>
          <button class="btn btn-outline btn-sm" style="margin-top:10px" @click="copyKey()">
            Copy to clipboard
          </button>
        </template>
      </div>
      <div class="modal-foot">
        <button class="btn btn-ghost" @click="closeCreate()">{{ created ? 'Done' : 'Cancel' }}</button>
        <button v-if="!created" class="btn btn-primary" :disabled="creating" @click="create()">
          {{ creating ? 'Creating…' : 'Create key' }}
        </button>
      </div>
    </div>
  </div>

  <ConfirmDialog :open="!!confirmRow" title="Revoke API key" danger confirm-label="Revoke"
                 :message="`Revoke '${confirmRow?.label ?? ''}'? Any integration using it will stop working immediately.`"
                 @CONFIRM="remove(confirmRow!); confirmRow = null" @CANCEL="confirmRow = null" />
</template>

<style scoped>
.act-btn{width:28px;height:28px;border:1px solid var(--color-border);border-radius:6px;
  background:var(--color-card);cursor:pointer;color:var(--color-text-sub);}
.act-btn:hover{border-color:var(--color-red);color:var(--color-red);}
</style>
