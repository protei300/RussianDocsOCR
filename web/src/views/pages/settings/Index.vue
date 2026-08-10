<script setup lang="ts">
/**
 * The form is generated from the server's own schema rather than hand-written.
 *
 * The recognition pipeline has a lot of knobs and they will churn; with a
 * schema-driven form, adding one is a backend-only change and the defaults
 * cannot drift between the two sides.
 */
import { computed, onMounted, ref } from 'vue'
import { useStore } from 'vuex'
import AppTopbar from '@/components/AppTopbar.vue'
import $api from '@/api'
import type { SettingDef } from '@/types'

const store = useStore()
const schema = ref<SettingDef[]>([])
const values = ref<Record<string, string>>({})
const original = ref<string>('')
const saving = ref(false)
const restartRequired = ref<string[]>([])

const dirty = computed(() => JSON.stringify(values.value) !== original.value)

const groups = computed(() => {
    const out = new Map<string, SettingDef[]>()
    for (const def of schema.value) {
        const list = out.get(def.group) ?? []
        list.push(def)
        out.set(def.group, list)
    }
    return [...out.entries()]
})

function load(): void {
    $api.settings.get().then((res) => {
        schema.value = res.schema
        values.value = { ...res.values }
        original.value = JSON.stringify(values.value)
    })
}

function save(): void {
    saving.value = true
    $api.settings.update(values.value)
        .then((res) => {
            values.value = { ...res.values }
            original.value = JSON.stringify(values.value)
            restartRequired.value = res.restart_required
            store.dispatch('ui/toast', { kind: 'success', title: 'Saved', message: 'Settings updated' })
        })
        .finally(() => { saving.value = false })
}

function discard(): void {
    values.value = JSON.parse(original.value)
}

onMounted(load)
</script>

<template>
  <AppTopbar :meta="dirty ? '● Unsaved changes' : undefined">
    <template #actions>
      <button v-if="dirty" class="btn btn-ghost" @click="discard()">Discard</button>
      <button class="btn btn-primary" :disabled="!dirty || saving" @click="save()">
        {{ saving ? 'Saving…' : 'Save changes' }}
      </button>
    </template>
  </AppTopbar>

  <div class="content">
    <div v-if="restartRequired.length" class="ephemeral-note" style="margin-bottom:14px">
      Restart required for: <strong>{{ restartRequired.join(', ') }}</strong> — these are baked into
      the pipeline when it is constructed, so the change applies on the next start.
    </div>

    <div v-for="[group, defs] in groups" :key="group" class="card card-pad" style="margin-bottom:16px">
      <div class="card-head" style="padding:0 0 14px">{{ group }}</div>
      <div v-for="def in defs" :key="def.key" class="setting-row">
        <div class="setting-info">
          <div class="field-label">
            {{ def.label }}
            <span v-if="def.restart_required" class="era-chip">restart</span>
          </div>
          <div class="field-help">{{ def.description }}</div>
        </div>
        <div class="setting-control">
          <select v-if="def.choices" v-model="values[def.key]" class="select">
            <option v-for="c in def.choices" :key="c" :value="c">{{ c }}</option>
          </select>
          <input v-else-if="def.type === 'int' || def.type === 'float'"
                 v-model="values[def.key]" class="input" type="number"
                 :min="def.min_value ?? undefined" :max="def.max_value ?? undefined"
                 :step="def.type === 'float' ? 0.05 : 1">
          <input v-else v-model="values[def.key]" class="input" type="text">
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.setting-row{display:flex;align-items:flex-start;gap:20px;padding:12px 0;
  border-bottom:1px solid var(--color-border-light);}
.setting-row:last-child{border-bottom:none;}
.setting-info{flex:1;min-width:0;}
.setting-control{width:220px;flex-shrink:0;}
.setting-control .input,.setting-control .select{width:100%;}
</style>
