<script setup lang="ts">
import { computed } from 'vue'
import { useStore } from 'vuex'

const store = useStore()
const toasts = computed(() => store.state.ui.toasts)

// The design system styles .toast.success / .error / .warning — the kind is a
// bare modifier class, not a 'toast-' prefix.
const GLYPH: Record<string, string> = { success: '✓', error: '!', info: 'i' }
</script>

<template>
  <div class="toast-stack">
    <div v-for="t in toasts" :key="t.id" :class="['toast', t.kind]">
      <div class="t-ic">{{ GLYPH[t.kind] ?? 'i' }}</div>
      <div class="t-body">
        <div class="t-title">{{ t.title }}</div>
        <div v-if="t.message" class="t-msg">{{ t.message }}</div>
      </div>
      <button class="t-close" aria-label="Dismiss" @click="store.dispatch('ui/dismiss', t.id)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round">
          <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </div>
  </div>
</template>
