<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useStore } from 'vuex'

defineProps<{ title?: string; meta?: string }>()
const route = useRoute()
const store = useStore()
const crumb = computed(() => (route.meta.crumb as string[] | undefined) ?? [])
const heading = computed(() => crumb.value[crumb.value.length - 1] ?? '')
const dark = computed(() => store.state.ui.dark)
</script>

<template>
  <header class="topbar">
    <div>
      <div class="crumbs">
        <span v-for="(c, i) in crumb" :key="i">{{ c }}<span v-if="i < crumb.length - 1"> / </span></span>
      </div>
      <h1 class="page-title">{{ title ?? heading }}</h1>
    </div>
    <div class="topbar-right">
      <span v-if="meta" class="topbar-meta">{{ meta }}</span>
      <slot name="actions" />
      <button class="icon-btn" :title="dark ? 'Light mode' : 'Dark mode'"
              @click="store.dispatch('ui/toggleDark')">{{ dark ? '☀' : '☾' }}</button>
    </div>
  </header>
</template>
