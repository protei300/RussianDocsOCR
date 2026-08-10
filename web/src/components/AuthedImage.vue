<script setup lang="ts">
import { computed, toRef } from 'vue'
import { useAuthedImage } from '@/composables/useAuthedImage'

const props = defineProps<{ url: string | null; alt?: string; imgClass?: string }>()
const { src, loading, failed } = useAuthedImage(toRef(props, 'url'))
const cls = computed(() => props.imgClass ?? '')
</script>

<template>
  <img v-if="src" :src="src" :alt="alt ?? ''" :class="cls">
  <div v-else-if="loading" :class="[cls, 'skel']"></div>
  <div v-else-if="failed" :class="[cls, 'thumb-fallback']" title="Image unavailable">n/a</div>
  <div v-else :class="[cls, 'thumb-fallback']">—</div>
</template>
