<script setup lang="ts">
defineProps<{ open: boolean; title: string; message: string; danger?: boolean; confirmLabel?: string }>()
const emit = defineEmits<{ CONFIRM: []; CANCEL: [] }>()
</script>

<template>
  <div v-if="open" class="modal-overlay" @click.self="emit('CANCEL')">
    <div class="modal" style="max-width:420px">
      <div class="modal-head">{{ title }}</div>
      <div class="modal-body">{{ message }}</div>
      <div class="modal-foot">
        <button class="btn btn-ghost" @click="emit('CANCEL')">Cancel</button>
        <button :class="['btn', danger ? 'btn-danger-outline' : 'btn-primary']"
                @click="emit('CONFIRM')">{{ confirmLabel ?? 'Confirm' }}</button>
      </div>
    </div>
  </div>
</template>
