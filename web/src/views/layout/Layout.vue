<script setup lang="ts">
import AppSidebar from '@/components/AppSidebar.vue'
import ToastStack from '@/components/ToastStack.vue'
</script>

<template>
  <div class="app">
    <AppSidebar />
    <main class="main"><router-view /></main>
  </div>
  <ToastStack />
</template>
