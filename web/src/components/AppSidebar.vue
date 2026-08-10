<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useStore } from 'vuex'

const store = useStore()
const router = useRouter()

const collapsed = computed(() => store.state.ui.sidebarCollapsed)
const initials = computed(() => store.getters['auth/initials'])
const user = computed(() => store.getters['auth/user'])

function logout(): void {
    store.dispatch('auth/logout')
    router.push({ name: 'Login' })
}
</script>

<template>
  <aside :class="['sidebar', { collapsed }]">
    <div class="sidebar-brand">
      <div class="dot"></div>
      <div>
        <div class="title"><span>Russian</span>Docs</div>
        <div class="sub">Document OCR</div>
      </div>
    </div>

    <button class="sidebar-toggle" aria-label="Toggle sidebar"
            @click="store.dispatch('ui/toggleSidebar')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"
           stroke-linecap="round" stroke-linejoin="round">
        <polyline points="15 18 9 12 15 6" />
      </svg>
    </button>

    <nav class="nav">
      <div class="nav-section">Recognition</div>

      <router-link class="nav-link" to="/documents" title="Documents">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="8" y1="13" x2="16" y2="13" /><line x1="8" y1="17" x2="13" y2="17" />
        </svg>
        Documents
      </router-link>

      <router-link class="nav-link" to="/integration" title="Integration demo">
        <!-- A plug going into a socket: this page is about connecting a site to the
             service, and the two prongs read at 18px where a more literal API icon
             (braces, a cloud, an arrow pair) does not. -->
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M9 2v6" /><path d="M15 2v6" />
          <path d="M6 8h12v3a6 6 0 0 1-6 6 6 6 0 0 1-6-6V8z" />
          <path d="M12 17v5" />
        </svg>
        Integration demo
      </router-link>

      <div class="nav-section">System</div>

      <router-link class="nav-link" to="/status" title="Status">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
        Status
      </router-link>

      <router-link class="nav-link" to="/api-keys" title="API keys">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
        </svg>
        API keys
      </router-link>

      <router-link class="nav-link" to="/settings" title="Settings">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
        Settings
      </router-link>

      <router-link class="nav-link" to="/logs" title="Logs">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" />
        </svg>
        Logs
      </router-link>
    </nav>

    <div class="sidebar-footer">
      <div class="user-row">
        <div class="avatar">{{ initials }}</div>
        <div>
          <div class="user-name">{{ user?.name ?? 'Operator' }}</div>
          <div class="user-role">{{ user?.role ?? '' }}</div>
        </div>
      </div>
      <button class="logout-btn" @click="logout">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
          <polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" />
        </svg>
        Log out
      </button>
    </div>
  </aside>
</template>
