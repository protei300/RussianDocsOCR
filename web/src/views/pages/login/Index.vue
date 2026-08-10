<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useStore } from 'vuex'
import $api from '@/api'

const PIN_LENGTH = 4
const store = useStore()
const router = useRouter()
const route = useRoute()

const pin = ref('')
const state = ref<'idle' | 'checking' | 'error' | 'success'>('idle')

function press(digit: string): void {
    if (state.value === 'checking' || pin.value.length >= PIN_LENGTH) return
    if (state.value === 'error') { state.value = 'idle'; pin.value = '' }
    pin.value += digit
    if (pin.value.length === PIN_LENGTH) void submit()
}

function backspace(): void {
    if (state.value !== 'checking') pin.value = pin.value.slice(0, -1)
}

async function submit(): Promise<void> {
    if (pin.value.length !== PIN_LENGTH) return
    state.value = 'checking'
    try {
        // The PIN is only ever checked server-side; never compare it here.
        const res = await $api.auth.pinLogin(pin.value)
        store.dispatch('auth/signIn', { token: res.access_token, user: res.user })
        state.value = 'success'
        setTimeout(() => router.push((route.query.redirect as string) || '/documents'), 250)
    } catch {
        state.value = 'error'
        setTimeout(() => { if (state.value === 'error') { pin.value = '' } }, 700)
    }
}

function onKey(event: KeyboardEvent): void {
    if (event.key >= '0' && event.key <= '9') press(event.key)
    else if (event.key === 'Backspace') backspace()
    else if (event.key === 'Enter') void submit()
}

onMounted(() => document.addEventListener('keydown', onKey))
onUnmounted(() => document.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="login-page">
    <div class="login-card">
      <div class="login-brand">
        <span class="brand-accent">Russian</span>Docs
        <div class="login-sub">Document recognition service</div>
      </div>

      <div :class="['pin-dots', state]">
        <span v-for="i in PIN_LENGTH" :key="i" :class="['pin-dot', { filled: pin.length >= i }]"></span>
      </div>

      <div class="keypad">
        <button v-for="d in ['1','2','3','4','5','6','7','8','9']" :key="d" class="key"
                :disabled="state === 'checking'" @click="press(d)">{{ d }}</button>
        <button class="key key-alt" :disabled="state === 'checking'" @click="backspace()">⌫</button>
        <button class="key" :disabled="state === 'checking'" @click="press('0')">0</button>
        <button class="key key-ok" :disabled="state === 'checking' || pin.length < PIN_LENGTH"
                @click="submit()">→</button>
      </div>

    </div>
  </div>
</template>

<style scoped>
.login-page{min-height:100vh;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#0D1A2D 0%,#12243d 55%,#F27405 320%);}
.login-card{width:380px;background:var(--color-card);border-radius:14px;padding:34px 30px;
  box-shadow:0 18px 50px rgba(0,0,0,.35);text-align:center;}
.login-brand{font-size:22px;font-weight:700;color:var(--color-text);}
.brand-accent{color:var(--color-accent);}
.login-sub{font-size:12px;color:var(--color-text-muted);font-weight:500;margin-top:4px;letter-spacing:.04em;}
.pin-dots{display:flex;justify-content:center;gap:14px;margin:28px 0;}
.pin-dot{width:14px;height:14px;border-radius:50%;border:2px solid var(--color-border);
  background:transparent;transition:all 160ms ease;}
.pin-dot.filled{background:var(--color-primary);border-color:var(--color-primary);}
.pin-dots.error{animation:shake .4s;}
.pin-dots.error .pin-dot{border-color:var(--color-red);background:var(--color-red);}
.pin-dots.success .pin-dot{border-color:var(--color-green);background:var(--color-green);}
@keyframes shake{0%,100%{transform:translateX(0)}20%{transform:translateX(-9px)}
  40%{transform:translateX(9px)}60%{transform:translateX(-6px)}80%{transform:translateX(6px)}}
.keypad{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.key{height:52px;border:1px solid var(--color-border);border-radius:10px;background:var(--color-card);
  font-family:inherit;font-size:19px;font-weight:600;color:var(--color-text);cursor:pointer;
  transition:all 120ms ease;}
.key:hover:not(:disabled){background:var(--color-primary-light);border-color:var(--color-primary);}
.key:disabled{opacity:.45;cursor:default;}
.key-ok{background:var(--color-primary);color:#fff;border-color:var(--color-primary);}
.key-alt{color:var(--color-text-sub);}
.login-foot{margin-top:22px;font-size:10px;letter-spacing:.08em;color:var(--color-text-muted);
  text-transform:uppercase;}
</style>
