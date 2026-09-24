<script setup lang="ts">
/**
 * One login page, two modes, decided by the server.
 *
 * The page asks `/auth/config` before rendering anything rather than guessing
 * from a build-time flag. That is what keeps the same bundle usable by the Go,
 * .NET and Kotlin services, which today report `mode: "pin"` and get exactly the
 * keypad they had before.
 *
 * The seeded credentials are printed on the page on purpose. This is a
 * demonstration service whose first account ships with a known password; hiding
 * it would only mean the person evaluating cannot get in. What makes it
 * defensible is the forced change — the account can do nothing else until the
 * password is replaced — and the warning says so rather than leaving it implied.
 */
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useStore } from 'vuex'
import $api from '@/api'
import type { AuthConfig } from '@/types'

const PIN_LENGTH = 4

const store = useStore()
const router = useRouter()
const route = useRoute()

const config = ref<AuthConfig | null>(null)
const loading = ref(true)
const state = ref<'idle' | 'checking' | 'error'>('idle')
const errorText = ref('')

const pin = ref('')
const username = ref('')
const password = ref('')

const mode = computed(() => config.value?.mode ?? 'pin')
const demo = computed(() => config.value?.demo_credentials ?? null)

function afterSignIn(mustChange: boolean): void {
    if (mustChange) {
        router.push({ name: 'ChangePassword' })
        return
    }
    const redirect = route.query.redirect as string | undefined
    router.push(redirect || { name: 'Documents' })
}

function press(digit: string): void {
    if (state.value === 'checking' || pin.value.length >= PIN_LENGTH) return
    pin.value += digit
    if (pin.value.length === PIN_LENGTH) void submitPin()
}

function backspace(): void { pin.value = pin.value.slice(0, -1) }

function onKey(event: KeyboardEvent): void {
    if (mode.value !== 'pin') return
    if (/^[0-9]$/.test(event.key)) press(event.key)
    else if (event.key === 'Backspace') backspace()
    else if (event.key === 'Enter') void submitPin()
}

async function submitPin(): Promise<void> {
    if (pin.value.length !== PIN_LENGTH) return
    state.value = 'checking'
    errorText.value = ''
    try {
        // The PIN is only ever checked server-side; never compare it here.
        const response = await $api.auth.pinLogin(pin.value)
        await store.dispatch('auth/signIn', { token: response.access_token, user: response.user })
        afterSignIn(false)
    } catch (error) {
        state.value = 'error'
        errorText.value = (error as Error).message || 'Wrong PIN'
        pin.value = ''
        setTimeout(() => { if (state.value === 'error') state.value = 'idle' }, 1200)
    }
}

async function submitLogin(): Promise<void> {
    if (!username.value || !password.value || state.value === 'checking') return
    state.value = 'checking'
    errorText.value = ''
    try {
        const response = await $api.auth.login(username.value, password.value)
        await store.dispatch('auth/signIn', {
            token: response.access_token,
            user: null,
            mustChangePassword: response.must_change_password,
        })
        afterSignIn(response.must_change_password)
    } catch (error) {
        state.value = 'error'
        errorText.value = (error as Error).message || 'Wrong username or password'
        password.value = ''
        setTimeout(() => { if (state.value === 'error') state.value = 'idle' }, 2500)
    }
}

function fillDemo(): void {
    if (!demo.value) return
    username.value = demo.value.username
    password.value = demo.value.password
}

onMounted(async () => {
    document.addEventListener('keydown', onKey)
    try {
        config.value = await $api.auth.config()
        await store.dispatch('auth/setMode', config.value.mode)
    } catch {
        // An older service (or one that is still starting) has no /auth/config.
        // Falling back to the keypad keeps this page usable against every
        // version that came before named accounts existed.
        config.value = { mode: 'pin', pin_required: true, users_enabled: false,
            downgrade_reason: null }
    } finally {
        loading.value = false
    }
})
onUnmounted(() => document.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="login-page">
    <div class="login-card">
      <div class="login-brand">
        <span class="brand-accent">Russian</span>Docs
        <div class="login-sub">Document recognition service</div>
      </div>

      <div v-if="loading" class="login-loading">Connecting…</div>

      <!-- PIN: unchanged from before named accounts, deliberately. -->
      <template v-else-if="mode === 'pin'">
        <div :class="['pin-dots', state]">
          <span v-for="i in PIN_LENGTH" :key="i"
                :class="['pin-dot', { filled: pin.length >= i }]"></span>
        </div>

        <div class="keypad">
          <button v-for="d in ['1','2','3','4','5','6','7','8','9']" :key="d" class="key"
                  :disabled="state === 'checking'" @click="press(d)">{{ d }}</button>
          <button class="key key-alt" :disabled="state === 'checking'"
                  @click="backspace()">⌫</button>
          <button class="key" :disabled="state === 'checking'" @click="press('0')">0</button>
          <button class="key key-ok" :disabled="state === 'checking' || pin.length < PIN_LENGTH"
                  @click="submitPin()">→</button>
        </div>
      </template>

      <!-- Named accounts. -->
      <template v-else>
        <!-- What makes a browser remember BOTH fields, not just the password:
             `name`/`id` it can key the credential on, the standard autocomplete
             tokens, and `readonly` rather than `disabled` while the request is in
             flight. A disabled field is not part of the form, and a password
             manager reading the form at submit time skips it — which is how the
             first version got its password saved with an empty username. -->
        <form class="login-form" method="post" action="#" @submit.prevent="submitLogin">
          <label class="field" for="username">
            <span class="field-label">Username</span>
            <input id="username" v-model="username" name="username" class="input" type="text"
                   autocomplete="username" autocapitalize="none" spellcheck="false"
                   autofocus :readonly="state === 'checking'" />
          </label>

          <label class="field" for="password">
            <span class="field-label">Password</span>
            <input id="password" v-model="password" name="password" class="input"
                   type="password" autocomplete="current-password"
                   :readonly="state === 'checking'" />
          </label>

          <div v-if="errorText" class="login-error">{{ errorText }}</div>

          <button class="btn btn-primary btn-block" type="submit"
                  :disabled="state === 'checking' || !username || !password">
            {{ state === 'checking' ? 'Signing in…' : 'Sign in' }}
          </button>
        </form>

        <div v-if="demo" class="demo-note">
          <div class="demo-head">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 9v4" /><path d="M12 17h.01" />
              <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
            </svg>
            Demonstration credentials
          </div>
          <div class="demo-creds">
            <code>{{ demo.username }}</code><span class="demo-sep">/</span><code>{{ demo.password }}</code>
            <button class="btn btn-ghost btn-sm" type="button" @click="fillDemo">Fill in</button>
          </div>
          <p class="demo-text">
            This account is seeded for the demo and <strong>must be given a new password at
            first sign-in</strong> — until then it cannot do anything else. The service
            re-creates it after every restart, because its storage is temporary.
          </p>
        </div>
      </template>

      <div v-if="config?.downgrade_reason" class="login-warning">
        {{ config.downgrade_reason }}
      </div>

      <div v-if="!loading && mode === 'pin' && errorText" class="login-error">{{ errorText }}</div>
    </div>
  </div>
</template>

<style scoped>
.login-page{min-height:100vh;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#0D1A2D 0%,#12243d 55%,#F27405 320%);padding:24px;}
.login-card{width:380px;background:var(--color-card);border-radius:14px;padding:34px 30px;
  box-shadow:0 18px 50px rgba(0,0,0,.35);text-align:center;}
.login-brand{font-size:22px;font-weight:700;color:var(--color-text);}
.brand-accent{color:var(--color-accent);}
.login-sub{font-size:12px;color:var(--color-text-muted);font-weight:500;margin-top:4px;
  letter-spacing:.04em;}
.login-form{display:flex;flex-direction:column;gap:14px;margin-top:22px;text-align:left;}
.field{display:flex;flex-direction:column;gap:6px;}
.field-label{font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;
  color:var(--color-text-muted);}
.login-form .input{width:100%;height:42px;padding:0 12px;border-radius:10px;
  border:1px solid var(--color-border);background:var(--color-bg);color:var(--color-text);
  font-family:inherit;font-size:14px;transition:border-color 120ms ease;}
.login-form .input:focus{outline:none;border-color:var(--color-primary);}
.btn-block{width:100%;height:44px;justify-content:center;margin-top:4px;}
.login-error{color:var(--color-red);font-size:13px;text-align:center;}
/* --- PIN: restored verbatim from before named accounts ---------------------- */
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
.key{height:52px;border:1px solid var(--color-border);border-radius:10px;
  background:var(--color-card);font-family:inherit;font-size:19px;font-weight:600;
  color:var(--color-text);cursor:pointer;transition:all 120ms ease;}
.key:hover:not(:disabled){background:var(--color-primary-light);border-color:var(--color-primary);}
.key:disabled{opacity:.45;cursor:default;}
.key-ok{background:var(--color-primary);color:#fff;border-color:var(--color-primary);}
.key-alt{color:var(--color-text-sub);}

.login-loading{color:var(--color-text-muted);padding:34px 0;font-size:13px;}

.login-warning{margin-top:16px;padding:10px 12px;border-radius:10px;text-align:left;
  border:1px solid var(--color-accent);background:var(--color-orange-light);
  color:var(--color-text);font-size:12px;line-height:1.5;}

/* Quieter than the form but impossible to miss: information, and the warning
   inside it is the part that matters. */
.demo-note{margin-top:22px;padding:14px;border-radius:12px;text-align:left;
  border:1px dashed var(--color-border);background:var(--color-row-alt);}
.demo-head{display:flex;align-items:center;gap:7px;font-size:11px;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase;color:var(--color-accent);}
.demo-head svg{width:15px;height:15px;flex:none;}
.demo-creds{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:10px 0 8px;}
.demo-creds code{font-family:ui-monospace,monospace;font-size:14px;background:var(--color-bg);
  border:1px solid var(--color-border);padding:3px 8px;border-radius:6px;color:var(--color-text);}
.demo-sep{color:var(--color-text-muted);}
.demo-text{margin:0;font-size:12px;line-height:1.55;color:var(--color-text-sub);}
</style>
