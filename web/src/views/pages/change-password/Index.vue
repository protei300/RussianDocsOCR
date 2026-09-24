<script setup lang="ts">
/**
 * Change your own password — forced at first sign-in, voluntary afterwards.
 *
 * The rule checklist is evaluated from patterns the **server** sent with
 * `/auth/config`, not from regexes written again here. Two copies of a
 * validation rule drift, and the drift shows up as a form that happily accepts
 * a password the server then rejects.
 *
 * On success the session is gone: the server bumps the account's token version,
 * which invalidates every token it ever issued for that account — this one
 * included, and any other device too. So the page signs out and sends the user
 * back to the login form rather than pretending the old token still works.
 */
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useStore } from 'vuex'
import $api from '@/api'
import type { PasswordRule } from '@/types'

const store = useStore()
const router = useRouter()

const rules = ref<PasswordRule[]>([])
const current = ref('')
const next = ref('')
const repeat = ref('')
const saving = ref(false)
const errorText = ref('')
const done = ref(false)

const forced = computed<boolean>(() => store.getters['auth/mustChangePassword'])
const user = computed(() => store.getters['auth/user'])
/** From the token's `sub` claim — the account name the password is saved under. */
const username = computed<string>(() => user.value?.username ?? user.value?.name ?? '')

/** Which rules the typed password satisfies, using the server's own patterns. */
const checks = computed(() => rules.value.map((rule) => ({
    ...rule,
    // `new RegExp` on a server-provided string: the patterns are written to be
    // valid in both Python's re and JS RegExp, and a malformed one must not take
    // the page down with it.
    ok: (() => {
        try { return new RegExp(rule.pattern).test(next.value) } catch { return false }
    })(),
})))

const allRulesMet = computed(() => checks.value.every((c) => c.ok))
const matches = computed(() => next.value.length > 0 && next.value === repeat.value)
const canSubmit = computed(() =>
    Boolean(current.value) && allRulesMet.value && matches.value && !saving.value)

async function submit(): Promise<void> {
    if (!canSubmit.value) return
    saving.value = true
    errorText.value = ''
    try {
        await $api.auth.changePassword(current.value, next.value)
        done.value = true
        // Deliberate: the token that made this call is already dead server-side.
        await store.dispatch('auth/passwordChanged')
        setTimeout(() => router.push({ name: 'Login' }), 1400)
    } catch (error) {
        errorText.value = (error as Error).message || 'Could not change the password'
    } finally {
        saving.value = false
    }
}

onMounted(async () => {
    try {
        const config = await $api.auth.config()
        rules.value = config.password_rules ?? []
    } catch {
        rules.value = []
    }
})
</script>

<template>
  <div class="login-page">
    <div class="login-card pw-card">
      <div class="login-brand">
        <span class="brand-accent">Russian</span>Docs
        <div class="login-sub">
          {{ forced ? 'Choose a password to continue' : 'Change your password' }}
        </div>
      </div>

      <div v-if="done" class="pw-done">
        <div class="pw-done-mark">✓</div>
        <p>Password changed. Every session for this account has been signed out —
           including this one. Redirecting to the sign-in page…</p>
      </div>

      <template v-else>
        <div v-if="forced" class="pw-forced">
          You are signed in as <strong>{{ user?.username ?? user?.name }}</strong> with the
          seeded demonstration password. Until it is replaced, this account cannot do
          anything except this page.
        </div>

        <form class="login-form" method="post" action="#" @submit.prevent="submit">
          <!-- The account this password belongs to, shown read-only. This is the
               field whose absence broke autofill: a browser offering to save the
               NEW password here had no username on the page to attach it to, so it
               saved the password with an empty username — and from then on filled
               the password alone at sign-in. Keeping it visible rather than hidden
               also tells the person which account they are changing. -->
          <label class="field" for="account">
            <span class="field-label">Account</span>
            <input id="account" name="username" class="input input-readonly" type="text"
                   autocomplete="username" :value="username" readonly tabindex="-1" />
          </label>

          <label class="field" for="current-password">
            <span class="field-label">Current password</span>
            <input id="current-password" v-model="current" name="current-password"
                   class="input" type="password" autocomplete="current-password"
                   autofocus :readonly="saving" />
          </label>

          <label class="field" for="new-password">
            <span class="field-label">New password</span>
            <input id="new-password" v-model="next" name="new-password" class="input"
                   type="password" autocomplete="new-password" :readonly="saving" />
          </label>

          <ul v-if="checks.length" class="pw-rules">
            <li v-for="rule in checks" :key="rule.code" :class="{ met: rule.ok }">
              <span class="pw-mark">{{ rule.ok ? '✓' : '•' }}</span>{{ rule.label }}
            </li>
          </ul>

          <label class="field" for="confirm-password">
            <span class="field-label">Repeat new password</span>
            <input id="confirm-password" v-model="repeat" name="confirm-password"
                   class="input" type="password" autocomplete="new-password"
                   :readonly="saving" />
            <span v-if="repeat && !matches" class="pw-mismatch">The two do not match</span>
          </label>

          <div v-if="errorText" class="login-error">{{ errorText }}</div>

          <button class="btn btn-primary btn-block" type="submit" :disabled="!canSubmit">
            {{ saving ? 'Saving…' : 'Change password' }}
          </button>
        </form>
      </template>
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
.login-form .input-readonly{background:var(--color-row-alt);color:var(--color-text-sub);
  cursor:default;}
.btn-block{width:100%;height:44px;justify-content:center;margin-top:4px;}
.login-error{color:var(--color-red);font-size:13px;text-align:center;}
.login-card{width:420px;}
.pw-forced{margin-top:20px;padding:12px;border-radius:10px;text-align:left;
  border:1px solid var(--color-accent);background:var(--color-orange-light);
  color:var(--color-text);font-size:12px;line-height:1.55;}
.pw-rules{list-style:none;margin:-2px 0 0;padding:0;display:grid;gap:5px;}
.pw-rules li{display:flex;align-items:center;gap:8px;font-size:12px;
  color:var(--color-text-muted);transition:color 140ms ease;}
.pw-rules li.met{color:var(--color-green);}
.pw-mark{width:12px;text-align:center;font-weight:700;}
.pw-mismatch{font-size:12px;color:var(--color-red);}
.pw-done{padding:22px 4px 8px;}
.pw-done-mark{width:48px;height:48px;margin:0 auto 14px;border-radius:50%;display:grid;
  place-items:center;background:var(--color-green-light);color:var(--color-green);
  font-size:25px;font-weight:700;}
.pw-done p{font-size:13px;line-height:1.6;color:var(--color-text-sub);margin:0;}
</style>
