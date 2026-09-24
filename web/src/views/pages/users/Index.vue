<script setup lang="ts">
/**
 * User management — visible only when the server runs named accounts, and only
 * to an administrator.
 *
 * The page never decides a rule for itself. "You cannot demote the last active
 * administrator" is enforced server-side and surfaced here as the message the
 * server sent, because a check duplicated in the UI is a check that will
 * eventually disagree with the one that matters. What the page *does* do is
 * avoid offering an action it knows will be refused — the row for your own
 * account has no delete button — so the common case does not need an error.
 */
import { computed, onMounted, ref } from 'vue'
import { useStore } from 'vuex'
import AppTopbar from '@/components/AppTopbar.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'
import $api from '@/api'
import { formatDate, formatTime } from '@/utils/format'
import type { PasswordRule, UserRow } from '@/types'

const store = useStore()

const users = ref<UserRow[]>([])
const roles = ref<string[]>([])
const rules = ref<PasswordRule[]>([])
const loading = ref(false)

const createOpen = ref(false)
const form = ref({ username: '', password: '', role: 'viewer', display_name: '' })
const creating = ref(false)

const resetFor = ref<UserRow | null>(null)
const resetPassword = ref('')
const confirmRow = ref<UserRow | null>(null)

const me = computed(() => store.getters['auth/user'])

/** Initials for the row avatar: two letters, first of each word. */
function initials(name: string): string {
    return (name || '?').trim().split(/\s+/).map((part) => part[0]).join('').slice(0, 2).toUpperCase()
}

/** Deterministic avatar colours from the name, the same trick the sidebar uses —
 *  a hashed hue rather than a palette, so a new account never lands unstyled. */
function avatarStyle(name: string): Record<string, string> {
    let hash = 0
    for (const char of name || '?') hash = (hash * 31 + char.charCodeAt(0)) >>> 0
    const hue = hash % 360
    return {
        '--avatar-from': `hsl(${hue} 62% 46%)`,
        '--avatar-to': `hsl(${(hue + 38) % 360} 68% 52%)`,
    }
}
const activeAdmins = computed(() =>
    users.value.filter((u) => u.role === 'admin' && u.is_active).length)

function ruleChecks(value: string) {
    return rules.value.map((rule) => {
        let ok = false
        try { ok = new RegExp(rule.pattern).test(value) } catch { ok = false }
        return { ...rule, ok }
    })
}

const newPasswordOk = computed(() => ruleChecks(form.value.password).every((r) => r.ok))
const resetPasswordOk = computed(() => ruleChecks(resetPassword.value).every((r) => r.ok))

function load(): void {
    loading.value = true
    $api.users.list()
        .then((res) => { users.value = res.items; roles.value = res.roles;
            rules.value = res.password_rules })
        .finally(() => { loading.value = false })
}

function create(): void {
    creating.value = true
    $api.users.create(form.value)
        .then(() => {
            createOpen.value = false
            form.value = { username: '', password: '', role: 'viewer', display_name: '' }
            store.dispatch('ui/toast', { kind: 'success', title: 'User created',
                message: 'They must change this password at first sign-in' })
            load()
        })
        .finally(() => { creating.value = false })
}

function setRole(row: UserRow, role: string): void {
    if (role === row.role) return
    $api.users.update(row.id, { role }).then(load).catch(load)
}

function toggleActive(row: UserRow): void {
    $api.users.update(row.id, { is_active: !row.is_active }).then(load).catch(load)
}

function doReset(): void {
    if (!resetFor.value) return
    const row = resetFor.value
    $api.users.resetPassword(row.id, resetPassword.value)
        .then(() => {
            store.dispatch('ui/toast', { kind: 'success', title: 'Password reset',
                message: `${row.username} must change it at next sign-in` })
            resetFor.value = null
            resetPassword.value = ''
            load()
        })
}

function remove(): void {
    const row = confirmRow.value
    if (!row) return
    $api.users.remove(row.id).then(() => { confirmRow.value = null; load() }).catch(() => {
        confirmRow.value = null
    })
}

onMounted(load)
</script>

<template>
  <AppTopbar>
    <template #actions>
      <button class="btn btn-primary" @click="createOpen = true">New user</button>
    </template>
  </AppTopbar>

  <div class="content">
    <div class="ephemeral-note">
      Accounts live in the temporary store and are re-seeded on every restart, exactly like
      the documents. The password hash is Argon2id — no password is stored in a readable
      form anywhere. Putting users behind a real database is left to the adopter: see
      <code>docs/auth.md</code>.
    </div>

    <div class="card u-overflow-hidden">
      <table class="table">
        <thead>
          <tr>
            <th>User</th>
            <th>Role</th>
            <th>State</th>
            <th>Created</th>
            <th>Last sign-in</th>
            <th class="u-text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in users" :key="row.id" :class="{ 'row-off': !row.is_active }">
            <td>
              <div class="urow">
                <span class="avatar-sm" :data-avatar-name="row.display_name || row.username"
                      :style="avatarStyle(row.display_name || row.username)">
                  {{ initials(row.display_name || row.username) }}
                </span>
                <div class="urow-ident">
                  <div class="urow-name">
                    {{ row.username }}
                    <span v-if="row.username === me?.username" class="badge badge-internal">you</span>
                  </div>
                  <div v-if="row.display_name && row.display_name !== row.username"
                       class="urow-sub">{{ row.display_name }}</div>
                  <span v-if="row.must_change_password" class="badge badge-no_recording">
                    must change password
                  </span>
                </div>
              </div>
            </td>
            <td>
              <select class="select select-role" :value="row.role"
                      @change="setRole(row, ($event.target as HTMLSelectElement).value)">
                <option v-for="r in roles" :key="r" :value="r">{{ r }}</option>
              </select>
            </td>
            <td>
              <span :class="['badge', row.is_active ? 'badge-done' : 'badge-archived']">
                <span class="bdot"></span>{{ row.is_active ? 'active' : 'disabled' }}
              </span>
            </td>
            <td class="u-muted">{{ formatDate(row.created_at) }}</td>
            <td class="u-muted">
              <template v-if="row.last_login_at">
                {{ formatDate(row.last_login_at) }} {{ formatTime(row.last_login_at) }}
              </template>
              <span v-else class="u-dash">—</span>
            </td>
            <td class="u-text-right">
              <!-- btn-sm, not act-btn: the latter is a fixed 28x28 icon button, and
                   text put inside it overlaps the next one. "Disable" cannot be said
                   with an icon unambiguously, so the labels stay and the button class
                   changes. -->
              <div class="row-actions">
                <button class="btn btn-outline btn-sm" title="Set a new password"
                        @click="resetFor = row; resetPassword = ''">Reset</button>
                <button class="btn btn-ghost btn-sm" :title="row.is_active
                          ? 'Sign this user out and block further sign-ins'
                          : 'Allow this user to sign in again'"
                        @click="toggleActive(row)">
                  {{ row.is_active ? 'Disable' : 'Enable' }}
                </button>
                <!-- No delete for your own row: the server refuses it, and an action
                     that always fails is worse than one that is not offered. -->
                <button v-if="row.username !== me?.username" class="btn btn-danger-outline btn-sm"
                        @click="confirmRow = row">Delete</button>
                <!-- A hidden copy of the very button it stands in for, with the dash
                     centred over it. Guessing a min-width drifts the moment the label
                     changes, and the drift shows up as one row of buttons sitting a few
                     pixels left of every other. -->
                <span v-else class="action-placeholder"
                      title="You cannot delete the account you are signed in as">
                  <button class="btn btn-danger-outline btn-sm" aria-hidden="true"
                          tabindex="-1">Delete</button>
                  <span class="placeholder-dash">—</span>
                </span>
              </div>
            </td>
          </tr>
          <tr v-if="!users.length && !loading">
            <td colspan="6" class="u-muted u-text-center">No users yet</td>
          </tr>
        </tbody>
      </table>
    </div>

    <p class="u-muted hint">
      Roles are ordered: <strong>viewer</strong> reads documents and status,
      <strong>operator</strong> also uploads, reprocesses and deletes them,
      <strong>admin</strong> also manages users, API keys, settings and logs.
      The last active administrator ({{ activeAdmins }} right now) cannot be demoted,
      disabled or deleted — otherwise nobody could undo it.
    </p>
  </div>

  <!-- create ------------------------------------------------------------- -->
  <div v-if="createOpen" class="modal-overlay" @click.self="createOpen = false">
    <div class="modal">
      <div class="modal-head">New user</div>
      <div class="modal-body">
        <label class="field-label">Username</label>
        <input v-model="form.username" class="input" autocomplete="off" />

        <label class="field-label">Display name <span class="u-muted">(optional)</span></label>
        <input v-model="form.display_name" class="input" autocomplete="off" />

        <label class="field-label">Role</label>
        <select v-model="form.role" class="input">
          <option v-for="r in roles" :key="r" :value="r">{{ r }}</option>
        </select>

        <label class="field-label">Initial password</label>
        <input v-model="form.password" class="input" type="text" autocomplete="off" />
        <ul class="pw-rules">
          <li v-for="rule in ruleChecks(form.password)" :key="rule.code" :class="{ met: rule.ok }">
            <span class="pw-mark">{{ rule.ok ? '✓' : '•' }}</span>{{ rule.label }}
          </li>
        </ul>
        <div class="field-help">
          They will be required to change it at first sign-in, so it is safe to hand over
          in person — and it is shown in the clear here for exactly that reason.
        </div>
      </div>
      <div class="modal-foot">
        <button class="btn btn-ghost" @click="createOpen = false">Cancel</button>
        <button class="btn btn-primary"
                :disabled="creating || !form.username || !newPasswordOk" @click="create">
          Create
        </button>
      </div>
    </div>
  </div>

  <!-- reset password ------------------------------------------------------ -->
  <div v-if="resetFor" class="modal-overlay" @click.self="resetFor = null">
    <div class="modal">
      <div class="modal-head">Reset password for {{ resetFor.username }}</div>
      <div class="modal-body">
        <input v-model="resetPassword" class="input" type="text" autocomplete="off" />
        <ul class="pw-rules">
          <li v-for="rule in ruleChecks(resetPassword)" :key="rule.code" :class="{ met: rule.ok }">
            <span class="pw-mark">{{ rule.ok ? '✓' : '•' }}</span>{{ rule.label }}
          </li>
        </ul>
        <div class="field-help">
          Every session this user currently has will be signed out immediately, and they
          must choose a new password at their next sign-in.
        </div>
      </div>
      <div class="modal-foot">
        <button class="btn btn-ghost" @click="resetFor = null">Cancel</button>
        <button class="btn btn-primary" :disabled="!resetPasswordOk" @click="doReset">Reset</button>
      </div>
    </div>
  </div>

  <!-- The component's contract: a required `open`, and UPPER-CASE events. The first
       version passed neither — v-if instead of open, @confirm/@cancel instead of
       @CONFIRM/@CANCEL — so the dialog never opened and Delete did nothing. vite
       build does not type-check, so only `vue-tsc` caught it. -->
  <ConfirmDialog :open="confirmRow !== null"
                 :title="confirmRow ? `Delete ${confirmRow.username}?` : ''"
                 message="The account is removed and its sessions stop working at once."
                 danger confirm-label="Delete"
                 @CONFIRM="remove()" @CANCEL="confirmRow = null" />
</template>

<style scoped>
/* Only layout lives here. Every visual token — badge, avatar, button, select —
   comes from _shared.css, so this page restyles with the rest of the app instead
   of drifting away from it. */
.hint { font-size: 12px; line-height: 1.6; margin-top: 14px; max-width: 72ch; }
.row-off { opacity: .55; }

.urow { display: flex; align-items: center; gap: 11px; }
.urow-ident { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.urow-name { display: flex; align-items: center; gap: 7px; font-weight: 600; }
.urow-sub { font-size: 12px; color: var(--color-text-muted); }
/* Width so the chip never stretches to the column and the rows stay aligned. */
.urow-ident .badge { align-self: flex-start; }
/* Prefixed rather than the obvious .user-name: _shared.css already defines that
   for the sidebar's account row, white on navy, and a global rule beats a scoped
   one on the same element — which rendered every username white on white. */

.select-role { height: 30px; padding: 0 28px 0 10px; font-size: 12px; min-width: 104px; }

/* .u-text-right is a single class and loses to `.table thead th` (one class, two
   elements) on specificity, so the header sat left while its buttons sat right.
   Scoping adds the data-v attribute, which is exactly the weight needed. */
.table thead th.u-text-right { text-align: right; }

.row-actions { display: flex; gap: 6px; justify-content: flex-end; align-items: center; }
/* Same footprint as the Delete button it stands in for, so the column does not
   jump between the row you are signed in as and every other row. */
.action-placeholder { position: relative; display: inline-flex; cursor: default; }
.action-placeholder .btn { visibility: hidden; }
.placeholder-dash {
    position: absolute; inset: 0;
    display: grid; place-items: center;
    color: var(--color-text-muted);
}

.pw-rules { list-style: none; margin: 8px 0 0; padding: 0; display: grid; gap: 5px; }
.pw-rules li {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: var(--color-text-muted); transition: color 140ms ease;
}
.pw-rules li.met { color: var(--color-green); }
.pw-mark { width: 12px; text-align: center; font-weight: 700; }
</style>
