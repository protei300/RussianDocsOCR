<script setup lang="ts">
/**
 * The journal of in-memory recognitions (POST /recognize).
 *
 * Each line says only THAT something was recognised: when, how it ended, how long
 * it took. There is no filename, no document type and nothing to open — the server
 * kept nothing — so the rows are deliberately not links and have no actions.
 */
import { formatDate, formatMs, formatTime } from '@/utils/format'
import type { RecognitionRow } from '@/types'

defineProps<{ items: RecognitionRow[]; total: number; loading?: boolean }>()

const LABEL: Record<RecognitionRow['outcome'], string> = {
    recognised: 'recognised',
    not_recognised: 'not recognised',
    rejected: 'rejected',
    busy: 'busy',
    error: 'error',
}
// Reuse the design system's status badges instead of new colours.
const BADGE: Record<RecognitionRow['outcome'], string> = {
    recognised: 'badge-done',
    not_recognised: 'badge-archived',
    rejected: 'badge-failed',
    busy: 'badge-queued',
    error: 'badge-failed',
}
</script>

<template>
  <div class="card u-overflow-hidden">
    <div class="card-head">
      Recognitions without storage <span class="u-muted">{{ total }}</span>
    </div>
    <table class="table">
      <thead>
        <tr>
          <th style="width:70px">#</th>
          <th style="width:180px">When</th>
          <th style="width:170px">Outcome</th>
          <th style="width:90px">HTTP</th>
          <th style="width:100px">Time</th>
          <th></th>
        </tr>
      </thead>
      <tbody v-if="!items.length">
        <tr><td colspan="6" class="journal-empty">
          {{ loading ? 'Loading…' : 'No recognitions yet' }}
        </td></tr>
      </tbody>
      <tbody v-else :class="{ 'u-opacity-50': loading }">
        <tr v-for="row in items" :key="row.id">
          <td class="u-mono u-muted">{{ row.id }}</td>
          <td>
            <div class="dval">{{ formatDate(row.at) }}</div>
            <div class="dtime">{{ formatTime(row.at) }}</div>
          </td>
          <td>
            <span :class="['badge', BADGE[row.outcome]]"><span class="bdot"></span>{{ LABEL[row.outcome] }}</span>
          </td>
          <td class="u-mono">{{ row.status_code }}</td>
          <td class="u-mono">{{ formatMs(row.processing_ms) }}</td>
          <td class="journal-note">not stored — nothing to open</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.dval{font-size:13px;font-weight:600;color:var(--color-text);}
.dtime{font-size:11px;color:var(--color-text-muted);font-variant-numeric:tabular-nums;}
.journal-empty{padding:22px;text-align:center;color:var(--color-text-muted);font-size:13px;}
.journal-note{font-size:11px;color:var(--color-text-muted);}
</style>
