<script setup lang="ts">
/**
 * The integration demo.
 *
 * Its purpose is stated on the page itself, in the banner, because a reader who lands here from a
 * link needs to know what they are looking at: this is not another way to process documents — it is
 * a worked example of a SITE calling the service, with the actual REST exchange and the TypeScript
 * that produced it shown beside the result.
 *
 * Everything here goes through `@/client/rdocs-client`, the single file an integrator copies.
 * Nothing on this page uses the app's own axios instance, so the demo cannot show something the
 * client does not actually support.
 *
 * **Layout: FOUR cards on one grid of two equal columns**, in reading order — the document and its
 * result on top, because they are what a visitor came to see, and the machinery (which service was
 * found, which calls were made) underneath. Every card therefore starts at the same left edge and
 * ends at the same right edge, and the grid columns are `minmax(0, 1fr)` rather than `1fr` so a wide
 * URL or a raw JSON blob cannot push the page sideways.
 */
import AppTopbar from '@/components/AppTopbar.vue'
import { colourForLabel, labelOrder, pct } from '@/utils/boxes'
import { useHooks } from './controller'
import {
    activeBoxIds, activeFieldName, apiKey, canvasFallback, canvasUrl, discovering, dragging,
    failure, file, filePreview, hovered, pinned, probes, progress, result, row, running,
    servedElsewhere, showLabels, showRaw, steps, target,
} from './model'
import { computed } from 'vue'

const { discover, pick, onDrop, run, cleanUp } = useHooks()

const order = computed(() => (result.value ? labelOrder(result.value.boxes) : []))

const STATE_TEXT: Record<string, string> = {
    unknown: 'not probed', probing: 'probing…', ready: 'ready', loading: 'loading models',
    down: 'not running', blocked: 'not running',
}

/** True when the chosen target is a different origin, which is the only case a key is needed for. */
const needsKey = computed(() => !!target.value && target.value.baseUrl !== '')

function colour(label: string): string {
    return colourForLabel(label, order.value)
}

/**
 * Whether a value should be set in monospace.
 *
 * Numbers, dates and codes line up in a column and are easier to check digit by digit; CAPITAL
 * CYRILLIC is the opposite — a monospace face mangles `Ш`, `Щ`, `Ж` and `Ы`, which is why the values
 * are proportional everywhere else. Same rule as the document page.
 */
function isCode(value: string): boolean {
    return /^[\d\s.,:/-]+$/.test(value)
}

/** Timings are seconds on the wire; the header shows the total in milliseconds. */
const totalMs = computed(() => {
    const total = result.value?.timings?.total
    return total ? Math.round(total * 1000) : (row.value?.processing_ms ?? null)
})
</script>

<template>
  <AppTopbar :meta="target ? `target: ${target.label}` : 'no service found'">
    <template #actions>
      <button class="btn btn-outline btn-sm" :disabled="discovering" @click="discover()">
        {{ discovering ? 'Probing…' : 'Re-detect service' }}
      </button>
    </template>
  </AppTopbar>

  <div class="content">
    <!-- Says what the page is. First thing on it, on purpose. -->
    <div class="card intro u-mb-lg">
      <div class="card-pad">
        <div class="intro-row">
          <div class="intro-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
                 stroke-linecap="round" stroke-linejoin="round">
              <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
              <line x1="13" y1="4" x2="11" y2="20" />
            </svg>
          </div>
          <div class="u-min0">
            <h2 class="intro-title">This page demonstrates integrating a website with the
              recognition service</h2>
            <p class="intro-text">
              It is a worked example, not another upload screen. A document goes to the service over
              its REST API through a <span class="u-mono">TypeScript</span> client — the single file
              <span class="u-mono">web/src/client/rdocs-client.ts</span>, which you copy into your
              own site — and every call it makes is listed below with its status and duration.
              Recognition is asynchronous: upload, poll, then read the fields. The same client works
              against <strong>all four implementations</strong> — Python, Go, .NET and Kotlin/JVM —
              which publish one contract and differ only in the port they listen on.
            </p>
          </div>
        </div>
      </div>
    </div>

    <div class="grid">
      <!-- ─── 1 · the document: the main action, so it comes first ──────────── -->
      <div class="card">
        <div class="card-head">
          <h3>1 · The document</h3>
          <span class="u-sub">multipart, part name <span class="u-mono">file</span></span>
        </div>
        <div class="card-pad">
          <label class="drop" :class="{ over: dragging }"
                 @dragover.prevent="dragging = true" @dragleave="dragging = false"
                 @drop.prevent="onDrop">
            <input type="file" accept="image/*" hidden
                   @change="pick(($event.target as HTMLInputElement).files)">
            <img v-if="filePreview" :src="filePreview" class="drop-thumb" alt="">
            <template v-else>
              <div class="drop-main">Drop a passport here</div>
              <div class="drop-sub">or click to choose · JPEG, PNG, WEBP, BMP, TIFF</div>
            </template>
          </label>

          <div class="send-row">
            <span class="u-sub u-mono u-min0 send-name">
              <template v-if="file">{{ file.name }} · {{ Math.round(file.size / 1024) }} KB</template>
              <template v-else>no file chosen</template>
            </span>
            <button class="btn btn-primary btn-sm" :disabled="!file || !target || running"
                    @click="run()">
              {{ running ? 'Recognising…' : 'Send to the service' }}
            </button>
          </div>

          <!-- The credential, as a NOTE rather than a form: for the service that served this page
               there is nothing to type, and a prominent empty field invites a reader to look for
               something that is missing. The input appears only when the target is another origin,
               where a session token genuinely does not apply. -->
          <p class="note">
            <strong>Credential.</strong> An integration authenticates with an
            <span class="u-mono">X-API-Key</span> header; keys are managed on the API keys page.
            <template v-if="needsKey">
              The chosen target is a different origin, so a session token does not apply — paste a
              key below.
            </template>
            <template v-else>
              Here the target is the service that served this page, so the browser session is reused
              and there is nothing to enter.
            </template>
          </p>
          <input v-if="needsKey" v-model="apiKey" class="input u-w-full u-mono u-mt-md"
                 placeholder="rdk_…">
        </div>
      </div>

      <!-- ─── 2 · the result, beside the document ───────────────────────────── -->
      <div class="card">
        <div class="card-head">
          <h3>2 · The result</h3>
          <div v-if="result" class="head-actions">
            <span class="badge" :class="result.recognised ? 'badge-done' : 'badge-failed'">
              {{ result.doc_type ?? 'not recognised' }}
            </span>
            <button class="btn btn-ghost btn-sm" @click="showLabels = !showLabels">
              {{ showLabels ? 'Hide labels' : 'Show labels' }}
            </button>
            <button class="btn btn-ghost btn-sm" @click="cleanUp()">Delete</button>
          </div>
          <span v-else-if="totalMs" class="u-sub">{{ totalMs }} ms</span>
        </div>

        <div class="card-pad">
          <p v-if="!result" class="u-dash u-text-center" style="padding:26px">
            Send a document and the corrected canvas, the detected regions and the fields appear
            here.
          </p>

          <template v-else>
            <!-- The canvas, WHOLE and inside the card.
                 The wrapper is inline-block so it shrink-wraps the scaled image, and the SVG is
                 `inset:0` over it — that is what keeps the boxes on the document at any size. The
                 image is bounded on BOTH axes: bounding only the width pushes a portrait passport
                 past the fold, and bounding neither is what made this card overflow. -->
            <div v-if="canvasUrl" class="shot">
              <div class="shot-wrap">
                <img :src="canvasUrl" class="shot-img" alt="corrected document">
                <svg v-if="result.canvas.width" class="shot-svg"
                     :viewBox="`0 0 ${result.canvas.width} ${result.canvas.height}`"
                     preserveAspectRatio="none">
                  <rect v-for="b in result.boxes" :key="b.id"
                        :x="b.x1" :y="b.y1" :width="b.x2 - b.x1" :height="b.y2 - b.y1"
                        :stroke="colour(b.label)" fill="none" stroke-width="2"
                        vector-effect="non-scaling-stroke"
                        :opacity="activeBoxIds.size && !activeBoxIds.has(b.id) ? 0.25 : 1"
                        @mouseenter="hovered = 'box:' + b.id" @mouseleave="hovered = null"
                        @click="pinned = pinned === 'box:' + b.id ? null : 'box:' + b.id" />
                </svg>
                <!-- Labels as HTML in percentages, not SVG text: SVG text scales with the viewBox
                     and turns unreadable on a large document. -->
                <template v-if="showLabels">
                  <span v-for="b in result.boxes" :key="'l' + b.id" class="shot-label"
                        :style="{ left: pct(b.x1, result.canvas.width),
                                  top: pct(b.y1, result.canvas.height),
                                  borderColor: colour(b.label),
                                  opacity: activeBoxIds.size && !activeBoxIds.has(b.id) ? 0.3 : 1 }">
                    {{ b.display }}
                  </span>
                </template>
              </div>
            </div>
            <p v-else class="u-dash u-text-center" style="padding:22px">
              No corrected canvas — the document was not recognised.
            </p>

            <p v-if="canvasFallback" class="note note-warn">
              This is the upload itself: recognition short-circuited, so there is no corrected
              canvas.
            </p>
            <p class="u-sub shot-note">
              {{ result.boxes.length }} regions ·
              coordinates are in <span class="u-mono">{{ result.coord_space }}</span> space, so they
              cannot be drawn on the original photo — the library does not retain the deskew angle.
            </p>

            <div class="panel-title">Fields · {{ result.fields.length }}</div>
            <!-- A TABLE, with its own styles rather than the document page's `.fld-row`: those
                 classes lay out inside that page's grid, and borrowed into this card the label and
                 the value collapsed into one another ("Last nameВАЛУЕВ"). Two real columns cannot
                 do that. -->
            <table v-if="result.fields.length" class="fields">
              <tbody>
                <tr v-for="f in result.fields" :key="f.name"
                    :class="{ active: activeFieldName === f.name, pinned: pinned === 'field:' + f.name }"
                    @mouseenter="hovered = 'field:' + f.name" @mouseleave="hovered = null"
                    @click="pinned = pinned === 'field:' + f.name ? null : 'field:' + f.name">
                  <td class="f-name">
                    <span class="f-dot" :style="{ background: colour(f.name) }"></span>
                    {{ f.display }}
                  </td>
                  <td class="f-val" :class="{ 'u-mono': isCode(f.value) }"
                      :lang="f.script === 'ru' ? 'ru' : 'en'">{{ f.value }}</td>
                </tr>
              </tbody>
            </table>
            <p v-else class="u-dash">No text fields were read.</p>

            <div class="panel-title u-mt-lg">Quality</div>
            <div class="q-pills">
              <span v-for="(v, k) in result.quality" :key="k" class="q-pill"
                    :class="String(v) === 'good' || String(v) === 'REAL' ? 'q-good' : 'q-bad'">
                {{ k }}: {{ v }}
              </span>
            </div>

            <div class="panel-title u-mt-lg">Timings, seconds</div>
            <!-- Same reason as the field table: `.key-value` is another page's grid, and here the
                 stage name ran straight into its number ("_doctype_angle0.027"). -->
            <table class="fields timings">
              <tbody>
                <tr v-for="(v, k) in result.timings" :key="k">
                  <td class="f-name u-mono">{{ k }}</td>
                  <td class="f-val u-mono">{{ v.toFixed(3) }}</td>
                </tr>
              </tbody>
            </table>

            <button class="btn btn-ghost btn-sm u-mt-lg" @click="showRaw = !showRaw">
              {{ showRaw ? 'Hide' : 'Show' }} raw JSON
            </button>
            <pre v-if="showRaw" class="raw-json u-mono">{{ JSON.stringify(result, null, 2) }}</pre>
          </template>
        </div>
      </div>

      <!-- ─── 3 · which service was found ──────────────────────────────────── -->
      <div class="card">
        <div class="card-head">
          <h3>3 · Service discovery</h3>
          <span class="u-sub"><span class="u-mono">GET /health</span> · no credential</span>
        </div>
        <div class="card-pad">
          <p class="u-sub u-mb-md">
            Runs automatically when the page opens. The target is not chosen by hand — the page
            probes the known ports and takes the first that answers, preferring the service that
            served this page.
          </p>
          <div class="probes">
            <!-- FOUR rows, one per implementation. The one serving this page is marked, not listed
                 twice: a fifth "This service" row put the same process in two places and left a
                 reader unable to say which of the four they were talking to. -->
            <div v-for="p in probes" :key="p.label" class="probe"
                 :class="[`st-${p.state}`, { chosen: target?.label === p.label }]">
              <span class="probe-dot"></span>
              <span class="probe-name">{{ p.label }}</span>
              <span class="probe-url u-mono">
                localhost:{{ p.port }}<template v-if="p.self"> · serves this page</template>
              </span>
              <span class="probe-state">{{ STATE_TEXT[p.state] }}</span>
              <span class="probe-badge">
                <!-- The duration belongs to the chosen row alone: on a row that did not answer it is
                     the probe timeout and nothing else. -->
                <template v-if="target?.label === p.label">
                  <span v-if="p.ms !== null" class="probe-ms u-mono">{{ p.ms }} ms</span>
                  <span class="badge badge-done">in use</span>
                </template>
              </span>
            </div>
          </div>
          <p v-if="servedElsewhere" class="note">
            This page is not served by any of the four — a dev server, most likely — so every probe
            above is cross-origin and needs the target to allow this origin.
          </p>
          <p v-if="probes.some((p) => p.state === 'blocked')" class="note">
            A port marked <em>not running</em> may also be running and refusing this origin: a
            cross-origin service has to list it in
            <span class="u-mono">CORS_ALLOWED_ORIGINS</span>. The browser withholds which of the two
            it was, deliberately, so the page cannot tell you either — and the two are reported as one
            state rather than guessed at.
          </p>
          <p v-if="!target && !discovering" class="note note-bad">
            No service answered. Start one and press <em>Re-detect service</em>.
          </p>
        </div>
      </div>

      <!-- ─── 4 · the calls, aligned under the result ───────────────────────── -->
      <div class="card">
        <div class="card-head">
          <h3>4 · The REST exchange</h3>
          <span v-if="totalMs" class="u-sub">recognised in {{ totalMs }} ms</span>
        </div>
        <div class="card-pad">
          <p v-if="!steps.length" class="u-dash u-text-center" style="padding:18px">
            Send a document and every call appears here.
          </p>
          <div v-for="(s, i) in steps" :key="i" class="step" :class="`sp-${s.state}`">
            <div class="step-head">
              <span class="step-method">{{ s.method }}</span>
              <span class="step-url u-mono u-min0">{{ s.url }}</span>
              <span class="step-meta u-mono">
                <template v-if="s.status">{{ s.status }}</template>
                <template v-else-if="s.state === 'pending'">…</template>
                <template v-if="s.ms !== null"> · {{ s.ms }} ms</template>
              </span>
            </div>
            <pre class="step-code u-mono">{{ s.code }}</pre>
            <p v-if="s.note" class="step-note">{{ s.note }}</p>
          </div>

          <div v-if="progress" class="prog u-mt-md">
            <div class="prog-bar"><div class="prog-fill" :style="{ width: progress.pct + '%' }"></div></div>
            <span class="u-sub">{{ progress.label }}</span>
          </div>
          <p v-if="failure" class="note note-bad">{{ failure }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ONLY variables that `public/_shared.css` actually defines: --color-border, --color-border-light,
   --color-row-alt, --color-text, --color-text-sub, --color-text-muted, --color-primary,
   --color-green, --color-red, --color-accent, --color-card, --radius-*, --shadow-card.
   `--border-color-light` and `--border-color-base` belong to a DIFFERENT design system, and an
   undefined custom property is invalid at computed-value time — the whole declaration is dropped,
   silently. Every border on this page was therefore missing until it was measured: the probe rows,
   the call cards, the canvas frame and the table separators all read back as `0px none`. */
.intro{border-left:3px solid var(--color-primary);}
.intro-row{display:flex;gap:16px;align-items:flex-start;}
.intro-icon{flex-shrink:0;width:40px;height:40px;border-radius:10px;display:grid;place-items:center;
  background:color-mix(in srgb, var(--color-primary) 12%, transparent);color:var(--color-primary);}
.intro-icon svg{width:22px;height:22px;}
.intro-title{margin:0 0 6px;font-size:16px;font-weight:650;}
.intro-text{margin:0;font-size:13px;line-height:1.6;color:var(--color-text-sub);}

/* TWO EQUAL COLUMNS, so all four cards share the page's left and right edges.
   `minmax(0,1fr)`, not `1fr`: a bare `1fr` is `minmax(auto,1fr)` and refuses to shrink below the
   min-content width of what is inside — the request URLs and the raw JSON here — which made the
   whole page scroll sideways (measured: 1284px of grid in a 1280px viewport). */
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start;}
@media (max-width:1180px){.grid{grid-template-columns:minmax(0,1fr);}}

.card-head{gap:12px;}
.head-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}

/* --- 1 · the document ------------------------------------------------------ */
.drop{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;
  min-height:190px;padding:16px;border:1.5px dashed var(--color-border);border-radius:10px;
  cursor:pointer;transition:border-color .15s,background .15s;}
.drop:hover,.drop.over{border-color:var(--color-primary);background:var(--color-row-alt);}
.drop-main{font-size:14px;font-weight:600;}
.drop-sub{font-size:12px;color:var(--color-text-muted);}
/* Bounded on both axes and `contain`: the preview must sit inside the box whatever the photo's
   aspect ratio, and a portrait passport bounded only by width is taller than the fold. */
.drop-thumb{max-width:100%;max-height:240px;object-fit:contain;border-radius:6px;}
.send-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:12px;}
.send-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}

.note{margin:14px 0 0;padding:10px 12px;border-radius:8px;background:var(--color-row-alt);
  font-size:12px;line-height:1.6;color:var(--color-text-sub);}
.note-warn{border-left:2px solid var(--color-accent);}
.note-bad{border-left:2px solid var(--color-red);color:var(--color-red);}

/* --- 2 · the result -------------------------------------------------------- */
.shot{display:flex;justify-content:center;}
/* inline-block, so the wrapper is exactly the size of the SCALED image and the absolute overlay
   lines up with it. A block wrapper would be the card's full width and the boxes would drift. */
.shot-wrap{position:relative;display:inline-block;line-height:0;max-width:100%;
  border-radius:8px;overflow:hidden;border:1px solid var(--color-border-light);}
/* As large as the column allows, and never cropped. `max-height` is in vh rather than px so a
   portrait document uses the full card width on a tall screen (a fixed 420px cap rendered an
   862x1255 passport at 288px wide, which is too small to read) while a panorama still fits. */
.shot-img{display:block;max-width:100%;max-height:78vh;width:auto;height:auto;}
.shot-svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:auto;}
.shot-svg rect{cursor:pointer;}
.shot-label{position:absolute;transform:translateY(-100%);padding:1px 5px;border-radius:4px;
  border:1px solid;background:var(--color-primary);color:#fff;font-size:9.5px;line-height:1.5;
  white-space:nowrap;pointer-events:none;}
.shot-note{margin:12px 0 16px;font-size:12px;line-height:1.55;}

.panel-title{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:var(--color-text-muted);margin-bottom:8px;}

/* Two real columns: the label shrinks to its content, the value takes the rest. `table-layout` stays
   auto on purpose so a long place of birth widens its own column instead of being clipped. */
.fields{width:100%;border-collapse:collapse;font-size:13px;}
.fields td{padding:7px 8px;border-bottom:1px solid var(--color-border-light);vertical-align:top;}
.fields tr:last-child td{border-bottom:none;}
.fields tbody tr{cursor:pointer;transition:background .12s;}
.fields tbody tr:hover{background:var(--color-row-alt);}
/* The overlay highlights the same row from the other direction — hovering a box marks its field. */
.fields tbody tr.active{background:color-mix(in srgb, var(--color-primary) 9%, transparent);}
.fields tbody tr.pinned{box-shadow:inset 2px 0 0 var(--color-primary);}
.f-name{width:1%;white-space:nowrap;color:var(--color-text-sub);}
.f-dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:7px;
  vertical-align:baseline;}
/* `anywhere`, not `break-word`: a place of birth runs past 40 characters and must wrap inside its
   cell rather than widen the card. */
.f-val{font-weight:600;overflow-wrap:anywhere;}
.timings tbody tr{cursor:default;}
.timings tbody tr:hover{background:transparent;}
.timings .f-name,.timings .f-val{font-size:11.5px;font-weight:400;}
.timings .f-val{text-align:right;color:var(--color-text-sub);}

/* --- 3 · discovery --------------------------------------------------------- */
/* A GRID, not a flex row. The flex version overlapped: `.probe-url` grew to its content and the
   state text sat on top of it, because neither could shrink. Fixed columns plus `min-width:0`
   means every cell truncates instead. */
.probes{display:flex;flex-direction:column;gap:6px;}
.probe{display:grid;grid-template-columns:9px minmax(70px,auto) minmax(0,1fr) auto auto;
  align-items:center;gap:10px;padding:8px 10px;border-radius:8px;
  border:1px solid var(--color-border-light);font-size:12px;
  /* Everything that is NOT the active service recedes, so the answer to "which one am I talking
     to" is visible without reading five statuses. */
  color:var(--color-text-muted);}
/* GREEN, and only here: the chosen row is the page's whole point. */
.probe.chosen{border-color:var(--color-green);
  background:color-mix(in srgb, var(--color-green) 8%, transparent);color:inherit;}
.probe.chosen .probe-name{color:var(--color-green);}
.probe-dot{width:9px;height:9px;border-radius:50%;background:var(--color-text-muted);opacity:.5;}
.probe.chosen .probe-dot{background:var(--color-green);opacity:1;
  box-shadow:0 0 0 3px color-mix(in srgb, var(--color-green) 22%, transparent);}
.st-loading.chosen .probe-dot{background:var(--color-accent);box-shadow:none;}
.st-probing .probe-dot{background:var(--color-primary);opacity:1;}
.probe-ms{color:var(--color-text-sub);}
.probe-name{font-weight:600;white-space:nowrap;}
.probe-url,.probe-state{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.probe-url{color:var(--color-text-muted);}
.probe-state{color:var(--color-text-sub);}
/* Always present, so the rows keep one column layout whether or not a badge is in them. */
.probe-badge{min-width:52px;display:flex;align-items:center;gap:8px;
  justify-content:flex-end;}

/* --- 4 · the exchange ------------------------------------------------------ */
.step{border:1px solid var(--color-border-light);border-radius:8px;padding:8px 10px;
  margin-bottom:8px;}
.step-head{display:flex;align-items:center;gap:8px;font-size:12px;}
.step-method{font-weight:700;font-size:11px;letter-spacing:.04em;color:var(--color-primary);}
.step-url{color:var(--color-text-sub);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.step-meta{margin-left:auto;color:var(--color-text-muted);flex-shrink:0;}
.sp-ok .step-meta{color:var(--color-green);}
.sp-fail{border-color:var(--color-red);}
.sp-fail .step-meta{color:var(--color-red);}
.step-code{margin:6px 0 0;padding:8px 10px;border-radius:6px;background:var(--color-row-alt);
  font-size:11.5px;line-height:1.5;white-space:pre-wrap;color:var(--color-text-sub);}
.step-note{margin:6px 0 0;font-size:12px;color:var(--color-red);}

.prog{display:flex;align-items:center;gap:10px;}
.prog-bar{flex:1;height:6px;border-radius:3px;background:var(--color-row-alt);overflow:hidden;}
.prog-fill{height:100%;background:var(--color-primary);transition:width .3s;}

/* Wide content scrolls inside its own box rather than widening the page. */
.step-code,:deep(.raw-json){overflow-x:auto;}
</style>
