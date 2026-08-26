# The implementation ↔ checker contract

Normative. Every port implements this; the checker knows nothing else about it.

## The boundary is a process, not a library

Each port ships **one executable** and the checker drives it with `exec`, reading
`stdout` and a dump directory. Never a library call, never HTTP.

Why not a library call: the checker would then have to be written in, or bound to,
each port's language, and would share that language's own float, rounding and sort
bugs with the thing it is judging.

Why not HTTP: it would force the *service* port to exist before the *library* port
could be graded, which is exactly backwards — the library is milestones 1–7, the
service is milestone 9.

## The executable

Name it `rdocs-conform` (`.exe` on Windows). Registered in `conformance/ports.json`.

### `rdocs-conform info`

Emits one JSON object on stdout describing the implementation:

```json
{
  "port": "go",
  "language": "Go 1.26.5",
  "versions": {"runtime": "go1.26.5", "onnxruntime": "1.21.1", "opencv": "4.13.0"},
  "device": "cpu",
  "ocr_device": "cpu",
  "providers": ["CPUExecutionProvider"],
  "model_format": "ONNX",
  "ocr_mode": "accurate",
  "stages_implemented": ["prepare", "doctype.label", "rotate"],
  "commit": "a1b2c3d"
}
```

`stages_implemented` is what makes a partial port gradeable: the checker skips
what an implementation does not claim, instead of failing it.

### `rdocs-conform recognize --image <path> [flags]`

Emits **the view model** on stdout and nothing else.

Flags, all optional: `--device cpu|gpu`, `--ocr accurate|fast`, `--img-size N`,
`--docconf F`, `--include-debug`.

### `rdocs-conform probe --image <path> --dump-dir <dir> [--upto <stage>] [flags]`

Writes one file per stage into `<dir>`, plus `stages.json` as an ordered index.
Same optional flags as `recognize`.

`--upto <stage>` stops **after** the named stage (inclusive). This is the mechanism
that makes each milestone verifiable before the pipeline is finished, and it is the
reason the harness exists rather than being written at the end.

Payload format per stage is fixed by `stages.md`: `.npy` for arrays, `.json`
otherwise. The `.npy` subset is defined in `npy-subset.md`.

## stdout discipline

**stdout carries only the payload.** All logging, warnings and progress go to
stderr.

This is not a stylistic preference. `document_processing` prints to stdout —
`Pipeline.warmup` swallows exceptions into a bare `print`, `process_img` prints
`[!] The document on picture has unknown type`, and every module prints
`[*] Loading model X!` at construction. The reference CLI must therefore redirect
the library's stdout to stderr, and a port that logs to stdout will produce
unparseable output that looks like a serialisation bug.

**stdout is UTF-8, always, whatever the machine's locale is.** The checker decodes
it as UTF-8 and has no way to detect that it received something else: the bytes of
a Cyrillic string in another code page are still bytes, and they arrive as damaged
text rather than as an error.

Measured, which is why this is normative rather than assumed: on a Windows machine
whose locale is cp1251, the Python reference — which had not pinned its stdout —
failed against its **own** goldens with 18 differences, every one a Cyrillic OCR
string turned into replacement characters, and the checker then died with
`UnicodeEncodeError` while printing the report. Nothing in that red mentioned the
console. Note where it did *not* bite: the per-stage dump files, which are written
with an explicit encoding, were byte-correct in the same run. Only the stream had
no encoding of its own.

A port must therefore set its own output encoding rather than inherit it: in Python
`sys.stdout.reconfigure(encoding="utf-8")`, in .NET `Console.OutputEncoding =
new UTF8Encoding(false)` (on Windows it otherwise follows the console code page),
in Go and on a modern JVM this already holds. The checker also pins
`PYTHONIOENCODING=utf-8` in the port's environment, but that is a second belt for
Python ports only — it is not the contract.

## Exit codes

| code | meaning | checker's reaction |
|---|---|---|
| `0` | ran | compare the output |
| `2` | stage or feature not implemented | **skip**, do not fail |
| `3` | input error (missing/undecodable image, bad flag) | fail the case |
| `1` | crash | fail the case, capture stderr |

`2` versus `1` is the important distinction: a port under construction must be
able to say "not yet" without being scored as broken. That is what lets M2 be
green while M6 does not exist.

## What the checker guarantees in return

* It never imports a port, and never imports `document_processing`. Only
  `conformance/refcli/` may import the library.
* It derives its case list from `service/seed_data/manifest.json`, so the cases and
  the service's seed data cannot drift apart.
* It reports the **first divergent stage** per case as the headline result.
