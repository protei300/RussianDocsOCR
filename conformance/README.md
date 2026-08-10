# Conformance harness

How an implementation of this library in another language is proved equivalent to the
Python reference.

## Why this exists

The library is being reimplemented in Go, .NET, Kotlin and C++. Comparing only the
final JSON of a port tells you *that* it diverged, never *where* — and on a
twelve-model pipeline that is a week of bisection. Comparing intermediate stages turns
it into "first divergence at `fields.bbox`", which is an hour.

Everything here supports that one idea.

## Quick start

```bash
# what is registered, and which cases have goldens
python -m conformance.runner list

# grade the reference against its own goldens: must be zero differences
python -m conformance.runner run --port python

# grade a port
python -m conformance.runner run --port go
python -m conformance.runner run --port go --profile gpu --verbose

# regenerate the goldens (deliberate, reviewable, its own commit)
python -m conformance.refcli regen
```

On Windows call the interpreter by absolute path — `conda run` mangles flags on this
machine:

```powershell
D:/miniconda3/envs/russiandocs/python.exe -m conformance.runner run --port python
```

## Layout

```
spec/       normative documents -- read stages.md first
refcli/     the Python REFERENCE cli; the only package here that may import
            document_processing
runner/     the CHECKER; imports no port and no library, drives everything by
            subprocess
cases/      golden data, one directory per document (~170 KB total)
tools/      operational helpers, not part of the contract
```

The dependency rule is the load-bearing part: a checker that shared code with the
thing it judges would share its bugs. The ABI is `subprocess`.

## What a golden case contains

```
cases/<slug>/
  case.json               the sample, the flags, the expected doc_type
  viewmodel.json          the final client-facing JSON
  stages/
    stages.json           ordered index -- this order IS pipeline order
    <stage>.json          small payloads, compared by value
    <stage>.digest.json   image stages: sha256 over the array bytes
```

**Image pixels are not committed.** Four image stages per case at ~2 MB each across
seven cases is 50+ MB of binary in git, growing with every document type. The digest
still answers "which stage diverged first"; for the magnitude — and to judge
relaxation R-02 — regenerate pixels locally:

```bash
python -m conformance.refcli regen --with-pixels --case DL_2011
```

Those `.npy` files are gitignored.

## The rules, in one paragraph

Numeric values within absolute **1e-3**; discrete values (document type, every OCR
string, quality verdicts, class labels, counts, the JSON key set) **exactly**. Never
byte-diff JSON — `round()` breaks ties differently in Python, Go, C# and Kotlin, so
float *text* legitimately differs while values agree. Timings and environment fields
are checked for presence only. A GPU run uses a separate numeric profile, because
absolute 1e-3 is the wrong metric for box coordinates: measured, CUDA differs from CPU
by up to 0.50 on values reaching 635 — sub-pixel — while every discrete outcome is
identical. Full detail in `spec/tolerances.md`.

## Proving the harness itself

A harness that has never been seen to fail is worthless. Two checks, both part of M0
and both currently green:

1. **The reference against its own goldens: exactly zero differences.** 7 cases, 153
   stages.
2. **A deliberately perturbed constant must fail AND name the right stage:**

```bash
python -m conformance.tools.perturb --set TextFields.CLS=0.8
python -m conformance.runner run --port python --case DL_2011   # -> fields.bbox
python -m conformance.tools.perturb --restore
```

Observed: `fields.bbox` reported as the first divergence (21 boxes → 19), followed by
the causal chain — `ocr.Living_region_ru.words` missing, then `join`, then the view
model. That is the localisation this harness is for.

Note that not every constant is detectable: perturbing the TextFields NMS **IOU**
from 0.2 to 0.45 changed nothing, because same-class text fields on these documents do
not overlap enough for NMS to behave differently. That is a property of the case set,
not a defect — but it is why the proof uses a confidence threshold instead.

## Adding a case

Add a document type to `service/seed_data/` (`python service/tools/build_seed_data.py`)
and the case appears automatically: `conformance/cases.py` derives the list from that
manifest, so the two cannot drift apart.

`INTPASSPORTADDR` has no case, because the repository has no anonymised sample of a
registration page. Its `address.*` stages and the `angle_rad` sign convention are
therefore unverified — see `spec/viewmodel.md`.
