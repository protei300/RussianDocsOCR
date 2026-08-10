# The `.npy` subset

Normative. Array payloads are exchanged as NumPy `.npy`, restricted to what is
listed here. Anything outside this subset is an error, not a best-effort guess.

## Why `.npy` at all

Considered and rejected:

* **JSON numbers** — 2–3× the size, no shape, and cross-language `repr` ambiguity
  on floats.
* **A flat float32 blob plus a JSON sidecar** — the sidecar describes the shape,
  so a transposed payload compares *equal* and passes. That is exactly the failure
  mode that costs a day here: NHWC vs NCHW, or `[1,T,C]` vs `[T,C]`.
* **protobuf / flatbuffers** — schema plus codegen for a comparison format.
* **MessagePack** — another dependency, in every language.

`.npy` wins on three counts: it is native on the judging side (Python is the
reference, so `np.load` is zero code and cannot itself be wrong); it is
self-describing, so dtype, shape and memory order travel with the bytes and a
mis-shaped payload fails loudly; and **every port needs a reader anyway**, because
`models/DocTypeAngles/ONNX/resources/centers.npz` is a zip of three `.npy` members.

## Supported

| aspect | requirement |
|---|---|
| version | **1.0 only** (`\x93NUMPY\x01\x00`) |
| byte order | little-endian |
| order | C-contiguous; `fortran_order: True` is an error |
| dtypes | `<f4`, `<f8`, `\|u1`, `<i8`, `<U<n>` |

`<U<n>` is fixed-width **UTF-32LE**, `n` code points, NUL-padded on the right.
`centers.npz`'s `labels` is `<U64`, i.e. 256 bytes per row. Naive byte slicing
yields `I\0\0\0N\0\0\0T…` instead of `INTPASSPORT_2011` — the reader must decode
UTF-32LE and trim trailing NULs.

## Not supported, deliberately

* **Pickled object arrays.** A code-execution vector. The library itself moved off
  `centers.pkl` to `centers.npz` with `allow_pickle=False` for this reason, and
  readers must pass the equivalent flag.
* `.npy` **v2/v3** headers (4-byte length, UTF-8 names) — nothing the reference
  writes needs them.
* big-endian, Fortran order, structured dtypes.

## Header details that bite

* The header is a **Python literal**, not JSON: single quotes, `True`/`False`, and
  a trailing comma in one-element tuples (`'shape': (3,)`). A JSON parser appears
  to work until it meets a 1-D array.
* The header must be padded with spaces so that **the data begins on a 64-byte
  boundary**, and must end with `\n`. Get the padding wrong and a hand-written
  reader will accept the file while NumPy rejects it — so writers must be verified
  by loading their output with NumPy, not with their own reader.
* A zero-length `shape` (`()`) is a scalar: one element, not zero.

## Verification

`conformance/runner` compares with `numpy.load(..., allow_pickle=False)`. A port's
writer is considered conformant when NumPy loads its output and a round-trip is
bit-identical in both directions. The Go implementation and its round-trip test
live in the spike at `D:\Grant\go-spike\internal\npy\` and will move into
`ports/go/internal/docproc/tensor/` in M1.
