# ports/ — the same library and service in four languages

`document_processing/` and `service/` are the Python reference. This directory holds
reimplementations of both, one per language, each graded against that reference by
[`../conformance/`](../conformance/) rather than by inspection.

The point is not redundancy. It is that **the integration layer is the deliverable** — how you hold a
non-thread-safe pipeline that eagerly loads 215 MB of weights, whose result object lives only until the
next call — and a rule stated once in Python is a claim, while the same rule arriving independently in
four languages is a design. Each port also stands alone: an integrator on the JVM copies
`ports/java`, not a translation guide.

## The four

| Port | Language / stack | Service port | Conformance | Docker |
|---|---|---|---|---|
| — | **Python 3.12** — the reference | 8002 | it defines the goldens | built, 7.72 GB (GPU); runs as `rdocs`, not root |
| [`go`](go/) | Go 1.24, cgo, gocv, `onnxruntime_go` | 8003 | **44/44, cpu + gpu** | built, 902 MB (CPU) / 6.73 GB (GPU) |
| [`dotnet`](dotnet/) | .NET 8, OpenCvSharp4, `Microsoft.ML.OnnxRuntime` | 8004 | **44/44, cpu + gpu** | built, 2.31 GB / 7.56 GB |
| [`java`](java/) | Kotlin 2.4 on JDK 21, `org.opencv`, ORT 1.21.1 | 8005 | **44/44, cpu + gpu** | built, 2.01 GB / 7.21 GB |
| `cpp` | — | — | not started | — |

"44/44" means every stage of the pipeline, on all seven sample documents, with **zero skips** — not a
green end-to-end comparison. The stage list is [`../conformance/spec/stages.md`](../conformance/spec/stages.md);
the tolerance rules are [`tolerances.md`](../conformance/spec/tolerances.md) (floats `abs <= 1e-3`,
discrete outputs exactly).

Honest comparisons, all measured rather than assumed:

- **Recognition speed is a three-way tie on CPU.** One file through all three services interleaved in
  the same minute: Python 595 ms, Kotlin 592 ms, .NET 824 ms. Almost all of the work is ONNX Runtime
  and OpenCV, which is the same C++ in every port.
- **Only the CPU image is dramatically smaller.** Go's 902 MB against Python's ~1.5–2 GB is real; the GPU
  images are 6.73 / 7.21 / 7.56 / 7.72 GB (Go, Kotlin, .NET, Python) — within 15 % of each other, because
  CUDA and cuDNN dominate and are the same bytes in all four. **No language makes a GPU image small.**
- **Startup and memory are where the ports win.** Twelve ONNX sessions build in 531 ms (Go), 665 ms
  (.NET), 513–803 ms (Kotlin — 513 ms measured inside the container) against ~15 s for the Python
  service on GPU — mostly the absence of Python imports.
- **A CPU image can carry GPU weight by accident.** The Kotlin CPU image is 2.01 GB because its fat jar
  bundles `onnxruntime_gpu`, ~450 MB of which is CUDA kernels a CPU host never runs. Known, measured,
  and deliberately not yet fixed — see the note at the top of `ports/java/docker/Dockerfile`.
- **Kotlin is the only port that pins the reference's exact ONNX Runtime**, 1.21.1. Neither Go nor
  .NET publishes an artefact at that patch.

## Which document answers which question

Two documents are **normative for every port** and live here, not inside one of them:

| | |
|---|---|
| [`CONVENTIONS.md`](CONVENTIONS.md) | how to write a port. What to avoid in each language, the numeric traps, the shape of the parallel groups and the lease. A decision that appears in none of the documents is a gap in the *design* — add it here rather than deciding locally. |
| [`MAPPING.md`](MAPPING.md) | which Python file becomes which file in each port, and the order of the cases in the three loader switches. **Verified against the trees**, not derived by rule. |

Then, inside each port, the same four:

| | |
|---|---|
| `README.md` | what it is, how to build it, what it needs from the machine, and the traps that are specific to this toolchain |
| `ARCHITECTURE.md` | how *this* implementation works — package tree, ownership, one recognition end to end, concurrency, resource lifetime, the error taxonomy, and what differs from Python on purpose |
| `DEVIATIONS.md` | numbered, where this port legitimately differs. `D-nn` are shared across ports; a language prefix (`N-nn` for .NET, `J-nn` for Kotlin) marks one that belongs to that language alone |
| `PLAN.md` | the milestone order it was built in, kept as a record of what was verified when |

Plus [`PORTING-LESSONS.md`](PORTING-LESSONS.md) — what the finished ports would tell the next one, and
[`base/Dockerfile.models`](base/) — the 215 MB model layer, built once and copied into every runtime
image so four ports do not pull four copies.

## Building and verifying one

Each port has its own prerequisites; the details are in its README, and none of them is guessable from
an error message. The verification is the same everywhere:

```bash
# from the repository root, with the port built
python -m conformance.runner run --port go        # or dotnet, java
python -m conformance.runner run --port go --profile gpu --device gpu

# the leak check the conformance harness structurally CANNOT do:
# it runs one document per process, so it cannot see a leak at all
<port>/bin/rdocs-conform soak --rounds 4 --threads 8
```

Two rules with no exceptions:

- **Nothing under `ports/` may contain a model, a sample, or a built binary.** Every port resolves
  models from the repository root through `RDOCS_MODELS_ROOT`; OpenCV builds and JDKs live outside the
  tree. Check it:
  `git ls-files ports/ | grep -E '\.(onnx|npz|jpg|exe|dll|so|jar)$'` must print nothing.
- **`conformance/runner` never imports a port**, and never imports `document_processing`. The interface
  is `exec` plus stdout plus a dump directory; the checker is Python because Python is the reference,
  and generating a golden and checking against one must be one code path.

### Without a single installed toolchain

Every port can be graded formally on a machine that has neither Go, nor .NET, nor a JDK —
proven in practice on a Windows host where none of the three was installed (2026-08-10):

- **Go** — the runtime image already contains the conformance CLI next to the service
  binary. Extract it into the (gitignored) host path the runner expects and run the
  runner in the same container, with python added on top:

  ```bash
  printf 'FROM russiandocs-go:cpu
USER root
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 python3-numpy
' > gopy.Dockerfile
  docker build -f gopy.Dockerfile -t rdocs-go-conform:check .
  docker run --rm -v <repo>:/host -w /host --entrypoint sh rdocs-go-conform:check -c     "cp /usr/local/bin/rdocs-conform /host/ports/go/bin/ && chmod +x /host/ports/go/bin/rdocs-conform && python3 -m conformance.runner run --port go"
  ```

- **.NET** — build and evaluate in `mcr.microsoft.com/dotnet/sdk:8.0-jammy` with
  GTK/Pango/Cairo installed (the OpenCvSharp native package is not headless). Watch one
  trap: the Linux restore rewrites `packages.lock.json` to `linux-x64` runtime packages —
  do not let that reach a commit, it breaks every Windows build.
- **Kotlin** — build in the port's own builder image, where OpenCV is already compiled;
  the JNI library needs `LD_LIBRARY_PATH=/opt/opencv/lib` to find its own neighbours.

`ports/*/bin/` is gitignored, so an extracted binary cannot reach a commit.

## Where the shared frontend fits

`web/` is one Vue SPA, served unchanged by all four services — a third of the project that is not
ported at all, because the REST contract is the same. It also carries
[`web/src/client/rdocs-client.ts`](../web/src/client/rdocs-client.ts): a dependency-free TypeScript
client, and the page at `/integration` that demonstrates a site calling any of the four (it finds
whichever is running).

That is the practical consequence of the contract being written down rather than implied: **the UI
cannot tell the implementations apart, and neither can a client.**
