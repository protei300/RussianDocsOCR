# Go port

A reimplementation of `document_processing/` and `service/` in Go. One of four
reference integrations (Python, Go, .NET, Kotlin/JVM, plus C++), graded against the
Python reference by `../../conformance/`.

**Read in this order:** [`../CONVENTIONS.md`](../CONVENTIONS.md) (normative for every port, not only
this one) → [`../MAPPING.md`](../MAPPING.md) (which file corresponds to which) →
[`DEVIATIONS.md`](DEVIATIONS.md) (where a port cannot or should not follow Python) →
[`ARCHITECTURE.md`](ARCHITECTURE.md) (how this implementation works). A decision that appears in none
of them is a gap in the design; add it there rather than deciding locally.

The first two used to live in this directory, because Go was the first port and there was nowhere else
to put them. They are shared by all four and now sit one level up — see [`../README.md`](../README.md)
for the whole set.

## Status: complete — M1 through M10.5

| Milestone | Scope | State |
|---|---|---|
| M1 | imaging, tensor, config | **done — `prepare` byte-identical on 7/7** |
| M2 | inference, model loader, DocTypeAngles | **done — type, DocConf, angle exact on 7/7** |
| M3 | quality group + the parallel primitive | **done** |
| M4 | borders (segmentation) + deskew | **done — R-01/R-02 relaxations written here** |
| M5 | text fields + word splitting | **done** |
| M6 | OCR | **done — every word, both engines** |
| M7 | view model + `recognize` | **done — 44/44 stages, zero skips, 7/7 cases** |
| M8 | GPU, the session mutex, the soak | **done — gpu profile 44/44; the 663 → 4018 MB leak found and fixed here** |
| M9 | the service | **done — verified against `service/seed_data`** |
| M10 / M10.5 | Docker (cpu + gpu, both built) / [`ARCHITECTURE.md`](ARCHITECTURE.md) | **done** |

Verify it rather than trusting the table:

```bash
python -m conformance.runner run --port go
python -m conformance.runner run --port go --profile gpu --device gpu
```

M2 confirmed, on all seven documents, that the document type, its confidence
(`quality.DocConf`), the 90-degree angle and its confidence all match the reference
exactly — including `angle_confidence` to the last digit. That single stage exercises
the ONNX binding, the declared-dtype casting, the `<U64` label decode from
`centers.npz`, the cosine-radius plus per-class-threshold semantics, the loader's
three-switch dispatch and the positional output-to-postprocessor mapping.

M3 added the four quality classifiers and the concurrency primitive. Worth noting it is
not a trivial stage: Glare and Blur classify **28 tiles each** on a 7×4 canvas and
aggregate them with different rules (Glare counts confidently-glared tiles; Blur weighs
three of five labels and ignores the rest, including from the denominator), while the
two spoofing heads classify the whole image and PrintSpoofing applies a 0.9 gate on top
of the model's own threshold. That gate genuinely fires — the SNILS case yields
`PrintSpoofing: FAKE` in both implementations — so the stage is substantive rather than
uniformly green.

M4 added border segmentation and deskew. Its last defect is worth reading before
writing another port: `cv2.convexHull` defaults to `clockwise=False`, every binding
makes that argument explicit, and the hull's orientation decides which vertices
Douglas-Peucker keeps — so the wrong default silently yields a different document quad.
See CONVENTIONS trap 15.

Building and running needs the toolchain, the OpenCV DLLs and the ONNX Runtime library.
Dot-source `env.ps1` (or `. ./env.sh` on Linux) rather than setting them by hand:

```powershell
. .\env.ps1
.\build.ps1 -Test
```

ONNX Runtime is loaded **by path at runtime**, not linked, so without `ORT_DLL` the CLI
exits 1 on every case with `set ORT_DLL/ORT_SO to a matching build` — which reads like a
broken build and is not one. It deliberately points at the same conda environment the
reference uses (1.21.1); comparing across two ORT versions would confound "Go differs
from Python" with "1.21 differs from 1.28".

For `--device gpu` the `nvidia/*/bin` directories also need to be on PATH — the same set
Python's `_enable_cuda_dlls()` registers.

Verify at any time:

```powershell
D:/miniconda3/envs/russiandocs/python.exe -m conformance.runner run --port go
```

Stages this build does not implement are reported through `info.stages_implemented`
and the checker **skips** them rather than failing — that is what makes a partial port
gradeable.


## Как библиотека, из своего приложения

```go
import (
    "github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/inference"
    "github.com/protei300/RussianDocsOCR/ports/go/internal/docproc/pipeline"
)

// SLOW: 12 сессий, 215 МБ весов. Один экземпляр на процесс.
rec, err := pipeline.NewRecognizer(pipeline.RecognizerOptions{
    Device:   inference.CPU,     // тип Device — строка: "cpu" | "gpu"
    OcrTier:  "accurate",
    Threads:  0,          // 0 — пусть ORT решает; 1 нужен только для сверки с golden
})
if err != nil { return err }
defer rec.Close()

// Results владеет изображениями; Close освобождает КАЖДОЕ промежуточное, а не только канвас.
res, err := rec.Run("passport.jpg", pipeline.RunOptions{Docconf: 0.5, ImgSize: 1500})
if err != nil { return err }
defer res.Close()

fmt.Println(res.DocType)              // "INTPASSPORT_2011" или "NONE"
for field, value := range res.Ocr {   // поля документа
    fmt.Printf("%s: %s
", field, value)
}
```

Пакеты лежат под `internal/`, потому что публичного API у порта пока нет — это осознанно (см.
`ports/README.md`, пункт про упаковку). Чтобы вызывать из другого модуля, вынесите нужные пакеты из
`internal/` или используйте порт как сервис.

`DocType == "NONE"` — не ошибка, а нормальный короткий возврат: документ не распознан.

Нативное: OpenCV 4.13 через cgo (**headless**, `-DWITH_GTK=OFF -DWITH_QT=OFF` — иначе gocv затащит Qt) и
`libonnxruntime`, путь к которой задаётся `ORT_SO`/`ORT_DLL`.

Три правила, которые не видны из подписи методов и стоят дороже всего:

1. **Экземпляр дорогой — создавайте один и держите.** Это 12 ONNX-сессий и 215 МБ весов; на GPU второй
   экземпляр это ещё и второй CUDA-контекст. Эталонный сервис заворачивает его в пул размера **1**.
2. **Один экземпляр — один документ за раз.** Порт не воспроизводит питоновскую нереентерабельность
   (состояние живёт в локальных переменных), но GPU-сессия сериализуется мьютексом вокруг вызова:
   8 потоков по одной CUDA-сессии без него — 600+ секунд против 6.6 с. Параллельте документы пулом
   экземпляров, а не вызовами одного.
3. **OCR остаётся на CPU даже когда детекторы на GPU.** Это не недоделка: замерено 13.7-кратное
   замедление на CUDA, потому что каждое слово имеет свою ширину и рантайм перекомпилирует граф.

Модели берутся из репозитория: путь ищется от рабочего каталога вверх до `document_processing/models`,
или задаётся переменной **`RDOCS_MODELS_ROOT`** (она указывает на КОРЕНЬ, а не на каталог моделей).
В своём приложении проще всего скопировать `document_processing/models` и `document_processing/config`
рядом с бинарём и выставить `RDOCS_MODELS_ROOT` на этот каталог — ровно так делают Dockerfile'ы портов.

Полная картина — устройство типов, владение изображениями, конкурентность и таксономия ошибок — в
[`ARCHITECTURE.md`](ARCHITECTURE.md). Если вам нужен не встроенный вызов, а отдельный сервис, он уже
написан: тот же порт содержит `rdocs-service`, а клиент к нему —
[`web/src/client/rdocs-client.ts`](../../web/src/client/rdocs-client.ts).

## Build

```powershell
.\build.ps1              # Windows
.\build.ps1 -Test        # and run the unit tests
```
```bash
./build.sh               # Linux
./build.sh --test
```

Windows needs MSYS2's prebuilt OpenCV; there is no reason to build it from source:

```powershell
C:\msys64\usr\bin\pacman.exe -S --needed mingw-w64-x86_64-opencv `
    mingw-w64-x86_64-gcc mingw-w64-x86_64-pkgconf mingw-w64-x86_64-qt6-base
```

## The three pinned dependency facts

1. **`onnxruntime_go v1.19.0`, not the newest.** Each tag hardcodes an
   `ORT_API_VERSION` and needs a shared library at least that new; the C API is
   backward compatible only one way. v1.19.0 vendors 21, matching the ONNX Runtime
   **1.21** the repository pins — so this port uses the same runtime *and* the same
   CUDA stack as Python, and its Docker image can reuse the same
   `nvidia/cuda:12.6.3-cudnn` base. v1.32.0 would demand ORT 1.28, whose CUDA build
   wants CUDA 12.8 and fails to load against 12.6 with a bare "Error 127".
2. **OpenCV 4.13 via gocv 0.43.0** (its README says 4.12 and is stale — its own
   default link flags name `opencv_core4130`). Python is on 4.12.0.88, and the two were
   measured bit-identical across every operation the pipeline performs, on ~14 million
   pixel values. Build headless: see `DEVIATIONS.md` D-07.
3. **cgo is mandatory**, and not only because of OpenCV: `onnxruntime_go` is itself a
   cgo binding. There is no pure-Go build of this port, and dropping cgo would be a
   change of *conformance policy*, not an optimisation — a pure-Go image path differs
   from OpenCV by up to 14 LSB on most pixels.

## Windows: the DLL trap

`build.ps1` copies four DLLs next to the binary and this is not optional. See
`DEVIATIONS.md` D-09: `C:\Windows\System32` may hold older copies of
`libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll` and `zlib1.dll`, and
Windows searches the application directory and System32 **before** PATH — so a binary
elsewhere dies at load with `0xC0000139`, silently, before `main()`.

Symptom to recognise: the binary fails, but the identical file runs when copied into
`C:\msys64\mingw64\bin`.

The same trap breaks `go test`, which builds into a temp directory — hence
`build.ps1 -Test` compiles test binaries into `bin\` and runs them there.

## Layout

```
cmd/rdocs-conform/     the conformance CLI (spec: ../../conformance/spec/cli.md)
cmd/rdocs-service/     the HTTP service                              (M9)
internal/docproc/      the library port
  imaging/             THE ONLY package that may import gocv
  tensor/              arrays, .npy/.npz, Python-semantics numerics
  config/              models_path.yaml, ocr_alphabets.json
  preprocess/ postprocess/ inference/ models/ modules/ pipeline/
internal/viewmodel/    PipelineResults -> the client JSON   (library side; D-01)
internal/svc/          the service port                              (M9)
```

`internal/docproc/**` must not import `internal/svc/**`, and the service reaches the
library only through `internal/svc/runtime` — the same boundary that keeps the Python
service testable without 215 MB of models.

**Nothing under `ports/` may contain a model, a sample or a copy of a `model.json`.**
The data lives once, in `document_processing/`, and is found via `RDOCS_MODELS_ROOT` or
by walking up from the executable.
