# .NET port

A reimplementation of `document_processing/` and `service/` on .NET 8. One of four reference
integrations (Python, Go, .NET, Kotlin/JVM, plus C++), graded against the Python reference by
[`../../conformance/`](../../conformance/).

**Read in this order:** [`../CONVENTIONS.md`](../CONVENTIONS.md) (normative for every port,
not only that one) → [`../MAPPING.md`](../MAPPING.md) (which file corresponds to which) →
[`DEVIATIONS.md`](DEVIATIONS.md) (where this port cannot or should not follow Python) →
[`ARCHITECTURE.md`](ARCHITECTURE.md) (how this implementation actually works). A decision that
appears in none of them is a gap in the *design* — add it there rather than deciding locally.

All four ports and how they compare: [`../README.md`](../README.md).

## Status: complete

| Milestone | Scope | State |
|---|---|---|
| M1 | imaging, tensors, config | **done — `prepare` 7/7** |
| M2 | inference, model loader, DocTypeAngles | **done — `doctype.label`, `rotate` 7/7** |
| M3 | quality group + the parallel primitive | **done — `quality` 7/7** |
| M4 | borders (segmentation) + deskew | **done — 7/7** |
| M5 | text fields + word splitting | **done — 7/7** |
| M6 | OCR | **done — per-word exact, 7/7** |
| M7 | view model + `recognize` | **done — 44/44 stages, zero skips** |
| M8 | GPU + soak | **done — 44/44 on the gpu profile; memory plateaus** |
| M9 | the service | **done — 37/37 wire checks, 7/7 recognitions, cpu and gpu** |
| M10 | Docker | **done — both images built, run and verified** |

Conformance: **PASS on 44/44 stages across all seven cases, zero skips, on both the `cpu` and
`gpu` profiles.** Two stages pass under relaxation R-02 (`borders.canvas`, `deskew.canvas`):
`warpPerspective` interpolation differs by at most one grey level on 0.02% of pixels between
OpenCV minors.

Tests: `dotnet test` — 11 library + 23 service = **34 passing**.


## Как библиотека, из своего приложения

Пакета в NuGet пока нет (см. `ports/README.md`), поэтому подключение — ссылкой на проект:

```bash
dotnet add reference path/to/ports/dotnet/src/RussianDocs.DocumentProcessing/RussianDocs.DocumentProcessing.csproj
```

Распознать один файл — десять строк:

```csharp
using RussianDocs.DocumentProcessing.Inference;   // Device
using RussianDocs.DocumentProcessing.Modules;     // OcrTier
using RussianDocs.DocumentProcessing.Pipeline;

// SLOW: 12 сессий, 215 МБ весов. Создаётся один раз на процесс.
using var recognizer = new Recognizer(Device.Cpu, intraOpThreads: 0, ocrTier: OcrTier.Accurate);

// Results владеет изображениями — `using` обязателен, иначе утечка вне управляемой памяти.
using Results results = recognizer.Run("passport.jpg", new RunOptions { Docconf = 0.5, ImgSize = 1500 });

Console.WriteLine(results.DocType);                    // "INTPASSPORT_2011" или "NONE"
foreach ((string field, string value) in results.Ocr)  // поля документа
    Console.WriteLine($"{field}: {value}");

// Проводной вид (тот же JSON, что отдаёт сервис) — если нужно отдать наружу:
var payload = Recognizer.BuildViewModel(results, includeDebug: false);
```

`doc_type == "NONE"` — не ошибка, а нормальный короткий возврат: документ не распознан, `Results`
заполнен, полей нет. Бросать здесь исключение сломало бы штатный сценарий.

Нативные зависимости, которые придётся довезти вместе с приложением: `OpenCvSharp4.official.runtime.*`
под вашу платформу (на Linux он **не** headless и требует GTK-стек — точный список в
`docker/Dockerfile`) и `Microsoft.ML.OnnxRuntime` либо `.Gpu`.

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

## What this port is for

The same thing the Go port is for: showing how to integrate the library correctly in a second
stack. The interesting file is [`src/RussianDocs.Service/Ml/PipelineRuntime.cs`](src/RussianDocs.Service/Ml/PipelineRuntime.cs),
which carries the ten numbered correctness rules from `service/ml/runtime.py` — including a note
on which two do **not** apply here and why the lease survives anyway.

Two things it does *not* buy:

- **Speed.** 422–943 ms per document, the same as the reference and the Go port. Almost all of
  that time is inside ONNX Runtime and OpenCV, which are the same C++ in all three.
- **Image size.** 2.31 GB (CPU) and 7.56 GB (GPU), *larger* than Go's 902 MB / 6.73 GB, mostly
  because OpenCvSharp's native package is not headless. See `D-07` in
  [`DEVIATIONS.md`](DEVIATIONS.md).

What it does buy: one build for both devices (`N-01`), a service with **zero** dependencies
beyond the two native libraries the library needs, and a Dockerfile a third the size of Go's
because there is no OpenCV or ONNX Runtime to compile or download.

## Quick start

```bash
cd ports/dotnet
dotnet build RussianDocs.sln -c Release
dotnet test RussianDocs.sln -c Release

# conformance, from the repository root
python -m conformance.runner run --port dotnet
python -m conformance.runner run --port dotnet --profile gpu

# the leak check the conformance harness structurally cannot perform
./src/RussianDocs.Conform/bin/Release/net8.0/rdocs-conform soak --rounds 4 --dir samples
```

Run the service:

```bash
DATA_DIR=/var/tmp/rdocs DEFAULT_API_KEY=rdk_dev JWT_SECRET=dev \
  ./src/RussianDocs.Service/bin/Release/net8.0/rdocs-service --addr :8004
```

Then <http://127.0.0.1:8004> — PIN `1234`. The log seeds itself from `service/seed_data/`, so
there is something to click immediately.

Docker (from the repository root):

```bash
docker build -f ports/base/Dockerfile.models -t russiandocs-models:latest .
docker build -f ports/dotnet/build/Dockerfile --target runtime-cpu -t russiandocs-dotnet:cpu .
docker run --rm -p 8004:8004 -e JWT_SECRET=... -e DEFAULT_API_KEY=... russiandocs-dotnet:cpu
```

Swap `runtime-cpu` for `runtime-gpu` and add `--gpus all` for the GPU image.

## Handling rules that are not negotiable

- **`DATA_DIR` must live outside the repository.** It holds uploaded documents, which are
  personal data.
- **`WARMUP_IMAGE` may only point at an anonymised `samples/` file.** Warmup re-reads it at
  every start of every deployment.
- **Nothing under `ports/` may contain a model, a sample or a copy of `model.json`.** All four
  ports resolve them from the repository root; four copies of 215 MB in git history is forever.
