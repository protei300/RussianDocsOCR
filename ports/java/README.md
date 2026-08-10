# Kotlin/JVM port

A reimplementation of `document_processing/` and `service/` in Kotlin on the JVM. One of four
reference integrations (Python, Go, .NET, Kotlin/JVM, plus C++), graded against the Python reference
by [`../../conformance/`](../../conformance/).

The directory is `ports/java/` rather than `ports/kotlin/` because the artefact is a JVM build and the
layout is Gradle's; the language inside it is Kotlin.

**Read in this order:** [`../CONVENTIONS.md`](../CONVENTIONS.md) (normative for every port, not
only that one) → [`../MAPPING.md`](../MAPPING.md) → [`DEVIATIONS.md`](DEVIATIONS.md) →
[`PLAN.md`](PLAN.md). A decision that appears in none of them is a gap in the *design* — add it there
rather than deciding locally.

All four ports and how they compare: [`../README.md`](../README.md).

## Status: complete — M0 through M10.5

| Milestone | Scope | State |
|---|---|---|
| Phase 0 | infrastructure spike | **done — all six gates green** |
| M0 | Gradle build, CLI skeleton, `ports.json` entry | **done** |
| M1 | imaging, tensors, config | **done — `prepare` byte-identical on 7/7** |
| M2–M7 | inference, quality, borders, fields, OCR, view model | **done — 44/44 stages, ZERO skips, on 7/7 cases** |
| M8 | GPU, the session mutex, the soak | **done — gpu profile 44/44; 460 documents, RSS plateau, 0 failures** |
| M9 | the service | **done — 30 contract tests; 7/7 recognitions identical to `service/seed_data`** |
| M10 | Docker | **written, NOT BUILT** — no daemon on the development machine; the file names its own likely first-build fixes |
| M10.5 | [`ARCHITECTURE.md`](ARCHITECTURE.md) | **done** |

Verify it rather than trusting the table:

```bash
./gradlew build                                                   # 10 + 30 tests
python -m conformance.runner run --port java                       # cpu:  44/44, 0 skips
python -m conformance.runner run --port java --profile gpu --device gpu
```


## Как библиотека, из своего приложения

Артефакта в Maven Central пока нет (см. `ports/README.md`), поэтому — либо `mavenLocal()` после
`./gradlew :docproc:publishToMavenLocal`, либо `implementation(project(":docproc"))` в составном билде.

```kotlin
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.pipeline.OcrTier
import net.russiandocs.docproc.pipeline.Recognizer
import net.russiandocs.docproc.pipeline.RunOptions

// J-01: на Windows нативные библиотеки надо подгрузить в правильном порядке ДО первого вызова.
net.russiandocs.docproc.NativeLibraries.load()

// SLOW: 12 сессий, 215 МБ весов. Один экземпляр на процесс.
Recognizer(Device.CPU, intraOpThreads = 0, ocrTier = OcrTier.ACCURATE).use { recognizer ->
    // Results владеет изображениями: `use` обязателен — Mat живёт вне Java-heap, и GC его не торопит.
    recognizer.run("passport.jpg", RunOptions(docconf = 0.5, imgSize = 1500)).use { results ->
        println(results.docType)                 // "INTPASSPORT_2011" или "NONE"
        results.ocr.forEach { (field, value) -> println("$field: $value") }

        // Проводной вид (тот же JSON, что отдаёт сервис):
        val payload = recognizer.buildViewModel(results, includeDebug = false)
    }
}
```

`docType == "NONE"` — не ошибка, а нормальный короткий возврат: документ не распознан.

Нативное: OpenCV 4.13 с Java-биндингами (собирается `tools/build-opencv.sh`, путь в
`RDOCS_OPENCV_HOME`) и `onnxruntime`/`onnxruntime_gpu` 1.21.1. **На Windows нужен JDK с актуальным
C-рантаймом** — иначе ORT не грузится, см. `DEVIATIONS.md` J-16.

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

## What Phase 0 established, on real runs

| question | answer |
|---|---|
| Do the models produce the reference's numbers? | **Bit-identical** — `max abs diff = 0.000e+00` on Glare and both DocTypeAngles heads |
| Dynamic-width OCR on a long-lived session? | **Yes** — six widths ascending then descending, no shape caching |
| Does CUDA actually bind, or is it merely listed? | **Yes, really** — 65.7 ms → 4.6 ms, 14.3× |
| Does OpenCV 4.13 build with `BUILD_JAVA=ON`? | **Yes**, 690/690 — and it produced the jar without ant |
| Do the needed OpenCV functions exist? | **23/23** |

Three of those results changed the plan, and they are recorded in [`DEVIATIONS.md`](DEVIATIONS.md):
**D-08 does not apply here** (the Java binding accepts a fractional rotation centre, which gocv
cannot), `rotatedRectangleIntersection` **exists** (Go lacks it), and the `dnn` module is **not
needed** because every port implements NMS by hand.

This port also pins **ONNX Runtime 1.21.1** — the reference's exact version. Neither Go nor .NET can:
neither publishes a CPU artefact at that patch.

## Building

### 1. OpenCV, once

There is no official `org.opencv` on Maven Central, and 4.13.0 is mandatory (contour approximation
changed in 4.8, and the goldens encode 4.13). So it is built here:

```bash
tools/build-opencv.sh ~/opencv-build
export RDOCS_OPENCV_HOME=~/opencv-build/build
```

Needs a C++ toolchain, cmake and ninja. **Not ant** — 4.13 assembles the jar itself. The cmake line in
that script is kept identical to the Docker stage's, on purpose: two builds that both claim 4.13.0
while differing in configuration is the hardest divergence to find.

### 2. The port

```bash
./gradlew build
java -jar conform/build/dist/rdocs-conform.jar info
```

A fat jar rather than Gradle's start scripts, because `conformance/ports.json` names one executable
the checker runs with `exec`, and `java -jar` is the same invocation on every platform.

### 3. Conformance

```bash
python -m conformance.runner run --port java
```

## Windows: two native-loading traps, both real

Neither can occur on Linux or in Docker. Both are the JVM's counterpart to `D-09` in the Go port — a
DLL resolved by base name to the wrong copy — and neither is guessable from its error message.

**1. ONNX Runtime versus the JDK's bundled C runtime.** `C:\Program Files\Java\jdk-21\bin` ships
`msvcp140.dll` at 14.31 (early 2022) and the directory of `java.exe` is searched before `System32`, so
`jvm.dll` loads that copy before any of this code runs and ORT cannot initialise against it:

```
UnsatisfiedLinkError: onnxruntime.dll: A dynamic link library (DLL) initialization routine failed
```

**Use a JDK whose bundled runtime is current** — Temurin, Zulu or Corretto builds of 21 track a newer
redistributable than Oracle 21.0.1. This one cannot be worked around in code; `NativeLibraries`
detects the failure and explains it instead.

**2. A MinGW-built OpenCV versus `System32`'s runtime.** `libopencv_java4130.dll` imports
`libstdc++-6.dll` and `libgcc_s_seh-1.dll`, and `System32` may carry its own copies, which win:

```
UnsatisfiedLinkError: libopencv_java4130.dll: The specified procedure could not be found
```

This one **is** fixable in code, and `NativeLibraries` does it: point `RDOCS_TOOLCHAIN_BIN` at the
toolchain's `bin` and the runtime is preloaded in dependency order before the JNI library, so the
import binds to the chosen copy.

```
set RDOCS_OPENCV_HOME=<opencv build dir>
set RDOCS_TOOLCHAIN_BIN=C:\msys64\mingw64\bin
```

## Handling rules that are not negotiable

- **`DATA_DIR` must live outside the repository.** It holds uploaded documents, which are personal
  data.
- **`WARMUP_IMAGE` may only point at an anonymised `samples/` file.** Warmup re-reads it at every
  start of every deployment.
- **Nothing under `ports/` may contain a model, a sample, or a built binary.** All four ports resolve
  models from the repository root, and the OpenCV jar and its 35 MB of native libraries are located
  through `RDOCS_OPENCV_HOME` rather than committed.
