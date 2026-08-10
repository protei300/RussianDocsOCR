# File mapping — all four implementations

One table, four columns. The source of truth for where a thing lives in each port —
**do not derive a file name by rule, look it up here.** The names are NOT mechanical: the same concept
is `postprocess/yolodetector.go`, `Postprocess/Detector.cs` and `postprocess/Detector.kt`, and three
Python modules collapse into one file in every port.

**Verified against the actual trees on 2026-08-06**, not predicted. The earlier version of this table
was written at M0 as a forecast and had drifted in three ways worth knowing about, because they are the
ways any such table drifts: it named files that were never created (the Python source turned out to be
dead code), it guessed names that the implementation chose differently, and it carried a Kotlin package
prefix the real code deliberately does not use. If you change a file name, change this table in the
same commit — a mapping that lies is worse than no mapping, because it is trusted.

Conventions that ARE mechanical, and therefore not repeated per row:

| | Go | C# | Kotlin |
|---|---|---|---|
| file names | `snake_case.go` | `PascalCase.cs` | `PascalCase.kt` |
| library root | `ports/go/internal/docproc/` | `ports/dotnet/src/RussianDocs.DocumentProcessing/` | `ports/java/docproc/src/main/kotlin/net/russiandocs/docproc/` |
| service root | `ports/go/internal/svc/` | `ports/dotnet/src/RussianDocs.Service/` | `ports/java/service/src/main/kotlin/net/russiandocs/service/` |
| namespace tail | `postprocess` | `RussianDocs.DocumentProcessing.Postprocess` | `net.russiandocs.docproc.postprocess` |

All three ports are complete: 44/44 conformance stages on all seven cases, zero skips, on the cpu and
gpu profiles. The **C++ port does not exist yet** and therefore has no column — add one when it starts,
rather than a column of dashes that reads as "missing files".

---

## Library — `document_processing/`

| Python | Go | C# | Kotlin |
|---|---|---|---|
| `config/__init__.py` + `models_path.yaml` | `config/modelspaths.go` | `Config/ModelPaths.cs` | `config/ModelPaths.kt` |
| `config/alphabets.py` + `ocr_alphabets.json` | `config/alphabets.go` | `Config/Alphabets.cs` | `config/Alphabets.kt` |
| `processing/preprocessing.py::BasePreprocessing` | `preprocess/preprocess.go` | `Preprocess/Preprocess.cs` | `preprocess/Preprocess.kt` |
| … `ClassificationPreprocessing` | `preprocess/classification.go` | `Preprocess/Preprocess.cs` | `preprocess/Preprocess.kt` |
| … `YoloPreprocessing` (letterbox) | `preprocess/yolo.go` | `Preprocess/Yolo.cs` | `preprocess/Yolo.kt` |
| … `OCRv2Preprocessing` | `preprocess/ocrv2.go` | `Preprocess/OcrV2.cs` | `preprocess/OcrV2.kt` |
| `processing/postprocessing.py` — base + `MultiClass` + `Metric` | `postprocess/postprocess.go`, `multiclass.go`, `metric.go` | `Postprocess/Postprocess.cs` | `postprocess/Postprocess.kt` |
| … `OCRProbsPostprocessing` | `postprocess/ocrprobs.go` | `Postprocess/OcrProbs.cs` | `postprocess/OcrProbs.kt` |
| … `YOLODetectorPostprocessing` **+** `PerClassYOLODetectorPostprocessing` | `postprocess/yolodetector.go` | `Postprocess/Detector.cs` | `postprocess/Detector.kt` |
| … `YOLOSegmentorPostprocessing` | `postprocess/yolosegmentor.go` | `Postprocess/Segmentor.cs` | `postprocess/Segmentor.kt` |
| `processing/models.py::ModelLoader` + the three switches | `models/loader.go` | `Models/Loader.cs` | `models/Loader.kt` |
| … `model.json` DTOs | `models/modeljson.go` | `Models/ModelJson.cs` | `models/ModelJson.kt` |
| … `YOLODetectionModel` | `models/detection.go` | `Models/DetectionModel.cs` | `models/Loader.kt` |
| … `YOLOSegmentionModel` | `models/segmentation.go` | `Models/DetectionModel.cs` | `models/SegmentationModel.kt` |
| `processing/inference.py::ModelInference` | `inference/onnx.go` | `Inference/Session.cs` | `inference/Session.kt` |
| … device resolution, `gpu_visible()` | `inference/device.go` | `Inference/DeviceResolution.cs` | `inference/DeviceResolution.kt` |
| `pipeline_modules/doctype_angles_classificator/` | `modules/doctypeangles.go` | `Modules/DocTypeAngles.cs` | `modules/DocTypeAngles.kt` |
| `pipeline_modules/{blur,glare}_detector/` + `quality.py` | `modules/quality.go`, `blur.go`, `glare.go` | `Modules/Quality.cs` | `modules/Quality.kt` |
| `pipeline_modules/{lcd,print}_spoofing_detector/` | `modules/spoofing.go` | `Modules/Quality.cs` | `modules/Quality.kt` |
| `pipeline_modules/doc_detector/doc_detector.py` | `modules/docdetector.go` | `Modules/DocDetector.cs` | `modules/DocDetector.kt` |
| `pipeline_modules/doc_detector/image_transformation.py` | `imaging/geometry.go` | `Imaging/Geometry.cs` | `imaging/Geometry.kt` |
| `pipeline_modules/deskewer/deskewer.py` | `modules/deskewer.go` | `Modules/Deskewer.cs` | `modules/Deskewer.kt` |
| `pipeline_modules/textfields_detector/` | `modules/textfieldsdetector.go` | `Modules/TextFieldsDetector.cs` | `modules/TextFieldsDetector.kt` |
| `pipeline_modules/words_detector/` | `modules/wordsdetector.go` | `Modules/TextFieldsDetector.cs` | `modules/TextFieldsDetector.kt` |
| `pipeline_modules/ocr_cyrillic/` **+** `ocr_latin/` (D-11: one type, `script` field) | `modules/ocrengine.go` | `Modules/OcrEngine.cs` | `modules/OcrEngine.kt` |
| `pipeline_modules/ocr_corrections.py` | `modules/ocrcorrections.go` | `Modules/OcrCorrections.cs` | `modules/OcrCorrections.kt` |
| `pipeline/pipeline.py::Pipeline.process_img` | `pipeline/pipeline.go` | `Pipeline/Recognizer.cs` | `pipeline/Recognizer.kt` |
| … `PipelineResults` | `pipeline/pipeline.go` | `Pipeline/Recognizer.cs` | `pipeline/Recognizer.kt` |
| … `OCROptions*` | `pipeline/ocroptions.go` | `Pipeline/OcrOptions.cs` | `pipeline/OcrOptions.kt` |
| … `_split_words` + `_duplicate_field_indices` | `pipeline/splitwords.go` | `Pipeline/SplitWords.cs` | `pipeline/SplitWords.kt` |
| … `_ocr_serial` + `_join_field` + `_fix_fms` (stub) | `pipeline/ocr.go` | `Pipeline/Ocr.cs` | `pipeline/Ocr.kt` |
| … the two parallel groups | `pipeline/parallel.go` | `Pipeline/Parallel.cs` | `pipeline/Parallel.kt` |
| … stage timings (WIRE names: `_doctype_angle`, …) | `pipeline/timings.go` | `Pipeline/Timings.cs` | `pipeline/Recognizer.kt` |
| `pipeline/probe.py` (`StageSink`) | `pipeline/probe.go`, `payloads.go` | `Pipeline/StageSink.cs` | `pipeline/StageSink.kt` |
| **`service/ml/transform.py`** — D-01, moves to the library | `viewmodel/viewmodel.go`, `boxes.go`, `fields.go`, `address.go`, `round.go` | `ViewModel/Builder.cs`, `Payload.cs` | `viewmodel/Builder.kt`, `Payload.kt` |
| `service/ml/labels.py` | `viewmodel/labels.go` | `ViewModel/Labels.cs` | `viewmodel/Labels.kt` |
| *(new)* OpenCV wrapper, owned images | `imaging/image.go`, `io.go`, `contours.go`, `crop.go`, `mask.go` | `Imaging/Image.cs`, `Io.cs`, `Contours.cs`, `Crop.cs`, `FloatMask.cs` | `imaging/Image.kt`, `Io.kt`, `Contours.kt` |
| *(new)* tensors, `.npy`, `.npz`, ops, CPython numerics | `tensor/npy.go`, `npz.go`, `ops.go`, `pynum.go` | `Tensors/NdArray.cs`, `Npy.cs`, `Npz.cs`, `Ops.cs`, `PyNum.cs` | `tensors/NdArray.kt`, `Npy.kt`, `Ops.kt`, `PyNum.kt` |
| *(new, Windows only)* native library loading | — (`D-09`: DLLs beside the binary) | — | `NativeLibraries.kt` (`J-01`) |

Rows where one port's cell repeats another row's file are not mistakes: `Modules/Quality.cs` really
does hold blur, glare and both spoofing detectors, and Go splits the same code across three files. The
table records what IS, not what would be tidy.

## Service — `service/`

| Python | Go | C# | Kotlin |
|---|---|---|---|
| `core/config.py` | `config/config.go` | `Config/Settings.cs` | `config/Settings.kt` |
| `core/models.py` | `model/model.go` | `Model/Document.cs` | `model/Document.kt` |
| `core/database.py` (`FileStore`) + the store interface | `store/store.go`, `filestore.go` | `Store/IDocumentStore.cs`, `FileStore.cs` | `store/FileStore.kt` |
| `core/auth.py` | `auth/auth.go` | `Auth/Auth.cs` | `auth/Tokens.kt` |
| `core/logging.py` | `logging/ring.go` | `Logging/LogRing.cs` | `logging/LogRing.kt` |
| `core/settings_schema.py` | `settingsschema/schema.go` | `Settings/SettingsSchema.cs` | `settings/SettingsSchema.kt` |
| `core/seed.py` | `seed/seed.go` | `Seed/SeedData.cs` | `seed/SeedData.kt` |
| *(new)* the seven error kinds | `errs/errs.go` | `Errors/ServiceErrors.cs` | `errors/ServiceErrors.kt` |
| `repositories/documents.py` | `repo/documents.go` | `Repositories/Documents.cs` | `repositories/Documents.kt` |
| `repositories/artifacts.py` | `repo/artifacts.go` | `Repositories/Artifacts.cs` | `repositories/Artifacts.kt` |
| `repositories/api_keys.py` | `repo/apikeys.go` | `Repositories/ApiKeys.cs` | `repositories/ApiKeys.kt` |
| `repositories/settings.py` | `repo/settings.go` | `Repositories/SettingsRepository.cs` | `repositories/SettingsRepository.kt` |
| **`ml/runtime.py`** — the deliverable, ten numbered rules | `runtime/runtime.go` | `Ml/PipelineRuntime.cs` | `ml/PipelineRuntime.kt` |
| `api/deps.py` (who is calling) | `api/deps.go` | `Api/Identity.cs` | `api/Identity.kt` |
| *(new)* the error contract — **write this first** | `api/errors.go` | `Api/ApiErrors.cs` | `api/ApiErrors.kt` |
| *(new)* query-parameter validation, pydantic-shaped 422 | `api/params.go` | `Api/ApiErrors.cs` | `api/ApiErrors.kt` |
| `main.py` router table | `api/router.go` | `Api/ApiServer.cs` | `api/ApiRoutes.kt` |
| `api/documents.py` | `api/documents.go` | `Api/ApiServer.Documents.cs` | `api/ApiDocuments.kt` |
| `api/{auth,api_keys,settings_api,status,logs}.py` | `api/misc.go` | `Api/ApiServer.Misc.cs` | `api/ApiMisc.kt` |
| `api/status.py` — host CPU/RAM/disk/GPU | `sysinfo/sysinfo*.go` | `Api/SysInfo.cs` | `api/SysInfo.kt` |
| `main.py` SPA catch-all | `api/misc.go` | `Api/ApiServer.cs` | `api/ApiMisc.kt` |
| `main.py` CORS middleware | `api/router.go` | `Program.cs` | `api/CorsFilter.kt` |
| `worker.py` | `worker/worker.go` | `Worker/RecognitionWorker.cs` | `worker/RecognitionWorker.kt` |
| … `build_search_text` | `worker/searchtext.go` | `Worker/SearchText.cs` | `worker/SearchText.kt` |
| `main.py` entry point | `cmd/rdocs-service/main.go` | `Program.cs` | `Application.kt` |

`api/errors.*` has no Python counterpart because FastAPI supplies the shape. It must be written
**first**, because everything else inherits its constraints: `{"detail": "<string>"}`, 401 not 403,
204-with-an-empty-body on DELETE, 202-with-the-full-row on `POST /documents`, `200` + JSON `null` on
`/progress`, `Cache-Control: private, no-cache` on images, and the list filter parameter named
`status`.

## The conformance CLI

| | Go | C# | Kotlin |
|---|---|---|---|
| entry point | `cmd/rdocs-conform/main.go` | `RussianDocs.Conform/Program.cs` | `conform/…/Main.kt` |
| `info` payload | same file | `RussianDocs.Conform/Probes.cs` | `conform/…/InfoPayload.kt` |

## Tests

| | Go | C# | Kotlin |
|---|---|---|---|
| library foundations | `*_test.go` beside each package | `tests/RussianDocs.DocumentProcessing.Tests/FoundationTests.cs` | `docproc/src/test/…/FoundationTests.kt` |
| service wire contract | `internal/svc/api/contract_test.go` | `tests/RussianDocs.Service.Tests/ContractTests.cs` | `service/src/test/…/ContractTests.kt` |
| the leak check conformance cannot do | `modules/soak_test.go` + `conform soak` | `conform soak` | `conform soak` |

---

## Switch case order

Cases must appear in Python's `match` order so the four files diff line for line. Recorded here so
nobody tidies them alphabetically.

**`newPreprocessor`** — from `processing/models.py::__load_preprocess`:

1. `Classification`
2. `YOLO`
3. `YOLOOBB` *(→ not implemented: the address path is deferred)*
4. `OCR` *(→ not implemented: legacy 31×200, no shipped config declares it)*
5. `OCRv2`
6. default → error naming the tag (`D-06`)

**`newPostprocessor`** — from `__load_postprocess`: `BinaryClass`, `MultiClass`, `Metric`, `OCRProbs`,
`YOLODetector`, `YOLOOBBDetector` *(not implemented)*, `YOLOSegmentor`, default → error.

**`newModel`** — from `ModelLoader.load`: `YOLODetectionModel`, `YOLOSegmentionModel`, `UnifiedModel`
(the default case), default → error.

A case that is not implemented is **present and returns an error**, never omitted: an omitted case
reads as an oversight and gets "helpfully" filled in differently by each port.
