# Архитектура

Публичный обзор устройства библиотеки — для тех, кто дорабатывает код или глубже
интегрирует пайплайн. Пользовательский гайд (установка, быстрый старт, API) — в
[README](../README.md).

## Общая идея

`Pipeline` — оркестратор: принимает изображение и последовательно (часть стадий —
параллельно) прогоняет его через набор ML-модулей, возвращая `PipelineResults`.

```
Изображение
  → Тип документа + угол (DocTypeAngles), поворот в вертикаль
  → [Проверки качества + границы документа]        (параллельно при low_quality=True)
  → Deskew (коррекция остаточного наклона)
  → Детекция текстовых полей
  → Разбиение полей на слова → OCR (кириллица / латиница+цифры) → CTC-декод
  → Дедупликация полей → результат
```

Для страницы регистрации (`INTPASSPORTADDR`) вместо обычной детекции полей
работает отдельный путь: детектор строк адреса (YOLO-OBB) → классификатор
«печатный/рукописный» → пословный OCR печатных строк.

## Слои и папки

```
russian_docs_ocr/document_processing/
  pipeline/pipeline.py    # Pipeline, PipelineResults, OCROptions*
  pipeline_modules/       # по подпапке на ML-модуль
  processing/             # инференс-слой
  config/                 # models_path.yaml, алфавиты OCR
  models/                 # артефакты моделей
```

### `pipeline/pipeline.py`
- `Pipeline` — какие модули запускать и в каком порядке.
- `PipelineResults` — хранит выходы, отдаёт удобные свойства (`ocr`, `doctype`,
  `quality`, `angle`, `timings`, …).
- `OCROptionsClass` и подклассы (`OCROptionsINTPassport`, `OCROptionsDL`, …) —
  какие поля есть у типа документа, каким движком их распознавать, нужно ли
  разбиение на слова.

### `pipeline_modules/`
Один подкаталог на модуль; каждый класс наследует `BaseModule`
(`base_module.py`) и реализует `predict()` / `predict_transform()`. Модуль
загружает свой артефакт по имени через `config/models_path.yaml`.

Основные модули: `doctype_angles_classificator`, `doc_detector`,
`textfields_detector`, `address_lines_detector`, `address_textkind_classifier`,
`words_detector`, детекторы качества (`blur`, `glare`, `lcd_spoofing`,
`print_spoofing`), OCR-движки (`ocr_cyrillic`, `ocr_latin`), `deskewer`.

### `processing/`
- `models.py` (`ModelLoader`) — собирает модель из JSON-конфига (препроцессинг →
  инференс → постпроцессинг) по полям `Inputs`/`Outputs`/`ModelType`.
- `preprocessing.py` / `postprocessing.py` — типизированные пайплайны
  (классификация, YOLO, YOLO-OBB, OCR v2 и т.д.).
- `inference.py` — тонкая обёртка над рантаймами ONNX / OpenVINO; приводит вход к
  объявленному в модели dtype и авто-резолвит GPU/CPU-провайдер.

## OCR-движки

Два поколения, выбор через `Pipeline(ocr=...)`:

- **v2 (по умолчанию)** — `OCRCyrillic` / `OCRLatin`, тиры `accurate`
  (MobileNetV4) и `fast` (EdgeNext). Вход — цветной BGR, нормализация зашита в
  граф; выход — softmax-матрица, декодируется greedy-CTC с масками алфавита.
  Разрешённый набор символов резолвится из `config/ocr_alphabets.json`.


## Форматы моделей

- Поддерживаются `ONNX` (по умолчанию) и `OpenVINO`.
- В каждой папке модели — `model.json` с описанием pre/postprocessing, поэтому
  смена формата = замена пары JSON + артефакт.
- Конвенция экспорта YOLO-детекторов — «wrap-and-bake»: конвертация NHWC/деление
  на 255 запекается в граф, чтобы модель была drop-in (deploy = замена
  `model.onnx`).

## Поддерживаемые типы документов

Внутренний паспорт (1997, 2011), страница регистрации (`INTPASSPORTADDR`),
загранпаспорт (2003, биометрический 2007), водительское удостоверение
(2011, 2020), СНИЛС (1996, 2019). Тип определяется `DocTypeAngles`.

## Как добавить модуль

1. Подпапка в `pipeline_modules/` с классом-наследником `BaseModule`
   (`model_name`, `predict()` / `predict_transform()`).
2. Артефакты в `models/<Name>/<FORMAT>/` (`model.onnx` + `model.json`) и запись
   в `config/models_path.yaml`.
3. Вызов модуля в `Pipeline.process_img()`.
4. При влиянии на извлечение текста — обновить нужный `OCROptions*`.
5. Тест в `tests/` (+ фикстуры в `tests/images/` при необходимости).

## Тесты

```bash
python -m pytest tests/            # из корня репозитория
```

Тесты грузят модели из `russian_docs_ocr/document_processing/models/...` и
фикстуры из `tests/images/...`, поэтому запускаются из корня репозитория.
