# Как внести вклад

Спасибо за интерес к проекту! Этот документ описывает, как поднять окружение,
прогнать тесты и оформить изменения.

## Требования

- Python **3.11+**
- Git

## Настройка окружения

```bash
git clone https://github.com/protei300/RussianDocsOCR.git
cd RussianDocsOCR

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
pip install -e .            # editable-установка пакета
```

Модели (`.onnx`, `.npz`, …) хранятся прямо в репозитории, отдельно ничего
скачивать не нужно.

## Запуск тестов

Из корня репозитория:

```bash
python -m pytest tests/
```

Тесты должны быть **зелёными** после любого изменения в `russian_docs_ocr/`.
Прогоняйте их локально до отправки PR. Если правите что-то в OCR/пайплайне —
дополнительно проверьте сквозной прогон на образцах:

```bash
python russian_docs_ocr/scripts/process_img.py -i samples/DL_2011/1_CR_DL_2010.jpg -f ONNX
```

## Структура проекта

```
russian_docs_ocr/document_processing/
  pipeline/          # класс Pipeline (оркестратор)
  pipeline_modules/  # по подпапке на ML-модуль (наследники BaseModule)
  processing/        # слой инференса: pre/postprocessing, загрузчик, ONNX/OpenVINO
  config/            # models_path.yaml, алфавиты OCR
  models/            # артефакты моделей (ONNX / OpenVINO)
russian_docs_ocr/scripts/   # CLI-скрипты (process_img, benchmark, …)
tests/               # юнит-тесты (+ tests/images/ с фикстурами)
samples/             # образцы документов по типам
```

Подробный обзор — в [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Стиль кода

- Код, комментарии и докстринги — **на английском** (README/CHANGELOG — на русском).
- Докстринги в стиле проекта: краткое описание + `Args:` / `Returns:`.
- Соблюдайте стиль окружающего кода (именование, отступы, плотность комментариев).
- Не заводите тяжёлые зависимости без необходимости; список — в `requirements.txt`.

## Ветки и Pull Request

- `main` — релизная ветка (помечается тегами `vX.Y.Z`).
- `dev` — ветка разработки, **PR отправляйте в неё**.
- Заводите тематическую ветку от `dev`, держите её сфокусированной на одной задаче.
- Перед PR: зелёный `pytest`, осмысленные сообщения коммитов.
- В описании PR укажите, что и зачем меняется; приложите вывод тестов при правках логики.

## Версионирование и CHANGELOG

Проект использует [SemVer](https://semver.org/lang/ru/):

- Несовместимое изменение публичного API → **MAJOR**.
- Обратносовместимая функциональность → **MINOR**.
- Исправления → **PATCH**.

При заметном изменении добавьте запись в [`CHANGELOG.md`](CHANGELOG.md) в секцию
готовящейся версии. Единственный источник версии — `__version__` в
`russian_docs_ocr/document_processing/__init__.py`.

## Добавление нового модуля

Кратко (детали — в [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)):

1. Создайте подпапку в `pipeline_modules/` с классом-наследником `BaseModule`.
2. Положите артефакты в `models/` и пропишите путь в `config/models_path.yaml`.
3. Подключите модуль в `Pipeline.process_img()`.
4. Добавьте тест в `tests/`.

## Сообщения об ошибках и идеи

Используйте шаблоны issue (баг / фича). Уязвимости — по процедуре из
[`SECURITY.md`](SECURITY.md), **не** через публичные issue.
