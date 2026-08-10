# -*- coding: utf-8 -*-
"""Разбор ОДНОГО документа всеми OCR-движками бок о бок.

Использование:
    python scripts/analyze_one.py path/to/doc.jpg
    python scripts/analyze_one.py path/to/doc.jpg --device gpu --modes accurate fast

Показывает: тип документа, метрики качества, таблицу полей (accurate | fast)
с пометкой подозрительных/пустых значений, и тайминги (total + стадия OCR).
Удобно для «плохих» сканов: видно, какой движок читает больше полей.
"""
import argparse
import sys
from pathlib import Path

# запуск из корня репозитория или из scripts/ — найдём document_processing
_HERE = Path(__file__).resolve().parent
for cand in (_HERE.parent, _HERE):
    if (cand / "document_processing").is_dir():
        sys.path.insert(0, str(cand))
        break

from document_processing import Pipeline  # noqa: E402


def suspicious(value: str) -> bool:
    """Грубая эвристика 'поле распозналось плохо': пусто, слишком коротко,
    или мало буквенно-цифровых символов относительно длины."""
    v = (value or "").strip()
    if len(v) < 2:
        return True
    alnum = sum(c.isalnum() for c in v)
    return alnum < max(2, len(v) * 0.5)


def analyze(image_path: str, device, modes):
    results = {}
    meta = {}
    for mode in modes:
        try:
            p = Pipeline(device=device, ocr=mode, verbose=False)
            r = p.process_img(image_path, ocr=True, check_quality=True, low_quality=True)
            results[mode] = r.ocr or {}
            meta[mode] = {
                "doctype": r.doctype,
                "quality": r.quality,
                "timings": r.timings,
            }
        except Exception as e:
            results[mode] = {}
            meta[mode] = {"error": repr(e)}
    return results, meta


def main():
    ap = argparse.ArgumentParser(description="Разбор одного документа всеми OCR-движками.")
    ap.add_argument("image", help="путь к изображению документа")
    ap.add_argument("--device", default=None, choices=["cpu", "gpu"],
                    help="устройство инференса (по умолчанию авто: GPU→CPU)")
    ap.add_argument("--modes", nargs="+", default=["accurate", "fast"],
                    choices=["accurate", "fast"], help="какие движки сравнивать")
    ap.add_argument("--img_size", type=int, default=1500)
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"[!] Файл не найден: {args.image}")
        return 2

    print(f"Изображение: {args.image}")
    print(f"Устройство : {args.device or 'auto'}")
    print(f"Движки     : {', '.join(args.modes)}\n")

    results, meta = analyze(args.image, args.device, args.modes)

    # шапка: doctype / качество / тайминги по каждому режиму
    for mode in args.modes:
        m = meta[mode]
        if "error" in m:
            print(f"[{mode}] ОШИБКА: {m['error']}")
            continue
        q = m["quality"]
        t = m["timings"]
        qstr = f"Glare={q.get('Glare')} Blur={q.get('Blur')} " \
               f"Print={q.get('PrintSpoofing')} LCD={q.get('LCDSpoofing')} DocConf={q.get('DocConf')}"
        print(f"[{mode}] doctype={m['doctype']} | {qstr}")
        print(f"        total={t.get('total', 0)*1000:.0f} ms, ocr={t.get('_ocr', 0)*1000:.0f} ms")
    print()

    # таблица полей
    keys = sorted(set().union(*[set(results[m]) for m in args.modes]) if results else set())
    if not keys:
        print("Полей не распознано ни одним движком.")
        return 0

    w = max((len(k) for k in keys), default=10)
    header = "  ".join(f"{m:<24}" for m in args.modes)
    print(f"{'ПОЛЕ':<{w}}  {header}")
    print("-" * (w + 2 + 26 * len(args.modes)))
    for k in keys:
        cells = []
        for m in args.modes:
            v = str(results[m].get(k, ""))
            mark = " ⚠" if suspicious(v) else "  "
            cells.append(f"{(v[:22] + mark):<24}")
        print(f"{k:<{w}}  " + "  ".join(cells))

    # сводка: сколько «подозрительных»/пустых полей на движок
    print("\nСводка (меньше — лучше): подозрительных/пустых полей")
    allkeys = set(keys)
    for m in args.modes:
        bad = sum(1 for k in allkeys if suspicious(str(results[m].get(k, ""))))
        print(f"  {m:<10}: {bad} из {len(allkeys)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
