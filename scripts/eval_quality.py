"""End-to-end OCR quality evaluation against per-image ground-truth JSONs.

Two supported ground-truth layouts:

1. ``samples/`` layout (default): ``<root>/<DOCTYPE_YEAR>/<name>.json`` next to the
   image, keys are pipeline field names (``Last_name_ru``, ``Licence_number``, ...).
   The folder name is the expected ``results.doctype``.

2. ``--damir`` layout: ``<root>/images/*.jpg`` + ``<root>/labels/<img>.txt`` where the
   label is a JSON dict of external-schema keys (``surname``, ``number``, ...) mapped to
   ``[value, confidence]`` pairs. Keys are remapped to pipeline field names via
   ``DAMIR_FIELD_MAP``. All images are expected to be internal passports.

Metrics: per-field and overall CER (levenshtein / len(gt), over fields with non-empty
GT), exact-match rate, and doctype accuracy. samples/ comparison is strict (uppercase +
whitespace collapse only). The Damir labels come from another OCR system with its own
conventions and known label errors (see docs/progress-log.md), so its comparison is
looser: gender compares by first letter, numbers ignore spacing, text ignores
punctuation, and a GT value of ``'-'`` means "not labeled" and is skipped — treat the
resulting numbers as cross-system agreement, not gold accuracy.

Usage:
    python scripts/eval_quality.py                        # eval samples/
    python scripts/eval_quality.py -i path/to/root        # other samples-layout root
    python scripts/eval_quality.py --damir D:/.../Passports_Damir
    python scripts/eval_quality.py --limit 3              # cap images per doctype

``tests/test_quality.py`` imports this module and asserts thresholds on the same
metrics — keep the metric definitions here as the single source of truth.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# External (Damir) label key -> (pipeline field name, comparison kind)
DAMIR_FIELD_MAP = {
    'surname': ('Last_name_ru', 'text'),
    'name': ('First_name_ru', 'text'),
    'middle_name': ('Middle_name_ru', 'text'),
    'birth_date': ('Birth_date', 'number'),
    'birth_place': ('Birth_place_ru', 'text'),
    'gender': ('Sex_ru', 'gender'),
    'issue_date': ('Issue_date', 'number'),
    'issued_by': ('Issue_organization_ru', 'text'),
    'subdivision': ('Issue_organisation_code', 'number'),
    'number': ('Licence_number', 'number'),
}

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.PNG', '.JPG')


def normalize(text, kind='strict'):
    """Comparison normalization. 'strict' = uppercase + collapse whitespace (samples/);
    the other kinds implement the looser Damir cross-system comparison."""
    text = ' '.join((text or '').upper().split())
    if kind == 'number':
        text = ''.join(ch for ch in text if ch not in ' .')
    elif kind == 'text':
        text = ' '.join(''.join(ch for ch in text if ch not in '.,"\'«»').split())
    elif kind == 'gender':
        text = text[:1]
    return text


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, gt, kind='strict'):
    """Character error rate, or None when the GT value is empty (not evaluable)."""
    pred, gt = normalize(pred, kind), normalize(gt, kind)
    if not gt:
        return None
    return levenshtein(pred, gt) / len(gt)


class MetricAggregator:
    """Accumulates per-field CER values and doctype hits."""

    def __init__(self):
        self.cer_sum = 0.0
        self.n_fields = 0
        self.n_exact = 0
        self.per_field = {}   # field -> [cer_sum, n, n_exact]
        self.doc_total = 0
        self.doc_correct = 0
        self.failures = []    # (image, message)

    def add_field(self, field, value):
        self.cer_sum += value
        self.n_fields += 1
        self.n_exact += (value == 0)
        stats = self.per_field.setdefault(field, [0.0, 0, 0])
        stats[0] += value
        stats[1] += 1
        stats[2] += (value == 0)

    def add_doctype(self, correct):
        self.doc_total += 1
        self.doc_correct += correct

    @property
    def mean_cer(self):
        return self.cer_sum / self.n_fields if self.n_fields else 0.0

    @property
    def exact_rate(self):
        return self.n_exact / self.n_fields if self.n_fields else 0.0

    @property
    def doctype_accuracy(self):
        return self.doc_correct / self.doc_total if self.doc_total else 0.0

    def summary(self):
        return (f'doctype {self.doc_correct}/{self.doc_total} '
                f'({100 * self.doctype_accuracy:.1f}%)  '
                f'mean CER={self.mean_cer:.4f}  '
                f'exact={self.n_exact}/{self.n_fields} ({100 * self.exact_rate:.1f}%)')


def _find_image(json_path):
    for ext in IMG_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def eval_samples(pipeline, root, limit=None):
    """Evaluate the samples/ layout. Returns (overall, {doctype: MetricAggregator})."""
    overall = MetricAggregator()
    by_doctype = {}
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir():
            continue
        agg = by_doctype.setdefault(folder.name, MetricAggregator())
        json_files = sorted(folder.glob('*.json'))
        if limit:
            json_files = json_files[:limit]
        for json_path in json_files:
            image = _find_image(json_path)
            if image is None:
                continue
            gt = json.loads(json_path.read_text(encoding='utf-8'))
            try:
                result = pipeline.process_img(str(image), ocr=True, check_quality=False)
            except Exception as exc:  # noqa: BLE001 - eval must survive any sample
                overall.failures.append((f'{folder.name}/{image.name}', repr(exc)))
                continue
            correct = result.doctype == folder.name
            overall.add_doctype(correct)
            agg.add_doctype(correct)
            if not correct:
                overall.failures.append(
                    (f'{folder.name}/{image.name}', f'doctype={result.doctype}'))
            predictions = result.ocr or {}
            for field, gt_value in gt.items():
                value = cer(predictions.get(field, ''), gt_value)
                if value is None:
                    continue
                overall.add_field(field, value)
                agg.add_field(field, value)
    return overall, by_doctype


def eval_damir(pipeline, root, limit=None):
    """Evaluate the Damir layout (internal passports, external GT schema)."""
    agg = MetricAggregator()
    root = Path(root)
    labels = sorted((root / 'labels').glob('*.txt'))
    if limit:
        labels = labels[:limit]
    for label_path in labels:
        image = root / 'images' / label_path.stem  # '001.jpg.txt' -> '001.jpg'
        if not image.exists():
            continue
        gt = json.loads(label_path.read_text(encoding='utf-8'))
        try:
            result = pipeline.process_img(str(image), ocr=True, check_quality=False)
        except Exception as exc:  # noqa: BLE001
            agg.failures.append((image.name, repr(exc)))
            continue
        doctype = result.doctype or '?'
        correct = doctype.startswith('INTPASSPORT_')
        agg.add_doctype(correct)
        if not correct:
            agg.failures.append((image.name, f'doctype={doctype}'))
        predictions = result.ocr or {}
        for key, (field, kind) in DAMIR_FIELD_MAP.items():
            gt_value = gt.get(key)
            if isinstance(gt_value, list):
                gt_value = gt_value[0]
            if gt_value == '-':  # Damir convention for "field not labeled"
                continue
            value = cer(predictions.get(field, ''), gt_value, kind)
            if value is None:
                continue
            agg.add_field(field, value)
    return agg


def _print_fields(agg):
    for field, (cer_sum, n, n_exact) in sorted(agg.per_field.items()):
        print(f'  {field:26s} CER={cer_sum / n:.4f}  exact={n_exact}/{n}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('-i', '--input', default=str(REPO_ROOT / 'samples'),
                        help='samples-layout root (default: repo samples/)')
    parser.add_argument('--damir', metavar='DIR',
                        help='additionally evaluate a Damir-layout dataset')
    parser.add_argument('--limit', type=int,
                        help='max images per doctype folder (or total for --damir)')
    parser.add_argument('-f', '--format', default='ONNX',
                        choices=['ONNX', 'OpenVINO'])
    parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--ocr', default='accurate',
                        choices=['accurate', 'fast', 'legacy'])
    args = parser.parse_args()

    from document_processing import Pipeline
    pipeline = Pipeline(model_format=args.format, device=args.device,
                        ocr=args.ocr, verbose=False)

    overall, by_doctype = eval_samples(pipeline, args.input, limit=args.limit)
    print(f'==== {args.input} ====')
    print(f'OVERALL: {overall.summary()}')
    for doctype, agg in sorted(by_doctype.items()):
        print(f'  {doctype:22s} {agg.summary()}')
    print('BY FIELD:')
    _print_fields(overall)
    for name, message in overall.failures:
        print(f'  [!] {name}: {message}')

    if args.damir:
        agg = eval_damir(pipeline, args.damir, limit=args.limit)
        print(f'\n==== {args.damir} ====')
        print(f'OVERALL: {agg.summary()}')
        print('BY FIELD:')
        _print_fields(agg)
        for name, message in agg.failures:
            print(f'  [!] {name}: {message}')


if __name__ == '__main__':
    main()
