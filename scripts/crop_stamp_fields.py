"""Cut upright registration stamps into fields for local OCR transcription.

No files are uploaded. Input images are not copied into the output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from service.ml.registration.config import RegistrationConfig
from service.ml.registration.fields_net import FieldsDetector
from service.ml.registration.ocr import StampOCR, crop, greedy


def collect(source: Path, output: Path, cfg: RegistrationConfig, *, with_ocr=False,
            pad_x=.3, pad_y=.15) -> int:
    if not np.isfinite([pad_x, pad_y]).all() or min(pad_x, pad_y) < 0:
        raise ValueError('Crop margins must be finite and nonnegative')
    images = ([source] if source.is_file() else sorted(p for p in source.iterdir()
              if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp') and p.is_file()))
    if not images:
        raise ValueError('No input images found')
    detector = FieldsDetector(cfg.fields_model, conf=cfg.fields_conf)
    ocr = StampOCR(cfg.ocr_dir) if with_ocr else None
    # Never overwrite an existing transcription, including a partial one.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'crops').mkdir()
    rows, stamps, seen = [], [], set()
    for image in images:
        raw = image.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        stamp_id = digest[:20]
        stamps.append({'stamp_id': stamp_id, 'source_sha256': digest,
                       'source_group': stamp_id})
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f'Cannot decode input: {image}')
        for i, box in enumerate(detector.predict(img)):
            patch = crop(img, box['quad'], pad_x, pad_y,
                         horizontal=box['field'] in ('house', 'korp', 'flat'))
            if patch is None:
                continue
            key = f'{stamp_id}_{i:03d}'
            name = f'crops/{key}_{box["field"]}.png'
            ok, encoded = cv2.imencode('.png', patch)
            if not ok:
                raise ValueError('Cannot encode crop')
            (output / name).write_bytes(encoded.tobytes())
            row = {'id': key, 'stamp_id': stamp_id, 'class': box['field'], 'crop': name, 'text': None,
                   'reviewed': False, 'unreadable': False, 'source_sha256': digest,
                   'quad': np.asarray(box['quad']).tolist(), 'confidence': float(box['conf'])}
            if ocr is not None:
                row['ocr_suggestion'] = greedy(ocr.probs(patch), ocr.abc)
            rows.append(row)
        img.fill(0)
    model_hash = hashlib.sha256(cfg.fields_model.read_bytes()).hexdigest()
    manifest = {'schema': 'stamp-field-transcription-v1', 'case_sensitive': False,
                'detector_sha256': model_hash, 'padding': {'along_height': pad_x, 'across_height': pad_y},
                'stamps': stamps, 'fields': rows}
    (output / 'annotations.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path, help='one upright stamp or a directory of stamps')
    ap.add_argument('--output', type=Path, required=True, help='new output directory')
    ap.add_argument('--ocr', action='store_true', help='add OCR suggestions, separate from ground truth')
    ap.add_argument('--pad-x', type=float, default=.3, help='margin along the line, as a share of height')
    ap.add_argument('--pad-y', type=float, default=.15, help='margin across the line, as a share of height')
    args = ap.parse_args()
    count = collect(args.input, args.output, RegistrationConfig(), with_ocr=args.ocr,
                    pad_x=args.pad_x, pad_y=args.pad_y)
    print(f'{count} field crops -> {args.output.resolve()}')


if __name__ == '__main__':
    main()
