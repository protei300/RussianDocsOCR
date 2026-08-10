"""
Visualize deskew correction: saves before/after images and detected angles.

Usage:
    python scripts/visualize_deskew.py -i samples/INTPASSPORT_2011/3_BG_INTPASSPORT_2011.jpg
"""

import sys
sys.path.append('.')

import argparse
from pathlib import Path

import cv2
import numpy as np

from document_processing import Pipeline
from document_processing.pipeline_modules.deskewer import DocDeskewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img', required=True, type=Path)
    parser.add_argument('-o', '--out_dir', type=Path, default=Path('deskew_debug'))
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    stem = args.img.stem

    pipeline = Pipeline(model_format='ONNX', device='cpu', verbose=False)
    deskewer = DocDeskewer()

    # Monkey-patch to intercept before/after
    original_deskew = pipeline.deskewer.deskew
    img_before = [None]
    img_after = [None]
    angles_found = [None]

    def patched_deskew(img, n_segments=1):
        img_before[0] = img.copy()
        # Run angle detection per segment and log
        if n_segments == 2:
            mid = img.shape[0] // 2
            a_upper = deskewer._find_angle(img[:mid])
            a_lower = deskewer._find_angle(img[mid:])
            angles_found[0] = (a_upper, a_lower)
        else:
            angles_found[0] = deskewer._find_angle(img)
        result = original_deskew(img, n_segments=n_segments)
        img_after[0] = result
        return result

    pipeline.deskewer.deskew = patched_deskew

    results = pipeline.process_img(args.img, ocr=False, check_quality=False)
    print(f"Doc type   : {results.doctype}")
    print(f"Angles     : {angles_found[0]}")

    if img_before[0] is None:
        print("Deskew stage was not reached")
        return

    b = cv2.cvtColor(img_before[0], cv2.COLOR_RGB2BGR)
    a = cv2.cvtColor(img_after[0], cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(args.out_dir / f"{stem}_1_before_deskew.jpg"), b)
    cv2.imwrite(str(args.out_dir / f"{stem}_2_after_deskew.jpg"), a)

    if b.shape != a.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]))
    sep = np.full((b.shape[0], 4, 3), 200, dtype=np.uint8)
    side = np.hstack([b, sep, a])
    cv2.putText(side, "BEFORE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(side, "AFTER", (b.shape[1] + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

    angle_label = str(angles_found[0])
    cv2.putText(side, f"angle: {angle_label}", (10, side.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)

    diff_path = args.out_dir / f"{stem}_3_diff.jpg"
    cv2.imwrite(str(diff_path), side)
    print(f"Saved: {diff_path}")


if __name__ == '__main__':
    main()
