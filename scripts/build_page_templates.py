"""Build the internal-passport page templates used by PageRegistrar.

A template is a *reference page* - a sharp, frontal photo of one page rectified
by the current Borders stack and resized to the canonical 1000x704 frame
(88x125 mm, ~11.4 px/mm) - with every person-specific zone erased by a layout
box (issuing authority, dates, codes, stamp, signatures, series/number, photo,
name lines, MRZ). What is left is the printed blank: header, captions, rules,
ornamental frame, emblem. A second reference is aligned to the first with SIFT
so both share one canonical frame; the registrar tries both and keeps the
better match per page.

The erase boxes are in canonical coordinates and were read off a coordinate
grid rendered over the aligned references; they are wider than the printed
zones on purpose. A smooth fill (inpainting at 1/4 scale) replaces the erased
pixels, so the shipped template carries no trace of the source document. The
feature mask is the ink contrast of the erased page, minus the boxes.

Provenance of the shipped set (models are NOT needed to reproduce it - the
rectified canvases are the input):
  ref a: Real_docs/CR_INTPASSPORT_1997/2_CR_INTPASSPORT_1997.jpg (defines the frame)
  ref b: samples/INTPASSPORT_2011/6_CR_INTPASSPORT_2011.jpg (aligned to a)

Usage:
    python scripts/build_page_templates.py --ref-a <spread photo> --ref-b <spread photo> \\
        -o document_processing/pipeline_modules/page_registration/templates
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

W, H = 1000, 704
PAGES = ['page2', 'page3']
# erase boxes [x0, y0, x1, y1] per page, canonical coordinates
BOXES = {
    'page2': [[175, 105, 965, 272], [145, 270, 445, 336], [535, 270, 905, 336],
              [0, 375, 365, 604], [300, 350, 650, 704], [635, 478, 955, 574],
              [930, 145, 1000, 610]],
    'page3': [[30, 170, 340, 580], [400, 30, 960, 145], [375, 165, 955, 258],
              [395, 252, 955, 304], [365, 310, 475, 354], [560, 300, 955, 358],
              [400, 355, 960, 480], [930, 145, 1000, 610], [0, 555, 1000, 704]],
}


def rectified_pages(photo: Path):
    """Two pages (BGR) of a spread photo via the Borders stack."""
    from document_processing import Pipeline
    pipe = Pipeline(model_format='ONNX', device='cpu', verbose=False)
    r = pipe.process_img(str(photo), ocr=False, find_text_fields=False, check_quality=False)
    canvas = cv2.cvtColor(r.img_with_fixed_perspective, cv2.COLOR_RGB2BGR)
    h = canvas.shape[0] // 2
    return canvas[:h], canvas[h:]


def align_to(ref, tgt):
    """Homography-align tgt to ref (both roughly rectified pages)."""
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY), None)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d2, d1, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.85 * n.distance]
    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    sx, sy = ref.shape[1] / tgt.shape[1], ref.shape[0] / tgt.shape[0]
    keep = np.linalg.norm(src * [sx, sy] - dst, axis=1) < 150
    Hm, mask = cv2.findHomography(src[keep], dst[keep], cv2.USAC_MAGSAC, 3.0, maxIters=10000)
    print(f'  aligned with {int(mask.sum())} inliers')
    return cv2.warpPerspective(tgt, Hm, (W, H), borderMode=cv2.BORDER_REPLICATE)


def level(page):
    """Rotate the reference so its print is level. The Borders quad follows the
    page edges, and on the seed document the print sat ~2 deg off them; a
    template inheriting that tilt tilts every registered page (measured: the
    MRZ lines came out at 2.5 deg and their OCR mixed). The projection-profile
    deskewer is run on the un-erased page at fine resolution, and the frame
    is rotated about its centre."""
    from document_processing.pipeline_modules.deskewer import DocDeskewer
    d = DocDeskewer(angle_range=5.0, angle_steps=201, min_angle=0.0, scale=1.0, coarse_steps=41)
    angle = d._find_angle(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
    print(f'  levelling by {angle:+.2f} deg')
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    return cv2.warpAffine(page, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def erase(aligned, boxes):
    erased = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in boxes:
        erased[y0:y1, x0:x1] = 1
    small = cv2.resize(aligned, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    m_small = cv2.dilate(cv2.resize(erased * 255, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST),
                         np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(small, m_small, 7, cv2.INPAINT_TELEA)
    bg = cv2.GaussianBlur(cv2.resize(filled, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 6)
    clean = aligned.copy()
    clean[erased == 1] = bg[erased == 1]
    # guard: nothing legible may survive inside an erased zone (ink contrast
    # of the fill must be far below the printed blank's)
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F))
    inside = float(lap[erased == 1].mean())
    outside = float(lap[erased == 0].mean())
    assert inside < 0.2 * outside, f'erase fill still carries ink: {inside:.2f} vs {outside:.2f}'
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    contrast = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (0, 0), 2.0)
    mask = (contrast > np.percentile(contrast, 60)).astype(np.uint8)
    mask[erased == 1] = 0
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    mask[cv2.dilate(erased, np.ones((15, 15), np.uint8)) == 1] = 0
    return clean, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ref-a', required=True, type=Path, help='spread photo defining the frame')
    ap.add_argument('--ref-b', type=Path, help='second spread photo, aligned to ref-a')
    ap.add_argument('-o', '--output', required=True, type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    refs = {'a': rectified_pages(args.ref_a)}
    if args.ref_b:
        refs['b'] = rectified_pages(args.ref_b)
    meta = {'doc_type': 'INTPASSPORT', 'page_mm': [88, 125], 'page_px': [W, H], 'pages': []}
    for pi, pname in enumerate(PAGES):
        entry = {'name': pname, 'refs': []}
        base = level(cv2.resize(refs['a'][pi], (W, H), interpolation=cv2.INTER_AREA))
        for tag, pages in refs.items():
            print(pname, tag)
            # every reference is levelled on its own: the SIFT alignment of b to
            # a can carry a rotation error of a few degrees (few, clustered
            # inliers on page 3), which would tilt every page registered to b
            aligned = base if tag == 'a' else level(align_to(base, pages[pi]))
            clean, mask = erase(aligned, BOXES[pname])
            img_name, mask_name = f'intpassport_{pname}_{tag}.png', f'intpassport_{pname}_{tag}_mask.png'
            cv2.imwrite(str(args.output / img_name), clean)
            cv2.imwrite(str(args.output / mask_name), mask * 255)
            entry['refs'].append({'image': img_name, 'mask': mask_name})
            print(f'  mask fraction {mask.mean():.3f}')
        meta['pages'].append(entry)
    (args.output / 'intpassport.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print('written to', args.output)


if __name__ == '__main__':
    main()
