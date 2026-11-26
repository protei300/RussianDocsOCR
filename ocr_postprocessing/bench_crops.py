"""Crops with ground truth: plain reading vs improved crop by crop (``CropImprover``) vs improved with the
whole stamp (``OCRPostprocessor.process_crops``).

    python -m ocr_postprocessing.bench_crops --probs probs.npz --crops crops.json --alphabet <model dir>

"Right" ignores spaces and dots, Ё = Е (decisions #159, #160). Per field class: share right, fixed, broken.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .bench import squash
from .crop import CropImprover
from .postprocess import OCRPostprocessor


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probs', type=Path, required=True)
    ap.add_argument('--crops', type=Path, required=True)
    ap.add_argument('--alphabet', type=Path, required=True)
    ap.add_argument('--rules', nargs='*', help='only these crop rules: date number code words')
    ap.add_argument('--no-stamp', action='store_true', help='skip the whole-stamp variant')
    ap.add_argument('--show', type=int, default=0)
    ap.add_argument('--all-streets', action='store_true')
    ap.add_argument('--margin', type=float)
    a = ap.parse_args()
    crops = json.loads(a.crops.read_text('utf-8'))
    probs = np.load(a.probs)
    t0 = time.time()
    if a.all_streets:
        CropImprover.all_streets = True
    if a.margin is not None:
        CropImprover.word_margin = a.margin
    imp = CropImprover(a.alphabet)
    if a.rules is not None:
        imp.rules_on = tuple(a.rules)
    print(f'загрузка (лоскут) {time.time() - t0:.1f} с')
    st, shows, t_crop = Counter(), defaultdict(list), []
    res_crop = {}
    for c in crops:
        t1 = time.time()
        r = imp.improve(c['label'], probs[f'm{c["i"]}'])
        t_crop.append(time.time() - t1)
        res_crop[c['i']] = r
    res_stamp = {}
    t_stamp = []
    if not a.no_stamp:
        pp = OCRPostprocessor(a.alphabet)
        by = defaultdict(list)
        for c in crops:
            by[c['stamp']].append(c)
        for cs in by.values():
            t1 = time.time()
            r = pp.process_crops([{'label': c['label'], 'probs': probs[f'm{c["i"]}']} for c in cs])
            t_stamp.append(time.time() - t1)
            for c, o in zip(cs, r['fields']):
                res_stamp[c['i']] = o
    for c in crops:
        gt = squash(c['gt'])
        b = squash(res_crop[c['i']]['raw']) == gt
        v = {'лоскут': squash(res_crop[c['i']]['text']) == gt}
        if res_stamp:
            v['штамп'] = squash(res_stamp[c['i']]['text']) == gt
        for g in (c['label'], 'все'):
            st[(g, 'n')] += 1
            st[(g, 'raw')] += b
            for k, ok in v.items():
                st[(g, k)] += ok
                st[(g, k, 'fixed')] += ok and not b
                st[(g, k, 'broken')] += b and not ok
        if a.show and v['лоскут'] != b:
            shows['исправлено' if v['лоскут'] else 'испорчено'].append(
                f'{c["label"]:11} | {res_crop[c["i"]]["raw"]} -> {res_crop[c["i"]]["text"]} | эталон {c["gt"]}')
    print(f'лоскут: медиана {np.median(t_crop) * 1000:.1f} мс, max {max(t_crop) * 1000:.0f} мс'
          + (f'; штамп: медиана {np.median(t_stamp) * 1000:.0f} мс, max {max(t_stamp) * 1000:.0f} мс' if t_stamp else ''))
    kinds = ['лоскут'] + (['штамп'] if res_stamp else [])
    head = f'{"поле":12} {"n":>5} {"без":>7}' + ''.join(f' {k:>7} {"+/-":>7}' for k in kinds)
    print(head)
    for g in sorted({k[0] for k in st}, key=lambda g: (g == 'все', g)):
        n = st[(g, 'n')]
        row = f'{g:12} {n:5} {st[(g, "raw")] / n:7.1%}'
        for k in kinds:
            row += f' {st[(g, k)] / n:7.1%} {"+" + str(st[(g, k, "fixed")]) + "/-" + str(st[(g, k, "broken")]):>7}'
        print(row)
    for k, v in shows.items():
        print(f'\n--- {k} (лоскут)')
        print('\n'.join(v[:a.show]))


if __name__ == '__main__':
    main()
