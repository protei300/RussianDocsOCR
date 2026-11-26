"""Measure the post-processing on crops with ground truth, grouped by stamp.

    python -m ocr_postprocessing.bench --probs probs.npz --crops crops.json --alphabet <model dir>

``crops.json`` - a list of {i, stamp, label, gt, ...}; ``probs.npz`` - matrix ``m<i>`` of every crop.
Per field class: the share of crops read right before (best path) and after, how many were fixed
and how many broken. "Right" ignores spaces and dots, Ё = Е (decisions #159, #160).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .postprocess import OCRPostprocessor


def squash(s: str) -> str:
    return re.sub(r'[\s.]', '', (s or '').upper().replace('Ё', 'Е'))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probs', type=Path, required=True)
    ap.add_argument('--crops', type=Path, required=True)
    ap.add_argument('--alphabet', type=Path, required=True)
    ap.add_argument('--out', type=Path, help='per-crop answers, JSON')
    ap.add_argument('--show', type=int, default=0, help='print this many fixed and broken crops')
    a = ap.parse_args()
    crops = json.loads(a.crops.read_text('utf-8'))
    probs = np.load(a.probs)
    t0 = time.time()
    pp = OCRPostprocessor(a.alphabet)
    print(f'загрузка {time.time() - t0:.1f} с')
    by = defaultdict(list)
    for c in crops:
        by[c['stamp']].append(c)
    st, ans, t0 = Counter(), [], time.time()
    times = []
    for sid, cs in by.items():
        t1 = time.time()
        r = pp.process_crops([{'label': c['label'], 'probs': probs[f'm{c["i"]}'], 'i': c['i']} for c in cs])
        times.append(time.time() - t1)
        for c, o in zip(cs, r['fields']):
            b, f = squash(o['raw']) == squash(c['gt']), squash(o['text']) == squash(c['gt'])
            for g in (c['label'], 'все'):
                st[(g, 'n')] += 1
                st[(g, 'before')] += b
                st[(g, 'after')] += f
                st[(g, 'fixed')] += f and not b
                st[(g, 'broken')] += b and not f
            ans.append({**c, 'raw': o['raw'], 'text': o['text'], 'stamp_fields': r['stamp']})
    print(f'{len(by)} штампов, {time.time() - t0:.1f} с; на штамп: медиана {np.median(times) * 1000:.0f} мс, '
          f'max {max(times) * 1000:.0f} мс')
    print(f'{"поле":12} {"n":>5} {"до":>7} {"после":>7} {"исправлено":>11} {"испорчено":>10}')
    for g in sorted({k[0] for k in st}, key=lambda g: (g == 'все', g)):
        n = st[(g, 'n')]
        print(f'{g:12} {n:5} {st[(g, "before")] / n:7.1%} {st[(g, "after")] / n:7.1%} '
              f'{st[(g, "fixed")]:11} {st[(g, "broken")]:10}')
    if a.show:
        for kind, cond in (('исправлено', lambda x: squash(x['text']) == squash(x['gt']) != squash(x['raw'])),
                           ('испорчено', lambda x: squash(x['raw']) == squash(x['gt']) != squash(x['text']))):
            print(f'\n--- {kind}')
            for x in [x for x in ans if cond(x)][:a.show]:
                print(f'{x["label"]:11} | {x["raw"]} -> {x["text"]} | эталон {x["gt"]}')
    if a.out:
        a.out.write_text(json.dumps(ans, ensure_ascii=False, indent=1), 'utf-8')


if __name__ == '__main__':
    main()
