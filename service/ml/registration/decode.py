"""Fields chosen on the CTC matrix rather than on the network's best path.

A candidate value - a house number, a dictionary street, a nearby department code -
is scored as a whole against the matrix, with a wildcard state that absorbs the
type word ("ДОМ №", "УЛ.") and any noise around it (``wildctc.WildCTC``). The
measured gains over the best path on the stamp test (decision #182): house
71.9 -> 79.5 %, flat 75.1 -> 80.1 %, street 50.8 -> 67.4 %, code 88.5 -> 90.8 %.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from .wildctc import WildCTC

LETTERS = ['', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'К']
#: log P margin to the free reading: a street of the found locality, a frequent street name
STREET_MARGIN_LOCAL, STREET_MARGIN_COMMON = 10.0, 15.0


def _num_forms(n1: int, let: str, n2: int | None) -> list[str]:
    s = str(n1) + let
    return [s] if n2 is None else [f'{s}/{n2}', f'{s}-{n2}', f'{s} {n2}']


def best_number(mats, index, greedy: str, gamma: float = 0.1, rounds: int = 2):
    """-> ('17А/2', log P) of the likeliest «number [letter] [/number]» over the matrices
    (crops with different margins - their scores add up). Coordinate ascent from the
    greedy reading: the first number over 1..999 (and one-digit neighbours of a
    four-digit reading), the letter, the second number."""
    wcs = [WildCTC(m.astype(np.float32), index, gamma) for m in mats if m is not None]
    if not wcs:
        return None
    m = re.match(r'(\d{1,4})([А-Я]?)(?:[/-](\d{1,3}))?', greedy or '')
    n1 = int(m.group(1)) if m and 0 < int(m.group(1)) < 10000 else 1
    let = m.group(2) if m and m.group(2) in LETTERS else ''
    n2 = int(m.group(3)) if m and m.group(3) and int(m.group(3)) > 0 else None
    big = sorted({int(s[:i] + c + s[i + 1:]) for s in [str(n1)] if len(s) == 4
                  for i in range(4) for c in '0123456789'} - {0})
    big = [v for v in big if v >= 1000]
    cache = {}

    def sc(n1, let, n2):
        k = (n1, let, n2)
        if k not in cache:
            cache[k] = max(sum(w.score([f]) for w in wcs) for f in _num_forms(n1, let, n2))
        return cache[k]
    best = sc(n1, let, n2)
    for _ in range(rounds):
        changed = False
        for v in list(range(1, 1000)) + big:
            s = sc(v, let, n2)
            if s > best:
                best, n1, changed = s, v, True
        for v in LETTERS:
            s = sc(n1, v, n2)
            if s > best:
                best, let, changed = s, v, True
        for v in [None] + list(range(1, 60)):
            s = sc(n1, let, v)
            if s > best:
                best, n2, changed = s, v, True
        if not changed:
            break
    return f'{n1}{let}' + (f'/{n2}' if n2 else ''), best


def street_choice(free: str, mats: list, ocr, local: list[str], common: list[str], n_pads: int,
                  ml: float = STREET_MARGIN_LOCAL, mc: float = STREET_MARGIN_COMMON) -> str | None:
    """A street from the dictionaries: the 25 names most like the reading among the
    found locality's streets, then among frequent names country-wide, each scored on
    the matrix of all street boxes (summed over crops). The best street of the
    locality wins if it is at most ``ml`` worse than the free reading, else the best
    frequent name within ``mc``. A number the reading has and the name has not
    («НАГОРНАЯ-2», «8-Я ЛИНИЯ») keeps the reading."""
    from rapidfuzz import fuzz, process
    if not free or not mats:
        return None
    tot, fs, n = defaultdict(float), 0.0, 0
    for pi in range(n_pads):
        ms = [m[pi].astype(np.float32) for m in mats if m[pi] is not None]
        if not ms:
            continue
        w = WildCTC(ocr.join(ms), ocr.abc.index, 0.1)
        n += 1
        fs += w.score(free.split())
        for kind, pool in (('local', local), ('common', common)):
            for c, _, _ in process.extract(free, pool, scorer=fuzz.ratio, limit=25) if pool else ():
                tot[(kind, c)] += w.score(c.split())
    for kind, m in (('local', ml), ('common', mc)):
        cs = [(s, c) for (k, c), s in tot.items() if k == kind]
        if cs:
            s, c = max(cs)
            if s >= fs - m * n:
                if set(re.findall(r'\d+', free)) - set(re.findall(r'\d+', c)):
                    return None
                return c
    return None


def code_candidates(read: str, codes) -> list[str]:
    """Department codes near the reading: up to two digits different, or the reading
    with one digit lost."""
    dg = re.sub(r'\D', '', read)
    out = []
    for k in codes:
        kd = k.replace('-', '')
        if len(kd) != 6:
            continue
        if len(dg) == 6 and sum(a != b for a, b in zip(kd, dg)) <= 2:
            out.append(k)
        elif len(dg) == 5 and any(kd[:i] + kd[i + 1:] == dg for i in range(6)):
            out.append(k)
    return out


def code_scores(read: str, mats: list, index, codes) -> dict | None:
    """-> {'read', 'scores': {code: log P}} for the reading and its dictionary neighbours,
    scored on the matrices of the first code box (``rules.code_by_place`` picks)."""
    wcs = [WildCTC(m.astype(np.float32), index, 0.1) for m in mats if m is not None]
    if not read or not wcs:
        return None
    cands = code_candidates(read, codes)
    return {'read': read, 'scores': {c: sum(w.score([c]) for w in wcs) for c in set(cands) | {read}}}
