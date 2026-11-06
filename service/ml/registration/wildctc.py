"""The date on the CTC matrix with noise between its parts: P(* day * month * year * | matrix).

A stamp date is printed and handwritten at once («__» ______ 20__ г., the gaps filled
by hand), with signatures, lines and quotes between the parts. No regular expression
over the best path takes it: the network's best path through the noise reads like
«08 ... 08 ... 20 08». Here every candidate date is scored whole against the matrix,
the noise is absorbed by a wildcard state ``W`` (a frame is blank, or any character
at weight ``gamma``), and the likeliest real date wins. The search is coordinate
ascent: year, day, month in turn, two rounds. On the stamp test (513 dates) the full
date went from 40.7 % (best path) to 62.2 %.
"""
from __future__ import annotations

import calendar
import datetime as dt

import numpy as np

MONTHS_GEN = ['ЯНВАРЯ', 'ФЕВРАЛЯ', 'МАРТА', 'АПРЕЛЯ', 'МАЯ', 'ИЮНЯ', 'ИЮЛЯ', 'АВГУСТА', 'СЕНТЯБРЯ',
              'ОКТЯБРЯ', 'НОЯБРЯ', 'ДЕКАБРЯ']
NEG = -1e30
#: a digit on a stamp also reads as its look-alike letter: that letter counts for the digit
LOOKALIKE = {'0': 'О', '3': 'З', '6': 'Б', '4': 'Ч', '7': 'Т'}


def _lse(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    m = np.where(m <= NEG, 0, m)
    with np.errstate(divide='ignore'):
        return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class WildCTC:
    def __init__(self, probs: np.ndarray, index: dict, gamma: float = 0.1):
        p = probs.astype(np.float64).copy()
        for dg, ch in LOOKALIKE.items():
            if dg in index and ch in index:
                p[:, index[dg]] += p[:, index[ch]]
        self.p = np.clip(p, 1e-30, 1.0)
        self.pw = self.p[:, 0] + gamma * (1 - self.p[:, 0])        # wildcard: blank or any character, penalised
        self.index = index
        self.cache: dict = {}

    def score(self, parts: list[str]) -> float:
        """log P(W p1 W p2 ... W | matrix); the parts are strings of alphabet characters."""
        key = tuple(parts)
        if key in self.cache:
            return self.cache[key]
        kinds, preds = [], []                      # kind: -1 wildcard, 0 blank, >0 character

        def add(kind, ps):
            kinds.append(kind)
            preds.append(ps)
            return len(kinds) - 1

        START = -2
        ends = [(START, None)]                     # (state, last character) - where the next part may enter from
        for seg in ['*'] + [x for p in parts for x in (p, '*')]:
            if seg == '*':
                w = add(-1, [i for i, _ in ends])
                preds[w].append(w)
                ends = ends + [(w, None)]
                continue
            for ch in seg:
                c = self.index.get(ch)
                if c is None:
                    self.cache[key] = NEG
                    return NEG
                cs = add(c, [i for i, lc in ends if lc != ch])
                preds[cs].append(cs)
                bs = add(0, [cs])
                preds[bs].append(bs)
                ends = [(cs, ch), (bs, None)]
        S, K = len(kinds), max(len(p) for p in preds)
        start = np.array([START in p for p in preds])
        P = np.full((S, K), S)
        for s, ps in enumerate(preds):
            ps = [x for x in ps if x != START]
            P[s, :len(ps)] = ps
        kinds = np.array(kinds)
        # forward pass in probabilities, normalised at every step (a sum instead of log-sum-exp: several times faster)
        emit = np.where(kinds[None] == -1, self.pw[:, None], self.p[:, np.maximum(kinds, 0)])
        alpha = np.where(start, emit[0], 0.0)
        ext = np.zeros(S + 1)                      # alpha plus one always-zero slot for absent predecessors
        scale = 0.0
        for t in range(1, self.p.shape[0]):
            z = alpha.sum()
            if z <= 0:
                self.cache[key] = NEG
                return NEG
            scale += np.log(z)
            np.divide(alpha, z, out=ext[:S])
            alpha = ext[P].sum(1) * emit[t]
        fin = [i for i, _ in ends if i >= 0]
        z = alpha[fin].sum()
        v = float(scale + np.log(z)) if z > 0 else NEG
        self.cache[key] = v
        return v


def forms_day(d: int) -> list[str]:
    return [str(d)] + ([f'{d:02d}'] if d < 10 else [])


def forms_month(m: int, words: bool = True) -> list[str]:
    """As a number, a word, the word with one letter lost (handwriting drops letters), its start («ОКТ.»)."""
    out = [str(m)] + ([f'{m:02d}'] if m < 10 else [])
    if words:
        w = MONTHS_GEN[m - 1]
        out += [w, w[:3]] + [w[:i] + w[i + 1:] for i in range(1, len(w))]
    return list(dict.fromkeys(out))


def forms_year(y: int) -> list[list[str]]:
    """The whole year, two digits, and «20 08»: the printed start of the form and the handwritten end, noise between."""
    return [[str(y)], [f'{y % 100:02d}'], [str(y)[:2], str(y)[2:]]]


def best_date(probs, index: dict, gamma: float = 0.1, min_year: int = 1950,
              init: tuple | None = None, rounds: int = 2) -> tuple[str, float]:
    """-> ('DD.MM.YYYY', log P) of the likeliest real date. ``probs`` is a matrix or a list of
    matrices of one date (crops with different margins): then the score is their sum - what one crop
    cuts off, another shows."""
    mats = probs if isinstance(probs, (list, tuple)) else [probs]
    wcs = [WildCTC(m, index, gamma) for m in mats]
    today = dt.date.today()

    def ok(d, m, y):
        return y <= today.year and d <= calendar.monthrange(y, m)[1] and dt.date(y, m, d) <= today

    def forms(axis, v):
        return forms_day(v) if axis == 'd' else forms_month(v) if axis == 'm' else forms_year(v)

    # the current date and its forms; while one part is searched, the other two keep their best forms
    d, m, y = init or (15, 6, 2005)
    if not (1 <= d <= 28 and ok(d, m, y)):
        d, m, y = 15, 6, 2005
    cur = {'d': d, 'm': m, 'y': y}
    form = {'d': forms_day(d)[0], 'm': forms_month(m)[0], 'y': forms_year(y)[0]}

    def sc(f):
        return sum(w.score([f['d'], f['m'], *f['y']]) for w in wcs)

    best = sc(form)
    for axis in 'dmy':                             # first the best forms of the initial date
        for fv in forms(axis, cur[axis]):
            s = sc({**form, axis: fv})
            if s > best:
                best, form[axis] = s, fv
    for _ in range(rounds):
        changed = False
        for axis, values in (('y', range(min_year, today.year + 1)), ('d', range(1, 32)), ('m', range(1, 13))):
            for v in values:
                c = {**cur, axis: v}
                if v == cur[axis] or not ok(c['d'], c['m'], c['y']):
                    continue
                for fv in forms(axis, v):
                    s = sc({**form, axis: fv})
                    if s > best:
                        best, cur, form[axis], changed = s, c, fv, True
        if not changed:
            break
    return f'{cur["d"]:02d}.{cur["m"]:02d}.{cur["y"]:04d}', best
