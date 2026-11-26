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
        v = self.cache.get(tuple(parts))
        return v if v is not None else self.score_many([parts])[0]

    def _graph(self, parts):
        """States of ``W p1 W p2 ... W``: kinds (-1 wildcard, 0 blank, >0 character), predecessors
        (-2 = the start), the final states; None if a character is not in the alphabet."""
        kinds, preds = [], []

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
                    return None
                cs = add(c, [i for i, lc in ends if lc != ch])
                preds[cs].append(cs)
                bs = add(0, [cs])
                preds[bs].append(bs)
                ends = [(cs, ch), (bs, None)]
        return kinds, preds, [i for i, _ in ends if i >= 0]

    def score_many(self, cands: list[list[str]]) -> list[float]:
        """``score`` of many candidates in one forward pass: their state graphs side by side, each
        normalised on its own at every frame - the same numbers as one at a time, several times faster."""
        keys = [tuple(c) for c in cands]
        todo = list(dict.fromkeys(k for k in keys if k not in self.cache))
        graphs = []
        for k in todo:
            g = self._graph(k)
            if g is None:
                self.cache[k] = NEG
            else:
                graphs.append((k, g))
        if graphs:
            kinds, P_rows, start, fin, off = [], [], [], [], [0]
            for _, (kd, pr, fn) in graphs:
                o = off[-1]
                kinds += kd
                for ps in pr:
                    start.append(-2 in ps)
                    P_rows.append([x + o for x in ps if x != -2])
                fin.append([x + o for x in fn])
                off.append(o + len(kd))
            S, K = off[-1], max(len(r) for r in P_rows)
            P = np.full((S, K), S)
            for i, r in enumerate(P_rows):
                P[i, :len(r)] = r
            kinds = np.array(kinds)
            starts = np.array(off[:-1])
            sizes = np.diff(off)
            emit = np.where(kinds[None] == -1, self.pw[:, None], self.p[:, np.maximum(kinds, 0)])
            alpha = np.where(np.array(start), emit[0], 0.0)
            ext = np.zeros(S + 1)
            scale = np.zeros(len(graphs))
            dead = np.zeros(len(graphs), bool)
            for t in range(1, self.p.shape[0]):
                z = np.add.reduceat(alpha, starts)
                dead |= z <= 0
                z = np.where(z > 0, z, 1.0)
                scale += np.log(z)
                np.divide(alpha, np.repeat(z, sizes), out=ext[:S])
                alpha = ext[P].sum(1) * emit[t]
            for j, (k, _) in enumerate(graphs):
                zf = alpha[fin[j]].sum()
                self.cache[k] = float(scale[j] + np.log(zf)) if zf > 0 and not dead[j] else NEG
        return [self.cache[k] for k in keys]


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
    if not (min_year <= y and 1 <= m <= 12 and 1 <= d <= 31 and ok(d, m, y)):
        d, m, y = 15, 6, 2005
    cur = {'d': d, 'm': m, 'y': y}
    form = {'d': forms_day(d)[0], 'm': forms_month(m)[0], 'y': forms_year(y)[0]}

    def sc(f):
        return sum(w.score([f['d'], f['m'], *f['y']]) for w in wcs)

    def warm(fs):                                  # one batch per matrix; ``sc`` then reads the cache
        for w in wcs:
            w.score_many([[f['d'], f['m'], *f['y']] for f in fs])

    warm([form] + [{**form, a: fv} for a in 'dmy' for fv in forms(a, cur[a])])
    best = sc(form)
    for axis in 'dmy':                             # first the best forms of the initial date
        for fv in forms(axis, cur[axis]):
            s = sc({**form, axis: fv})
            if s > best:
                best, form[axis] = s, fv
    for _ in range(rounds):
        changed = False
        for axis, values in (('y', range(min_year, today.year + 1)), ('d', range(1, 32)), ('m', range(1, 13))):
            warm([{**form, axis: fv} for v in values for fv in forms(axis, v)])
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
