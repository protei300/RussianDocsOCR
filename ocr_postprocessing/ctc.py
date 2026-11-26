"""The OCR output as it comes from the stamp OCR: a CTC probability matrix ``[T, C]``.

Column 0 is the blank, column ``i > 0`` is ``alphabet[i - 1]``; the alphabet is the
``Outputs[0].Alphabet`` of the model's ``model.json``. Everything in this package works
on that matrix, not on a string: a date, a number, a street or a department code is
scored whole against the matrix (``wildctc``), which is where the gain over the
plain best-path reading comes from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CTCAlphabet:
    alphabet: str                       # without blank: column i > 0 -> alphabet[i - 1]
    index: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.index = {ch: i + 1 for i, ch in enumerate(self.alphabet)}

    @classmethod
    def from_model_json(cls, path: str | Path) -> 'CTCAlphabet':
        """The alphabet of an OCR model folder (``model.json``) or of the file itself."""
        p = Path(path)
        if p.is_dir():
            p = p / 'model.json'
        return cls(json.loads(p.read_text('utf-8'))['Outputs'][0]['Alphabet'])


def as_matrix(probs) -> np.ndarray:
    """The model's output as it comes (``[1, T, C]``) or already ``[T, C]`` -> float32 ``[T, C]``."""
    m = np.asarray(probs, np.float32)
    return m[0] if m.ndim == 3 else m


def greedy(probs: np.ndarray, abc: CTCAlphabet) -> str:
    """Best path: the argmax of every row, repeats merged, blanks dropped."""
    idx = as_matrix(probs).argmax(axis=1)
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0:
            out.append(abc.alphabet[i - 1])
        prev = i
    return ''.join(out)


def join(mats: list[np.ndarray], abc: CTCAlphabet) -> np.ndarray:
    """Matrices of several crops of one field in a row, a space between them."""
    mats = [np.asarray(m, np.float32) for m in mats]
    if len(mats) == 1:
        return mats[0]
    sep = np.zeros((2, mats[0].shape[1]), np.float32)
    sep[0, abc.index[' ']] = 1
    sep[1, 0] = 1
    out = [mats[0]]
    for m in mats[1:]:
        out += [sep, m]
    return np.vstack(out)


class MatrixJoiner:
    """What ``decode.street_choice`` needs of an OCR: ``join`` and ``abc``."""

    def __init__(self, abc: CTCAlphabet):
        self.abc = abc

    def join(self, mats):
        return join(mats, self.abc)
