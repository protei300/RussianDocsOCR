"""One crop of one field in, its reading improved out - without the rest of the stamp.

What a crop alone allows: the format of its field (a regular expression) and the CTC matrix -
the likeliest reading that fits the format, scored whole on the matrix - plus dictionaries
that need no context (department codes, the words of KLADR and FMS names)::

    imp = CropImprover('model_dir_or_alphabet')
    imp.improve('date', probs)   # -> {'raw', 'text', 'value', 'changed', 'rule'}

Labels: the client's crops (``date, addr_line, issuer_line, code, house, flat, korp``) and the
service's boxes (``date, place, street, house, building, flat, code``).

* **date** - a plain reading that is already a whole valid date stays; otherwise the likeliest real
  date on the matrix (``wildctc.best_date``, started from the reading), written in the stamp's form;
* **house, flat, korp / building** - a reading of the number's form stays; otherwise the likeliest
  «number [letter] [/number]» on the matrix (``decode.best_number``);
* **code** - ``NNN-NNN``: the reading (look-alike letters as digits) and its neighbours in the FMS
  code list, scored on the matrix; a code that is in the list wins over one that is not;
* **lines** (address, issuer, place, street) - a word missing from the vocabulary of KLADR names
  and FMS unit names is replaced by its nearest vocabulary word, only if the matrix finds the new
  line hardly less likely (``postprocess._replace_words``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from . import rules
from .ctc import CTCAlphabet, as_matrix, greedy
from .decode import best_number, code_candidates
from .dictionaries import AddressBook, DepartmentCodes
from .postprocess import References, date_init, date_text, strict_date, strip_date_year_suffix
from .textnorm import norm
from .wildctc import WildCTC, best_date

NUMBER_LABELS = {'house', 'flat', 'korp', 'building'}
LINE_LABELS = {'addr_line', 'issuer_line', 'place', 'street'}
RE_NUMBER = re.compile(r'\d{1,4}(?:\s?-?\s?[А-Я])?(?:\s?/\s?\d{1,4}[А-Я]?)?')
RE_CODE = re.compile(r'\d{3}-\d{3}')


class CropImprover:
    """One per process; the vocabulary and the code list are built once and kept in memory."""

    #: log P a vocabulary word may cost the line on the matrix
    word_margin = 10.0
    #: rules on: 'date', 'number', 'code', 'words'
    rules_on = ('date', 'number', 'code', 'words')
    all_streets = False

    def __init__(self, alphabet, refs: References | None = None, vocabulary: bool = True):
        if hasattr(alphabet, 'alphabet'):
            alphabet = CTCAlphabet(alphabet.alphabet)
        elif isinstance(alphabet, Path) or (isinstance(alphabet, str) and Path(alphabet).exists()):
            alphabet = CTCAlphabet.from_model_json(alphabet)
        else:
            alphabet = CTCAlphabet(str(alphabet))
        self.abc = alphabet
        self.refs = refs or References()
        self.codes = DepartmentCodes(self.refs.fms_csv)
        self._codes = list(self.codes._load())
        self.vocab: list[str] = []
        self._vocab_set: set[str] = set()
        if vocabulary:
            self._build_vocab()

    def _build_vocab(self) -> None:
        """Words of 4+ letters: names of regions, districts, localities (KLADR), frequent street names,
        FMS unit names. Kept as a list for the fuzzy search and a set for the look-up."""
        book = AddressBook(self.refs.addressru)
        book.preload(key=rules._key)
        words = set()

        def add(text):
            words.update(w for w in re.findall(r'[А-ЯЁ]+', norm(text)) if len(w) >= 4)
        for r in book.regions():
            add(r.name)
            for p in book.children(r.key):
                add(p.name)
            for p in book.localities(r):
                add(p.name)
        if self.refs.street_names.exists():
            for n in json.loads(self.refs.street_names.read_text('utf-8')):
                add(n)
        if self.all_streets and book._zip is not None:          # every street name of KLADR (1.2 M rows)
            for name in book._zip_names:
                if name.endswith('.json') and name[:1].isdigit() and name != '00.json':
                    for row in json.loads(book._zip.read(name).decode('utf-8')):
                        if 500 <= int(row[2]) < 600:
                            add(row[1])
        for names in self.codes._load().values():
            for n in names:
                add(n)
        self._vocab_set = words
        self.vocab = sorted(words)
        self._by_len: dict[int, list[str]] = {}
        for w in self.vocab:
            self._by_len.setdefault(len(w), []).append(w)

    # ------------------------------------------------------------------------------------ public
    def improve(self, label: str, probs: np.ndarray) -> dict:
        m = as_matrix(probs)
        raw = greedy(m, self.abc)
        out = {'raw': raw, 'text': raw, 'rule': None}
        if label == 'date' and 'date' in self.rules_on:
            out.update(self._date(raw, m))
        elif label in NUMBER_LABELS and 'number' in self.rules_on:
            out.update(self._number(raw, m))
        elif label == 'code' and 'code' in self.rules_on:
            out.update(self._code(raw, m))
        elif label in LINE_LABELS and 'words' in self.rules_on and self.vocab:
            t = self._words(raw, m)
            if t != raw:
                out.update(text=t, rule='словарь слов')
        out['changed'] = out['text'] != raw
        return out

    # ------------------------------------------------------------------------------------ fields
    def _date(self, raw: str, m: np.ndarray) -> dict:
        v = strict_date(raw)
        if v:
            cleaned = strip_date_year_suffix(raw)
            return {'value': v, 'text': cleaned}
        best = None
        for init in dict.fromkeys((date_init(raw), None)):
            r = best_date([m], self.abc.index, init=init)
            if best is None or r[1] > best[1]:
                best = r
        v = best[0]
        same = date_init(raw) == tuple(int(x) for x in v.split('.'))
        text = strip_date_year_suffix(raw) if same else date_text(v, [m], self.abc.index)
        return {'value': v, 'text': text, 'rule': 'дата по матрице'}

    def _number(self, raw: str, m: np.ndarray) -> dict:
        t = norm(raw).strip()
        if RE_NUMBER.fullmatch(t):
            return {'value': rules.num_value(t)}
        if not re.search(r'\d', raw):
            return {}
        r = best_number([m], self.abc.index, rules.num_value(raw))
        if not r:
            return {}
        return {'value': r[0], 'text': r[0], 'rule': 'номер по матрице'}

    def _code(self, raw: str, m: np.ndarray) -> dict:
        mm = re.search(r'([0-9ОЗБЧТ]{3})\s*[-–.]?\s*([0-9ОЗБЧТ]{2,3})', norm(raw))
        if not mm:
            return {}
        read = f'{rules.fix_num(mm.group(1))}-{rules.fix_num(mm.group(2))}'
        if RE_CODE.fullmatch(read) and read in self.codes:
            return {'value': read, 'text': read, 'rule': None if read == raw else 'код: формат'}
        w = WildCTC(m, self.abc.index, 0.1)
        cands = [c for c in code_candidates(read, self._codes)]
        allc = list(dict.fromkeys(cands + [read]))
        sc = dict(zip(allc, w.score_many([[c] for c in allc])))
        known = [c for c in allc if c in self.codes]
        pick = max(known, key=sc.get) if known else read
        if pick != read and sc[pick] < sc.get(read, -1e30) - 2.0:
            pick = read
        return {'value': pick, 'text': pick, 'rule': 'код по справочнику' if pick != read else 'код: формат'}

    def words(self, text: str, m: np.ndarray) -> str:
        """The vocabulary correction of a line already improved otherwise (the matrix judges against it)."""
        return self._words(text, m) if self.vocab and 'words' in self.rules_on else text

    def _words(self, raw: str, m: np.ndarray) -> str:
        vs = self._vocab_set

        def fix(mo):
            w = mo.group(0)
            if len(w) < 5 or w in vs or w in rules.ALL_T or w in rules.NUM_T:
                return w
            pool = [x for n in range(len(w) - 1, len(w) + 2) for x in self._by_len.get(n, ())]
            hit = process.extractOne(w, pool, scorer=fuzz.ratio, score_cutoff=80)
            if not hit:
                return w
            c = hit[0]
            if Levenshtein.distance(w, c) > 2 or w in c or c in w or rules.stem(w) == rules.stem(c):
                return w
            return c
        new = re.sub(r'[А-ЯЁ]+', fix, raw)
        if new != raw:
            w = WildCTC(m, self.abc.index, 1e-4)
            a, b = w.score_many([[new], [raw]])
            if a < b - self.word_margin:
                return raw
        return new
