"""Fields of one stamp in, the same fields improved out.

Input - the fields of one registration stamp, as the field network of the client's
pipeline cuts them, each with the CTC matrix of the stamp OCR::

    [{'label': 'date' | 'addr_line' | 'issuer_line' | 'code' | 'house' | 'flat' | 'korp',
      'probs': np.ndarray [T, C]},          # softmax rows of the OCR (blank + alphabet)
     ...]                                   # any other keys are passed through

Output - the same list, in the same order, each field with ``raw`` (the plain best
path), ``text`` (the improved reading, in the form the field is written on the
stamp) and ``changed``; plus ``stamp`` - what the fields say together:
``date, region, district, locality, street, house, building, flat, code``.

How a field is improved (every step was measured on the stamp test, see README):

* **date** - every real date is scored whole against the matrix with a wildcard for
  the noise between its parts (``wildctc.best_date``), then written in the form the
  stamp uses («18 ОКТЯБРЯ 2007 Г.», «21.09.2023»), the form also scored on the matrix;
* **house, flat, korp** - the likeliest «number [letter] [/number]» on the matrix
  (``decode.best_number``);
* **code** - the reading and its department-code neighbours (FMS), scored on the
  matrix; the one that agrees with the place (region = first two digits) wins;
* **address lines** - cut into pieces by type words (region, district, locality,
  street, numbers) and linked through KLADR: region -> locality -> street, backed by
  the department code; a street is chosen among the found locality's streets on the
  matrix; a misread name word is replaced by the dictionary's word;
* **issuer lines** - words fixed by the unit names of the department code (FMS).

The code helps every other field: it gives the region (and with it the pool of
localities and streets) and the unit name, which names the city or district.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from . import rules
from .ctc import CTCAlphabet, MatrixJoiner, as_matrix, greedy, join
from .decode import best_number, code_scores, street_choice
from .dictionaries import AddressBook, DepartmentCodes
from .textnorm import MONTHS_GEN, RE_DATE_NUM, RE_DATE_WORD, digits_fixed, norm
from .wildctc import WildCTC, best_date

LABELS = ('date', 'addr_line', 'issuer_line', 'code', 'house', 'flat', 'korp')      # the client's field network
NET_CLASSES = ('date', 'place', 'street', 'house', 'building', 'flat', 'code')    # the service's fields_m
NUM_LABEL = {'house': 'house', 'flat': 'flat', 'korp': 'building'}
#: number type words inside an address line -> stamp field
NUM_WORDS = {'Д': 'house', 'ДОМ': 'house', 'КВ': 'flat', 'КВАРТИРА': 'flat', 'КОРП': 'building',
             'КОРПУС': 'building', 'К': 'building', 'СТР': 'structure', 'СТРОЕНИЕ': 'structure'}
STAMP_FIELDS = ('date', 'region', 'district', 'locality', 'street', 'house', 'building', 'structure', 'flat',
                'code')

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / 'service' / 'ml' / 'registration' / 'assets'


def _default_ref_path(env_name: str, filename: str) -> Path:
    explicit = os.environ.get(env_name)
    if explicit:
        return Path(explicit)
    models_root = Path(os.environ.get('RDOCS_MODELS_ROOT') or REPO_ROOT)
    refs_dir = Path(os.environ.get('RDOCS_REGISTRATION_REFS')
                    or (models_root / 'document_processing' / 'models' / 'RegistrationRefs'))
    candidate = refs_dir / filename
    if candidate.is_file():
        return candidate
    return ASSETS / filename


@dataclass
class References:
    """Where the dictionaries are. The same environment variables as the service."""
    addressru: Path = None          # KLADR/FIAS tree, AddressRU layout (folder or flat zip)
    fms_csv: Path = None            # department codes ``code,name``
    street_names: Path = None       # frequent street names of all KLADR, JSON list

    def __post_init__(self):
        self.addressru = Path(self.addressru or _default_ref_path('RDOCS_ADDRESSRU', 'addressru.zip'))
        self.fms_csv = Path(self.fms_csv or _default_ref_path('RDOCS_FMS_UNIT', 'fms_unit_2026-09-22.csv'))
        self.street_names = Path(self.street_names or _default_ref_path('RDOCS_STREET_NAMES', 'street_names_kladr.json'))

    def available(self) -> bool:
        return all(p.is_file() for p in (self.addressru, self.fms_csv, self.street_names))


# ------------------------------------------------------------------------------------------ dates
DATE_TAILS = ('', ' Г.', ' Г', 'Г.', ' ГОД', ' ГОДА')
_RE_DATE_YEAR_SUFFIX = re.compile(r'\s*Г\.?\s*$', re.IGNORECASE)


def strip_date_year_suffix(text: str) -> str:
    """Drop trailing 'г.' / 'г' from a stamp date text (blank-border artifact)."""
    return _RE_DATE_YEAR_SUFFIX.sub('', text.strip()).strip()


def date_text(value: str, mats: list[np.ndarray], index: dict) -> str:
    """'DD.MM.YYYY' -> the stamp's own form of it, the likeliest on the matrix: day with or without a
    zero, month as a word or a number, the parts by spaces or dots, with any trailing 'г.' dropped."""
    d, m, y = (int(x) for x in value.split('.'))
    days = [f'{d:02d}'] + ([str(d)] if d < 10 else [])
    cands = []
    for dd in days:
        for yy in (str(y), f'{y % 100:02d}'):                # «29.03.99» - a two-digit year is written too
            cands += [f'{dd} {MONTHS_GEN[m - 1]} {yy}{t}' for t in DATE_TAILS]
            cands += [f'{dd}.{m:02d}.{yy}{t}' for t in DATE_TAILS]
            cands += [f'{dd} {m:02d} {yy}{t}' for t in DATE_TAILS]
    wcs = [WildCTC(x, index, 1e-4) for x in mats]           # almost no wildcard: the whole text is scored
    for w in wcs:
        w.score_many([[c] for c in cands])
    best = max(cands, key=lambda c: sum(w.score([c]) for w in wcs))
    return strip_date_year_suffix(best)


def date_init(raw: str) -> tuple | None:
    """The date of the plain reading, if it has one - the start of the search."""
    t = digits_fixed(norm(raw))
    m = RE_DATE_NUM.search(t)
    if m:
        d, mo, y = int(m['d']), int(m['m']), int(m['y'])
    else:
        m = RE_DATE_WORD.search(t)
        if not m:
            return None
        mw = m['m']
        mo = next((i + 1 for i, w in enumerate(MONTHS_GEN) if w[:3] == mw[:3]), None)
        if mo is None:
            return None
        d, y = int(rules.fix_num(m['d'])), int(rules.fix_num(m['y']))
    if y < 100:
        y += 2000 if y < 50 else 1900
    return (d, mo, y) if 1 <= d <= 31 and 1 <= mo <= 12 else None


_RE_STRICT_NUM = re.compile(r'(?<!\d)(\d{2})[ .](\d{2})[ .]((?:19|20)\d{2})(?!\d)')
_RE_STRICT_WORD = re.compile(r'(?<!\d)(\d{1,2}) (' + '|'.join(MONTHS_GEN) + r') ((?:19|20)\d{2})(?!\d)')


def strict_date(raw: str) -> str | None:
    """'DD.MM.YYYY' if the reading holds exactly one whole date in a strict form («18 09 2018 Г.»,
    «18 СЕНТЯБРЯ 2018»), a real day not in the future; else None."""
    import datetime as dt
    t = norm(raw)
    hits = [(int(m[1]), int(m[2]), int(m[3])) for m in _RE_STRICT_NUM.finditer(t)]
    hits += [(int(m[1]), MONTHS_GEN.index(m[2]) + 1, int(m[3])) for m in _RE_STRICT_WORD.finditer(t)]
    if len(hits) != 1:
        return None
    d, m, y = hits[0]
    try:
        v = dt.date(y, m, d)
    except ValueError:
        return None
    return f'{d:02d}.{m:02d}.{y}' if v <= dt.date.today() and y >= 1950 else None


# ---------------------------------------------------------------------------------------- address
def address_pieces(text: str) -> list[tuple[str, str]]:
    """An address line -> pieces (kind, text): kind 'street', 'house' / 'flat' / 'building' /
    'structure' (the number), or 'place' (region, district, locality - the rules tell them apart)."""
    out = []
    for chunk in norm(text).split(','):
        toks = rules.tokens(chunk)
        if not toks:
            continue
        cur_kind, cur = None, []

        def flush():
            if cur:
                out.append((cur_kind or 'place', ' '.join(cur)))
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < len(toks) else ''
            if t in NUM_WORDS and (re.match(r'[\dОЗБ]', nxt) or t not in ('Д', 'К')):
                flush()
                cur_kind, cur = NUM_WORDS[t], []          # «Д. 58», «КВ. 495»; «Д. ИВАНОВКА» is a village
                continue
            if t in rules.STREET_T and cur_kind not in ('street',):
                flush()
                cur_kind, cur = 'street', [t]
                continue
            if cur_kind in NUM_WORDS.values() and cur:     # «Д. 5 КВ 7»: a number takes one token
                flush()
                cur_kind, cur = None, []
            if t in rules.REGION_T | rules.DISTRICT_T | rules.LOCALITY_T and cur_kind == 'street':
                flush()
                cur_kind, cur = None, []
            cur.append(t)
        flush()
    return out


def _replace_words(raw: str, names: list[str], mat: np.ndarray | None = None, index: dict | None = None,
                   cutoff: float = 80, margin: float = 6.0) -> str:
    """Words of ``raw`` that are misreadings of the dictionary ``names`` -> the dictionary's word.

    Not touched: a word in another case of the same name («ОМСКОЙ», the name is «ОМСКАЯ»), numbers, type
    words, words shorter than 4 letters; a word that is another word of the same root («ЧАПАЕВА» and
    «ЧАПАЕВСК», «БИРОБИДЖАНСКИЙ» and «БИРОБИДЖАН») - more than two letters apart; a word cut by the edge of
    the crop («ЛОЯРОСЛАВЕЦКОМ» inside «МАЛОЯРОСЛАВЕЦКОМ»). With the line's matrix a change is kept only if
    the matrix finds the new line at most ``margin`` (log P) less likely than the reading."""
    vocab = {w for n in names for w in re.findall(r'[А-ЯЁ]+', norm(n)) if len(w) >= 4}
    if not vocab:
        return raw
    stems = {rules.stem(w) for w in vocab}
    vl = list(vocab)

    def fix(m):
        w = m.group(0)
        if len(w) < 4 or w in vocab or w in rules.ALL_T or w in rules.NUM_T or rules.stem(w) in stems:
            return w
        hit = process.extractOne(w, vl, scorer=fuzz.ratio)
        if not hit or hit[1] < cutoff:
            return w
        c = hit[0]
        if Levenshtein.distance(w, c) > 2 or w in c or c in w:
            return w
        return c
    new = re.sub(r'[А-ЯЁ]+', fix, raw)
    if new != raw and mat is not None:
        w = WildCTC(mat, index, 1e-4)
        if w.score([new]) < w.score([raw]) - margin:
            return raw
    return new


# ---------------------------------------------------------------------------------------------- main
class OCRPostprocessor:
    """One per process: the dictionaries are loaded once and kept in memory. Two inputs:
    ``process_crops`` - crops of the client's field network (labels ``LABELS``), ``process_boxes`` -
    boxes of the service's field network ``fields_m`` (labels ``NET_CLASSES``)."""

    def __init__(self, alphabet: CTCAlphabet | str | Path, refs: References | None = None, preload: bool = True,
                 vocabulary: bool = True):
        if hasattr(alphabet, 'alphabet'):                          # any alphabet object: the service's too
            alphabet = CTCAlphabet(alphabet.alphabet)
        elif isinstance(alphabet, Path) or (isinstance(alphabet, str) and Path(alphabet).exists()):
            alphabet = CTCAlphabet.from_model_json(alphabet)          # a model folder or its model.json
        else:
            alphabet = CTCAlphabet(str(alphabet))                     # the alphabet itself
        self.abc = alphabet
        self.refs = refs or References()
        import json
        self.dicts = rules.Dicts(AddressBook(self.refs.addressru), DepartmentCodes(self.refs.fms_csv))
        self.common_streets = json.loads(self.refs.street_names.read_text('utf-8')) \
            if self.refs.street_names.exists() else []
        self._codes = list(self.dicts.codes._load())
        # everything in memory before the first stamp: localities of every region with search keys
        self.load_seconds = self.dicts.book.preload(key=rules._key) if preload else 0.0
        for r in self.dicts.regions:
            self.dicts.locs(r.key)
        self._joiner = MatrixJoiner(self.abc)
        # the crop-level rules with their word vocabulary (``crop.CropImprover``), shared
        from .crop import CropImprover
        self.crops = CropImprover(self.abc, self.refs) if vocabulary else None

    # ---------------------------------------------------------------------------- public
    def process_crops(self, fields: list[dict]) -> dict:
        idx = self.abc.index
        out = [{k: v for k, v in f.items() if k != 'probs'} for f in fields]
        mats = [as_matrix(f['probs']) for f in fields]
        for o, m in zip(out, mats):
            o['raw'] = greedy(m, self.abc)
            o['text'] = o['raw']
        by = defaultdict(list)
        for i, f in enumerate(fields):
            by[f['label']].append(i)

        # dates: each date field on its own (a stamp may carry two)
        dates = []
        for i in by.get('date', []):
            v = self._date(mats[i], out[i]['raw'])
            out[i]['value'] = v
            raw_or_fmt = (
                out[i]['raw']
                if date_init(out[i]['raw']) == tuple(int(x) for x in v.split('.'))
                else date_text(v, [mats[i]], idx)
            )
            out[i]['text'] = strip_date_year_suffix(raw_or_fmt)
            dates.append(v)
        # numbers of their own fields
        nums = {'house': [], 'building': [], 'flat': []}
        for lab, fld in NUM_LABEL.items():
            for i in by.get(lab, []):
                raw = out[i]['raw']
                v = None
                if re.search(r'\d', raw):
                    r = best_number([mats[i]], idx, rules.num_value(raw))
                    v = r[0] if r else None
                v = v or raw
                out[i]['value'] = v
                # the stamp's own form of the same number stays («11-А», not «11А»)
                same = re.fullmatch(r'\d+(?:\s*[-/]?\s*[А-Я])?(?:\s*/\s*\d+[А-Я]?)?', norm(raw).strip())                     and rules.num_value(raw) == v
                out[i]['text'] = norm(raw).strip() if same else v
                nums[fld].append(v)
        # department code: the reading and its FMS neighbours on the matrix
        codes, code_i = None, None
        for i in by.get('code', []):
            m = re.search(r'([0-9ОЗБЧТ]{3})\s*[-–.]?\s*([0-9ОЗБЧТ]{2,3})', norm(out[i]['raw']))
            if m:
                read = f'{rules.fix_num(m.group(1))}-{rules.fix_num(m.group(2))}'
                codes, code_i = code_scores(read, [mats[i]], idx, self._codes), i
                break

        # address: the pieces of every line as the readings the rules know
        reads = defaultdict(list)
        street_lines = []
        for i in by.get('addr_line', []):
            for kind, txt in address_pieces(out[i]['raw']):
                cls = 'place' if kind == 'place' else 'street' if kind == 'street' else kind
                reads[cls].append({'text': txt})
                if kind == 'street':
                    street_lines.append(i)
        for fld, vs in nums.items():
            reads[fld] += [{'text': v} for v in vs]
        # the unit name of the issuer lines names the region / city: an extra place hint, after the address
        issuer = ' '.join(out[i]['raw'] for i in by.get('issuer_line', []))
        reads['code'] = [{'text': codes['read']}] if codes else []
        f = rules.stamp_fields(dict(reads), self.dicts, dates[0] if dates else None, None, codes)
        if 'region' not in f and issuer:
            r = self.dicts.book.region_in_text(issuer)
            if r is not None:
                f['region'] = r.name
        # street: dictionary names on the matrices of the lines that carry it
        self._street(f, reads, [mats[i] for i in dict.fromkeys(street_lines)], by, out, mats)

        # back to the fields: code, address and issuer lines with dictionary words
        if code_i is not None and f.get('code'):
            out[code_i]['text'] = out[code_i]['value'] = f['code']
        names = [f[k] for k in ('region', 'district', 'locality', 'street') if f.get(k)]
        for i in by.get('addr_line', []):
            out[i]['text'] = _replace_words(out[i]['raw'], names, mats[i], idx)
        unit = self.dicts.codes.names(f['code']) if f.get('code') in self.dicts.codes else []
        for i in by.get('issuer_line', []):
            out[i]['text'] = _replace_words(out[i]['raw'], unit + names, mats[i], idx)
        # then the words no dictionary of this stamp knows: the vocabulary of all KLADR and FMS names
        if self.crops is not None:
            for i in by.get('addr_line', []) + by.get('issuer_line', []):
                out[i]['text'] = self.crops.words(out[i]['text'], mats[i])
        for o in out:
            o['changed'] = o['text'] != o['raw']
        return {'fields': out, 'stamp': {k: f[k] for k in STAMP_FIELDS if f.get(k)}}

    # ---------------------------------------------------------------------------- place on the matrix
    #: fields chosen on the matrices of the place boxes, in this order (each narrows the next)
    place_ctc = ('region', 'district', 'locality')
    place_margin = 8.0
    #: these fields, once found by the rules, are replaced on the matrix only by a piece typed as them
    place_typed_only = ('region',)
    street_ml, street_mc, street_limit = 10.0, 15.0, 150   # the street: margins for the locality's streets / frequent names           # log P a dictionary name may lose to the free reading of its box
    place_limit = 20

    def _place_ctc(self, f: dict, boxes: list) -> None:
        """Region, district, locality chosen on the matrices of the place boxes, as the street is: for
        every place piece the dictionary names most like its words - regions; the region's districts; the
        region's localities (all cities when the region is unknown) - each scored whole on the box's
        matrices; the best one at most ``place_margin`` worse than the piece's own reading wins. A piece
        typed as another field, or already taken by another field, is not tried."""
        d = self.dicts
        for field in self.place_ctc:
            reg = next((r for r in d.regions if r.name == f.get('region')), None)
            if field == 'locality':
                if reg is not None and reg.type == 'ГОРОД':
                    continue
                pool = d.locs(reg.key) if reg is not None else d.book.cities()
                keys = d.book.loc_keys.get(reg.key) if reg is not None else None
            elif field == 'district':
                if reg is None:
                    continue
                pool, keys = d.areas(reg.key), None
            else:
                pool, keys = d.regions, None
            if not pool:
                continue
            if keys is None or len(keys) != len(pool):
                keys = [rules._key(x.name) for x in pool]
            others = [k for k in ('region', 'district', 'locality') if k != field]
            skip = {rules._key(f[k]) for k in others if f.get(k)}
            best = None
            for raw, ms in boxes:
                for seg in rules.segments([raw]):
                    if seg['kind'] not in (None, field) or not seg['key']                             or (field in self.place_typed_only and f.get(field) and seg['kind'] != field) \
                            or any(fuzz.ratio(seg['key'], k) >= 85 for k in skip):
                        continue
                    free = ' '.join(seg['words'])
                    cands = process.extract(seg['key'], keys, scorer=fuzz.ratio, limit=self.place_limit)
                    names = list(dict.fromkeys(pool[i].name for _, _, i in cands))
                    tot, fs = defaultdict(float), 0.0
                    for m in ms:
                        if m is None:
                            continue
                        w = WildCTC(m, self.abc.index, 0.1)
                        w.score_many([free.split()] + [n.split() for n in names])
                        fs += w.score(free.split())
                        for n in names:
                            tot[n] += w.score(n.split())
                    if not tot:
                        continue
                    sc, name = max((v, k) for k, v in tot.items())
                    n_m = sum(m is not None for m in ms)
                    gain = sc - fs + (5.0 if seg['kind'] == field else 0.0)
                    if sc >= fs - self.place_margin * n_m and (best is None or gain > best[0]):
                        best = (gain, name)
            if best is not None:
                f[field] = best[1]

    street_after_place = True     # the street pool from the locality chosen on the matrix

    def _found_place(self, f: dict, loc, region):
        """The dictionary objects of the region and locality as finally chosen (``_place_ctc`` may have
        changed them): the same names, the rules' objects kept when they agree."""
        d = self.dicts
        if f.get('region') and (region is None or region.name != f['region']):
            region = next((r for r in d.regions if r.name == f['region']), region)
        if f.get('locality') and (loc is None or loc.name != f['locality']) and region is not None:
            same = [x for x in d.locs(region.key) if x.name == f['locality']]
            if same:
                loc = sorted(same, key=lambda x: (x.type not in ('ГОРОД', 'Г'), len(x.key)))[0]
        return loc, region

    # ---------------------------------------------------------------------------- date
    #: which crops of a date box the date is searched on: 'margin' (the date's margin, measured best:
    #: 85.4 % against 84.0 % for both summed on the 515-stamp test), 'plain', 'both'
    date_crops = 'margin'
    #: a plain reading that is already a whole valid date (strict pattern) is taken as is
    date_greedy_first = True

    def _date(self, mat, raw: str) -> str:
        """The likeliest real date on the matrix: coordinate ascent started from the plain reading's date
        and from a neutral one (15.06.2005); a search started far away gets stuck in a worse date (the
        reading «24 12 2019» came out as 12.02.2001)."""
        if self.date_greedy_first:
            v = strict_date(greedy(mat[0] if isinstance(mat, list) else mat, self.abc))
            if v:
                return v
        starts = [s for s in (date_init(raw), None) if s is not None or True]
        best = None
        for init in dict.fromkeys(starts):
            v, sc = best_date(mat if isinstance(mat, list) else [mat], self.abc.index, init=init)
            if best is None or sc > best[1]:
                best = (v, sc)
        return best[0]

    # ---------------------------------------------------------------------------- street
    def _street(self, f: dict, reads: dict, mats: list, by: dict, out: list, all_mats: list) -> None:
        d = self.dicts
        texts = rules.dedupe_texts([r['text'] for r in reads.get('street', [])])
        free = ' '.join(t for t in rules.tokens(' '.join(texts)) if t not in rules.STREET_T)
        pl = rules.place_fields(dict(reads), f.get('code'), d)
        loc, region = pl.get('_loc'), pl.get('_region')
        local = []
        if loc is not None:
            local = [p.name for p in d.book.streets(loc)]
        elif region is not None and region.type == 'ГОРОД':
            local = [p.name for p in d.book.children(region.key) if p.level == 'street']
        if free and mats:
            st = street_choice(free, [[m] for m in mats], self._joiner, local, self.common_streets, 1)
            if st:
                f['street'] = st
            return
        # no street type word: a handwritten line «ПУШКИНА» - a street of the found locality, if it is one
        if not local:
            return
        for i in by.get('addr_line', []):
            if any(k != 'place' for k, _ in address_pieces(out[i]['raw'])):
                continue
            key = rules.name_key(out[i]['raw'], drop=rules.ALL_T).replace(' ', '')
            if not key or any(fuzz.ratio(key, rules._key(f.get(k, ''))) >= 75 for k in ('region', 'district',
                                                                                          'locality')):
                continue
            hit = process.extractOne(key, [rules._key(n) for n in local], scorer=fuzz.ratio)
            if hit and hit[1] >= 80:
                f['street'] = local[hit[2]]
                return

    # ---------------------------------------------------------------------------- boxes of fields_m
    def process_boxes(self, boxes: list[dict]) -> dict:
        """Boxes of the service's field network in, the same boxes with ``raw`` / ``text`` and the stamp's
        fields out. A box: ``{'label': one of NET_CLASSES, 'probs': [matrix of the plain crop, matrix of
        the crop with the date margin] (or one matrix for both), 'quad': 4 corners on the stamp (reading
        order and the overlap of date boxes), 'conf'}``. This is the stamp pipeline of the service
        (``service/ml/registration/stamp.py``)."""
        idx = self.abc.index
        out = [{k: v for k, v in b.items() if k != 'probs'} for b in boxes]
        pads = []
        for b in boxes:
            m = b['probs']
            ms = [as_matrix(x) if x is not None else None
                  for x in (m if isinstance(m, (list, tuple)) else [m, m])]
            pads.append(ms)
        for o, ms in zip(out, pads):
            o['raw'] = o['text'] = greedy(ms[0], self.abc) if ms[0] is not None else ''
        order = defaultdict(list)
        for i, b in enumerate(boxes):
            if pads[i][0] is not None:
                order[b['label']].append(i)

        def yx(i):
            q = np.asarray(boxes[i]['quad'], np.float64)
            return q[:, 1].mean(), q[:, 0].mean()
        for c, ii in order.items():
            if all('quad' in boxes[i] for i in ii):
                ii.sort(key=yx)
        reads = {c: [{'text': out[i]['raw']} for i in ii] for c, ii in order.items()}
        mats = {c: [pads[i] for i in ii] for c, ii in order.items()}
        d = self.dicts
        # date: deduped boxes, the date margin, one joined matrix
        date = None
        di = [i for i, b in enumerate(boxes) if b['label'] == 'date']
        if di and all('quad' in boxes[i] for i in di):
            qs = _dedupe([(i, np.asarray(boxes[i]['quad'], np.float64)) for i in di])
            qs.sort(key=lambda x: (np.mean(x[1][:, 1]), np.mean(x[1][:, 0])))
            di = [i for i, _ in qs]
        dm = [pads[i][1] for i in di if pads[i][1] is not None]
        d0 = [pads[i][0] for i in di if pads[i][0] is not None]
        if dm:
            src = {'margin': [join(dm, self.abc)], 'plain': [join(d0, self.abc)] if d0 else [join(dm, self.abc)],
                   'both': [join(dm, self.abc)] + ([join(d0, self.abc)] if d0 else [])}[self.date_crops]
            date = self._date(src, ' '.join(out[i]['raw'] for i in di))
        # numbers: every box of house / building / flat on its own
        nums = {}
        for c in ('house', 'building', 'flat'):
            vals = []
            for r, ms in zip(reads.get(c, []), mats.get(c, [])):
                v = best_number(ms, idx, rules.num_value(r['text']))
                vals.append(v[0] if v else None)
            nums[c] = vals
        # code: the reading and its FMS neighbours, scored on the first code box
        codes, read = None, ''
        for r in reads.get('code', []):
            m = re.search(r'([0-9ОЗБЧТ]{3})\s*[-–.]?\s*([0-9ОЗБЧТ]{2,3})', norm(r['text']))
            if m:
                read = f'{rules.fix_num(m.group(1))}-{rules.fix_num(m.group(2))}'
                break
        if read and mats.get('code'):
            codes = code_scores(read, mats['code'][0], idx, self._codes)
        f = rules.stamp_fields(reads, d, date, nums, codes)
        if self.place_ctc:
            self._place_ctc(f, [(out[i]['raw'], pads[i]) for i in order.get('place', [])])
        # street: dictionary names scored on the matrix, among the streets of the locality found
        texts = rules.dedupe_texts([r['text'] for r in reads.get('street', [])])
        free = ' '.join(t for t in rules.tokens(' '.join(texts)) if t not in rules.STREET_T)
        if free and mats.get('street'):
            pl = rules.place_fields(reads, rules.code_field(reads, d), d)
            loc, region = pl.get('_loc'), pl.get('_region')
            if self.street_after_place:
                loc, region = self._found_place(f, loc, region)
            local = []
            if loc is not None:
                local = [p.name for p in d.book.streets(loc)]
            elif region is not None and region.type == 'ГОРОД':
                local = [p.name for p in d.book.children(region.key) if p.level == 'street']
            st = street_choice(free, mats['street'], self._joiner, local, self.common_streets, 2,
                               self.street_ml, self.street_mc, self.street_limit)
            if st:
                f['street'] = st
        stamp = {k: v for k, v in f.items() if k in STAMP_FIELDS and v}
        # back to the boxes
        names = [stamp[k] for k in ('region', 'district', 'locality', 'street') if stamp.get(k)]
        for c, ii in order.items():
            for j, i in enumerate(ii):
                o = out[i]
                if c in ('place', 'street'):
                    o['text'] = _replace_words(o['raw'], names, pads[i][0], idx)
                elif c in nums and j < len(nums[c]) and nums[c][j]:
                    o['text'] = nums[c][j]
                elif c == 'code' and stamp.get('code'):
                    o['text'] = stamp['code']
        if date and len(di) == 1 and pads[di[0]][0] is not None:
            out[di[0]]['text'] = date_text(date, [pads[di[0]][0]], idx)
        for o in out:
            o['changed'] = o['text'] != o['raw']
        return {'fields': out, 'stamp': stamp}


def _dedupe(items: list, thr: float = 0.5) -> list:
    """Nearly coincident boxes of one field (the network puts two on one line) - one, or the date is
    read twice. Overlap is the intersection over the smaller area."""
    import cv2
    keep = []
    for i, q in items:
        a = np.float32(q)
        if all(cv2.intersectConvexConvex(a, np.float32(k))[0] / max(1e-6, min(cv2.contourArea(a),
               cv2.contourArea(np.float32(k)))) < thr for _, k in keep):
            keep.append((i, q))
    return keep
