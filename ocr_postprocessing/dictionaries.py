"""Dictionaries of step 9: addresses (KLADR/FIAS in the AddressRU layout) and department codes.

AddressRU: a file is named by an object's full code and lists its children
``[code fragment, name, type code, (postcode)]``; a child's full code = file name + fragment, and
the child has its own file when it has children. ``00.json`` holds the regions,
``socrbase.json`` the type names. A branch is loaded by code on demand - the ~98 000 files are
never read in full. Federal cities (Moscow 77, Saint Petersburg 78, Sevastopol 92) are regions
whose streets sit directly in the region file.

Department codes: ``code,name`` (hflabs/fms-unit). The first two digits of a code are the code of
the subject of the federation (770-063 -> 77, Moscow).

Everything returned is in capitals.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .textnorm import norm

try:                                    # faster; may be missing from an environment
    from rapidfuzz import fuzz, process
except ImportError:                     # standard-library fallback
    fuzz = process = None
    import difflib

# Full type name (socrbase.json, capitals) -> how stamps abbreviate it.
TYPE_ABBR: dict[str, list[str]] = {
    'РЕСПУБЛИКА': ['РЕСП', 'РЕСПУБЛИКА'], 'КРАЙ': ['КРАЙ', 'КР'], 'ОБЛАСТЬ': ['ОБЛ', 'ОБЛАСТЬ'],
    'АВТОНОМНЫЙ ОКРУГ': ['АО'], 'АВТОНОМНАЯ ОБЛАСТЬ': ['АОБЛ'],
    'ГОРОД': ['Г', 'ГОР', 'ГОРОД'], 'РАЙОН': ['Р-Н', 'РАЙОН'],
    'ПОСЕЛОК': ['П', 'ПОС'], 'ПОСЕЛОК ГОРОДСКОГО ТИПА': ['ПГТ'], 'РАБОЧИЙ ПОСЕЛОК': ['РП'],
    'СЕЛО': ['С', 'СЕЛО'], 'ДЕРЕВНЯ': ['Д', 'ДЕР'], 'СТАНИЦА': ['СТ-ЦА', 'СТ'], 'ХУТОР': ['Х'],
    'АУЛ': ['АУЛ'], 'УЛИЦА': ['УЛ'], 'ПРОСПЕКТ': ['ПР-КТ', 'ПР-Т', 'ПРОСП'], 'ПЕРЕУЛОК': ['ПЕР'],
    'ПЛОЩАДЬ': ['ПЛ'], 'ПРОЕЗД': ['ПР-Д'], 'ШОССЕ': ['Ш'], 'БУЛЬВАР': ['Б-Р'], 'НАБЕРЕЖНАЯ': ['НАБ'],
    'ТУПИК': ['ТУП'], 'МИКРОРАЙОН': ['МКР'], 'КВАРТАЛ': ['КВ-Л'], 'ТЕРРИТОРИЯ': ['ТЕР'],
    'АЛЛЕЯ': ['АЛ'], 'ЛИНИЯ': ['ЛИН'], 'СПУСК': ['СП'],
}


@dataclass(frozen=True)
class Place:
    key: str          # full code (name of its children file)
    name: str         # capitals
    type: str         # full type name in capitals, '' if unknown
    level: str        # 'region' | 'area' | 'locality' | 'street'

    @property
    def abbr(self) -> str:
        return (TYPE_ABBR.get(self.type) or [self.type])[0]


def best(query: str, places, limit: int = 10, cutoff: float = 60.0):
    """Fuzzy search by name: [(Place, score 0..100)], best first."""
    q = norm(query)
    if not q or not places:
        return []
    names = [p.name for p in places]
    if process is not None:
        res = process.extract(q, names, scorer=fuzz.WRatio, limit=limit, score_cutoff=cutoff)
        return [(places[i], float(s)) for _, s, i in res]
    scored = []
    for i, name in enumerate(names):
        sm = difflib.SequenceMatcher(None, q, name, autojunk=False)
        if sm.real_quick_ratio() * 100 < cutoff or sm.quick_ratio() * 100 < cutoff:
            continue
        r = sm.ratio() * 100
        if r >= cutoff:
            scored.append((places[i], r))
    scored.sort(key=lambda x: -x[1])
    return scored[:limit]


class AddressBook:
    """Lazy view of the AddressRU tree. ``AddressBook(None)`` is an empty book (no dictionary
    installed): every lookup finds nothing and step 9 leaves addresses as read."""

    def __init__(self, root: Path | None):
        """``root`` is the folder of AddressRU files or a zip of them (flat, as in the repository:
        one archive instead of ~98 000 small files)."""
        self.root = Path(root) if root else None
        self._kids: dict[str, tuple[Place, ...]] = {}
        self._locs: dict[str, tuple[Place, ...]] = {}
        self._socr: dict[int, str] | None = None
        self._zip: zipfile.ZipFile | None = None
        self._zip_names: set[str] = set()
        self._zip_lock = threading.Lock()     # ZipFile reads are not safe from several threads
        if self.root is not None and self.root.suffix.lower() == '.zip' and self.root.is_file():
            # the whole archive in memory (27 MB): a branch is parsed from RAM, never from the disk
            self._zip = zipfile.ZipFile(io.BytesIO(self.root.read_bytes()))
            self._zip_names = set(self._zip.namelist())
        self.loc_keys: dict[str, list[str]] = {}    # region -> search keys of its localities (``preload``)

    def _has(self, name: str) -> bool:
        if self._zip is not None:
            return name in self._zip_names
        return bool(self.root) and (self.root / name).exists()

    def _load(self, name: str):
        if self._zip is not None:
            with self._zip_lock:
                return json.loads(self._zip.read(name).decode('utf-8'))
        return json.loads((self.root / name).read_text('utf-8'))

    @property
    def available(self) -> bool:
        return self._has('00.json')

    def _socr_names(self) -> dict[int, str]:
        if self._socr is None:
            raw = self._load('socrbase.json') if self.available else {}
            self._socr = {int(k): norm(v) for k, v in raw.items()}
        return self._socr

    def _level(self, type_code: int, is_region: bool) -> str:
        if is_region:
            return 'region'
        if 500 <= type_code < 600:
            return 'street'
        if type_code == 201 or self._socr_names().get(type_code, '') in ('РАЙОН', 'УЛУС'):
            return 'area'
        return 'locality'

    def children(self, key: str) -> tuple[Place, ...]:
        if key in self._kids:
            return self._kids[key]
        out: tuple[Place, ...] = ()
        if self._has(f'{key}.json'):
            names = self._socr_names()
            rows = self._load(f'{key}.json')
            out = tuple(Place(r[0] if key == '00' else key + r[0], norm(r[1]),
                              names.get(int(r[2]), ''), self._level(int(r[2]), key == '00'))
                        for r in rows)
        self._kids[key] = out
        return out

    def regions(self) -> tuple[Place, ...]:
        return self.children('00')

    def region(self, code2: str | None) -> Place | None:
        return next((r for r in self.regions() if r.key == code2), None) if code2 else None

    def localities(self, region: Place) -> tuple[Place, ...]:
        """Cities and settlements of a region (through its districts), no streets. A federal city
        is its own locality."""
        if region.key in self._locs:
            return self._locs[region.key]
        out, seen, stack = [], set(), [region.key]
        while stack:
            for p in self.children(stack.pop()):
                if p.level == 'street' or p.key in seen:
                    continue
                seen.add(p.key)
                if p.level == 'area':
                    stack.append(p.key)
                else:
                    out.append(p)
                    if any(c.level != 'street' for c in self.children(p.key)):
                        stack.append(p.key)          # settlements under a city
        if region.type == 'ГОРОД':
            out.append(Place(region.key, region.name, 'ГОРОД', 'locality'))
        self._locs[region.key] = tuple(out)
        return self._locs[region.key]

    def cities(self) -> tuple[Place, ...]:
        """Cities directly under a region, country-wide (86 small files): finds the region of a
        stamp that names only its city."""
        if '__cities__' not in self._locs:
            out = []
            for r in self.regions():
                if r.type == 'ГОРОД':
                    out.append(Place(r.key, r.name, 'ГОРОД', 'locality'))
                out += [p for p in self.children(r.key) if p.type == 'ГОРОД']
            self._locs['__cities__'] = tuple(out)
        return self._locs['__cities__']

    def region_in_text(self, text: str) -> Place | None:
        """A region named in free text, any case form («ПО ВЛАДИМИРСКОЙ ОБЛАСТИ» -> ВЛАДИМИРСКАЯ):
        a word sharing the first 6 letters with a region name. Only an unambiguous hit counts."""
        words = {w for w in re.findall(r'[А-Я]{6,}', norm(text))}
        hits = {r for r in self.regions() for w in words
                if len(r.name) >= 6 and w[:6] == r.name[:6]}
        return hits.pop() if len(hits) == 1 else None

    def preload(self, cache_dir: Path | None = None, key=None) -> float:
        """Every region with its districts and localities (268 000) and their search keys in memory, from
        an index built once from the archive and kept in ``cache_dir`` (rebuilt when the archive changes,
        by its SHA-256). ``key`` - the search-key function (``rules._key``). -> seconds spent. Streets stay
        lazy: one locality's streets are parsed from the in-memory archive in well under a millisecond."""
        import time
        t0 = time.time()
        if self.root is None or not self.available:
            return 0.0
        sha = hashlib.sha256(self.root.read_bytes()).hexdigest()[:16] if self.root.is_file() else 'dir'
        cache_dir = Path(cache_dir or os.environ.get('RDOCS_REFS_CACHE')
                         or Path.home() / '.cache' / 'ocr_postprocessing')
        f = cache_dir / f'addressru_index_{sha}.json.gz'
        if f.exists():
            data = json.loads(gzip.decompress(f.read_bytes()).decode('utf-8'))
        else:
            data = {}
            for r in self.regions():
                locs = self.localities(r)
                data[r.key] = {'kids': [[p.key, p.name, p.type, p.level] for p in self.children(r.key)
                                        if p.level != 'street'],
                               'locs': [[p.key, p.name, p.type, p.level] for p in locs],
                               'keys': [key(p.name) if key else p.name for p in locs]}
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix('.tmp')
            tmp.write_bytes(gzip.compress(json.dumps(data, ensure_ascii=False).encode('utf-8'), 6))
            tmp.replace(f)
        for rk, v in data.items():
            self._locs[rk] = tuple(Place(*x) for x in v['locs'])
            self.loc_keys[rk] = v['keys']
        self.cities()
        return time.time() - t0

    def streets(self, locality: Place) -> tuple[Place, ...]:
        return tuple(p for p in self.children(locality.key) if p.level == 'street')


class DepartmentCodes:
    """Department code 'ddd-ddd' -> names of the issuing unit (capitals)."""

    def __init__(self, csv_path: Path | None):
        self.path = Path(csv_path) if csv_path else None
        self._map: dict[str, list[str]] | None = None

    def _load(self) -> dict[str, list[str]]:
        if self._map is None:
            self._map = {}
            if self.path and self.path.exists():
                with open(self.path, encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        self._map.setdefault(row['code'].strip(), []).append(norm(row['name']))
        return self._map

    def __contains__(self, code: str) -> bool:
        return code in self._load()

    def names(self, code: str) -> list[str]:
        return self._load().get(code, [])

    @staticmethod
    def region_of(code: str | None) -> str | None:
        return code[:2] if code and len(code) >= 3 and code[:2].isdigit() else None
