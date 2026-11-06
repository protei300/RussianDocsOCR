"""Fields of a stamp from the readings of the field network's boxes: format rules and
dictionaries (decisions #180, #182, #183).

Input - the readings per network class (``date, place, street, house, building, flat,
code``), each ``{text, cx, cy, h, w}``. Output - the 10 fields of the stamp test. The
rules are general, never fitted to one stamp:

* **type words are dropped** («Г.», «ГОР.», «УЛ.», «ДОМ №», «КВ.», «КОРП.», «ОБЛ.»,
  «Р-Н», «РЕСП.» ...), but they decide what a piece of the place is: region, district
  or locality;
* **case** is reduced to a stem (the ending is cut): «УФЫ» and «УФА», «КУРСКА» and
  «КУРСК», «МОСКОВСКОЙ» and «МОСКОВСКАЯ» are one; the dictionary is searched by the
  stem and the answer is the dictionary's name;
* **dictionaries**: KLADR (AddressRU) - region -> districts and localities -> streets,
  linked by the KLADR code; FMS - department codes (the first two digits are the
  region); a region missing from the stamp comes from the code or the city;
* **numbers** (house, flat, building) by format; a digit counts its look-alike letter.

``canon(field, value)`` is the same common form for comparing with the ground truth,
which is written in several ways too («С.КРУТОГОРЬЕ» and «КРУТОГОРЬЕ»).
"""
from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz

from .dictionaries import AddressBook, DepartmentCodes, Place
from .textnorm import DIGIT_LOOKALIKE, norm

REGION_T = {'ОБЛ', 'ОБЛАСТЬ', 'ОБЛАСТИ', 'КРАЙ', 'КРАЯ', 'КРАЮ', 'КРАЕ', 'РЕСП', 'РЕСПУБЛИКА', 'РЕСПУБЛИКИ',
            'РЕСПУБЛИКЕ', 'АО', 'АВТОНОМНЫЙ', 'АВТОНОМНОГО', 'ОКРУГ', 'ОКРУГА', 'ОКРУГЕ'}
DISTRICT_T = {'Р-Н', 'Р-НА', 'Р-НЕ', 'РН', 'РАЙОН', 'РАЙОНА', 'РАЙОНЕ', 'РАЙОНУ', 'Р-НУ', 'УЛУС', 'УЛУСА'}
LOCALITY_T = {'Г', 'ГОР', 'ГОРОД', 'ГОРОДА', 'ГОРОДЕ', 'С', 'СЕЛО', 'СЕЛА', 'СЕЛЕ', 'П', 'ПОС', 'ПОСЕЛОК',
              'ПОСЕЛКА', 'ПГТ', 'РП', 'Д', 'ДЕР', 'ДЕРЕВНЯ', 'ДЕРЕВНИ', 'СТ', 'СТ-ЦА', 'СТАНИЦА', 'СТАНИЦЫ',
              'Х', 'ХУТОР', 'ХУТОРА', 'АУЛ', 'АУЛА', 'НП'}
STREET_T = {'УЛ', 'УЛИЦА', 'УЛИЦЫ', 'УЛИЦЕ', 'ПЕР', 'ПЕРЕУЛОК', 'ПЕРЕУЛКА', 'ПР', 'ПР-Т', 'ПР-КТ', 'ПРОСП',
            'ПРОСПЕКТ', 'ПРОСПЕКТА', 'ПЛ', 'ПЛОЩАДЬ', 'ПР-Д', 'ПРОЕЗД', 'ПРОЕЗДА', 'Ш', 'ШОССЕ', 'Б-Р',
            'БУЛЬВАР', 'НАБ', 'НАБЕРЕЖНАЯ', 'МКР', 'МИКРОРАЙОН', 'КВ-Л', 'КВАРТАЛ', 'ТУП', 'ТУПИК', 'ТЕР',
            'СНТ', 'ДНТ', 'С/Т', 'Т', 'ПРОЕЗ'}
NUM_T = {'ДОМ', 'Д', 'ДМ', 'ОМ', 'КВ', 'КВАРТ', 'КВАРТИРА', 'КОРП', 'КОРПУС', 'КОР', 'К', 'СТР', 'СТРОЕНИЕ', 'Н'}
FILLER = {'В', 'ПО', 'НА'}
ALL_T = REGION_T | DISTRICT_T | LOCALITY_T | STREET_T | FILLER

#: case endings, longest first; a stem shorter than three letters is not cut
ENDINGS = sorted(['ОГО', 'ЕГО', 'ОМУ', 'ЕМУ', 'ЫМИ', 'ИМИ', 'ОЙ', 'ЕЙ', 'ОМ', 'ЕМ', 'ЫМ', 'ИМ', 'УЮ', 'ЮЮ', 'АЯ',
                  'ЯЯ', 'ИЙ', 'ЫЙ', 'ОЕ', 'ЕЕ', 'ЫЕ', 'ИЕ', 'ЫХ', 'ИХ', 'АХ', 'ЯХ', 'АМ', 'ЯМ', 'А', 'Я', 'Ы',
                  'И', 'Е', 'У', 'Ю', 'Ь', 'О', 'Й'], key=len, reverse=True)


def tokens(text: str) -> list[str]:
    s = norm(text).replace('Ё', 'Е')
    s = re.sub(r'["«»()]', ' ', s)
    s = re.sub(r'(?<=[А-Я])\.(?=[А-Я])', '. ', s)               # «Г.САМАРА» -> «Г. САМАРА»
    return [t.strip('.,-:;') for t in re.split(r'[\s,.;:]+', s) if t.strip('.,-:;')]


def stem(w: str) -> str:
    """Stem: the ending is cut if 3 letters remain (2 for a one-letter ending: «УФЫ», «УФА»)."""
    for e in ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= (2 if len(e) == 1 else 3):
            return w[:-len(e)]
    return w


def name_key(text: str, drop=ALL_T) -> str:
    """A name without type words, word by word as stems: «Г. САМАРЫ» and «САМАРА» -> «САМАР»."""
    return ' '.join(stem(t) for t in re.split(r'[\s-]+', ' '.join(t for t in tokens(text) if t not in drop)) if t)


def fix_num(tok: str) -> str:
    return ''.join(DIGIT_LOOKALIKE.get(ch, ch) for ch in tok)


def num_value(text: str) -> str:
    """A house / flat / building number without type words: «15-А» -> «15А», «8 / 77» -> «8/77»."""
    toks = [t for t in tokens(text.replace('№', ' ')) if t not in NUM_T]
    s = ' '.join(toks)
    s = re.sub(r'(?<=\d)\s*[-]\s*(?=[А-Я])', '', s)
    s = re.sub(r'\s*/\s*', '/', s)
    m = None
    for t in s.split(' '):
        ft = fix_num(t) if sum(ch.isdigit() for ch in t) >= sum(ch.isalpha() for ch in t) else t
        m = re.match(r'\d+[А-Я]?(?:/\d+[А-Я]?)?', ft)
        if m:
            return m.group(0)
    return s.replace(' ', '')


def canon(field: str, value: str | None) -> str:
    """The common form of a field for comparison: numbers by format, names as stems without types, the code as digits."""
    v = str(value or '')
    if not v.strip():
        return ''
    if field in ('house', 'flat', 'building', 'structure'):
        return num_value(v)
    if field == 'code':
        return re.sub(r'\D', '', fix_num(norm(v)))
    if field == 'date':
        return v
    return name_key(v).replace(' ', '')


def canon_region(value: str | None, d: 'Dicts') -> str:
    """A region as its dictionary name (both sides of a comparison): «КБР» and
    «КАБАРДИНО-БАЛКАРСКАЯ РЕСПУБЛИКА», «СТАВРАПОЛЬСКИЙ КРАЙ» and «СТАВРОПОЛЬСКИЙ» are one."""
    v = str(value or '').strip()
    if not v:
        return ''
    raw = ' '.join(t for t in tokens(v) if t not in ALL_T)
    ab = REGION_ABBR.get(raw.replace(' ', ''))
    if ab and d.region_by_key2(ab):
        return _key(d.region_by_key2(ab).name)
    p, s = d.match(name_key(v).replace(' ', ''), d.regions, 88)
    return _key(p.name) if p else name_key(v).replace(' ', '')


# ----------------------------------------------------------------------------------------- dictionaries
class Dicts:
    def __init__(self, book: AddressBook, codes: DepartmentCodes):
        self.book, self.codes = book, codes
        self.regions = list(book.regions())

    @lru_cache(maxsize=None)  # noqa: B019 - one per process
    def areas(self, rkey: str) -> tuple[Place, ...]:
        return tuple(p for p in self.book.children(rkey) if p.level == 'area')

    @lru_cache(maxsize=None)  # noqa: B019
    def locs(self, rkey: str) -> tuple[Place, ...]:
        r = next(x for x in self.regions if x.key == rkey)
        return self.book.localities(r)

    def region_by_key2(self, code2: str | None) -> Place | None:
        return self.book.region(code2)

    @staticmethod
    def match(key: str, places, cutoff: float) -> tuple[Place | None, float]:
        """The best place for a name stem (``name_key``) and its score."""
        if not key or not places:
            return None, 0.0
        def score(name):
            k = _key(name)
            r = fuzz.ratio(key, k)
            if len(key) >= 7 and len(k) > len(key):
                r = max(r, 0.95 * fuzz.partial_ratio(key, k))   # part of a long name split across lines
            return r
        sc = sorted(((score(p.name), p) for p in places), key=lambda x: -x[0])
        s, p = sc[0]
        return (p, s) if s >= cutoff else (None, s)


@lru_cache(maxsize=200000)
def _key(name: str) -> str:
    return name_key(name, drop=set()).replace(' ', '')


def kind_of(text: str) -> str | None:
    ts = set(tokens(text))
    if ts & REGION_T:
        return 'region'
    if ts & DISTRICT_T or re.search(r'Р-?Н\b', norm(text)):
        return 'district'
    if ts & LOCALITY_T:
        return 'locality'
    return None


# ----------------------------------------------------------------------------------------- fields
def numbers(reads: dict, nums: dict | None = None) -> dict:
    """House, building, flat: by the type in the reading («ДОМ 7 КВ.2» in one box), else by the box
    class; a second «house» box with no flat of its own is the flat (decision #180). ``nums`` - the number
    of every box on the CTC matrix (``decode.best_number``): it replaces the reading of a one-number box."""
    out = {}
    for cls, field in (('house', 'house'), ('building', 'building'), ('flat', 'flat')):
        for i, r in enumerate(reads.get(cls, [])):
            t = norm(r['text'])
            parts = re.split(r'\b(ДОМ|Д|КВ|КОРП|КОРПУС|СТР)\b\.?', t)
            if len(parts) > 3:                             # several types in one box
                for i in range(1, len(parts) - 1, 2):
                    f = {'ДОМ': 'house', 'Д': 'house', 'КВ': 'flat', 'КОРП': 'building', 'КОРПУС': 'building',
                         'СТР': 'structure'}[parts[i]]
                    v = num_value(parts[i + 1])
                    if v and f not in out:
                        out[f] = v
                continue
            v = num_value(t)
            by_ctc = (nums or {}).get(cls) or []
            if i < len(by_ctc) and by_ctc[i]:
                v = by_ctc[i]
            if not v or not re.match(r'\d', v) and field != 'building':
                continue
            if field not in out:
                out[field] = v
            elif field == 'house' and 'flat' not in out and not reads.get('flat'):
                out['flat'] = v
    return out


def code_field(reads: dict, d: Dicts) -> str | None:
    texts = [r['text'] for r in reads.get('code', [])]
    for t in texts:
        m = re.search(r'([0-9ОЗБЧТ]{3})\s*[-–.]?\s*([0-9ОЗБЧТ]{3})', norm(t))
        if m:
            c = f'{fix_num(m.group(1))}-{fix_num(m.group(2))}'
            return c
    return None


#: common region abbreviations on stamps -> KLADR region code
REGION_ABBR = {'ЧР': '20', 'ЧЕЧР': '20', 'КБР': '07', 'КЧР': '09', 'РБ': '02', 'РД': '05', 'РТ': '16',
               'РСО': '15', 'РИ': '06', 'УР': '18', 'РК': '91', 'РМЭ': '12', 'РМ': '13', 'РХ': '19', 'РС': '14',
               'ХМАО': '86', 'ЯНАО': '89', 'ЛО': '47', 'МО': '50'}
TYPE_GLUE = sorted(['РЕСП', 'КРАЙ', 'ОБЛ', 'ГОР', 'Г', 'ПОС', 'С', 'П', 'СТ-ЦА', 'Р-Н'], key=len, reverse=True)
TYPE_TAIL = ('ОБЛАСТЬ', 'ОБЛАСТИ', 'ОБЛ', 'КРАЙ', 'Р-Н')
#: an adjective: «ОМСКОЙ», «РОСТОВСКАЯ», «ТИХВИНСКИЙ», «КИРОВСКОМ»
ADJ = re.compile(r'(СК|ЦК|Н|В|Ч)(АЯ|ОЙ|ИЙ|ЫЙ|ОМ|ОГО|ОМУ|ИЕ|ИИ|ИЯ|ОЕ|ЕЕ)$')


def segments(texts: list[str]) -> list[dict]:
    """All place boxes as one text -> pieces «type + name», split at commas and at type words. A type
    at the end of a line («...КРАЙ, Р-Н» / «ПРЕДГОРНЫЙ») goes to the next piece, a type after an adjective
    name («ОМСКОЙ ОБЛАСТИ», «С. МОСКОВСКАЯ ОБЛ.») to its own; a type glued to the name in front or behind
    («РЕСПЧЕЧЕНСКАЯ», «ГМОСКВА», «РОСТОВСКАЯОБЛ.») is split off, but the whole word is tried too
    («ГОРЛОВКА» is a city, not «ГОР. ЛОВКА»). A one-letter scrap that is not a type is dropped."""
    toks, whole = [], {}
    for t in texts:
        for part in norm(t).split(','):
            for w in tokens(part):
                tail = next((x for x in TYPE_TAIL if w.endswith(x) and len(w) - len(x) >= 5), None)
                g = next((x for x in TYPE_GLUE if w.startswith(x) and len(w) - len(x) >= 4 and w not in ALL_T
                          and not any(w.startswith(r[:5]) for r in ('ГОРНО', 'СТАВР', 'СЕВАС', 'ПЕТРО'))), None)
                if tail and w not in ALL_T:
                    toks += [w[:-len(tail)], tail]
                elif g and len(g) >= 3:
                    toks += [g, w[len(g):]]
                    whole[w[len(g):]] = w
                elif len(w) == 1 and w not in ALL_T and not w.isdigit():
                    continue
                else:
                    toks.append(w)
            toks.append(',')
    segs, cur = [], {'kind': None, 'words': []}

    def close():
        nonlocal cur
        if cur['words'] or cur['kind']:
            segs.append(cur)
        cur = {'kind': None, 'words': []}
    for w in toks:
        if w == ',':
            if cur['words']:
                close()
            continue
        k = ('region' if w in REGION_T else 'district' if w in DISTRICT_T else 'locality' if w in LOCALITY_T
             else None)
        if k:
            if cur['words'] and k in ('region', 'district') and (cur['kind'] is None
                                                                  or ADJ.search(cur['words'][-1])):
                cur['kind'] = k                          # a type after the name: «ОМСКОЙ ОБЛАСТИ»
                close()
            else:
                if cur['words']:
                    close()
                cur['kind'] = k
        elif w not in FILLER:
            cur['words'].append(w)
    close()
    for sg in segs:
        sg['raw'] = ' '.join(sg['words'])
        sg['key'] = name_key(sg['raw']).replace(' ', '')
        sg['keys'] = [sg['key']] + [name_key(whole[w]).replace(' ', '') for w in sg['words'] if w in whole]
        if len(sg['words']) > 1:                         # noise next to the name: «ТОР. САМАРЫ», «ОНД 1 СЛАВЯНСК»
            sg['keys'] += [k for k in (name_key(w).replace(' ', '') for w in sg['words']) if len(k) >= 4]
    return [sg for sg in segs if sg['key']]


def adj_nom(raw: str) -> str:
    """An adjective district in the nominative: «КИРОВСКОМ» -> «КИРОВСКИЙ», «ФРУНЗЕНСКИИ» -> «ФРУНЗЕНСКИЙ»."""
    return re.sub(r'(СК|ЦК)(ИИЙ|ИИ|ИЯ|ОМ|ОМУ|ОГО|ОЙ)\b', r'\1ИЙ', raw)


def _match(sg, pool, d):
    """The best place for a piece: by the whole name, by a glued word, and by single words (slightly worse)."""
    best = (None, 0.0)
    for k in sg['keys']:
        p, s = d.match(k, pool, 0)
        if k != sg['key']:
            s -= 3
        if p is not None and s > best[1]:
            best = (p, s)
    return best


def _center(words, pool):
    """A city from the adjective of a district or region: «ТИХВИНСКИЙ» -> ТИХВИН, «НОВОСИБИРСКОЙ» ->
    НОВОСИБИРСК, «АСТРАХАНСКОЙ» -> АСТРАХАНЬ (a stamp naming a city district and the region but not the
    city itself: the ground truth writes the centre)."""
    for w in words:
        m = re.match(r'(.+?)СК(АЯ|ОЙ|ИЙ|ЫЙ|ОМ|ОГО|ОМУ|ОЕ|ИЕ)$', w)
        if not m:
            continue
        b = m.group(1)
        names = {b, b + 'СК', b + 'Ь'}
        hit = [p for p in pool if p.name in names]
        if hit:
            return sorted(hit, key=lambda p: (p.type not in ('ГОРОД', 'Г'), len(p.name)))[0]
    return None


def place_fields(reads: dict, code: str | None, d: Dicts) -> dict:
    """Region, district, locality from the ``place`` boxes: pieces (``segments``), each scored in the
    dictionary as a region, a district and a locality; assigned by the best match, the piece's type
    weighs more. A region missing from the stamp comes from the department code or the found city. A
    federal city (Moscow, Saint Petersburg) is both region and locality; an untyped piece after it is a
    city district («Г. САНКТ-ПЕТЕРБУРГ НЕВСКИЙ»)."""
    segs = segments([r['text'] for r in reads.get('place', [])])
    out, region = {}, None
    feds = [x for x in d.regions if x.type == 'ГОРОД']
    # 0) a federal city - the region at once
    for sg in segs:
        if sg['kind'] in (None, 'locality'):
            p, sc = _match(sg, feds, d)
            if p is not None and sc >= 85:
                region, out['region'] = p, p.name
                sg['fed'] = True
                break
    # 1) region
    best = (0.0, None, None)
    for sg in segs if region is None else ():
        if sg['kind'] not in (None, 'region'):
            continue
        ab = REGION_ABBR.get(sg['raw'].replace(' ', ''))
        if ab and d.region_by_key2(ab):
            best = max(best, (100.0, id(sg), (sg, d.region_by_key2(ab))), key=lambda x: x[0])
            continue
        p, sc = d.match(sg['key'], d.regions, 0)
        need = 70 if sg['kind'] == 'region' else 85
        if p and sc >= need and sc > best[0]:
            best = (sc, id(sg), (sg, p))
    region_words = []
    if best[2]:
        sg, region = best[2]
        out['region'] = region.name
        region_words = sg['words']
        segs = [x for x in segs if x is not sg]
    elif region is None:
        typed = [x for x in segs if x['kind'] == 'region']
        if typed:
            out['region'] = typed[0]['raw']
            region_words = typed[0]['words']
            segs = [x for x in segs if x is not typed[0]]
    if region is None and code:
        region = d.region_by_key2(code[:2])
    # 2) locality: the best piece (typed as a locality or untyped); of several typed ones the last, the
    #    smallest («Г. ЖЕЛЕЗНОВОДСК, П. ИНОЗЕМЦЕВО», «Г. МОСКВА П. ВНУКОВСКОЕ»); the city above it - district
    pool = d.locs(region.key) if region else d.book.cities()
    cand = []
    for i, sg in enumerate(segs):
        if sg.get('fed') or sg['kind'] not in (None, 'locality') or (sg['kind'] is None and region is not None
                                                                      and region.type == 'ГОРОД'):
            continue                               # in Moscow and Saint Petersburg an untyped piece is a city district
        p, sc = _match(sg, pool, d)
        cand.append((sc + (10 if sg['kind'] == 'locality' else 0) + 0.01 * i, sg, p, sc))
    typed_locs = [c for c in cand if c[1]['kind'] == 'locality' and c[2] is not None and c[3] >= 75]
    typed_locs = list({c[2].key: c for c in typed_locs}.values())   # a repeated stamp line is one locality
    loc, upper, pick = None, None, None
    if len(typed_locs) >= 2:
        pick, upper = typed_locs[-1], typed_locs[-2]
    elif cand:
        pick = max(cand, key=lambda x: x[0])
    if pick:
        tot, sg, p, sc = pick
        if p is not None and sc >= 75:
            loc = p
            out['locality'] = p.name
            if region is None:
                region = d.region_by_key2(p.key[:2])
        elif sg['kind'] == 'locality' and not re.search(r'\d', sg['raw']):
            out['locality'] = sg['raw']            # untyped and not in the dictionary - not a locality (rather a district)
        if 'locality' in out:
            segs = [x for x in segs if x is not sg]
    if upper is not None:
        out['district'] = upper[2].name
        segs = [x for x in segs if x is not upper[1]]
    fed = any(x.get('fed') for x in segs)
    if fed and 'locality' not in out:
        out['locality'] = region.name
    segs = [x for x in segs if not x.get('fed')]
    # 3) district: a piece typed as a district, else the best remaining one among the region's districts;
    #    in a city - a remaining untyped piece: a city district, absent from the districts dictionary
    apool = d.areas(region.key) if region else ()
    dc = []
    for sg in segs if 'district' not in out else ():
        p, sc = _match(sg, apool, d)
        dc.append((sc + (20 if sg['kind'] == 'district' else 0), sg, p, sc))
    area = None
    if dc:
        tot, sg, p, sc = max(dc, key=lambda x: x[0])
        if p is not None and sc >= 85:
            out['district'] = p.name
            area = p
        elif sg['kind'] == 'district' or (sg['kind'] is None and not re.search(r'\d', sg['raw'])
                                          and len(sg['key']) >= 4):
            out['district'] = adj_nom(sg['raw'])
    # 4) no locality - the centre by the adjective of the district (found in the dictionary) or region
    if 'locality' not in out and region is not None:
        dwords = [w for x in segs if x['kind'] == 'district' for w in x['words']]
        for ws in ([dwords, region_words] if area is not None else [region_words, dwords]):
            c = _center(ws, pool)
            if c is not None:
                out['locality'], loc = c.name, c
                out['_weak'] = True                  # inferred, not read: the chain may replace it
                break
    if 'region' not in out and region:
        out['region'] = region.name
    out['_region'], out['_loc'] = region, loc
    return out


def dedupe_texts(texts: list[str], thr: float = 75) -> list[str]:
    """Two boxes on one line read almost the same - keep one (the first)."""
    out = []
    for t in texts:
        k = name_key(t).replace(' ', '')
        if k and all(fuzz.ratio(k, name_key(o).replace(' ', '')) < thr for o in out):
            out.append(t)
    return out


def street_field(reads: dict, place: dict, d: Dicts) -> str | None:
    texts = dedupe_texts([r['text'] for r in reads.get('street', [])])
    if not texts:
        return None
    toks = [t for t in tokens(' '.join(texts)) if t not in STREET_T]
    if len(toks) > 1 and len(toks[0]) == 1 and not toks[0].isdigit():
        toks = toks[1:]                                    # a scrap of a type: «У. ГОРЬКОГО», «П. НАГАТЫХ»
    raw = ' '.join(toks)
    loc, region = place.get('_loc'), place.get('_region')
    pool = ()
    if loc is not None:
        pool = d.book.streets(loc)
    elif region is not None and region.type == 'ГОРОД':
        pool = tuple(p for p in d.book.children(region.key) if p.level == 'street')
    p, s = d.match(name_key(raw, drop=STREET_T).replace(' ', ''), pool, 72)
    return p.name if p else raw


# ----------------------------------------------------------------------------------------- the chain
@lru_cache(maxsize=None)
def _loc_keys(d: Dicts, rkey: str):
    locs = d.locs(rkey)
    return locs, [_key(p.name) for p in locs]


def parents(p: Place, d: Dicts) -> tuple[Place | None, Place | None]:
    """The district and the city above a locality, by its KLADR code: SS RRR GGG PPP (region, district, city, settlement)."""
    area = city = None
    if p.key[2:5] != '000':
        area = next((a for a in d.areas(p.key[:2]) if a.key == p.key[:5]), None)
    if len(p.key) == 11 and p.key[5:8] != '000':
        city = next((c for c in d.book.children(p.key[:5] if area else p.key[:2]) if c.key == p.key[:8]), None)
    return area, city


def _fms_text(code: str | None, d: Dicts) -> str:
    return ' '.join(d.codes.names(code)) if code and code in d.codes else ''


def chain(segs: list[dict], code: str | None, d: Dicts, gamma_reg: float = 1.0, fixed: str | None = None,
          min_reg: float = 60.0) -> tuple | None:
    """Region, locality and district together, as the dictionary links them: the locality is searched
    among the region's localities, the district comes from the locality's code (SS RRR ...) and is checked
    against a district piece if there is one; the region is backed by the department code (its first
    digits) and the FMS unit name («... В Г. МАЙКОПЕ»). -> (score, region, locality, district-or-city
    above it, locality score) of the best link, or None. ``fixed`` - a region already confirmed (written
    and found in the dictionary): the locality is searched only there; otherwise a region with less than
    ``min_reg`` evidence is never chosen by the chain (a locality looks like something in any region)."""
    from rapidfuzz import process
    reg = {}
    for sg in segs:
        if sg['kind'] not in (None, 'region'):
            continue
        ab = REGION_ABBR.get(sg['raw'].replace(' ', ''))
        if ab:
            reg[ab] = max(reg.get(ab, 0), 100.0)
        for p, s, _ in process.extract(sg['key'], [_key(r.name) for r in d.regions], scorer=fuzz.ratio, limit=3):
            r = d.regions[_]
            s = s if sg['kind'] == 'region' else s - 10
            reg[r.key] = max(reg.get(r.key, 0), s)
    fms = _fms_text(code, d)
    if code and d.region_by_key2(code[:2]):
        reg[code[:2]] = reg.get(code[:2], 0) + (40 if fms else 15)
    rcands = [fixed] if fixed else [k for k in sorted(reg, key=lambda k: -reg[k])[:4] if reg[k] >= min_reg]
    reg.setdefault(fixed, 0.0)
    fms_keys = {stem(w) for w in tokens(fms) if len(w) >= 4}
    best = None
    for rk in rcands:
        region = d.region_by_key2(rk)
        if region is None or region.type == 'ГОРОД':
            continue
        locs, keys = _loc_keys(d, rk)
        rkey = _key(region.name)
        for sg in segs:
            if sg['kind'] not in (None, 'locality') or fuzz.ratio(sg['key'], rkey) >= 80:
                continue                           # the region piece («КРАСНОДАРСКИЙ») is not a locality
            for k in sg['keys']:
                for _, sl, i in process.extract(k, keys, scorer=fuzz.ratio, limit=5):
                    if sl < 70:
                        continue
                    L = locs[i]
                    area, city = parents(L, d)
                    up = city or area
                    sa = 0.0
                    for o in segs:
                        if o is sg or up is None:
                            continue
                        sa = max(sa, fuzz.ratio(o['key'], _key(up.name)) + (10 if o['kind'] == 'district' else 0))
                    bonus = 15 if stem(L.name) in fms_keys or (up and stem(up.name) in fms_keys) else 0
                    tot = gamma_reg * reg[rk] + sl + (10 if sg['kind'] == 'locality' else 0) \
                        + (0.5 * sa if sa >= 75 else 0) + bonus
                    if best is None or tot > best[0]:
                        best = (tot, region, L, up, sl)
    return best


def code_by_place(cs: dict | None, region: Place | None, loc: Place | None, area: Place | None,
                  d: Dicts, margin: float = 2.0) -> str | None:
    """A code among the reading's neighbours (``decode.code_scores``): of the place's region and with a
    unit name that mentions the found locality or district; chosen on the matrix, the reading itself if
    it is at most ``margin`` worse than the best code that agrees with the place."""
    if not cs:
        return None
    read, sc = cs['read'], cs['scores']
    keys = {stem(x.name) for x in (loc, area) if x is not None}
    def fits(c):
        ws = {stem(w) for w in tokens(_fms_text(c, d)) if len(w) >= 4}
        return (region is None or c[:2] == region.key[:2]) and (not keys or bool(ws & keys))
    ok = [(s, c) for c, s in sc.items() if c in d.codes and fits(c)]
    if not ok:
        return None
    s, c = max(ok)
    rs = sc.get(read, -1e30)
    if read in d.codes and fits(read):
        return read
    return c if s >= rs - margin else None


def stamp_fields(reads: dict, d: Dicts, date: str | None = None, nums: dict | None = None,
                 codes: dict | None = None, use_chain: bool = True) -> dict:
    out = dict(numbers(reads, nums))
    code = code_field(reads, d)
    pl = place_fields(reads, code, d)
    if use_chain and not (pl.get('_region') is not None and pl['_region'].type == 'ГОРОД'):
        segs = segments([r['text'] for r in reads.get('place', [])])
        cur, creg = pl.get('_loc'), pl.get('_region')
        # the region is confirmed if it is written and found in the dictionary, or agrees with the code
        sure = creg is not None and (pl.get('region') == creg.name) and (
            any(sg['kind'] == 'region' or sg['raw'].replace(' ', '') in REGION_ABBR for sg in segs)
            or (code is not None and code[:2] == creg.key))
        ch = chain(segs, code, d, fixed=creg.key if sure else None)
        if ch is not None and (cur is None or ch[2].key != cur.key):
            tot, region, L, up, sl = ch
            weak = cur is None or pl.get('_weak') or cur.key[:2] != region.key
            if weak and (sl >= 80 if cur is None else sl >= 90):
                pl['region'], pl['_region'] = region.name, region
                pl['locality'], pl['_loc'] = L.name, L
        if pl.get('_loc') is not None:
            area, city = parents(pl['_loc'], d)
            up = city or area
            if up is not None and not pl.get('district'):
                pl['district'] = up.name                  # the district (or city) above the locality, from the dictionary
    cfix = code_by_place((codes or {}), pl.get('_region'), pl.get('_loc'),
                         parents(pl['_loc'], d)[0] if pl.get('_loc') is not None else None, d)
    code = cfix or code
    if code:
        out['code'] = code
    out.update({k: v for k, v in pl.items() if not k.startswith('_') and v})
    st = street_field(reads, pl, d)
    if st:
        out['street'] = st
    if date:
        out['date'] = date
    return out
