"""Special marks of the vehicle registration certificate (STS): words torn by a
line break, and the leasing record.

The special marks are printed by the registry's printer into a narrow area and
wrapped at its edge WITHOUT a hyphen, so a word can be torn in two: «ЛИЗИ» at
the end of one line, «НГА» at the start of the next. Measured on 140 real STS
backs (external set, 2026-10-07): 4 such tears on known words - «ЛИЗИ|НГА»,
«ЛИЗИН|ГОДАТЕЛЬ», «КОМ|ПАНИЯ», «АВТОЛ|ИЗИНГ». The position of the line end does
NOT tell a tear from a break between words (lines that tear end at 0.90-0.92 of
the document's widest line, lines that do not at 0.87-1.00), so the tear is
recognised by vocabulary: the two pieces are glued only when together they make
a known word and neither piece is a known word on its own. Unknown words (a
company name outside the list, a contract number) stay as read - a wrong glue
would merge two real words, which is worse than leaving a torn one.

Pure functions of strings, like dates.py.
"""
import re

from .dates import to_ddmmyyyy

#: Words that stand as they are (no ending needed) and stems that only occur with
#: an ending. Special-marks terms, the words seen on real backs (2026-10-07) and
#: lessor names (public leasing companies). A word is known when it is a WORD,
#: or a word or stem plus one of _ENDINGS: «ЛИЗИНГ» + «А», «РЕГИСТРАЦИ» + «И».
#: A bare stem is not a word - «РЕГИСТРАЦИ» at a line end is a torn piece.
WORDS = (
    'ЛИЗИНГОДАТЕЛЬ', 'ЛИЗИНГОПОЛУЧАТЕЛЬ', 'ЛИЗИНГ', 'ДОГОВОР', 'СОБСТВЕННИК',
    'ВЛАДЕЛЕЦ', 'ДУБЛИКАТ', 'ВЗАМЕН', 'УВЭОС', 'ВЫДАН', 'УЧЕТ', 'ТЕНТ',
    'ФУРГОН', 'РЕФРИЖЕРАТОР', 'ОБТЕКАТЕЛЬ', 'МОЩНОСТЬ', 'ДЕЙСТВИТЕЛЬНО',
    'ОБОРУДОВАНО', 'УСТАНОВЛЕНО', 'АДРЕС',
    'АВТОЛИЗИНГ', 'ИНТЕРЛИЗИНГ', 'РОСЛИЗИНГ', 'ЕВРОПЛАН', 'ГАЗПРОМБАНК',
    'СБЕРБАНК', 'СОВКОМБАНК', 'КАРКАДЕ', 'АЛЬФАМОБИЛЬ', 'МЭЙДЖОР', 'ЭЛЕМЕНТ',
)
STEMS = (
    'ЛИЗИНГОВ', 'СОБСТВЕННОСТ', 'ВЛАДЕЛЬЦ', 'ОБЩЕСТВ', 'ОГРАНИЧЕНН',
    'ОТВЕТСТВЕННОСТ', 'АКЦИОНЕРН', 'ПУБЛИЧН', 'КОМПАНИ', 'УТРАЧЕНН',
    'РЕГИСТРАЦИ', 'ВРЕМЕНН', 'ЗАМЕН', 'СМЕН', 'ИЗМЕНЕНИ', 'ПЛАТФОРМ', 'ВОРОТ',
    'ПОДРАЗДЕЛЕНИ', 'КОНСТРУКЦИ', 'БАЛТИЙСК',
)
#: Noun and adjective endings of the case forms the marks use.
_ENDINGS = ('А', 'Я', 'У', 'Ю', 'Е', 'И', 'Ы', 'О', 'Ь', 'ОМ', 'ЕМ', 'ЁМ', 'ОВ', 'ЕВ', 'АМ',
            'ЯМ', 'АХ', 'ЯХ', 'ОЙ', 'ЕЙ', 'ИЙ', 'ЫЙ', 'АЯ', 'ЯЯ', 'ОЕ', 'ЕЕ', 'ЫЕ', 'ИЕ',
            'ИЯ', 'ИИ', 'ИЮ', 'ЬЮ', 'ОЮ', 'ЕЮ', 'ОГО', 'ЕГО', 'ОМУ', 'ЕМУ', 'ЫМ', 'ИМ',
            'ЫХ', 'ИХ', 'АМИ', 'ЯМИ', 'ЫМИ', 'ИМИ')

_LETTERS = re.compile(r'[^А-ЯЁA-Z0-9]')


def _clean(word: str) -> str:
    return _LETTERS.sub('', (word or '').upper())


def _known(word: str) -> bool:
    """A vocabulary word as is, or a word or stem with a case ending."""
    if word in WORDS:
        return True
    return any(word.startswith(base) and word[len(base):] in _ENDINGS
               for base in WORDS + STEMS)


def glue_torn_words(lines):
    """Words of a multi-line field, line by line -> words with tears glued.

    ``lines`` is a list of lines, each a list of the words read on it, top to
    bottom. The last word of a line and the first of the next are glued when
    together they are a known word and neither alone is: «ЛИЗИ» + «НГА» ->
    «ЛИЗИНГА», while «ЛИЗИНГОДАТЕЛЬ» + «АО» stay two words.

    Deliberately NOT glued by the length of the line, although the registry does
    wrap by character count: on 140 real backs (2026-10-07) the rule «a full
    line goes on into the next one» glued about half of its cases wrongly
    («КРОНШ.» + «БАЗЫ», «КВТ» + «Л.С», a date line onto the line above) - the
    width differs between documents (lines of about 26 characters on some, 40
    and more on others) and the reading of real marks is often noisy.
    """
    out = []
    for line in lines:
        words = [w for w in line if w]
        if not words:
            continue
        if out:
            tail, head = _clean(out[-1]), _clean(words[0])
            if (tail and head and _known(tail + head)
                    and not _known(tail) and not _known(head)):
                out[-1] = out[-1] + words[0]
                words = words[1:]
        out.extend(words)
    return out


_NUMBER = re.compile(r'№\s*([0-9A-ZА-ЯЁ][0-9A-ZА-ЯЁ/\-]*)')
#: «ДОГ ЛИЗ АХ ЭЛ/УЛН-123/ДЛ», «ПО ДОГ ЛИЗИНГА 12/34-СКТ»: the number is the first
#: token with a digit after the abbreviated «договор лизинга», within three tokens.
_ABBR_NUMBER = re.compile(r'\bДОГ\w*\.?\s+(?:ЛИЗ\w*\.?\s+)?((?:\S+\s+){0,2}?\S*\d\S*)')
_DATE = re.compile(r'(\d{1,2}[.,]\d{1,2}[.,]\d{4})')
_SHORT_LEASE = re.compile(r'\bЛ\s*\.\s*Д\b')       # the 2010 edition: «<nn> <nnnn> Л.Д <дата>»
_LEASE_CONTRACT = re.compile(r'\bПО\s+ДЛ\b')        # «по договору лизинга»
_UNTIL = re.compile(r'\bЛИЗИНГ\w*\s+(?:ДЕЙСТВ\w*\s+)?ДО\W{0,2}' + _DATE.pattern)
#: What follows the lessor's name on real marks: the contract, the lessee,
#: the registration term, the validity, the next mark (engine power ...).
_LESSOR_END = re.compile(r'[.,;(]|№|\bДОГ\w*|\bПО\s+ДЛ\b|ЛИЗИНГОПОЛУЧАТЕЛЬ|\bВРЕМ\w*'
                         r'|\bДЕЙСТВ\w*|\bМОЩНОСТ\w*|\bСРОК\w*|\bДО\b')


def _normal(date_text):
    return to_ddmmyyyy(date_text.replace(',', '.')) if date_text else None


def parse_leasing(text: str):
    """The leasing record in the special marks, or None when there is none.

    Returns ``{'leasing': True, 'role', 'lessor', 'contract_number',
    'contract_date', 'contract_date_normalized', 'until', 'until_normalized'}``;
    a part that is not printed or not found is None. ``role`` is
    'lessor_named' when the marks name the lessor («ЛИЗИНГОДАТЕЛЬ ...») and
    'lessee' when they only say the owner is the lessee («ЛИЗИНГОПОЛУЧАТЕЛЬ»).
    ``until`` is the end of the leasing term («ЛИЗИНГ ДО 31.12.2027»). The
    lessee's own name is never taken out: it is the owner, already read.

    The text is the reading AFTER glue_torn_words: a torn «ЛИЗИ НГ» is not found.
    Never guesses: no leasing word -> None, and every part is taken only where
    the marks print it. Real marks are abbreviated freely and read noisily, so
    expect the flag far more often than the parts (140 real backs, 2026-10-07).
    """
    if not text:
        return None
    t = ' '.join(text.upper().replace('Ё', 'Е').split())
    if not ('ЛИЗИНГ' in t or _SHORT_LEASE.search(t) or _LEASE_CONTRACT.search(t)):
        return None

    lessor = None
    role = None
    if 'ЛИЗИНГОДАТЕЛЬ' in t:
        role = 'lessor_named'
        after = t.split('ЛИЗИНГОДАТЕЛЬ', 1)[1]
        lessor = _LESSOR_END.split(after, maxsplit=1)[0].strip(' :-') or None   # quotes stay
    elif 'ЛИЗИНГОПОЛУЧАТЕЛЬ' in t:
        role = 'lessee'

    number = _NUMBER.search(t) or _ABBR_NUMBER.search(t)
    date_text = None
    m = re.search(r'\bОТ\s+' + _DATE.pattern, t)
    if m:
        date_text = m.group(1)
    else:
        short = _SHORT_LEASE.search(t)
        if short:
            m = _DATE.search(t, short.end())
            date_text = m.group(1) if m else None
    until = _UNTIL.search(t)
    until_text = until.group(1) if until else None
    return {
        'leasing': True,
        'role': role,
        'lessor': lessor,
        'contract_number': number.group(1).strip() if number else None,
        'contract_date': date_text,
        'contract_date_normalized': _normal(date_text),
        'until': until_text,
        'until_normalized': _normal(until_text),
    }
