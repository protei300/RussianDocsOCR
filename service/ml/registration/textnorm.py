"""Text normalisation for registration stamps.

Project rule (user, 2026-10-22): text is read, compared and returned in CAPITALS only - no
lowercase anywhere, neither in the ground truth nor in network outputs.
"""
from __future__ import annotations

import re

# Latin letters that look like Cyrillic ones (networks and people mix them up).
LAT2CYR = str.maketrans({
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р',
    'T': 'Т', 'X': 'Х', 'Y': 'У', 'a': 'А', 'c': 'С', 'e': 'Е', 'o': 'О', 'p': 'Р', 'x': 'Х',
    'y': 'У', 'k': 'К', 'm': 'М', 'Ё': 'Е',
})
PUNCT = str.maketrans({'—': '-', '–': '-', '−': '-', '«': '"', '»': '"', '„': '"', '“': '"',
                       '”': '"', '’': "'", '`': "'", ' ': ' '})

# Look-alikes, fixed only by context (inside a number / inside a word).
DIGIT_LOOKALIKE = {'О': '0', 'З': '3', 'Б': '6', 'Ч': '4', 'I': '1', 'І': '1', 'Т': '7', 'S': '5'}
LETTER_LOOKALIKE = {'0': 'О', '3': 'З', '6': 'Б', '4': 'Ч', '1': 'І'}


def norm(text: str | None) -> str:
    """Capitals, Cyrillic instead of Latin look-alikes, one kind of each mark, single spaces."""
    if not text:
        return ''
    s = text.upper().translate(PUNCT).translate(LAT2CYR)
    return re.sub(r'\s+', ' ', s).strip()


def fix_digits_in_number(token: str) -> str:
    """In a token with at least as many digits as letters: look-alike letters -> digits."""
    digits = sum(ch.isdigit() for ch in token)
    letters = sum(ch.isalpha() for ch in token)
    if digits == 0 or digits < letters:
        return token
    return ''.join(DIGIT_LOOKALIKE.get(ch, ch) for ch in token)


# ----------------------------------------------------------------------------------- months

MONTHS = ['ЯНВАРЬ', 'ФЕВРАЛЬ', 'МАРТ', 'АПРЕЛЬ', 'МАЙ', 'ИЮНЬ', 'ИЮЛЬ', 'АВГУСТ', 'СЕНТЯБРЬ',
          'ОКТЯБРЬ', 'НОЯБРЬ', 'ДЕКАБРЬ']
MONTHS_GEN = ['ЯНВАРЯ', 'ФЕВРАЛЯ', 'МАРТА', 'АПРЕЛЯ', 'МАЯ', 'ИЮНЯ', 'ИЮЛЯ', 'АВГУСТА', 'СЕНТЯБРЯ',
              'ОКТЯБРЯ', 'НОЯБРЯ', 'ДЕКАБРЯ']
MONTH_FORMS: dict[str, int] = {}
for _i, (_a, _b) in enumerate(zip(MONTHS, MONTHS_GEN), 1):
    MONTH_FORMS[_a] = _i
    MONTH_FORMS[_b] = _i
# Month stems, found inside glued reads too («ФКМАЙРА» -> May).
MONTH_STEMS = [('ЯНВ', 1), ('ФЕВ', 2), ('МАРТ', 3), ('АПР', 4), ('МАЯ', 5), ('МАЙ', 5), ('ИЮН', 6),
               ('ИЮЛ', 7), ('АВГ', 8), ('СЕНТ', 9), ('ОКТ', 10), ('НОЯБ', 11), ('ДЕК', 12)]

# ----------------------------------------------------------------------------------- patterns

RE_DATE_NUM = re.compile(r'(?<!\d)(?P<d>\d{1,2})\s*[./,\- ]\s*(?P<m>\d{1,2})\s*[./,\- ]\s*'
                         r'(?P<y>\d{4}|\d{2})(?!\d)')
RE_DATE_WORD = re.compile(r'(?<!\d)(?P<d>[0-9ОЗ]{1,2})[\s"\'.,]*(?P<m>[А-Я]{3,9})\.?\s*'
                          r'(?P<y>[0-9ОЗ]{4}|[0-9ОЗ]{2})(?![0-9])')
# A year; may be followed by a stray digit of a glued read («20123»). Match on digit-fixed text.
RE_YEAR = re.compile(r'(?<!\d)(19[89]\d|20[0-4]\d)')
RE_YEAR_ONLY = re.compile(r'^(19[89]\d|20[0-4]\d)\s*Г?\.?$')


def digits_fixed(text: str) -> str:
    """Every token that is mostly digits gets its look-alike letters turned into digits
    («200Б Г.» -> «2006 Г.»); words are left alone."""
    return ' '.join(fix_digits_in_number(t) for t in text.split(' '))
RE_CODE = re.compile(r'(?<![0-9])(?P<a>[0-9ОЗ]{3})\s*[-–—.]?\s*(?P<b>[0-9ОЗ]{3})(?![0-9])')
RE_HOUSE = re.compile(r'^\d{1,4}(\s*[-/]?\s*[А-Я]|\s*/\s*\d{1,4}[А-Я]?)?$')
RE_FLAT = re.compile(r'^\d{1,4}[А-Я]?$')

MIN_YEAR = 1997          # the current passport form
