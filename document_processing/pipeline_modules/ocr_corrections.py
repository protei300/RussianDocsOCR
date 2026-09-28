"""Field-level text corrections shared by the OCR engines.

These are the semantic post-OCR fixes (date formatting, sex normalization,
driver-class filtering, stray-dot stripping) applied on top of the raw decoded
string - the single source used by OCRLatin/OCRCyrillic.
"""
from datetime import datetime


def check_ddmmyyyy(date: str) -> str:
    """Normalize a recognized date to ``dd.mm.yyyy`` (best-effort)."""
    date = date.replace('O', '0').replace('-', '.')
    pure_nums = ''.join(c for c in date if c.isnumeric())
    if len(pure_nums) == 8:
        return datetime.strptime(pure_nums, '%d%m%Y').strftime('%d.%m.%Y')
    return date


def strip_day_quote(word: str) -> str:
    """'28,' -> '28': a day number followed by one comma or dot is the closing quote.

    The 1998 birth certificate prints «28» ИЮЛЯ 2010 г.; a field box with a
    margin around the letters (2026-09-27) takes in the closing quote », which is
    not in the Cyrillic alphabet, and the engine reads it as a comma or a dot
    right after the day - '28, ИЮЛЯ 2010'. The canonical date was right; the
    reading was not. Only a WORD that is one or two digits plus that single mark
    is touched, so a digit date ('22.06.2010', one word) and every other word
    pass through.
    """
    if len(word) in (2, 3) and word[-1] in ',.' and word[:-1].isdigit():
        return word[:-1]
    return word


def check_en_sex(sex: str) -> str:
    """Standardize Latin sex text to 'M' or 'F'."""
    to_check = sex.lstrip('.').upper().replace('.', '')
    return 'M' if 'M' in to_check else 'F'


def check_rus_sex(sex: str) -> str:
    """Standardize Cyrillic sex text to 'М' or 'Ж'."""
    to_check = sex.lstrip('.').upper().replace('.', '')
    return 'М' if 'М' in to_check else 'Ж'


def check_driver_class(driver_class: str) -> str:
    """Keep only valid driver-category characters."""
    allowed = set('ABCDEM1')
    return ''.join(c for c in driver_class.replace(' ', '') if c in allowed)


def check_vin(vin: str) -> str:
    """Replace the letter O with the digit 0 in a VIN.

    ISO 3779 excludes the letters I, O and Q from a VIN, so an O here is always
    a misread 0 - the Latin engine confuses the two on real STS photos (issue
    #17: a VIN starting 'WF0DX' came out as 'WFODX'). Only O is mapped,
    by decision: I and Q have no single safe digit (I could be 1 or a stray
    stroke), and nothing else is touched.
    """
    return vin.replace('O', '0')


def strip_edge_dots(name: str) -> str:
    """Strip stray leading dots the detector/OCR sometimes prepends to names."""
    return name.lstrip('.')
