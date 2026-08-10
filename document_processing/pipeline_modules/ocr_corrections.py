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


def strip_edge_dots(name: str) -> str:
    """Strip stray leading dots the detector/OCR sometimes prepends to names."""
    return name.lstrip('.')
