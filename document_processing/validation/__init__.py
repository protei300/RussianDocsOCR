"""Field validation over OCR results (see docs/validation-checks.md).

Recognition-quality checks, not authenticity checks: a failing hard check means
the text was almost certainly misread.
"""
from .fields import field_checks
from .mrz import check_digit, cross_check, parse_mrz, validate, validate_mrz

__all__ = ['check_digit', 'cross_check', 'field_checks', 'parse_mrz', 'validate',
           'validate_mrz']
