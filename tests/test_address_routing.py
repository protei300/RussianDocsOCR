import pytest
from document_processing.pipeline.pipeline import Pipeline


class TestAddressDigitRouting:
    """Per-word OCR routing decision: an eng+nums output is treated as a number
    word (house/building/flat number) when it is digit-dominated."""

    @pytest.mark.parametrize("en_text,expected", [
        ("67", True),       # house number
        ("103", True),
        ("76", True),
        ("5", True),
        ("16", True),
        ("5A", True),       # house number with letter suffix (digits >= letters)
        ("1PH", False),     # eng+nums misread of Cyrillic ЧРН (letters outnumber digits)
        ("KVLIFBCKAA", False),  # eng+nums misread of Cyrillic word
        ("P-H", False),     # Р-Н, no digits
        (",CT", False),     # СТ, no digits
        ("", False),
        (None, False),
    ])
    def test_is_number_token(self, en_text, expected):
        assert Pipeline._is_number_token(en_text) is expected
