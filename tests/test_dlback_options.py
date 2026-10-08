"""OCR options for the driving-licence back side (DLBACK_ALL).

The classifier gains a class for the back of the Russian licence (the categories
table), both forms in one class like INTPASSPORTADDR_ALL. `make_options`
dispatches on substrings in order, and 'dlback' contains 'dl': without its own
branch the back side would get the front-side options and the pipeline would
look for name/birth-date fields on a categories table. There is no field model
for the back yet, so it gets the empty base options.
"""
import pytest

from document_processing.pipeline.pipeline import OCROptionsClass, OCROptionsDL


@pytest.mark.parametrize('doc_type', ['DLBACK', 'DLBACK_ALL'])
def test_back_side_does_not_get_front_side_options(doc_type):
    options = OCROptionsClass.make_options(doc_type)
    assert not isinstance(options, OCROptionsDL)
    assert type(options) is OCROptionsClass
    assert not options.ru_fields and not options.en_fields


@pytest.mark.parametrize('doc_type', ['DL', 'DL_2011', 'DL_2020'])
def test_front_side_still_reaches_dl_options(doc_type):
    assert isinstance(OCROptionsClass.make_options(doc_type), OCROptionsDL)


def test_the_suffix_splits_off_cleanly():
    assert 'DLBACK_ALL'.rsplit('_', maxsplit=1) == ['DLBACK', 'ALL']
