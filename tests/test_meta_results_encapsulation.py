"""``PipelineResults`` must hand out its metadata as a copy, not as its own state.

Written BEFORE the fix and required to fail on ``715069f``: a fix without a
check that was red first repairs a guess, not a defect.

These tests deliberately never mention the private attribute that holds the
state. They ask only what a caller can observe -- write into what you were
given, then ask again -- so they test the contract rather than the way it
happens to be implemented today.

Nothing here loads a model or reads a sample: ``PipelineResults`` is built by
hand and filled with plain values. That is on purpose. A test that needs
material can go quiet when the material disappears, and this suite exists
precisely because a quiet check is indistinguishable from a passing one.
"""
import numpy as np
import pytest

from document_processing.pipeline.pipeline import PipelineResults


@pytest.fixture
def results():
    """A results object filled the way the pipeline fills it.

    The fixture writes through the private attribute on purpose: here it is
    playing the part of the pipeline, which is inside the module and is the only
    thing allowed to write. Every ASSERTION below stays on the public surface --
    the private name appears where state is produced, never where the contract
    is checked.

    Writing this fixture through the public property instead is not a style
    difference, it is the exact failure this task exists to prevent: the write
    lands in a discarded copy, the object stays empty, and the tests go red for
    a reason that has nothing to do with what they test. That happened here
    before the comment was written.
    """
    r = PipelineResults()
    r._meta_results.update({
        'DocType': 'DL_2011',
        'OCR': {'Surname': 'ИВАНОВ', 'Number': '1234567890'},
        'Quality': {'Glare': 'NO', 'Blur': 'NO'},
    })
    return r


class TestMetaResultsIsHandedOutAsACopy:
    def test_a_new_key_does_not_reach_the_state(self, results):
        """The classic leak: a caller adds a key and the pipeline now carries it."""
        handed_out = results.meta_results
        handed_out['Hacked'] = 'written from outside'

        assert 'Hacked' not in results.meta_results, (
            'meta_results handed out its own dict: a key added by a caller '
            'became part of the pipeline state'
        )

    def test_a_nested_write_does_not_reach_the_state(self, results):
        """A shallow copy would pass the test above and still fail this one."""
        handed_out = results.meta_results
        handed_out['Quality']['Glare'] = 'tampered'

        assert results.quality.get('Glare') == 'NO', (
            'a nested dict was shared: writing through it changed the quality '
            'verdict the pipeline reports'
        )

    def test_ocr_is_a_copy(self, results):
        ocr = results.ocr
        ocr['Surname'] = 'ПОДМЕНА'

        assert results.ocr['Surname'] == 'ИВАНОВ', 'ocr handed out live state'

    def test_quality_is_a_copy(self, results):
        quality = results.quality
        quality['Glare'] = 'tampered'

        assert results.quality['Glare'] == 'NO', 'quality handed out live state'

    def test_full_report_still_carries_the_values(self, results):
        """The other direction, and the reason it is here.

        Copying on the way out has a failure mode of its own: writes start
        landing in a discarded copy and every reader sees empty. That failure is
        SILENT -- it looks like clean data rather than lost data -- so the suite
        has to assert that the values still arrive, not only that they cannot be
        tampered with.
        """
        report = results.full_report

        assert report['DocType'] == 'DL_2011'
        assert report['OCR']['Surname'] == 'ИВАНОВ'
        assert report['Quality']['Glare'] == 'NO'


class TestTheBoundaryWeDoNotProtect:
    """Images are handed out live, and that is a decision, not an oversight.

    Callers ask for ``rotated_image`` to work with the array; copying a
    full-size image on every attribute access would cost far more than the
    protection is worth. The dictionary structure is protected, the pixel
    payload is not.

    This test is green before the fix and after it. It is not evidence that the
    defect existed -- it is a marker on the edge of the guarantee, so the
    boundary stays a named decision instead of quietly becoming a hole.
    """

    def test_rotated_image_is_the_stored_array_not_a_copy(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        r = PipelineResults()
        r._meta_results['Angle90'] = {'warped_img': image, 'angle': 0}

        assert r.rotated_image is image, (
            'rotated_image started copying. That may be an improvement, but it '
            'is a change of contract and of cost - decide it deliberately'
        )
