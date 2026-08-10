import pytest
from document_processing.pipeline.pipeline import Pipeline

UNIQUE = ('Licence_number', 'Issue_organisation_code')


def bbox(conf, label):
    # detector bbox layout: [x1, y1, x2, y2, conf, cls, label]
    return [0, 0, 1, 1, conf, 0, label]


class TestDuplicateFieldDedup:
    def test_drops_lower_confidence_duplicate(self):
        bboxes = [bbox(0.6, 'Licence_number'), bbox(0.9, 'Licence_number')]
        drop = Pipeline._duplicate_field_indices(bboxes, UNIQUE)
        assert drop == {0}, 'Should drop the lower-confidence Licence_number'

    def test_single_field_kept(self):
        bboxes = [bbox(0.6, 'Licence_number'), bbox(0.9, 'First_name_ru')]
        assert Pipeline._duplicate_field_indices(bboxes, UNIQUE) == set()

    def test_dedups_each_unique_field(self):
        bboxes = [
            bbox(0.5, 'Licence_number'),         # 0 -> drop
            bbox(0.8, 'Licence_number'),         # 1 keep
            bbox(0.7, 'Issue_organisation_code'),# 2 keep
            bbox(0.4, 'Issue_organisation_code'),# 3 -> drop
            bbox(0.9, 'Birth_date'),             # 4 keep (not unique-listed)
        ]
        assert Pipeline._duplicate_field_indices(bboxes, UNIQUE) == {0, 3}

    def test_empty(self):
        assert Pipeline._duplicate_field_indices([], UNIQUE) == set()
