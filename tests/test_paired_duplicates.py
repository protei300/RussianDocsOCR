"""A line the field detector labels as BOTH the Russian and the English field.

TextFields NMS runs per class, so a <name>_ru box and a <name>_en box never
suppress each other - on purpose, because real ru/en pairs overlap at 0.2-0.3
(external passport) and up to 0.5 (driving licence). A retrained detector put
Birth_place_ru (0.92) and Birth_place_en (0.62) on the same box of an external
passport (2026-09-26); the pipeline keeps the more confident one.
"""
from document_processing.pipeline.pipeline import Pipeline

drop = Pipeline._paired_duplicate_indices


def test_same_line_labeled_both_ways_keeps_the_confident_label():
    boxes = [[513, 384, 801, 423, 0.922, 11, 'Birth_place_ru'],
             [513, 384, 801, 422, 0.619, 12, 'Birth_place_en']]
    assert drop(boxes) == {1}


def test_the_english_label_wins_when_it_is_the_confident_one():
    boxes = [[100, 10, 300, 40, 0.55, 11, 'Birth_place_ru'],
             [100, 10, 300, 40, 0.90, 12, 'Birth_place_en']]
    assert drop(boxes) == {0}


def test_a_real_ru_en_pair_side_by_side_is_kept():
    # «Г. ЧЕЛЯБИНСК / USSR»: two boxes on one line, not overlapping
    boxes = [[407, 291, 571, 317, 0.95, 11, 'Birth_place_ru'],
             [582, 289, 652, 316, 0.93, 12, 'Birth_place_en']]
    assert drop(boxes) == set()


def test_stacked_lines_overlapping_like_a_licence_are_kept():
    # ru above en, overlapping about 0.4 - the driving-licence layout
    boxes = [[100, 100, 400, 140, 0.9, 1, 'Last_name_ru'],
             [100, 116, 400, 156, 0.9, 3, 'Last_name_en']]
    assert drop(boxes) == set()


def test_different_fields_on_the_same_box_are_not_this_rule():
    boxes = [[100, 10, 300, 40, 0.9, 11, 'Birth_place_ru'],
             [100, 10, 300, 40, 0.6, 2, 'Last_name_en']]
    assert drop(boxes) == set()
