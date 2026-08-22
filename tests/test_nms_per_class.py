"""Per-class NMS thresholds (PerClassYOLODetectorPostprocessing).

A multi-line field is labeled one box per line, so its lines are boxes of the
SAME class and suppress each other. The numbers below are not invented: they
come from the shipped MRZ labeling - a line 34x its own height, lines 1.87
heights apart - rotated by the angles actually measured on the passport
canvases.
"""
import numpy as np

from document_processing.processing.postprocessing import PerClassYOLODetectorPostprocessing

LABELS = ['Face', 'MRZ', 'Last_name_ru']


def mrz_line_boxes(angle_deg, height=30.0, ratio=34.2, step_ratio=1.87):
    """Axis-aligned boxes of two adjacent MRZ lines on a document tilted by `angle`."""
    a = np.deg2rad(angle_deg)
    length = ratio * height
    w = length * np.cos(a) + height * np.sin(a)
    h = length * np.sin(a) + height * np.cos(a)
    step = step_ratio * height
    dx, dy = step * np.sin(a), step * np.cos(a)
    first = [0.0, 0.0, w, h]
    second = [dx, dy, dx + w, dy + h]
    return np.array([first, second], dtype=float)


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def run(post, boxes, cls_idx):
    conf = np.array([0.9, 0.8])
    cls_ids = np.array([cls_idx, cls_idx])
    return post.nms_indices(boxes, conf, cls_ids)


def test_tilt_makes_adjacent_lines_overlap_past_the_shared_threshold():
    """Sanity check on the fixture itself: 5 degrees really does exceed 0.2."""
    boxes = mrz_line_boxes(5.0)
    assert iou(boxes[0], boxes[1]) > 0.2
    assert iou(mrz_line_boxes(1.0)[0], mrz_line_boxes(1.0)[1]) < 0.2


def test_shared_threshold_drops_the_second_line_on_a_tilted_document():
    post = PerClassYOLODetectorPostprocessing(labels=LABELS, iou=0.2, cls=0.5)
    assert len(run(post, mrz_line_boxes(5.0), 1)) == 1


def test_per_class_override_keeps_both_lines():
    post = PerClassYOLODetectorPostprocessing(labels=LABELS, iou=0.2, cls=0.5,
                                              iou_per_class={'MRZ': 0.6})
    assert len(run(post, mrz_line_boxes(5.0), 1)) == 2


def test_override_does_not_leak_to_other_classes():
    """The shared 0.2 is what keeps the ru/en field pairs of an external
    passport from swallowing each other - it must stay untouched."""
    post = PerClassYOLODetectorPostprocessing(labels=LABELS, iou=0.2, cls=0.5,
                                              iou_per_class={'MRZ': 0.6})
    assert len(run(post, mrz_line_boxes(5.0), 2)) == 1


def test_no_override_configured_behaves_exactly_as_before():
    plain = PerClassYOLODetectorPostprocessing(labels=LABELS, iou=0.2, cls=0.5)
    empty = PerClassYOLODetectorPostprocessing(labels=LABELS, iou=0.2, cls=0.5,
                                               iou_per_class={})
    boxes = mrz_line_boxes(5.0)
    assert run(plain, boxes, 1) == run(empty, boxes, 1)


def test_unknown_class_index_falls_back_to_the_shared_threshold():
    post = PerClassYOLODetectorPostprocessing(labels=LABELS, iou=0.2, cls=0.5,
                                              iou_per_class={'MRZ': 0.6})
    assert post.iou_for(99) == 0.2
    assert post.iou_for(1) == 0.6


def test_upright_lines_are_kept_even_with_the_shared_threshold():
    """The fix must not be papering over a case that already worked."""
    post = PerClassYOLODetectorPostprocessing(labels=LABELS, iou=0.2, cls=0.5)
    assert len(run(post, mrz_line_boxes(0.0), 1)) == 2
