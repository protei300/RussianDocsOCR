"""Regression tests for PipelineResults timing accounting.

Stages that run concurrently (quality checks + border detection, see
Pipeline._quality_and_borders_parallel) each report their own wall time. Summing
those into 'total' inflates it above the real processing time and keeps it flat
when parallelisation actually saves time - which also silently corrupted the
metric scripts/benchmark.py records. 'total' must therefore count a concurrent
group once, by the group's own elapsed time.

Paths are relative to tests/ (see conftest.py, which chdirs there).
"""
from pathlib import Path
from time import time

import pytest

from document_processing import Pipeline
from document_processing.pipeline.pipeline import PipelineResults

SAMPLES_DIR = Path('../samples')


class TestTimingAccounting:
    def test_sequential_stages_are_summed(self):
        r = PipelineResults()
        r.timings = {'_a': 0.5, '_b': 0.25}
        assert r.timings['total'] == 0.75

    def test_concurrent_group_counted_once(self):
        r = PipelineResults()
        r.timings = {'_seq': 0.5}
        # three stages of 0.2 each that actually took 0.25 together
        r.add_concurrent_group('_group', 0.25,
                               {'_x': 0.2, '_y': 0.2, '_z': 0.2})

        t = r.timings
        assert t['total'] == 0.75, 'group must count as its own elapsed time, not the sum'
        for key in ('_x', '_y', '_z'):
            assert key in t, 'per-stage detail must stay in the report'
        assert t['_group'] == 0.25

    def test_total_never_below_slowest_member(self):
        r = PipelineResults()
        r.add_concurrent_group('_group', 0.30, {'_slow': 0.29, '_fast': 0.01})
        assert r.timings['total'] >= r.timings['_slow']

    def test_repeated_stage_is_accumulated(self):
        r = PipelineResults()
        r.timings = {'_type': .2}
        r.timings = {'_type': .3}
        assert r.timings['_type'] == .5
        assert r.timings['total'] == .5

    def test_same_stage_before_inside_and_after_group(self):
        r = PipelineResults()
        r.timings = {'_borders': .2}
        r.add_concurrent_group('_group', .4, {'_borders': .3, '_quality': .35})
        r.timings = {'_borders': .1}
        assert r.timings['_borders'] == .6
        assert r.timings['total'] == .7

    def test_repeated_group_counts_each_wall_time_once(self):
        r = PipelineResults()
        r.add_concurrent_group('_group', .4, {'_x': .3, '_y': .35})
        r.add_concurrent_group('_group', .2, {'_x': .1, '_y': .15})
        assert r.timings['_group'] == .6
        assert r.timings['_x'] == .4
        assert r.timings['total'] == .6


def test_total_does_not_exceed_wall_clock():
    """Concurrent quality+borders stage must report group elapsed time, not the sum."""
    from time import perf_counter, sleep

    class _SlowQuality:
        def __init__(self, name, label):
            self.model_name = name
            self._label = label

        def predict(self, img):
            sleep(0.04)
            return {self.model_name: (self._label, 0.99)}

    class _SlowBorders:
        def predict_transform(self, img, **_kw):
            sleep(0.04)
            return {'Borders': {'bbox': [], 'warped_img': img}}

    pipe = object.__new__(Pipeline)
    pipe.results = PipelineResults()
    pipe.results._meta_results['DocTypeAngles'] = {'doc_type': 'DL_2011'}
    pipe.glare = _SlowQuality('Glare', 'good')
    pipe.blur = _SlowQuality('Blur', 'good')
    pipe.print_spoofing = _SlowQuality('PrintSpoofing', 'REAL')
    pipe.lcd_spoofing = _SlowQuality('LCDSpoofing', 'REAL')
    pipe.doc_detector = _SlowBorders()

    start = perf_counter()
    pipe._quality_and_borders_parallel(None)
    wall = perf_counter() - start

    timings = pipe.results.timings
    total = timings['total']
    members = ('_glare', '_blur', '_print_spoofing', '_lcd_spoofing', '_doc_detector')
    member_sum = sum(timings[k] for k in members)

    assert '_quality_and_borders' in timings
    assert total <= wall + 0.005, f'reported total {total:.3f}s exceeds wall clock {wall:.3f}s'
    assert total >= max(timings[k] for k in members) - 0.005
    assert total < member_sum * 0.75, (
        f'parallel group total {total:.3f}s looks like a serial sum ({member_sum:.3f}s)'
    )
