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
    """End-to-end: reported total must stay within measured wall time."""
    images = sorted(SAMPLES_DIR.joinpath('DL_2011').glob('*.jpg'))
    if not images:
        pytest.skip('No DL_2011 sample available')

    pipeline = Pipeline(model_format='ONNX', device='cpu')
    pipeline.process_img(images[0], img_size=1500)  # warm caches

    start = time()
    result = pipeline.process_img(images[0], img_size=1500)
    wall = time() - start

    total = result.timings['total']
    assert total <= wall, f'reported total {total:.3f}s exceeds wall clock {wall:.3f}s'
    stages = {k: v for k, v in result.timings.items() if k != 'total'}
    assert total >= max(stages.values()), 'total must cover at least the slowest stage'
