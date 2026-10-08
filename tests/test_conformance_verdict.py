"""The conformance verdict must not read better than what happened.

A case whose probe crashed has no stages, hence no differences, and the
classification used to read it as clean: on 2026-10-07 a Go run without
RDOCS_MODELS_ROOT failed in all 10 cases and printed `VERDICT: CLEAN`. These tests
are the negative control of the checker itself: a run that compared nothing, or
only part of what it was asked to, is never green - in words and in exit code.

No models, no port: the reports are built by hand.
"""
import pytest

from conformance import deviations as deviations_mod
from conformance.runner.__main__ import _hard_error
from conformance.runner.compare import StageResult
from conformance.runner.report import CaseReport, RunReport, render


def classified(cases, port='go'):
    """A run classified the way the runner does it, with no declared deviations."""
    run = RunReport(port=port, profile='cpu', cases=cases)
    run.deviations = deviations_mod.classify(run, [], port)
    return run


def good_case(slug='A'):
    return CaseReport(slug=slug, doc_type='X', stages=[
        StageResult(stage='prepare', ok=True)])


def crashed_case(slug='B'):
    return CaseReport(slug=slug, doc_type='X', error='probe exited 3: no models')


@pytest.mark.parametrize('cases,errored', [
    ([crashed_case('A'), crashed_case('B')], 2),   # every case crashed
    ([good_case('A'), crashed_case('B')], 1),      # one of two
])
def test_a_case_that_was_not_compared_is_never_clean(cases, errored):
    run = classified(cases)
    assert run.deviations.verdict() == 'CLEAN', 'the classification alone still reads it clean'
    assert run.verdict == f'NOT VERIFIED ({errored} of {len(cases)} cases not compared)'
    assert 'VERDICT: NOT VERIFIED' in render(run)
    assert _hard_error(run), 'the exit code must fail too'
    assert not run.ok


def test_an_empty_run_is_not_a_pass():
    run = classified([])
    assert run.verdict == 'NOTHING VERIFIED'
    assert _hard_error(run)


def test_a_fully_compared_clean_run_stays_clean():
    """The positive control: the fix must not turn every run red."""
    run = classified([good_case('A'), good_case('B')], port='python')
    assert run.verdict == 'CLEAN'
    assert not _hard_error(run)
    assert run.ok
