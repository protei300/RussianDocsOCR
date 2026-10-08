"""Every document type the library recognises has a conformance case (decision #144).

The ports are graded only on the cases the manifest lists. A type with no case is
not graded at all, and the port's conformance stays green over its absence: on
2026-10-07 the ports «caught up» with a green run while none of them could read an
STS - there was no STS case. The types the classifier knows come from its own
centroids, so a newly trained type without a case fails here at once.

KNOWN_GAPS lists the types still without a case, each with the reason. It may only
shrink: a gap that got a case must be removed from it (asserted below).
"""
from pathlib import Path

import numpy as np
import pytest

from conformance.cases import load_cases

REPO = Path(__file__).resolve().parents[1]
CENTERS = REPO / 'document_processing' / 'models' / 'DocTypeAngles' / 'ONNX' / 'resources' / 'centers.npz'

KNOWN_GAPS = {
    'DLBACK_ALL': 'back of a driving licence: no material yet (the type reads no fields)',
    'INTPASSPORTADDR_ALL': 'registration page: no anonymised sample yet (address path)',
    'SNILS_2019': 'SNILS of the 2019 form: no sample yet',
}


def recognised_types():
    if not CENTERS.is_file():
        pytest.skip('DocTypeAngles weights are not fetched (scripts/fetch_models.py)')
    labels = np.load(CENTERS, allow_pickle=False)['labels']
    return {str(x) for x in labels}


def test_every_recognised_type_has_a_conformance_case():
    covered = {c.doc_type for c in load_cases()}
    missing = recognised_types() - covered - set(KNOWN_GAPS)
    assert not missing, (f'types without a conformance case: {sorted(missing)} - add a case '
                         f'(sample + manifest row + goldens) before porting, decision #144')


def test_the_known_gaps_only_shrink():
    covered = {c.doc_type for c in load_cases()}
    closed = covered & set(KNOWN_GAPS)
    assert not closed, f'these gaps have a case now - remove them from KNOWN_GAPS: {sorted(closed)}'
    stale = set(KNOWN_GAPS) - recognised_types()
    assert not stale, f'not recognised any more - remove from KNOWN_GAPS: {sorted(stale)}'
