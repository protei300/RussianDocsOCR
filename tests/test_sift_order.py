"""Template matching does not depend on the order OpenCV hands keypoints over.

OpenCV orders SIFT keypoints with std::sort and cuts its ``nfeatures`` budget
with std::nth_element; the order of ties depends on the C++ library it was built
with. MAGSAC samples by index, so the Windows wheel and Linux reached different
decisions on one STS card (conformance D-07/D-08, 2026-10-08). ``detect_features``
sorts the keypoints itself and cuts the budget from that order. No models needed.
"""
import random

import cv2
import numpy as np

from document_processing.pipeline_modules.page_registration.page_registration import detect_features


class ShuffledSift:
    """A SIFT stand-in that returns the same keypoints in a given order."""

    def __init__(self, kp, desc, seed):
        self.kp, self.desc, self.seed = kp, desc, seed

    def detectAndCompute(self, gray, mask):
        idx = list(range(len(self.kp)))
        random.Random(self.seed).shuffle(idx)
        return [self.kp[i] for i in idx], self.desc[idx]


def keypoints():
    rng = np.random.default_rng(0)
    kp = [cv2.KeyPoint(float(x), float(y), 3.0, 0.0, float(r), 0)
          for x, y, r in zip(rng.uniform(0, 500, 300), rng.uniform(0, 500, 300), rng.uniform(0, 1, 300))]
    # ties on response, broken only by position
    kp += [cv2.KeyPoint(float(x), 10.0, 3.0, 0.0, 0.5, 0) for x in range(20)]
    desc = np.arange(len(kp) * 4, dtype=np.float32).reshape(len(kp), 4)   # row i belongs to kp i
    return kp, desc


def signature(kp, desc):
    return [(k.pt, k.response) for k in kp], desc.tolist()


def test_the_order_handed_over_does_not_matter():
    kp, desc = keypoints()
    # no budget: the tied group (response 0.5) sits mid-list, a budget of 100 would cut it
    # away and leave the tie-breaking unchecked
    results = [signature(*detect_features(ShuffledSift(kp, desc, s), None, None, None)) for s in range(5)]
    assert all(r == results[0] for r in results[1:])


def test_the_budget_keeps_the_strongest_and_descriptors_follow_their_keypoints():
    kp, desc = keypoints()
    got_kp, got_desc = detect_features(ShuffledSift(kp, desc, 1), None, None, 50)
    responses = [k.response for k in got_kp]
    assert len(got_kp) == 50 and responses == sorted(responses, reverse=True)
    assert min(responses) >= sorted((k.response for k in kp), reverse=True)[49]
    for k, d in zip(got_kp, got_desc):
        i = next(j for j, o in enumerate(kp) if o.pt == k.pt and o.response == k.response)
        assert (d == desc[i]).all()
