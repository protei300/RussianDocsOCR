"""Vehicle registration certificate (STS) straightened by its printed blank.

A card in a plastic sleeve gets the sleeve's edge from the border detector, and the
canvas comes out skewed. The pipeline therefore registers the card against the
cleaned blank of its type (page_registration/templates/sts*.json) and takes the
template geometry when the match is strong (Pipeline._register_card).

No models here: the blank itself, put into a photo in perspective, is the card.
"""
import cv2
import numpy as np
import pytest

from document_processing.pipeline_modules.page_registration import PageRegistrar
from document_processing.pipeline_modules.page_registration.page_registration import TEMPLATES_DIR

TYPES = ['STS_1996', 'STSBACK_1996', 'STS_2019', 'STSBACK_2019']


def photo_of(card_gray, corners, size=(1600, 1900)):
    """The card warped onto a plain background at `corners` (TL, TR, BR, BL)."""
    h, w = card_gray.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(corners))
    card = cv2.cvtColor(card_gray, cv2.COLOR_GRAY2RGB)
    bg = np.full((size[1], size[0], 3), (90, 110, 130), np.uint8)
    warped = cv2.warpPerspective(card, M, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, size) > 0
    bg[inside] = warped[inside]
    return bg


@pytest.mark.parametrize('doc_type', TYPES)
def test_every_sts_type_has_templates_and_a_static_mask(doc_type):
    reg = PageRegistrar(doc_type)
    assert reg.page_names == ['card']
    for tpl in reg.pages[0][1]:
        static = tpl.mask.mean()
        assert 0.3 < static < 0.8, f'{tpl.name}: mask keeps {static:.2f} of the card'
        assert len(tpl.kp) > 500, 'too few features on the printed blank'


@pytest.mark.parametrize('doc_type', TYPES)
def test_a_card_in_perspective_is_found_to_a_few_pixels(doc_type):
    reg = PageRegistrar(doc_type)
    card = reg.pages[0][1][0].gray
    corners = np.float32([[260, 180], [1320, 240], [1380, 1700], [210, 1640]])
    photo = photo_of(card, corners)

    (r,) = reg.register(photo)

    assert r.ok and r.inliers >= 40
    err = np.linalg.norm(r.quad - corners, axis=1).max()
    assert err < 4.0, f'card corners off by {err:.1f} px'


def test_the_fill_paints_past_the_photo_instead_of_smearing_its_edge():
    reg = PageRegistrar('STS_2019')
    photo = np.zeros((200, 200, 3), np.uint8)
    photo[:, :100] = 255                       # the left half is "card", the right is off the photo
    M = np.float64([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    smeared = reg.warp_matrix(photo[:, :100], M, scale=0.2)
    painted = reg.warp_matrix(photo[:, :100], M, scale=0.2, fill=(10, 20, 30))
    assert (smeared[:, 150:] == 255).all(), 'the default repeats the edge (the passport path)'
    assert (painted[:, 150:] == (10, 20, 30)).all(axis=-1).all()


def test_the_templates_carry_no_full_colour_print():
    """Shipped grey: the red blank number and the stamp were cleaned and colour is not needed."""
    for path in TEMPLATES_DIR.glob('sts*.jpg'):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert img is not None and img.ndim == 2, path.name
