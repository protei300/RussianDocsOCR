import pytest
from document_processing.pipeline_modules import *
from document_processing.processing.models import ModelLoader
from pathlib import Path
from document_processing.pipeline_modules.doc_detector.image_transformation import xywh2xyxy, iou
import numpy as np
import cv2

@pytest.fixture
def module():
    return WordsDetector(model_format='ONNX', device='cpu')

@pytest.fixture
def model():
    model_loader = ModelLoader()
    return model_loader(Path('../document_processing/models/Words/ONNX/model.json'))


@pytest.fixture
def load_imgs():
    """Pair each image with ITS label, by name.

    This used to zip two independent globs together. Path.glob returns entries in
    filesystem order, which is neither sorted nor the same across platforms, so the
    Nth image was matched with whatever label happened to be Nth - on Linux that
    graded DL_2020/6.jpg against another document's 9 words while the local run
    passed on lucky ordering. Deriving the label from the image makes the pairing
    a fact rather than a coincidence, and the assertions below turn a missing or
    stray fixture into a named failure.
    """
    pairs = []
    for img in sorted(Path('images/Words').glob('*/images/*.*')):
        lbl = img.parent.parent / 'labels' / f'{img.stem}.txt'
        assert lbl.is_file(), f'no label for {img}'
        pairs.append((img, lbl))
    assert pairs, 'no word fixtures under images/Words'
    return pairs

class TestWordsDetector:

    iou_tolerance = 0.8
    def test_model(self, model, load_imgs):
        for img_file, lbl_file in load_imgs:

            img = cv2.imread(img_file.as_posix())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            #Reading ground truth
            lbls = lbl_file.read_text().splitlines()
            lbl = [lbl.split() for lbl in lbls]
            lbl = np.array(lbl).astype(float)
            lbl[..., 1:] = xywh2xyxy(lbl[..., 1:])
            lbl[..., [1, 3]] *= w
            lbl[..., [2, 4]] *= h
            ind = np.lexsort([lbl[..., 1], lbl[..., 0],])
            lbl = lbl[ind]
            t_lbl, t_coords = np.split(lbl, [1,], axis=-1)

            #Preparing result from model
            result = np.array(model.predict(img))[..., [0,1,2,3,5]]
            result = result.astype(np.float32)
            r_coords, r_lbl = np.split(result, [4,], axis=-1)

            #tests
            assert r_coords.shape[0] == t_coords.shape[0], (
                f'Word count mismatch for {img_file}: '
                f'detected {r_coords.shape[0]}, expected {t_coords.shape[0]}'
            )
            # order-independent: every ground-truth word must be matched by some
            # detected box with IoU >= tolerance (pairwise IoU, best per GT box).
            iou_matrix = iou(r_coords[:, None, :], t_coords[None, :, :])  # [pred, gt]
            best_iou_per_gt = iou_matrix.max(axis=0)
            assert (best_iou_per_gt >= self.iou_tolerance).all(), (
                f'BBoxes dont match for img {img_file}: '
                f'min best IoU {best_iou_per_gt.min():.2f} < {self.iou_tolerance}'
            )


    def test_module(self, module, load_imgs):
        '''
        Testing predict_transform function
        '''

        img_file, lbl_file = next(iter(load_imgs))
        result = module.predict_transform(img_file)

        #check if result dict has field equal to module name
        assert module.model_name in result.keys(), 'No key field!'
        #check if bbox field exists in result dict
        assert 'bbox' in result[module.model_name].keys(), 'No key field bbox!'
        assert 'warped_img' in result[module.model_name].keys(), 'No warped_img field in results'

