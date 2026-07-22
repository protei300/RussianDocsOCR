import pytest
from pathlib import Path
from russian_docs_ocr.document_processing.pipeline_modules import DocTypeAngles
from russian_docs_ocr.document_processing.processing.models import ModelLoader


IMAGES = Path('tests/images/DocTypeAngles')


@pytest.fixture
def model():
    model_loader = ModelLoader()
    return model_loader(Path('russian_docs_ocr/document_processing/models/DocTypeAngles/ONNX/model.json'))


@pytest.fixture
def module():
    return DocTypeAngles(model_format='ONNX', device='cpu')


def _expected(img: Path):
    """Filenames are <DOCTYPE_FAMILY>_<angle>.jpg (e.g. EXTPASSPORTBIO_180.jpg).
    The model returns the doctype with a year suffix (EXTPASSPORTBIO_2007), so
    tests compare the family only."""
    doctype, angle = img.stem.split('_', maxsplit=1)
    return doctype, int(angle)


def _family(doctype: str) -> str:
    """Strip the trailing year from a model doctype (EXTPASSPORTBIO_2007 -> EXTPASSPORTBIO)."""
    return doctype.rsplit('_', maxsplit=1)[0]


class TestDocTypeAngle:
    def test_model(self, model):
        """Raw model.predict returns [(doc_type, dist, thresh), (angle, angle_conf)]."""
        for img in IMAGES.iterdir():
            doctype_expected, angle_expected = _expected(img)
            doctype_meta, angle_meta = model.predict(img)
            assert _family(doctype_meta[0]) == doctype_expected, f'Wrong doctype for {img.name}: {doctype_meta[0]}'
            assert angle_meta[0] == angle_expected, f'Wrong angle for {img.name}: {angle_meta[0]}'

    def test_module(self, module):
        """predict_transform returns {model_name: flat payload} incl. warped_img."""
        for img in IMAGES.iterdir():
            doctype_expected, angle_expected = _expected(img)
            result = module.predict_transform(img)

            assert module.model_name in result, f'Missing {module.model_name!r} key'
            meta = result[module.model_name]
            assert _family(meta['doc_type']) == doctype_expected, f'Wrong doctype for {img.name}'
            assert meta['angle'] == angle_expected, f'Wrong angle for {img.name}'
            assert 'warped_img' in meta, 'Missing warped_img'
            assert 'doc_type_confidence' in meta and 'angle_confidence' in meta

    def test_module_predict_has_no_warped_img(self, module):
        """predict() returns the same payload minus the rotated image."""
        img = next(iter(IMAGES.iterdir()))
        meta = module.predict(img)[module.model_name]
        assert 'warped_img' not in meta
        assert _family(meta['doc_type']) == _expected(img)[0]
