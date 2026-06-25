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
        """predict_transform returns DocType/Angle90 metas and the rotated image."""
        for img in IMAGES.iterdir():
            doctype_expected, angle_expected = _expected(img)
            result = module.predict_transform(img)

            assert 'DocType' in result and 'Angle90' in result, 'Missing DocType/Angle90 keys'
            assert _family(result['DocType']['doc_type']) == doctype_expected, f'Wrong doctype for {img.name}'

            angle_meta = result['Angle90']
            assert 'angle' in angle_meta and 'warped_img' in angle_meta, 'Missing angle/warped_img'
            assert angle_meta['angle'] == angle_expected, f'Wrong angle for {img.name}'
