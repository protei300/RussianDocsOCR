import numpy as np
import pytest
from pathlib import Path
from document_processing.pipeline_modules import DocTypeAngles

SAMPLES_DIR = Path('../samples')

#: The bound for test_module_accuracy is the class list OF THE MODEL, read from
#: its own centroids - not the set of directories that happen to sit in samples/.
#:
#: Classes the model knows that samples/ does not cover. The two classes that
#: are uncovered in the closed tree - the registration page and the 2019 SNILS -
#: have their material here: `samples/INTPASSPORTADDR_ALL/` (one image) and
#: `samples/SNILS_2019/` (two).
#:
#: The four vehicle registration certificate classes (STS_*, weight set v7) and
#: the licence back side (DLBACK_ALL) remain uncovered until anonymised
#: independent samples arrive.
KNOWN_UNCOVERED_DOC_TYPES = frozenset({'STS_1996', 'STSBACK_1996', 'STS_2019', 'STSBACK_2019',
                                       'DLBACK_ALL'})


def required_doc_types(module):
    """Doc types the model claims to distinguish, straight from its centroids."""
    labels = module.model.postprocessings[0].labels
    return frozenset(str(x) for x in labels)


@pytest.fixture(scope='module')
def module():
    return DocTypeAngles(model_format='ONNX', device='cpu')


@pytest.fixture(scope='module')
def model(module):
    return module.model


class TestDocTypeAngles:
    def test_model(self, model):
        """Raw model returns (doctype_meta, angle_meta) for each image."""
        checked = 0
        for doc_dir in sorted(SAMPLES_DIR.iterdir()):
            if not doc_dir.is_dir():
                continue
            for img_file in list(doc_dir.glob('*.jpg'))[:1]:
                checked += 1
                doctype_meta, angle_meta = model.predict(img_file)
                doc_type, dist, thresh = doctype_meta
                angle, conf = angle_meta
                assert isinstance(doc_type, str), f'doc_type must be a string, got {type(doc_type)}'
                assert angle in (0, 90, 180, 270), f'Unexpected angle value: {angle}'
                assert 0.0 <= conf <= 1.0, f'Angle confidence out of range: {conf}'
        assert checked, 'no sample images found under samples/'

    def test_module(self, module):
        """Module predict() follows the standard {model_name: payload} contract."""
        doc_dir = next(d for d in sorted(SAMPLES_DIR.iterdir()) if d.is_dir())
        img_file = next(doc_dir.glob('*.jpg'))
        result = module.predict(img_file)
        assert module.model_name in result, f'Missing {module.model_name} key'
        meta = result[module.model_name]
        for key in ('doc_type', 'doc_type_confidence', 'angle', 'angle_confidence'):
            assert key in meta, f'Missing {key} in payload'
        assert 'warped_img' not in meta, 'predict() must not return warped_img'

    def test_module_accuracy(self, module):
        """Predicted doc type matches samples directory name, over every type."""
        required = required_doc_types(module)
        seen_types = set()
        for doc_dir in sorted(SAMPLES_DIR.iterdir()):
            if not doc_dir.is_dir():
                continue
            expected_type = doc_dir.name
            for img_file in list(doc_dir.glob('*.jpg'))[:2]:
                result = module.predict(img_file)
                predicted = result[module.model_name]['doc_type']
                assert predicted == expected_type, (
                    f'{img_file.name}: expected {expected_type!r}, got {predicted!r}'
                )
                seen_types.add(expected_type)

        missing = required - seen_types
        assert missing == KNOWN_UNCOVERED_DOC_TYPES, '\n'.join([
            'coverage of the model class list changed.',
            f'  uncovered now : {sorted(missing)}',
            f'  known gaps    : {sorted(KNOWN_UNCOVERED_DOC_TYPES)}',
            f'  newly missing : {sorted(missing - KNOWN_UNCOVERED_DOC_TYPES)}'
            '  (material disappeared - fix that)',
            f'  newly covered : {sorted(KNOWN_UNCOVERED_DOC_TYPES - missing)}'
            '  (material arrived - strike it from KNOWN_UNCOVERED_DOC_TYPES)',
            f'  checked       : {sorted(seen_types)}',
        ])

    def test_module_transform(self, module):
        """predict_transform() adds a valid upright RGB image to the payload."""
        doc_dir = next(d for d in sorted(SAMPLES_DIR.iterdir()) if d.is_dir())
        img_file = next(doc_dir.glob('*.jpg'))
        result = module.predict_transform(img_file)
        assert module.model_name in result, f'Missing {module.model_name} key'
        meta = result[module.model_name]
        for key in ('doc_type', 'doc_type_confidence', 'angle', 'angle_confidence',
                    'warped_img'):
            assert key in meta, f'Missing {key} in payload'
        assert meta['doc_type'] == doc_dir.name
        assert meta['angle'] in (0, 90, 180, 270)
        warped = meta['warped_img']
        assert isinstance(warped, np.ndarray) and warped.ndim == 3 and min(warped.shape[:2]) > 50
