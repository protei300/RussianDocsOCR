import pytest
from pathlib import Path
from document_processing.pipeline_modules import *
from document_processing.processing.models import ModelLoader

SAMPLES_DIR = Path('../samples')

#: The bound for test_module_accuracy is the class list OF THE MODEL, read from
#: its own centroids - not the set of directories that happen to sit in samples/.
#: The difference is not arithmetic. Folder names say how many types we were
#: given; the model's labels say how many it is supposed to tell apart. A test
#: built on the first number checks "what we were handed" and calls it coverage -
#: the very disease this test was fixed for, one level up.
#:
#: A class the model knows and the material does not cover therefore FAILS here.
#: That is deliberate: missing material is a gap in what we can claim, and it has
#: to be visible as a refusal rather than as a quiet "checked what there was".


#: Classes the model knows that samples/ does not cover. **Empty in this
#: repository, and that is a statement about this tree's material, not a
#: weakening of the check.** The two classes that are uncovered in the closed
#: tree - the registration page and the 2019 SNILS - have their material here:
#: `samples/INTPASSPORTADDR_ALL/` and `samples/SNILS_2019/`, two images each.
#: The sample sets of the two repositories are not the same set, so this line
#: is read off THIS one; copying the closed tree's value would declare a gap
#: that does not exist here and would hide a real one behind it.
#:
#: Compared for EQUALITY, not membership, and that is the whole point. An
#: xfail-style "this is expected to fail" would swallow a NEW gap: the test is
#: already failing, so losing the material for a tenth class would change nothing
#: in the report. Equality separates the two - a new gap makes the set bigger and
#: turns the test red, and filling a known gap makes it smaller and ALSO turns it
#: red, demanding that the entry be struck from here. Self-cleaning in both
#: directions, and a known gap stays distinguishable from fresh breakage.
#:
#: With the set empty the equality is strict in the useful direction: every class
#: the model claims must be covered by material in this tree, and the day one is
#: not, the test says which.
KNOWN_UNCOVERED_DOC_TYPES = frozenset()


def required_doc_types(module):
    """Doc types the model claims to distinguish, straight from its centroids."""
    labels = module.model.postprocessings[0].labels
    return frozenset(str(x) for x in labels)


@pytest.fixture
def model():
    model_loader = ModelLoader()
    return model_loader(Path('../document_processing/models/DocTypeAngles/ONNX/model.json'))


@pytest.fixture
def module():
    return DocTypeAngles(model_format='ONNX', device='cpu')


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
        # Nothing above runs when samples/ is empty, and a test that predicted
        # nothing must not report success.
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
        """Predicted doc type matches samples directory name, over every type.

        The loop alone would pass on whatever the directory happens to hold - it
        takes "however many were given" for "as many as are needed", and a set
        shrunk to one type would still report success. The set check below is the
        lower bound, and it is a set rather than a count on purpose.
        """
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
        """predict_transform() adds the upright image to the payload."""
        doc_dir = next(d for d in sorted(SAMPLES_DIR.iterdir()) if d.is_dir())
        img_file = next(doc_dir.glob('*.jpg'))
        result = module.predict_transform(img_file)
        assert module.model_name in result, f'Missing {module.model_name} key'
        meta = result[module.model_name]
        for key in ('doc_type', 'doc_type_confidence', 'angle', 'angle_confidence',
                    'warped_img'):
            assert key in meta, f'Missing {key} in payload'
