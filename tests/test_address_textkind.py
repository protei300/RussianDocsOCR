import pytest
from pathlib import Path
from russian_docs_ocr.document_processing.pipeline_modules import AddressTextKindClassifier


@pytest.fixture
def module():
    return AddressTextKindClassifier(model_format='ONNX', device='cpu')


class TestAddressTextKindClassifier:
    def test_module_contract(self, module):
        """predict() returns (label, probability) under the module name."""
        img_file = next(Path('tests/images/AddressTextKind').iterdir())
        result = module.predict(img_file)
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        label, prob = result[module.model_name]
        assert label in ('printed', 'handwritten'), f'Unexpected label {label!r}'
        assert 0.0 <= prob <= 1.0, f'P(handwritten) out of range: {prob}'

    def test_classification(self, module):
        """Each fixture is classified to the kind encoded in its filename."""
        files = list(Path('tests/images/AddressTextKind').glob('*.png'))
        assert files, 'No AddressTextKind fixtures found'
        for img_file in files:
            ground_truth = img_file.stem.split('_')[0]  # 'printed' | 'handwritten'
            label, prob = module.predict(img_file)[module.model_name]
            assert label == ground_truth, (
                f'{img_file.name}: expected {ground_truth!r}, got {label!r} (p_hw={prob:.3f})'
            )
