from pathlib import Path
from typing import List, Union

import numpy as np

from ..base_module import BaseModule
from ..ocr_batch import predict_batch_padded
from ..ocr_corrections import check_ddmmyyyy, check_en_sex, check_driver_class

_CFG_KEY = {'accurate': 'OCRLatinAccurate', 'fast': 'OCRLatinFast'}


class OCRLatin(BaseModule):
    """v2 Latin OCR engine (MobileNetV4 'accurate' / EdgeNext 'fast').

    Recognizes Latin text (transliterated names, document numbers, dates) plus
    digits and punctuation. Decoding (greedy CTC + per-step alphabet masking)
    happens inside the model wrapper via ``OCRProbsPostprocessing``; ``predict``
    returns the decoded string, ``predict_fv`` the raw softmax matrix ``[1, T, C]``
    for the regex-constrained (beam-search) path.
    """

    def __init__(self, tier: str = 'accurate', model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        assert tier in _CFG_KEY, f"tier must be one of {list(_CFG_KEY)}"
        self.tier = tier
        self.model_name = 'OCRLatin'
        super().__init__(self.model_name, model_format=model_format, device=device,
                         verbose=verbose, cfg_key=_CFG_KEY[tier])

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Recognize text -> {'OCRLatin': {'ocr_output': <str>}}."""
        ocr_output = self.model.predict(img)[0]
        return {self.model_name: {'ocr_output': ocr_output}}

    def predict_fv(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Raw softmax CTC matrix (no decode), wrapped for the regex path."""
        ocr_raw_output = self.model.predict_fv(img)
        return {self.model_name: {'ocr_raw_output': ocr_raw_output}}

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        pass

    def predict_batch(self, patches: List[np.ndarray]) -> List[str]:
        """Decode many word patches in one padded-batch inference call.

        GPU-only optimization (see ocr_batch.py): avoids one CUDA graph
        recompile per distinct patch width. On CPU, use `predict` per patch
        instead (batching does not help there).
        """
        return predict_batch_padded(self.model, patches)

    def fix_errors(self, field_type: str, text: str) -> str:
        """Apply field-specific corrections (dates, sex, driver class)."""
        if field_type in ('Issue_date', 'Expiration_date', 'Birth_date'):
            try:
                return check_ddmmyyyy(text)
            except ValueError:
                return text
        if field_type == 'Sex_en':
            return check_en_sex(text)
        if field_type == 'Driver_class':
            return check_driver_class(text)
        return text
