from pathlib import Path
from typing import List, Union

import numpy as np

from ..base_module import BaseModule
from ..ocr_batch import predict_batch_padded
from ..ocr_corrections import check_ddmmyyyy, check_rus_sex, strip_edge_dots

_CFG_KEY = {'accurate': 'OCRCyrillicAccurate', 'fast': 'OCRCyrillicFast'}

_RU_NAME_FIELDS = ('Last_name_ru', 'First_name_ru', 'Birth_place_ru',
                   'Living_region_ru', 'Middle_name_ru', 'Issue_organization_ru')

#: Date fields, whatever engine reads them. NEITHER engine reads the printed
#: separator of a digit date: '22.06.2010' comes back as '22/06/2010' from both.
#: The Latin engine has always repaired that in fix_errors, so the damage only
#: became visible when birth-certificate dates moved to the Cyrillic route (they
#: had to: the 2018 blank spells its months out). Same repair, same list of
#: fields - BY NAME, not by content, so the normalization can never reach a
#: series or a document number that happens to hold eight digits.
_DATE_FIELDS = ('Issue_date', 'Birth_date', 'Expiration_date',
                'Father_birth_date', 'Mother_birth_date')


class OCRCyrillic(BaseModule):
    """v2 Cyrillic OCR engine (MobileNetV4 'accurate' / EdgeNext 'fast').

    Recognizes Cyrillic text (Russian names, places, organizations) plus digits
    and punctuation. Decoding (greedy CTC + per-step alphabet masking) happens
    inside the model wrapper via ``OCRProbsPostprocessing``; ``predict`` returns
    the decoded string, ``predict_fv`` the raw softmax matrix ``[1, T, C]`` for
    the regex-constrained (beam-search) path.
    """

    def __init__(self, tier: str = 'accurate', model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        assert tier in _CFG_KEY, f"tier must be one of {list(_CFG_KEY)}"
        self.tier = tier
        self.model_name = 'OCRCyrillic'
        super().__init__(self.model_name, model_format=model_format, device=device,
                         verbose=verbose, cfg_key=_CFG_KEY[tier])

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Recognize text -> {'OCRCyrillic': {'ocr_output': <str>}}."""
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
        """Apply field-specific corrections (dates, sex, stray dots on names)."""
        if field_type in _DATE_FIELDS:
            # Rewrites only when the text holds exactly eight digits, so a date
            # spelled out in words passes through untouched - «15 ОКТЯБРЯ 2020 Г.»
            # has six, and SNILS reaches this per word ('26', 'СЕНТЯБРЯ', '1997')
            # with four at most.
            try:
                return check_ddmmyyyy(text)
            except ValueError:
                return text
        if field_type == 'Sex_ru':
            return check_rus_sex(text)
        if field_type in _RU_NAME_FIELDS:
            return strip_edge_dots(text)
        return text
