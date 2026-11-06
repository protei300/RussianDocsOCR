"""Where the registration reader finds its models and dictionaries.

The two networks are pinned by models.lock.json and fetched with the rest of
the weight set. Environment variables can select an explicit installation.
No address dictionaries are required for the raw five-field reading.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _models() -> Path:
    root = Path(os.environ.get('RDOCS_MODELS_ROOT', Path(__file__).resolve().parents[3]))
    return root / 'document_processing' / 'models'


@dataclass
class RegistrationConfig:
    #: common printed/handwritten YOLO11s-OBB, five classes
    fields_model: Path = field(default_factory=lambda: _env(
        'RDOCS_STAMP_FIELDS_MODEL', str(_models() / 'StampFields/ONNX/model.onnx')))
    #: stamp OCR v3: a folder with ``model.onnx`` + ``model.json`` (OCRv2 contract)
    ocr_dir: Path = field(default_factory=lambda: _env(
        'RDOCS_STAMP_OCR', str(_models() / 'OCR/stamp_v3/ONNX')))

    fields_conf: float = 0.4            # selected on the detector validation split
    stamp_conf_fallback: float = 0.92   # type NONE + a stamp at least this sure -> a cut-out stamp
    spread_aspect: float = 1.2          # a canvas wider than this * height is a spread
    cut_outside_share: float = 0.03     # page cut = more than this share of it outside the frame

    def missing(self) -> list[str]:
        need = {'RDOCS_STAMP_FIELDS_MODEL': (self.fields_model,),
                'RDOCS_STAMP_OCR': (self.ocr_dir / 'model.onnx', self.ocr_dir / 'model.json')}
        return [k for k, paths in need.items() if not all(p.is_file() for p in paths)]
