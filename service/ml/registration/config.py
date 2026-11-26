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
    root = Path(os.environ.get('RDOCS_MODELS_ROOT') or Path(__file__).resolve().parents[3])
    return root / 'document_processing' / 'models'


def _default_ocr_dir() -> Path:
    explicit = os.environ.get('RDOCS_STAMP_OCR')
    if explicit:
        return Path(explicit)
    v4 = _models() / 'OCR/stamp_v4/ONNX'
    if (v4 / 'model.onnx').is_file() and (v4 / 'model.json').is_file():
        return v4
    return _models() / 'OCR/stamp_v3/ONNX'


@dataclass
class RegistrationConfig:
    #: printed/handwritten YOLO-OBB; class order, thresholds and input size in model.json
    fields_model: Path = field(default_factory=lambda: _env(
        'RDOCS_STAMP_FIELDS_MODEL', str(_models() / 'StampFields/ONNX/model.onnx')))
    #: stamp OCR (prefers v4 when installed, falls back to v3): folder with ``model.onnx`` + ``model.json``
    ocr_dir: Path = field(default_factory=_default_ocr_dir)
    #: optional dictionary bundle (addressru.zip, fms_unit_*.csv, street_names_kladr.json) for ocr_postprocessing
    refs_dir: Path = field(default_factory=lambda: _env(
        'RDOCS_REGISTRATION_REFS', str(_models() / 'RegistrationRefs')))
    #: enable dictionary-backed CTC postprocessing when references are installed
    postprocess: bool = field(default_factory=lambda: os.environ.get(
        'RDOCS_STAMP_POSTPROCESS', '1').lower() not in ('0', 'false', 'no'))

    fields_conf: float | None = None    # None -> read Confidence/CLS from model.json (fallback 0.4)
    stamp_conf_fallback: float = 0.92   # type NONE + a stamp at least this sure -> a cut-out stamp
    spread_aspect: float = 1.2          # a canvas wider than this * height is a spread
    cut_outside_share: float = 0.03     # page cut = more than this share of it outside the frame
    #: residual tilt before field detection; coarse OBB cropping remains in page.py
    stamp_deskew: bool = field(default_factory=lambda: os.environ.get(
        'RDOCS_STAMP_DESKEW', '1').lower() not in ('0', 'false', 'no'))

    def ref_paths(self) -> tuple[Path, Path, Path]:
        """Resolve ``(addressru, fms_csv, street_names)`` paths."""
        assets = Path(__file__).resolve().parent / 'assets'

        def pick(env_name: str, filename: str) -> Path:
            explicit = os.environ.get(env_name)
            if explicit:
                return Path(explicit)
            cand = self.refs_dir / filename
            return cand if cand.is_file() else (assets / filename)

        return (
            pick('RDOCS_ADDRESSRU', 'addressru.zip'),
            pick('RDOCS_FMS_UNIT', 'fms_unit_2026-09-22.csv'),
            pick('RDOCS_STREET_NAMES', 'street_names_kladr.json'),
        )

    def missing(self) -> list[str]:
        need = {'RDOCS_STAMP_FIELDS_MODEL': (self.fields_model,),
                'RDOCS_STAMP_OCR': (self.ocr_dir / 'model.onnx', self.ocr_dir / 'model.json')}
        return [k for k, paths in need.items() if not all(p.is_file() for p in paths)]
