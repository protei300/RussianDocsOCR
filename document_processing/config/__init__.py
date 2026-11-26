import os
from pathlib import Path

import yaml

FILE = Path(__file__).resolve()
DEFAULT_CFG_PATH = FILE.parent.joinpath('models_path.yaml')
ROOT = FILE.parents[1]


def models_root() -> Path:
    """Package root containing ``models/`` (honors ``RDOCS_MODELS_ROOT``)."""
    override = os.environ.get('RDOCS_MODELS_ROOT')
    return (Path(override) / 'document_processing') if override else ROOT


with open(DEFAULT_CFG_PATH, 'r', encoding='utf-8') as f:
    _RAW_CFG = yaml.safe_load(f)


def resolve_model_dir(key: str) -> Path:
    """Resolve a model directory from ``models_path.yaml``, honoring ``RDOCS_MODELS_ROOT``."""
    raw = _RAW_CFG[key]
    raw_path = Path(str(raw).replace('\\', os.sep))
    if raw_path.is_absolute():
        return raw_path
    override = os.environ.get('RDOCS_MODELS_ROOT')
    if override:
        return Path(override) / 'document_processing' / raw_path
    return Path(DEFAULT_CFG[key])


DEFAULT_CFG = {}
for _key, _value in _RAW_CFG.items():
    _rel = str(_value).replace('\\', os.sep)
    if Path(_rel).is_absolute():
        DEFAULT_CFG[_key] = _rel
    else:
        DEFAULT_CFG[_key] = models_root().joinpath(_rel).as_posix()
