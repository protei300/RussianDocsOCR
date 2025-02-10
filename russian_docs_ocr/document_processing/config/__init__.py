import os
from pathlib import Path

import yaml

FILE = Path(__file__).resolve()
DEFAULT_CFG_PATH = FILE.parent.joinpath('models_path.yaml')
ROOT = FILE.parents[1]

with open(DEFAULT_CFG_PATH, 'r') as f:
    DEFAULT_CFG = yaml.safe_load(f)
    for key, value in DEFAULT_CFG.items():
        if not Path(value).is_absolute():
            DEFAULT_CFG[key] = ROOT.joinpath(str(value).replace('\\', os.sep)).as_posix()
