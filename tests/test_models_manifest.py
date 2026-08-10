"""The deployed model files must match document_processing/models.lock.json.

Weights live outside git, so the manifest is the only record of what the set is
supposed to contain. Two 3.x bugs were exactly this drift going unnoticed: an
OpenVINO .ir that still held pre-retrain Borders weights, and a MaskFilter left
at the previous checkpoint's value when model.onnx was swapped. Both would have
shown up here the moment they were introduced.

No network: this only hashes what is already on disk.

Paths are relative to tests/ (see conftest.py, which chdirs there).
"""
import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path('..').resolve()
MANIFEST = REPO_ROOT / 'document_processing' / 'models.lock.json'
MODELS_DIR = REPO_ROOT / 'document_processing' / 'models'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope='module')
def manifest():
    if not MANIFEST.exists():
        pytest.fail(f'missing manifest {MANIFEST} - run scripts/build_models_manifest.py')
    return json.loads(MANIFEST.read_text(encoding='utf8'))


def test_manifest_is_well_formed(manifest):
    assert manifest['files'], 'manifest lists no files'
    assert manifest['base_url'].endswith('/'), 'base_url must end with a slash'
    assets = [e['asset'] for e in manifest['files']]
    assert len(assets) == len(set(assets)), 'two files map to the same asset name'
    for entry in manifest['files']:
        assert not Path(entry['path']).is_absolute(), entry['path']
        assert '..' not in Path(entry['path']).parts, entry['path']
        assert len(entry['sha256']) == 64, entry['path']


def test_deployed_models_match_manifest(manifest):
    """Every pinned file is present, the right size, and the right bytes."""
    problems = []
    for entry in manifest['files']:
        path = MODELS_DIR / entry['path']
        if not path.exists():
            problems.append(f'{entry["path"]}: missing')
            continue
        if path.stat().st_size != entry['size']:
            problems.append(f'{entry["path"]}: size {path.stat().st_size} != {entry["size"]}')
            continue
        if sha256(path) != entry['sha256']:
            problems.append(f'{entry["path"]}: checksum differs from the manifest')

    assert not problems, (
        'deployed models do not match models.lock.json:\n  ' + '\n  '.join(problems) +
        '\n\nEither the manifest is stale (a model changed -> run '
        'scripts/build_models_manifest.py) or the files are (run scripts/fetch_models.py).'
    )


def test_no_untracked_model_artifacts_shadow_the_set(manifest):
    """A model directory the manifest does not know about is a deployment risk.

    Local A/B folders (Borders/ONNX_old, TextFields/ONNX_legacy_backup, threshold
    sweeps) are legitimate and gitignored, so this only warns about ones holding a
    model.json that the loader could be pointed at by model_format=.
    """
    pinned = {(MODELS_DIR / e['path']).resolve() for e in manifest['files']}
    strays = [p for p in MODELS_DIR.rglob('model.json') if p.resolve() not in pinned]
    if strays:
        names = ', '.join(str(p.relative_to(MODELS_DIR)) for p in sorted(strays))
        pytest.skip(f'local model variants present (not published): {names}')
