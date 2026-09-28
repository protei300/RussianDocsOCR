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


# --- per-release layout (since models-v8) -------------------------------------

import importlib.util

from document_processing.fetch_models import entry_url

#: The first weight set whose NEW files are published under content-addressed
#: names. Files already published in v8 and older keep their plain flat names.
FIRST_CONTENT_ADDRESSED = 9


def _builder():
    spec = importlib.util.spec_from_file_location(
        'build_models_manifest', REPO_ROOT / 'scripts' / 'build_models_manifest.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_file_names_its_release_and_every_network_its_version(manifest):
    if not manifest.get('release_url'):
        pytest.skip('manifest predates the per-release layout')
    builder = _builder()
    newest = {}
    for e in manifest['files']:
        assert e.get('release', '').startswith('models-v'), e['path']
        assert builder.set_number(e['release']) <= builder.set_number(manifest['models_version']), \
            f'{e["path"]} lives in a set newer than the manifest'
        if builder.set_number(e['release']) >= FIRST_CONTENT_ADDRESSED:
            assert e['sha256'][:builder.SHA_IN_NAME] in e['asset'], \
                f'{e["path"]}: a file published since v{FIRST_CONTENT_ADDRESSED} must carry its checksum in the name'
        net = builder.network_of(e['path'])
        if builder.set_number(e['release']) > builder.set_number(newest.get(net, '')):
            newest[net] = e['release']
    assert manifest['networks'] == {n: r.replace('models-', '') for n, r in newest.items()}


def test_entry_url_prefers_mirror_then_own_release_then_base():
    m = {'base_url': 'https://x/models-v9/', 'release_url': 'https://x/{release}/'}
    old = {'asset': 'A__ONNX__model.onnx', 'release': 'models-v4'}
    assert entry_url(m, old) == 'https://x/models-v4/A__ONNX__model.onnx'
    assert entry_url(m, old, 'file:///mirror') == 'file:///mirror/A__ONNX__model.onnx'
    legacy = {'base_url': 'https://x/models-v7'}          # no release fields at all
    assert entry_url(legacy, {'asset': 'b'}) == 'https://x/models-v7/b'


def test_place_entries_carries_unchanged_files_and_addresses_new_ones():
    builder = _builder()
    previous = {'models_version': 'v8', 'files': [
        {'path': 'A/ONNX/model.onnx', 'sha256': 'a' * 64, 'asset': 'A__ONNX__model.onnx', 'release': 'models-v4'},
        {'path': 'B/ONNX/model.onnx', 'sha256': 'b' * 64, 'asset': 'B__ONNX__model.onnx', 'release': 'models-v8'},
    ]}
    entries = [
        {'path': 'A/ONNX/model.onnx', 'sha256': 'a' * 64, 'asset': 'x'},   # unchanged
        {'path': 'B/ONNX/model.onnx', 'sha256': 'c' * 64, 'asset': 'x'},   # retrained
        {'path': 'C/ONNX/model.json', 'sha256': 'd' * 64, 'asset': 'x'},   # new network
    ]
    builder.place_entries(entries, previous, 'v9', flatten=True)
    assert entries[0]['release'] == 'models-v4' and entries[0]['asset'] == 'A__ONNX__model.onnx'
    assert entries[1]['release'] == 'models-v9' and entries[1]['asset'] == f'B__ONNX__model.{"c" * 12}.onnx'
    assert entries[2]['release'] == 'models-v9' and entries[2]['asset'] == f'C__ONNX__model.{"d" * 12}.json'
    assert builder.network_versions(entries) == {'A': 'v4', 'B': 'v9', 'C': 'v9'}


def test_converting_an_old_manifest_finds_each_file_in_its_first_release():
    builder = _builder()
    previous = {'models_version': 'v8', 'files': [
        {'path': 'A/ONNX/model.onnx', 'sha256': 'a' * 64, 'asset': 'A__ONNX__model.onnx'}]}
    entries = [{'path': 'A/ONNX/model.onnx', 'sha256': 'a' * 64, 'asset': 'x'}]
    history = lambda: {('A/ONNX/model.onnx', 'a' * 64): ('models-v5', 'A__ONNX__model.onnx')}
    builder.place_entries(entries, previous, 'v8', flatten=True, history=history)
    assert entries[0]['release'] == 'models-v5'
