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


def _find_stray_model_jsons(models_dir: Path, manifest: dict) -> list[Path]:
    pinned = {(models_dir / e['path']).resolve() for e in manifest['files']}
    return sorted(p for p in models_dir.rglob('model.json') if p.resolve() not in pinned)


def test_no_untracked_model_artifacts_shadow_the_set(manifest):
    """A model directory the manifest does not know about is a deployment risk.

    In CI (or when STRICT_MODELS_MANIFEST=1) any stray model.json is a hard failure
    so a stale or unpinned artifact can never slip past release checks. Locally,
    developers may keep untracked A/B folders alongside the pinned set; those are
    surfaced as a warning rather than silently skipped.
    """
    import os
    import warnings

    strays = _find_stray_model_jsons(MODELS_DIR, manifest)
    if strays:
        names = ', '.join(str(p.relative_to(MODELS_DIR)) for p in strays)
        if os.environ.get('CI') or os.environ.get('STRICT_MODELS_MANIFEST') == '1':
            pytest.fail(f'untracked model.json files outside models.lock.json: {names}')
        warnings.warn(f'local untracked model.json present (strictly forbidden in CI): {names}',
                      UserWarning, stacklevel=1)


def test_stray_model_json_detector_catches_unpinned_artifact(tmp_path):
    """Negative control: proves the stray-artifact detector actually flags unpinned models."""
    pinned_cfg = tmp_path / 'Blur' / 'ONNX' / 'model.json'
    pinned_cfg.parent.mkdir(parents=True)
    pinned_cfg.write_text('{}', encoding='utf-8')
    stray_cfg = tmp_path / 'OCR' / 'stale_v0' / 'ONNX' / 'model.json'
    stray_cfg.parent.mkdir(parents=True)
    stray_cfg.write_text('{}', encoding='utf-8')

    fake_manifest = {'files': [{'path': 'Blur/ONNX/model.json'}]}
    assert _find_stray_model_jsons(tmp_path, fake_manifest) == [stray_cfg]


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


def test_fetch_models_safe_dest_and_http_exception_retry(tmp_path, monkeypatch):
    import http.client
    import urllib.error
    from document_processing import fetch_models

    with pytest.raises(ValueError, match='escapes'):
        fetch_models._safe_dest(tmp_path, '../outside.onnx')
    with pytest.raises(ValueError, match='unsafe'):
        fetch_models._safe_dest(tmp_path, str(tmp_path / 'abs.onnx'))
    with pytest.raises(ValueError, match='unsupported URL scheme'):
        fetch_models.download('ftp://example.com/a.onnx', tmp_path / 'a.onnx', '0' * 64)

    payload = b'hello-model'
    expected_sha = hashlib.sha256(payload).hexdigest()
    attempts = {'n': 0}

    class FakeResp:
        def __init__(self, fail_mid: bool):
            self._fail = fail_mid
            self._done = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, _n=-1):
            if self._fail:
                raise http.client.IncompleteRead(b'hel', len(payload) - 3)
            if self._done:
                return b''
            self._done = True
            return payload

    def fake_urlopen(req, timeout=60):
        attempts['n'] += 1
        return FakeResp(fail_mid=(attempts['n'] == 1))

    monkeypatch.setattr(fetch_models.urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(fetch_models.time, 'sleep', lambda _: None)
    dest = tmp_path / 'ok.onnx'
    fetch_models.download('https://example.com/ok.onnx', dest, expected_sha, expected_size=len(payload))
    assert attempts['n'] == 2
    assert dest.read_bytes() == payload

    calls_404 = {'n': 0}

    def fake_404(req, timeout=60):
        calls_404['n'] += 1
        raise urllib.error.HTTPError(req.full_url, 404, 'Not Found', {}, None)

    monkeypatch.setattr(fetch_models.urllib.request, 'urlopen', fake_404)
    with pytest.raises(RuntimeError):
        fetch_models.download('https://example.com/missing.onnx', tmp_path / 'missing.onnx', expected_sha)
    assert calls_404['n'] == 1
