"""Download the model weights listed in document_processing/models.lock.json.

The weights are not in git; they are published as assets of the public repo's
release, pinned by the tag in the manifest's base_url. Run this once after
cloning (or pip-installing), and again after pulling a release that changed
models.

  python scripts/fetch_models.py            # from a repo checkout
  rdocs-fetch-models                        # from a pip install
  ... --check                               # verify only, exit 1 on mismatch
  ... --force                               # re-download everything

This file lives inside the package so that a pip-installed copy can fetch the
weights into its own ``document_processing/models`` next to the code, exactly
as a repo checkout does: the target is derived from this file's location
(``site-packages`` and the repo root are the same case). ``scripts/fetch_models.py``
stays as a thin path-loading wrapper because the Docker model stage and CI call
it before any dependencies are installed - it must not import the package.

Only files whose checksum does not match are transferred, so a release that
changed one model.json costs a kilobyte rather than 215 MB.

Environment:
  RDOCS_MODELS_ROOT  directory CONTAINING document_processing/models - the same
                     variable the Go/.NET/Kotlin ports use to locate them.
  RDOCS_MODELS_URL   base URL override. This is the air-gapped path: mirror the
                     assets internally, point this at them, no code changes.

Standard library only, on purpose: this runs before dependencies are installed
and inside the Docker model stage, neither of which has boto3 or requests.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Repo checkout: <root>/document_processing/fetch_models.py -> <root>.
# Installed package: site-packages/document_processing/fetch_models.py -> site-packages.
REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / 'document_processing' / 'models.lock.json'
TIMEOUT = 60
RETRIES = 3


def models_dir() -> Path:
    root = os.environ.get('RDOCS_MODELS_ROOT')
    base = Path(root) if root else REPO_ROOT
    return base / 'document_processing' / 'models'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def state_of(entry: dict, dest: Path) -> str:
    """'ok' | 'missing' | 'size' | 'corrupt' - why a file needs fetching."""
    if not dest.exists():
        return 'missing'
    if dest.stat().st_size != entry['size']:
        return 'size'
    return 'ok' if sha256(dest) == entry['sha256'] else 'corrupt'


def download(url: str, dest: Path, expected_sha: str) -> None:
    """Fetch to a temporary neighbour, verify, then move into place.

    Never writes a partial file to the final name: a half-downloaded model
    surfaces as a baffling inference error rather than an obvious failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'rdocs-fetch-models'})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response, \
                    tmp.open('wb') as out:
                shutil.copyfileobj(response, out, 1 << 20)
            got = sha256(tmp)
            if got != expected_sha:
                raise ValueError(f'checksum mismatch: expected {expected_sha[:12]}, got {got[:12]}')
            tmp.replace(dest)
            return
        except (urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            if attempt < RETRIES:
                print(f'    retry {attempt}/{RETRIES - 1} after {exc}', flush=True)
    raise RuntimeError(f'{url}: {last}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='verify only, download nothing')
    ap.add_argument('--force', action='store_true', help='re-download even if the checksum matches')
    ap.add_argument('--jobs', type=int, default=4, help='parallel downloads (default 4)')
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit(f'no manifest at {MANIFEST} - run scripts/build_models_manifest.py')
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    # `or`, not a get() default: the Docker model stage exports the variable as
    # an EMPTY string via `ENV RDOCS_MODELS_URL=${RDOCS_MODELS_URL}`, and an empty
    # base URL turns every asset into "unknown url type: '/<asset>'".
    base_url = os.environ.get('RDOCS_MODELS_URL') or manifest['base_url']
    if not base_url.endswith('/'):
        base_url += '/'

    target = models_dir()
    print(f'model set {manifest["models_version"]} -> {target}')
    print(f'source {base_url}\n')

    todo, ok = [], 0
    for entry in manifest['files']:
        dest = target / entry['path']
        state = 'forced' if args.force else state_of(entry, dest)
        if state == 'ok':
            ok += 1
            continue
        todo.append((entry, dest, state))
        print(f'  {state:<8} {entry["path"]}')

    if not todo:
        print(f'all {ok} files present and verified')
        return 0

    if args.check:
        print(f'\n{len(todo)} file(s) missing or mismatched, {ok} ok')
        print('run again without --check to fix '
              '(rdocs-fetch-models or scripts/fetch_models.py)')
        return 1

    total = sum(e['size'] for e, _, _ in todo)
    print(f'\nfetching {len(todo)} file(s), {total / 1e6:.1f} MB')

    failures = []

    def fetch(item):
        entry, dest, _ = item
        try:
            download(base_url + entry['asset'], dest, entry['sha256'])
            print(f'  ok  {entry["path"]}', flush=True)
        except Exception as exc:  # noqa: BLE001 - reported below, not swallowed
            failures.append((entry['path'], exc))
            print(f'  FAIL {entry["path"]}: {exc}', flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(fetch, todo))

    if failures:
        print(f'\n{len(failures)} file(s) failed:')
        for path, exc in failures:
            print(f'  {path}: {exc}')
        return 1
    print(f'\ndone: {len(todo)} fetched, {ok} already present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
