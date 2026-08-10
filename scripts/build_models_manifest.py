"""Generate document_processing/models.lock.json - the checksum manifest that
scripts/fetch_models.py downloads against.

The file list comes from `git ls-files`, not from a directory walk: the models
tree also accumulates local experiment folders (Borders/ONNX_old,
TextFields/ONNX_legacy_backup, threshold sweeps) which are gitignored and must
never end up in the published set.

The manifest is also an integrity record. Two bugs this release came from an
artifact set nobody had pinned: a stale OpenVINO .ir that silently kept
pre-retrain weights, and a MaskFilter left at the previous checkpoint's value
when model.onnx was swapped. `fetch_models.py --check` turns both into a
one-second failure.

Usage:
  python scripts/build_models_manifest.py                     # refresh in place
  python scripts/build_models_manifest.py --base-url https://…/models/3.0.1/
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / 'document_processing' / 'models'
MANIFEST = REPO_ROOT / 'document_processing' / 'models.lock.json'
# Weights are published as assets of the PUBLIC repo - they are MIT like the
# code, and the corporate build pulls the same files, so no private channel is
# needed.
#
# THE TAG IS THE WEIGHTS' OWN, NOT THE LIBRARY'S. Both were the library version
# once, and it produced exactly the drift this manifest exists to stop: v3.0.2
# was tagged on 1 Aug, Borders v4 landed on 4 Aug without a version bump, and the
# v4 weights were published under v3.0.2 - a code tag whose models knew nothing
# about birth certificates. Weights change on their own cadence, so they get
# their own identity: retraining no longer needs a code release, and a code
# release no longer needs 225 MB re-uploaded.
DEFAULT_BASE_URL = 'https://github.com/protei300/RussianDocsOCR/releases/download/models-{models_version}/'


def asset_name(rel_path: str, flatten: bool) -> str:
    """Remote object name for a file.

    Release assets are a flat namespace, so 'Borders/ONNX/model.onnx' is
    published as 'Borders__ONNX__model.onnx'. Object storage keeps directories,
    so there --no-flatten leaves the path alone. Storing the name per file means
    the downloader never has to know which of the two it is talking to.
    """
    return rel_path.replace('/', '__') if flatten else rel_path


# NOTE: deliberately no reference to document_processing.__version__ here. Taking
# the weight-set name from the library version is what published v4 weights under
# the v3.0.2 tag; the two are independent axes and the manifest names only its own.


def git_tracked_model_files() -> list:
    """Bootstrap source, from back when the weights were still in git."""
    out = subprocess.run(
        ['git', 'ls-files', '--', str(MODELS_DIR.relative_to(REPO_ROOT).as_posix())],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return sorted(REPO_ROOT / p for p in out)


def pinned_files(previous: dict, extra: list) -> list:
    """Which files make up the set.

    Once the weights left git, `git ls-files` returns nothing here — so the
    manifest itself is what defines membership, and this only refreshes sizes
    and checksums for the paths already in it. A new model is an explicit act
    (--add), never a side effect of something appearing on disk: the models tree
    also holds local A/B folders (Borders/ONNX_old, TextFields/ONNX_legacy_backup,
    threshold sweeps) that must never reach the published set.
    """
    paths = [MODELS_DIR / e['path'] for e in previous.get('files', [])]
    if not paths:
        paths = git_tracked_model_files()
    for item in extra:
        candidate = (MODELS_DIR / item).resolve()
        if not str(candidate).startswith(str(MODELS_DIR.resolve())):
            sys.exit(f'--add outside the models directory: {item}')
        if candidate not in {p.resolve() for p in paths}:
            paths.append(candidate)
    return sorted(set(paths))


def report_strays(pinned: list) -> None:
    """Name model.json files on disk the manifest does not cover.

    Not an error — these are usually deliberate local variants — but a model
    that was added and never pinned would otherwise ship only on the machine
    that made it, which is the drift this manifest exists to prevent.
    """
    covered = {p.resolve() for p in pinned}
    strays = [p for p in MODELS_DIR.rglob('model.json') if p.resolve() not in covered]
    if strays:
        print('\nnot in the manifest (add with --add if they should be published):')
        for p in sorted(strays):
            print(f'  ? {p.relative_to(MODELS_DIR).as_posix()}')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-url', default=None,
                    help='where the files are published; keep the trailing slash. '
                         '"{models_version}" in it is substituted. Defaults to the value '
                         'already in the manifest, else the public repo release URL.')
    ap.add_argument('--models-version', default=None,
                    help='name of THIS weight set, e.g. v4. Independent of the library '
                         'version. Defaults to the value already in the manifest.')
    ap.add_argument('--no-flatten', action='store_true',
                    help='keep directory structure in asset names (object storage); '
                         'default flattens them for GitHub release assets')
    ap.add_argument('--add', nargs='*', default=[], metavar='PATH',
                    help='paths (relative to document_processing/models) to add to the set')
    args = ap.parse_args()

    previous = {}
    if MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text(encoding='utf8'))

    models_version = args.models_version or previous.get('models_version')
    if not models_version:
        sys.exit('--models-version is required the first time (e.g. --models-version v4); '
                 'it names the weight set and is NOT the library version')
    # An explicit --base-url wins. Otherwise the previous one is reused ONLY if it
    # still names this weight set: carrying it over blindly across a rename is how
    # a manifest ends up pointing at the tag of a different set.
    if args.base_url:
        base_url = args.base_url
    elif previous.get('base_url') and models_version in previous['base_url']:
        base_url = previous['base_url']
    else:
        base_url = DEFAULT_BASE_URL
    base_url = base_url.replace('{models_version}', models_version)
    if not base_url.endswith('/'):
        base_url += '/'

    files = pinned_files(previous, args.add)
    if not files:
        sys.exit('nothing to pin: no existing manifest and no tracked files under '
                 'document_processing/models')

    entries, total = [], 0
    for path in files:
        if not path.exists():
            sys.exit(f'tracked but missing on disk: {path}')
        size = path.stat().st_size
        total += size
        rel = path.relative_to(MODELS_DIR).as_posix()
        entries.append({
            'path': rel,
            'asset': asset_name(rel, flatten=not args.no_flatten),
            'size': size,
            'sha256': sha256(path),
        })
        print(f'  {rel:<52} {size / 1e6:8.2f} MB')

    manifest = {
        'models_version': models_version,
        'base_url': base_url,
        'files': entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')

    old = {e['path']: e['sha256'] for e in previous.get('files', [])}
    new = {e['path']: e['sha256'] for e in entries}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(p for p in set(old) & set(new) if old[p] != new[p])

    print(f'\n{len(entries)} files, {total / 1e6:.1f} MB -> {MANIFEST.relative_to(REPO_ROOT)}')
    print(f'model set {models_version}  base_url {base_url}')
    if previous:
        print(f'added {len(added)}, changed {len(changed)}, removed {len(removed)}')
        for p in added:
            print(f'  + {p}')
        for p in changed:
            print(f'  ~ {p}')
        for p in removed:
            print(f'  - {p}')
    report_strays(files)


if __name__ == '__main__':
    main()
