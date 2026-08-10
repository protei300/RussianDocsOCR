"""Lay out the model files under the flat asset names the manifest expects,
ready to upload to a GitHub release.

Release assets have no directories, so the manifest gives each file an `asset`
name with the path folded in ('Borders/ONNX/model.onnx' ->
'Borders__ONNX__model.onnx'). Renaming 29 files by hand is exactly the kind of
step that silently produces one wrong name and a download that 404s months
later, so it is done from the same manifest the downloader reads.

  python scripts/stage_release_assets.py                 # -> dist/models-<set>/
  python scripts/stage_release_assets.py --out /tmp/x    # elsewhere

Then, with the GitHub CLI. Note the tag names the WEIGHT SET, not the library
version — publishing v4 weights under a library tag is what put birth-certificate
models inside the v3.0.2 release, whose code predated them:

  gh release create models-<set> dist/models-<set>/* --repo protei300/RussianDocsOCR \
      --title "models <set>" --notes "..."

Files are hardlinked when the filesystem allows, so staging 225 MB costs no
extra disk.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / 'document_processing' / 'models.lock.json'
MODELS_DIR = REPO_ROOT / 'document_processing' / 'models'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None, help='staging directory (default dist/models-<set>)')
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit(f'no manifest at {MANIFEST} - run scripts/build_models_manifest.py')
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))

    out = Path(args.out) if args.out else REPO_ROOT / 'dist' / f'models-{manifest["models_version"]}'
    out.mkdir(parents=True, exist_ok=True)

    seen, total = set(), 0
    for entry in manifest['files']:
        src = MODELS_DIR / entry['path']
        if not src.exists():
            sys.exit(f'missing: {src}')
        if entry['asset'] in seen:
            sys.exit(f'duplicate asset name {entry["asset"]!r} - the flattening collided')
        seen.add(entry['asset'])

        dst = out / entry['asset']
        dst.unlink(missing_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        total += entry['size']

    print(f'{len(seen)} assets, {total / 1e6:.1f} MB -> {out}')
    tag = f'models-{manifest["models_version"]}'
    print(f'\nupload with:\n  gh release create {tag} {out}/* '
          f'--repo protei300/RussianDocsOCR --title "models {manifest["models_version"]}"')
    print('\nthen verify a clean fetch:')
    print('  python scripts/fetch_models.py --check')


if __name__ == '__main__':
    main()
