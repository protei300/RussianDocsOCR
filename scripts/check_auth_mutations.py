"""Reintroduce each fixed auth defect in turn and confirm its test goes red.

    python scripts/check_auth_mutations.py

A test that passes on the fixed code proves nothing on its own: the first
route-coverage test in this service passed on broken code too, and the first
concurrency test passed with its lock removed. So every test in
tests/service/test_auth_security.py is paired here with the exact regression it
exists to catch, the regression is applied to the working tree, the test is run,
and the file is restored — whatever happens, in a finally.

Run it after anything that touches service/api, service/core or
service/repositories, and especially after resolving a merge conflict in those
files: a conflict resolution is exactly where a fix disappears without a trace.
Exit status is non-zero if any regression goes uncaught — or if an anchor no
longer exists, which means the code moved and this table needs updating, not
that the check passed.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable

MUTATIONS = [
    ("viewer may delete a document",
     'service/api/documents.py',
     '_identity=Depends(require_api_or_operator)) -> Response:',
     '_identity=Depends(require_api_or_viewer)) -> Response:',
     'test_a_viewer_can_read_and_nothing_else'),
    ("viewer may mint an API key",
     'service/api/api_keys.py',
     'from service.api.deps import require_admin',
     'from service.api.deps import require_viewer as require_admin',
     'test_a_viewer_cannot_mint_an_api_key'),
    ("PIN mode honours a named-account token",
     'service/api/deps.py',
     '        if "uid" in claims:\n            return None\n',
     '',
     'test_pin_mode_refuses_a_token_issued_for_a_named_account'),
    ("the published default secret signs tokens",
     'service/core/auth.py',
     '    if configured and configured != DEFAULT_JWT_SECRET:\n        return configured',
     '    if configured:\n        return configured',
     'test_the_published_default_secret_cannot_sign_a_token'),
    ("a real ADMIN_PASSWORD is advertised",
     'service/api/auth.py',
     '    if settings.admin_password != DEFAULT_ADMIN_PASSWORD:\n        return None\n',
     '',
     'test_a_real_admin_password_is_never_published'),
    ("no write lock around user mutations",
     'service/repositories/users.py',
     '_WRITE = threading.RLock()',
     'import contextlib\n_WRITE = contextlib.nullcontext()',
     'test_concurrent_demotions_cannot_remove_every_administrator'),
    ("sign-in writes back its stale copy",
     'service/repositories/users.py',
     '        fresh = db.get_user(user.id)\n',
     '        fresh = user\n',
     'test_a_sign_in_does_not_undo_a_demotion_made_while_it_was_hashing'),
    ("the store hands out live objects",
     'service/core/database.py',
     '            return dataclasses.replace(user) if user is not None else None',
     '            return user',
     'test_reads_hand_out_copies_not_the_indexed_object'),
    ("lockout counted per account only",
     'service/core/auth.py',
     '        for key in (_throttle_key(identity, client), _throttle_key(_ANY_IDENTITY, client)):',
     '        for key in (_throttle_key(identity, client),):',
     'test_rotating_usernames_does_not_escape_the_lockout'),
    ("client address written to the audit log",
     'service/api/auth.py',
     'audit_repo.record(db, action="login_failed", actor=body.username.strip())',
     'audit_repo.record(db, action="login_failed", actor=body.username.strip(), detail=client)',
     'test_the_audit_log_holds_no_client_address'),
    ("any characters allowed in a username",
     'service/repositories/users.py',
     '    if not _USERNAME.match(name):',
     '    if False:',
     'test_usernames_that_could_impersonate_are_refused'),
]

results = []
for label, rel, old, new, test in MUTATIONS:
    path = ROOT / rel
    original = path.read_text(encoding='utf-8')
    if old not in original:
        results.append((label, 'ANCHOR NOT FOUND — the code moved; update this table'))
        continue
    path.write_text(original.replace(old, new, 1), encoding='utf-8')
    try:
        run = subprocess.run(
            [PY, '-m', 'pytest', 'tests/service/test_auth_security.py', '-q', '-p', 'no:warnings',
             '-k', test, '-x'], cwd=ROOT, capture_output=True, text=True, timeout=600)
        verdict = 'caught' if run.returncode != 0 else 'MISSED — test still passes'
    finally:
        path.write_text(original, encoding='utf-8')
    results.append((label, verdict))

width = max(len(l) for l, _ in results)
for label, verdict in results:
    print(f'  {label:<{width}}  {verdict}')
missed = [l for l, v in results if v != 'caught']
print(f'\n{len(results) - len(missed)}/{len(results)} regressions caught')
sys.exit(1 if missed else 0)
