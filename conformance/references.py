"""Do the register's pointers lead anywhere in THIS tree?

## Why this exists

`validate()` checks the SHAPE of a declaration — its dates, its scope, the event
that retires it. Nothing checked whether the things it points AT exist. They stopped
existing quietly: the public register names three commits and a file that live only
in the corporate tree, carried over word for word when the register was ported. The
entry explaining to a port author why their difference is legitimate sends them for
proof to a commit their clone does not contain.

That is the half of the prose nobody meant to leave unchecked. A declaration's text
divides in two:

* **pointers and numbers** — a commit id, a file name: they have a subject, and one
  cheap command settles them;
* **judgements** — "the ports do not implement the ladder yet": no subject inside
  the tree, and machining them would produce a check that cannot fail honestly.

Only the first half is handled here, and only its pointers. Numbers are left alone
on purpose: `9 cases of 9, 21 differences` cannot be matched against a run without
guessing which run it meant.

## What this does NOT claim

A pointer that does not resolve here is **not** declared wrong. It is one of three:
a genuine dangling reference, a deliberate one to the sibling tree, or one to
something living outside the repository altogether — and this tree can tell none of
them apart, because it sees only itself. So the report says the only thing a run can
verify: *this pointer leads nowhere here*. Which of the three it is belongs to
whoever wrote the entry.

The third case was not foreseen; it appeared on the first run, and the docstring said
"two" until it did. That is the very drift this module checks for, in this module.

That is the same rule that governs the register itself: a statement is made by
whoever can check it.

Impure on purpose (subprocess, filesystem) — which is why it is not part of
`deviations.py`, whose promise of purity the ports rely on.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from conformance.paths import REPO

#: Fields scanned. `reason` and `basis` are where evidence is cited; `removed_when`
#: names a future EVENT and has nothing to resolve, and the scope fields are already
#: checked by `validate()` against the case list.
SCANNED_FIELDS = ("reason", "basis")

#: A hex run is only treated as a commit when it stands in a citing position - in
#: parentheses, or right after the word "commit". Without that, `value fingerprint
#: 60f0d944` reads as a commit and the check invents a defect: the entry that first
#: exercised this module carries exactly such a fingerprint.
_COMMIT = re.compile(r"\(([0-9a-f]{7,40})\)|commit\s+([0-9a-f]{7,40})\b")

#: Anything that looks like a repository path. Bare names are looked up by basename
#: as well, because entries cite files the way people speak of them.
_FILE = re.compile(r"\b[\w][\w./-]*\.(?:json|py|md)\b")

#: An author may say WHERE a pointer lives. Then it is not expected to resolve here,
#: and silence is correct - the alternative is a permanent complaint about something
#: legitimate, which trains the reader to skip the section.
#:
#: The list gained its third case the moment the check first ran: the file cited by
#: D-02 is neither dangling nor in the sibling tree - it lives in the working notes,
#: outside both repositories. A pointer can legitimately leave the repository, and a
#: check that knows only 'here' and 'the other tree' would call that a defect.
_ELSEWHERE = re.compile(r"(corporate|sibling tree|notes|outside (this|the) (tree|repository))"
                        r"[\s/]+[^,;]{0,40}$", re.IGNORECASE)


@dataclass(frozen=True)
class Dangling:
    entry_id: str
    field: str
    kind: str      # "commit" | "file"
    token: str

    def __str__(self) -> str:
        return f"{self.entry_id}: {self.kind} {self.token!r} (in {self.field}) does not resolve here"


def _commit_exists(token: str) -> bool:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "cat-file", "-t", token],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return True      # cannot ask git -> do not invent a finding
    return out.returncode == 0 and out.stdout.strip() == "commit"


def _file_exists(token: str) -> bool:
    if (REPO / token).is_file():
        return True
    name = Path(token).name
    # Bounded search: the places an entry could reasonably cite. A repo-wide rglob
    # would walk the models and the samples for every token.
    for folder in ("conformance", "docs", "service", "scripts", "tests"):
        root = REPO / folder
        if root.is_dir() and any(root.rglob(name)):
            return True
    return False


def dangling(deviations) -> list[Dangling]:
    """Pointers in the register that lead nowhere in this tree."""
    found: list[Dangling] = []
    for dev in deviations:
        for field in SCANNED_FIELDS:
            text = getattr(dev, field, "") or ""
            for match in _COMMIT.finditer(text):
                token = match.group(1) or match.group(2)
                if _ELSEWHERE.search(text[:match.start()]):
                    continue
                if not _commit_exists(token):
                    found.append(Dangling(dev.id, field, "commit", token))
            for match in _FILE.finditer(text):
                token = match.group(0)
                if _ELSEWHERE.search(text[:match.start()]):
                    continue
                if not _file_exists(token):
                    found.append(Dangling(dev.id, field, "file", token))
    return found


def render(found: list[Dangling]) -> list[str]:
    """The section for the report. A notice, never a refusal - see the module docstring."""
    if not found:
        return []
    lines = ["POINTERS THAT LEAD NOWHERE IN THIS TREE (not necessarily wrong):"]
    lines += [f"  {item}" for item in found]
    lines += [
        "  Each is one of three: a stale reference, a deliberate one to the sibling",
        "  tree, or one to something living outside the repository altogether.",
        "  This tree cannot tell them apart - it sees only itself. Whoever wrote the",
        "  entry can: say where it lives ('corporate <id>', 'notes/<file>'), and it",
        "  stops being reported. The register is NOT discarded over this: a",
        "  declaration with a bad citation still absorbs the difference it names.",
    ]
    return lines
