#!/usr/bin/env python3
"""Refuse to publish anything the project has declared must not be here.

Two lists, one check. They are stored apart because they live differently:

* `publication/withheld.json` - material that WAS here and was taken out by an
  owner decision (photographs of real documents and text recognised from them,
  withdrawn by release 4.4.1). It grows with every such decision.
* `publication/not-exported.json` - paths that never belonged here at all: the
  personal research workspaces, the assistant's own files, the FMS dictionary.
  Constant; it changes only when the mirror filter changes.

Merged into one file they would inherit the worst of both - the constant half
stops being re-read ("obviously still true"), and a new entry in the growing
half gets lost among hundreds. Split, they keep their own rhythms. What must NOT
be split is the check: this tool reads the union, and the union is assembled by
machine, never by a person remembering to run two commands.

The two halves are also checked differently, and that difference is the point:

* `internal` paths must be absent from the working tree AND from the whole
  object graph. If one ever appears in history, it leaked - even if a later
  commit deleted it again.
* `personal-data` paths must be absent from the PRESENT: the working tree and
  the tip of every branch. History is deliberately not rewritten (decision 20),
  so old commits and tags still contain them, and that is not a violation.

A checker that treated both the same would either miss a leak or refuse a lawful
transfer.

Why the lists live here, inside the repository they guard: absence is the one
fact a repository cannot state about itself. "This file is missing because it
was withdrawn" and "this file is missing because nobody has copied it over yet"
look identical from both sides of a mirror. Only a declaration tells them apart,
and a declaration that stays in someone's notes does not travel with a clone.

Stdlib only, like `scripts/fetch_models.py`: this runs in release procedures and
in CI, before anything is installed.

Usage:

    python scripts/check_withheld.py                 # this repository
    python scripts/check_withheld.py --scope all     # including remote branches
    python scripts/check_withheld.py --tree DIR      # a plain directory
    python scripts/check_withheld.py --explain       # what to do about a refusal
    python scripts/check_withheld.py --allow-released=58,61   # in a pipeline

`--allow-released` takes a list of decision numbers rather than being a yes/no
switch, and that is the whole of its value: a boolean written once into a
pipeline config stops being re-read, which is the same fading this check exists
to prevent, one level up. Naming the decisions makes the NEXT release turn the
run red again, so somebody has to look at it and say yes by number.

`--tree` exists for the rebuild-from-snapshot route (`git archive` produces a
directory, not a repository), which is how 117 photographs of real documents
were published in the first place. A guard that only understood repositories
would not be watching that door.

Exit codes: 0 clean, 1 something declared is present, 2 the tool could not run.
Two is not a pass.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIST_DIR = "publication"
LIST_FILES = ("withheld.json", "not-exported.json")
INSTRUCTIONS = "publication/README.md"

REFUSAL_ADVICE = """
What to do now - read this before "fixing" anything:

  * If a transfer brought these paths in, DROP THEM from the transfer. They are
    kept out on purpose; the transfer is what is wrong, not the list.
  * DO NOT restore missing files to make a patch apply. A patch that fails with
    "No such file or directory" on a listed path is the guard working, not a
    broken checkout. Restoring the file produces exactly the outcome this tool
    exists to prevent.
  * DO NOT delete an entry to make this command green. Entries are retired only
    by a decision of the repository owner, and retiring one means moving it to
    "retired" with that decision recorded - never deleting it.

Full instructions: {instructions}
""".strip()


class GuardError(Exception):
    """The tool could not perform the check - as opposed to: found a violation."""


def git(*args: str, cwd: Path = REPO) -> list[str]:
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise GuardError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def digest(entries: list[dict]) -> str:
    """Digest of what the list SAYS, not of the file's bytes.

    It covers the match kind together with the path, because a lost asterisk
    (`Melnikov/*` becoming `Melnikov`) leaves the entry count unchanged while
    silently opening a whole directory. The guard permits everything it has not
    been told about, so damage to the list always opens it more quietly than it
    closes it - the list needs a check of its own, not only the tree does.
    """
    body = "".join(f"{e['match']}\t{e['path']}\n"
                   for e in sorted(entries, key=lambda e: (e["match"], e["path"])))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_one(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        raise GuardError(
            f"a declaration file is missing: {path}\n"
            "Without it this command cannot say anything, and a missing list "
            "must never read as a clean result.")
    except json.JSONDecodeError as exc:
        raise GuardError(f"{path} is not valid JSON: {exc}")

    entries = doc.get("entries", [])
    if not entries:
        raise GuardError(f"{path} is empty - refusing to report a clean result "
                         "from an empty list.")
    for e in doc.get("retired", []):
        if not str(e.get("decision") or "").strip():
            raise GuardError(
                f"{path}: retired entry without a decision: {e.get('path')}\n"
                "Retiring a path means the owner released it, and a release "
                "without a recorded decision is indistinguishable from someone "
                "quietly making the guard green.")
        if not str(e.get("retired_on") or "").strip():
            raise GuardError(
                f"{path}: retired entry without a date: {e.get('path')}\n"
                "When a path was released is part of the release.")
    declared = doc.get("integrity", {})
    if declared.get("entries") != len(entries) or declared.get("sha256") != digest(entries):
        raise GuardError(
            f"{path} does not match its own integrity record.\n"
            f"  declared: {declared.get('entries')} entries, "
            f"{str(declared.get('sha256'))[:12]}...\n"
            f"  actual:   {len(entries)} entries, {digest(entries)[:12]}...\n"
            "Either the list was edited without updating the record, or it was "
            "damaged in transit. Both need a person to look, not a rerun.")
    return doc


def load_lists(list_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    docs: dict[str, dict] = {}
    union: list[dict] = []
    for name in LIST_FILES:
        doc = load_one(list_dir / name)
        docs[name] = doc
        union.extend(doc["entries"])
    return union, docs


def check_list_edits(docs: dict[str, dict]) -> list[str]:
    """Did somebody edit the lists to make this command pass?

    Two ways to silence a guard by editing its own list, and both are checked
    against the committed version, because the committed version is the one a
    person reviewed.

    * A path DISAPPEARS from the file entirely.
    * A path MOVES from "entries" to "retired". Retiring means "the owner
      released this path, it may come back" - so a retired path is no longer
      refused, and moving one during a publish run silences the guard exactly
      where it should speak. An acceptance probe found this: retire the entry,
      recompute the integrity digest honestly, drop the file in, and all three
      defences missed at once.

    Hence the rule this enforces: **a release is its own commit.** Retiring is a
    decision of the repository owner and belongs in a change somebody reviews,
    never in the working copy of the person publishing right now. A retirement
    already committed is fine - it was reviewed - and gets reported at every run
    instead of vanishing (see `released_but_present`).
    """
    problems: list[str] = []
    for name, doc in docs.items():
        try:
            committed_raw = git("show", f"HEAD:{LIST_DIR}/{name}")
        except GuardError:
            continue  # not committed yet - nothing to compare against
        try:
            committed = json.loads("\n".join(committed_raw))
        except json.JSONDecodeError:
            continue
        def active(d: dict) -> set[str]:
            return {e["path"] for e in d.get("entries", [])}
        def retired(d: dict) -> set[str]:
            return {e["path"] for e in d.get("retired", [])}
        vanished = (active(committed) | retired(committed)) - (active(doc) | retired(doc))
        problems += [f"{name}: entry deleted outright: {p}" for p in sorted(vanished)]
        newly_retired = (active(committed) & retired(doc)) - retired(committed)
        problems += [f"{name}: retired in the working copy, not in a reviewed "
                     f"commit: {p}" for p in sorted(newly_retired)]
    return problems


def released_but_present(entries_retired: list[dict], paths: list[str]) -> list[tuple[dict, str]]:
    """Retired paths that are actually here - allowed, but never silent.

    A retirement is a decision, and decisions get re-read. Printing these at
    every run keeps a release visible long after the commit that made it has
    scrolled out of anyone's memory; silence would make "released once" and
    "never withheld" look the same, which is the confusion this whole tool
    exists to remove.
    """
    return hits(entries_retired, paths)


def hits(entries: list[dict], paths: list[str]) -> list[tuple[dict, str]]:
    """Every (entry, path) pair where a declared path is present.

    Matching ignores case, deliberately and on every platform. Two reasons, and
    the second is the one that bites:

    * Windows treats `1_CR_INTPASSPORT_2001.JPG` and `...jpg` as the same file
      while git records the case it was given - so a case-flipped copy of a
      withdrawn photograph is the same file to the operating system and a
      different one to a case-sensitive comparison. This list already contains
      entries in both cases; the guard must not depend on which one was typed.
    * `fnmatch.fnmatch` folds case on Windows and not on Linux, so a
      case-sensitive rule here would additionally mean the guard behaves
      differently in CI than on the machine that runs the release.

    The cost is over-blocking on a case-sensitive filesystem where two files
    differ only in case. That is the safe direction: a guard that stops one file
    too many is a nuisance, a guard that lets one through is the incident it
    exists to prevent.
    """
    found = []
    globs = [(e, e["path"].lower()) for e in entries if e.get("match") == "glob"]
    exact = {e["path"].lower(): e for e in entries if e.get("match") != "glob"}
    for path in paths:
        low = path.lower()
        entry = exact.get(low)
        if entry is not None:
            found.append((entry, path))
            continue
        for entry, pattern in globs:
            if fnmatch.fnmatchcase(low, pattern):
                found.append((entry, path))
                break
    return found


def repo_paths_now(cwd: Path, scope: str) -> list[str]:
    """The present: working tree, plus branch tips depending on scope.

    Branches only, never tags: tags are the past. `v4.4.0` still contains every
    photograph release 4.4.1 withdrew, and it is supposed to - the owner decided
    not to rewrite history. Counting tags here made the first run of this tool
    report all 616 withdrawn paths as present, which is worse than useless: the
    obvious way to silence that false alarm is to delete entries from the list,
    i.e. the one fix this guard forbids.

    `--others` matters as much as `--cached`: a file carried in by a patch or a
    copy sits in the working tree UNTRACKED, and a guard that only read the index
    would wave it through. The first negative control caught exactly that - the
    tool reported "clean" with a withheld path lying on disk.
    """
    paths = set(git("ls-files", "--cached", "--others", "--exclude-standard",
                    cwd=cwd))
    if scope == "worktree":
        return sorted(paths)
    globs = ["refs/heads"] if scope == "local" else ["refs/heads", "refs/remotes"]
    for ref in git("for-each-ref", "--format=%(refname)", *globs, cwd=cwd):
        try:
            paths.update(git("ls-tree", "-r", "--name-only", ref, cwd=cwd))
        except GuardError:
            continue
    return sorted(paths)


def repo_paths_ever(cwd: Path) -> list[str]:
    """Every path that ever existed in the object graph, all refs."""
    out = []
    for line in git("rev-list", "--all", "--objects", cwd=cwd):
        parts = line.split(" ", 1)
        if len(parts) == 2:
            out.append(parts[1])
    return out


def dir_paths(root: Path) -> list[str]:
    paths = []
    for base, dirs, files in os.walk(root):
        if ".git" in dirs:
            dirs.remove(".git")
        for name in files:
            paths.append(Path(base, name).relative_to(root).as_posix())
    return paths


def report(violations: list[tuple[str, dict, str]]) -> None:
    print(f"DECLARED PATHS PRESENT: {len(violations)}", file=sys.stderr)
    for where, entry, path in violations[:40]:
        decision = entry.get("decision")
        tail = f" (decision {decision})" if decision else ""
        print(f"  [{where}] {path}  <- {entry.get('reason', '?')}{tail}",
              file=sys.stderr)
    if len(violations) > 40:
        print(f"  ... and {len(violations) - 40} more", file=sys.stderr)
    print("", file=sys.stderr)
    print(REFUSAL_ADVICE.format(instructions=INSTRUCTIONS), file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Refuse to publish paths the project has declared must not "
                    "be in this repository.")
    ap.add_argument("--tree", metavar="DIR",
                    help="check a plain directory (e.g. `git archive` output) "
                         "instead of this repository")
    ap.add_argument("--lists", default=None, metavar="DIR",
                    help=f"where the declaration files live (default: {LIST_DIR}/)")
    ap.add_argument("--scope", choices=("worktree", "local", "all"),
                    default="local",
                    help="how far to look: the working tree, plus local branches "
                         "(default), or plus remote branches as well. The default "
                         "answers 'is what I am about to publish clean'; --scope "
                         "all answers 'is what is already published clean', which "
                         "is a different question and can fail on a branch you "
                         "are not touching.")
    ap.add_argument("--explain", action="store_true",
                    help="print what to do about a refusal, and exit")
    ap.add_argument("--allow-released", metavar="DECISIONS", default=None,
                    help="decisions whose released paths may be present, e.g. "
                         "--allow-released=58,61. Without it, a released path "
                         "found in the tree is a refusal. It takes a LIST rather "
                         "than being a yes/no switch on purpose: a boolean, once "
                         "written into a pipeline config, stops being re-read - "
                         "the same fading this whole check exists to prevent, one "
                         "level up. Naming the decisions means the next release "
                         "turns the run red again and needs a person to say yes "
                         "to it, by number. Do not 'simplify' it back.")
    args = ap.parse_args()

    if args.explain:
        print(REFUSAL_ADVICE.format(instructions=INSTRUCTIONS))
        return 0

    list_dir = Path(args.lists) if args.lists else REPO / LIST_DIR
    try:
        entries, docs = load_lists(list_dir)
        internal = [e for e in entries if e.get("reason") == "internal"]
        retired_entries = [e for doc in docs.values() for e in doc.get("retired", [])]
        released: list[tuple[dict, str]] = []
        unclaimed: list[str] = []
        violations: list[tuple[str, dict, str]] = []

        if args.tree:
            root = Path(args.tree).resolve()
            if not root.is_dir():
                raise GuardError(f"not a directory: {root}")
            paths = dir_paths(root)
            violations += [("tree", e, p) for e, p in hits(entries, paths)]
            released = released_but_present(retired_entries, paths)
            scope = f"directory {root} ({len(paths)} files)"
        else:
            dropped = check_list_edits(docs)
            if dropped:
                print("A DECLARATION LIST WAS EDITED AWAY FROM ITS COMMITTED "
                      "VERSION:", file=sys.stderr)
                for path in dropped[:20]:
                    print(f"  {path}", file=sys.stderr)
                print("", file=sys.stderr)
                print(REFUSAL_ADVICE.format(instructions=INSTRUCTIONS),
                      file=sys.stderr)
                return 1
            now = repo_paths_now(REPO, args.scope)
            violations += [("present", e, p) for e, p in hits(entries, now)]
            released = released_but_present(retired_entries, now)
            ever = repo_paths_ever(REPO)
            violations += [("in history", e, p) for e, p in hits(internal, ever)]
            looked_at = {"worktree": "the working tree only",
                         "local": "the working tree and local branches",
                         "all": "the working tree, local and remote branches"}
            scope = (f"repository {REPO}: {looked_at[args.scope]} "
                     f"({len(now)} paths), and the whole object graph for "
                     f"never-exported paths ({len(ever)} objects)")
            if args.scope != "all":
                scope += ("\n  NOT checked: remote branches. A green result here "
                          "says nothing about what is already published - for "
                          "that, run --scope all.")

        if released:
            # A released path is not a violation - the owner decided it may be
            # here. But it is also the one place where this whole mechanism rests
            # on a person reading something, and printing alone does not survive
            # CI: nobody reads the output of a green run. So a release passes only
            # when the run was told, by number, which decisions to expect.
            allowed = {d.strip() for d in (args.allow_released or "").split(",")
                       if d.strip()}
            present_decisions = {str(e.get("decision") or "?") for e, _ in released}
            unclaimed = sorted(present_decisions - allowed)
            print(f"RELEASED BY DECISION, AND PRESENT: {len(released)}")
            for entry, path in released[:20]:
                print(f"  {path}  <- released {entry.get('retired_on')} by "
                      f"decision {entry.get('decision')}")
            if len(released) > 20:
                print(f"  ... and {len(released) - 20} more")
            unused = sorted(allowed - present_decisions)
            if unused:
                print(f"  declared but not found here: decision(s) "
                      f"{', '.join(unused)} - harmless, but check you meant them")
            if unclaimed:
                print("")
                print("  NOT DECLARED BY THIS RUN: decision(s) "
                      f"{', '.join(unclaimed)}", file=sys.stderr)
                print("  Pass --allow-released=<decisions> to say these releases "
                      "are expected here.", file=sys.stderr)
                print("  Naming them is the point: the next release turns this "
                      "run red again, and somebody has to look at it.",
                      file=sys.stderr)
            else:
                print("  Declared by this run. Verify these are the decisions "
                      "you think they are.")
            print("")

        if violations:
            report(violations)
            return 1

        if unclaimed:
            # Deliberately after the violations report: a real violation is the
            # more urgent thing to read, and an undeclared release must not
            # shadow it.
            print("undeclared release - see above", file=sys.stderr)
            return 1

        print(f"clean: {scope}")
        for name, doc in docs.items():
            print(f"  {name}: {len(doc['entries'])} entries, integrity verified")
        return 0
    except GuardError as exc:
        print(f"CANNOT CHECK: {exc}", file=sys.stderr)
        print("A check that could not run is not a pass.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
