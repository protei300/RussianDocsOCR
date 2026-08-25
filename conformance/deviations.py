"""Declared deviations: differences that are known, explained and expected.

## Why this exists

A gate answers one question — did it match — and that answer is only useful while
"no" means "something broke". Two legitimate reds were standing at once (ports do not
implement the MRZ re-read ladder; the view model is about to gain a `normalized`
property), each announced, each explained. Nothing was wrong with either. What was
wrong is that reading the gate had become an act of memory: subtract the known reds,
and whatever is left is the real one.

That is the mirror image of a check that cannot fail. A check that always fails
carries no information either — and it decays quietly, because every single exception
looks reasonable on the day it is added. Two today, five in a month, and nobody
remembers which ones still apply.

So the subtraction is done by the machine, from a list that has to be maintained:

* every entry names its scope, its reason, its basis, and the event that will retire
  it — an exception with no exit condition is a permanent loosening in disguise;
* an entry that matches nothing in a run where it applies is reported as STALE, not
  silently kept: a list nobody prunes becomes the very thing it was meant to prevent —
  a check that cannot fail;
* the verdict gains a third outcome. `CLEAN` / `DECLARED` / `UNDECLARED`. Only the
  last is a red, and declared entries never mask an undeclared difference: the two are
  classified per difference, not per run.

This is deliberately NOT a tolerance. Tolerances say "this much numeric noise is
fine, always"; a deviation says "this exact stage of this exact case differs for this
written reason until this event". The spec already asked for the same shape on the
device axis (`spec/tolerances.md`, `gpu-deviations.json`); this applies it to the
second axis.

Pure: json, re and datetime only. The checker imports this.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from conformance.paths import REPO

#: Declared deviations live next to the cases, not inside them: an entry usually spans
#: several cases, and a per-case file would have to repeat the reason.
DEVIATIONS_FILE = REPO / "conformance" / "deviations.json"

#: A declared deviation older than this is not automatically wrong, but it means the
#: gate is being held up by memory rather than by itself. Reported, and escalated to
#: the arbiter rather than decided here.
STALE_AFTER_DAYS = 14


@dataclass(frozen=True)
class Deviation:
    """One declared difference. Every field is required for a reason.

    `ports`, `cases`, `stages` are the scope; `paths` narrows further to specific
    JSON paths when a stage differs only in part (the MRZ line inside a view model,
    not the whole view model).
    """

    id: str
    ports: tuple[str, ...]
    cases: tuple[str, ...]
    stages: tuple[str, ...]
    paths: tuple[str, ...]
    reason: str
    basis: str          # user decision number, or bus frame id
    declared: str       # ISO date
    owner: str
    removed_when: str   # the EVENT that retires it, never a date
    #: Two kinds of entry, and the distinction must reach the report, not only the
    #: file. "deviation": the case IS run and a difference against the golden is
    #: expected (the MRZ ladder, the normalized date). "withdrawal": the sample was
    #: taken out, so the case is NOT run at all — there is nothing to diff. Writing a
    #: withdrawal as a deviation would say "checked, with an expected difference" where
    #: the truth is "not checked, no sample" — which makes the suite look more thorough
    #: than it is, the exact failure this whole mechanism exists to prevent.
    kind: str = "deviation"

    @property
    def is_withdrawal(self) -> bool:
        return self.kind == "withdrawal"

    def applies_to_port(self, port: str) -> bool:
        return "*" in self.ports or port in self.ports

    def covers(self, case: str, stage: str, path: str) -> bool:
        if not _match_any(self.cases, case):
            return False
        if not _match_any(self.stages, stage):
            return False
        if not self.paths:
            return True
        return any(re.search(pattern, path) for pattern in self.paths)

    def age_days(self, today: date | None = None) -> int | None:
        """Age in days, or None when the date cannot be read.

        None rather than a number, because the first version used -1 as the "cannot
        read" sentinel — and an entry declared one day in the FUTURE produces exactly
        -1 legitimately. `validate()` then called a perfectly good date unparseable and
        discarded the whole list, which silently switched the mechanism off. Caught by
        its own validator on the second entry ever written; the lesson is not about
        dates but about sentinels that collide with real values.
        """
        today = today or datetime.now(timezone.utc).date()
        try:
            return (today - date.fromisoformat(self.declared)).days
        except ValueError:
            return None


def _match_any(patterns: tuple[str, ...], value: str) -> bool:
    return any(p == "*" or p == value or re.fullmatch(p, value) for p in patterns)


def withdrawal_for(case_slug: str, deviations: list[Deviation]) -> Deviation | None:
    """The declared withdrawal covering this case, or None.

    Matched by case alone: a withdrawn case is never run, so there is no stage and no
    path to narrow by — the statement is about the case as a whole.

    None is the answer that matters most. A case whose sample is absent and whose
    absence is NOT declared here is a red, exactly like an undeclared difference: the
    suite must not quietly shrink. This is what replaced the free-text `disabled` flag
    in the seed manifest, which could say anything and named neither owner nor the
    event that ends it.
    """
    for dev in deviations:
        if dev.is_withdrawal and _match_any(dev.cases, case_slug):
            return dev
    return None


@dataclass
class Classification:
    """What the run's differences turned out to be."""

    declared: dict[str, list[str]] = field(default_factory=dict)   # id -> ["case/stage: path"]
    undeclared: list[str] = field(default_factory=list)            # "case/stage: path"
    applicable: list[Deviation] = field(default_factory=list)
    stale: list[Deviation] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.declared and not self.undeclared

    @property
    def red(self) -> bool:
        return bool(self.undeclared)

    def verdict(self) -> str:
        if self.undeclared:
            return "UNDECLARED"
        if self.declared:
            return "DECLARED"
        return "CLEAN"


def load(path: Path | None = None) -> list[Deviation]:
    """Read the declared list. A missing file means no deviations, not an error:
    the ordinary state of a healthy project is an empty list."""
    path = path or DEVIATIONS_FILE
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for entry in raw.get("deviations", []):
        out.append(Deviation(
            id=entry["id"],
            ports=tuple(entry.get("ports") or ("*",)),
            cases=tuple(entry.get("cases") or ("*",)),
            stages=tuple(entry.get("stages") or ("*",)),
            paths=tuple(entry.get("paths") or ()),
            reason=entry["reason"],
            basis=entry["basis"],
            declared=entry["declared"],
            owner=entry["owner"],
            removed_when=entry["removed_when"],
            kind=entry.get("kind", "deviation"),
        ))
    return out


def validate(deviations: list[Deviation]) -> list[str]:
    """Complaints about the list itself, so a malformed entry cannot silently
    absorb differences it was never meant to cover."""
    problems = []
    seen = set()
    for dev in deviations:
        if dev.id in seen:
            problems.append(f"{dev.id}: duplicate id")
        seen.add(dev.id)
        age = dev.age_days()
        if age is None:
            problems.append(f"{dev.id}: unparseable declared date {dev.declared!r}")
        elif age < 0:
            # Distinct message on purpose: a future date is readable but wrong, and
            # saying "unparseable" about it sends the reader after the wrong thing.
            problems.append(f"{dev.id}: declared date {dev.declared!r} is in the future "
                            f"(by {-age} day(s)) — the age metric would lie")
        if not dev.removed_when.strip():
            problems.append(f"{dev.id}: no retiring event — an exception without an "
                            f"exit condition is a permanent loosening")
        if dev.kind not in ("deviation", "withdrawal"):
            problems.append(f"{dev.id}: unknown kind {dev.kind!r} "
                            f"(expected 'deviation' or 'withdrawal')")
        if dev.is_withdrawal:
            # A withdrawal says "this case is not run because its sample was taken
            # out". It must name the cases it withdraws, and it must not use "*":
            # a wildcard withdrawal would silently excuse any case that happens to
            # lack a sample, which is the undeclared-withdrawal red we want to keep.
            if not dev.cases or "*" in dev.cases:
                problems.append(f"{dev.id}: withdrawal must name specific cases, "
                                f"never '*' — a wildcard would excuse any missing sample")
        elif "*" in dev.cases and "*" in dev.stages and not dev.paths:
            problems.append(f"{dev.id}: covers every case and every stage — that is a "
                            f"general loosening, which this mechanism must never be")
    return problems


def classify(run, deviations: list[Deviation], port: str) -> Classification:
    """Split a run's differences into declared and undeclared.

    Per DIFFERENCE, not per run: one declared entry must not turn a case green when
    something else in the same stage also moved.
    """
    result = Classification()
    result.applicable = [d for d in deviations if d.applies_to_port(port)]
    matched_ids = set()

    for case in run.cases:
        for stage in case.stages:
            if stage.ok or stage.skipped:
                continue
            for diff in stage.diffs:
                where = f"{case.slug}/{stage.stage}: {diff.path}"
                covering = next((d for d in result.applicable
                                 if d.covers(case.slug, stage.stage, diff.path)), None)
                if covering is None:
                    result.undeclared.append(where)
                else:
                    matched_ids.add(covering.id)
                    result.declared.setdefault(covering.id, []).append(where)

    # An entry that applies here and matched nothing has outlived its difference.
    # Reported rather than dropped: removing it is a decision, and a silent drop would
    # hide that the gate got better.
    result.stale = [d for d in result.applicable if d.id not in matched_ids]
    return result


def render(cls: Classification, today: date | None = None) -> list[str]:
    """The section that goes into every gate report, including the health metric."""
    lines: list[str] = []
    if cls.declared:
        lines.append("declared deviations matched (not a failure):")
        for dev_id, places in sorted(cls.declared.items()):
            lines.append(f"  {dev_id}: {len(places)} difference(s)")
            for place in places[:6]:
                lines.append(f"      {place}")
            if len(places) > 6:
                lines.append(f"      ... {len(places) - 6} more")
    if cls.undeclared:
        lines.append("UNDECLARED differences — this is the red:")
        for place in cls.undeclared[:20]:
            lines.append(f"  {place}")
        if len(cls.undeclared) > 20:
            lines.append(f"  ... {len(cls.undeclared) - 20} more")
    if cls.stale:
        lines.append("STALE declarations — matched nothing here, propose removal:")
        for dev in cls.stale:
            lines.append(f"  {dev.id} (declared {dev.declared}, retires when: {dev.removed_when})")

    # The metric: not decoration. A growing list, or an old entry, means the gate is
    # being held up by people remembering things.
    active = len(cls.applicable)
    ages = [a for a in (d.age_days(today) for d in cls.applicable) if a is not None]
    oldest = max(ages) if ages else 0
    lines.append(f"declared deviations active: {active}; oldest: {oldest} day(s)")
    if active and oldest > STALE_AFTER_DAYS:
        lines.append(f"  ATTENTION: oldest declaration exceeds {STALE_AFTER_DAYS} days — "
                     f"the gate is being held up by memory; escalate to the arbiter")
    return lines
