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

#: Ages are measured against the UTC date; every stamp in this project is written in
#: local time (UTC+5). An entry declared just after local midnight therefore reads one
#: day into the future through no fault of its own — which is how this constant was
#: found: the very first `absent` entry, written at 01:47 local, was rejected as
#: future-dated, and a rejected entry does not fail alone. The whole list is dropped
#: (`it absorbs nothing until fixed`), so one timezone artefact would have quietly
#: switched off the subtraction for every other entry too. A day of skew is a clock;
#: anything beyond it is a wrong date, and still fails.
CLOCK_SKEW_DAYS = 1


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

    @property
    def about_a_difference(self) -> bool:
        """Can this entry ever match a difference in a run?

        Only a deviation can. A withdrawn case produces no stages at all, so a
        withdrawal can never appear in `classify`'s diff loop — and without this
        filter every one of them would be reported STALE ("matched nothing here,
        propose removal") on every single run. An instrument that cries every time
        says nothing, and the first person to act on that advice would delete the
        declarations that keep six absent samples honest.
        """
        return self.kind == "deviation"

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


def state_line(deviations: list[Deviation], voided: bool = False) -> str:
    """One line naming which of the three states the register is in.

    Empty and voided are DIFFERENT states that used to look identical downstream —
    both produced an empty list and neither said so. "Nothing is declared" is the
    ordinary state of a healthy project; "everything that was declared has been
    thrown away" is a broken instrument. Reading a gate must never require knowing
    which one happened.
    """
    if voided:
        return "declared register: DISCARDED (see the complaints above) — not empty"
    if not deviations:
        return "declared register: empty — nothing is declared, which is the healthy state"
    kinds: dict[str, int] = {}
    for dev in deviations:
        kinds[dev.kind] = kinds.get(dev.kind, 0) + 1
    parts = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
    return f"declared register: {len(deviations)} entr{'y' if len(deviations) == 1 else 'ies'} ({parts})"


def void_notice(complaints: list[str]) -> list[str]:
    """The block a run prints when its register was thrown away.

    Loud, and specific about the repair, because the obvious repair is forbidden. The
    visible symptom is a set of cases going red for no stated reason, and the two
    things a reader reaches for first — delete the entry that is complaining, or
    regenerate the goldens — both turn a broken register into a permanently smaller
    suite. That failure shape (a loud red whose most natural fix is the forbidden one)
    has now been met three times in two days, so it is written down here rather than
    rediscovered.
    """
    lines = [
        "!!! THE DECLARED REGISTER WAS DISCARDED — it is unusable, not empty !!!",
        f"  {len(complaints)} complaint(s):",
    ]
    lines += [f"    {complaint}" for complaint in complaints]
    lines += [
        "  Until they are fixed this run is graded WITHOUT any declaration:",
        "  withdrawn cases are graded as missing samples, and declared differences",
        "  are graded as ordinary reds. Those reds are an artefact of this state.",
        "  Do NOT quiet them by deleting the entry that complains or by regenerating",
        "  goldens — either one turns a broken register into a smaller suite, which is",
        "  exactly the silent shrinkage this mechanism exists to prevent. Fix the dates.",
    ]
    return lines


@dataclass
class Classification:
    """What the run's differences turned out to be."""

    declared: dict[str, list[str]] = field(default_factory=dict)   # id -> ["case/stage: path"]
    undeclared: list[str] = field(default_factory=list)            # "case/stage: path"
    applicable: list[Deviation] = field(default_factory=list)
    stale: list[Deviation] = field(default_factory=list)
    #: Declarations that matched nothing because NOTHING THEY COVER WAS RUN — every
    #: case in their scope is withdrawn or absent from this tree. Kept apart from
    #: `stale` because the two look identical and mean opposite things: a stale entry
    #: has outlived its difference and should go, while one of these has never been
    #: given the chance to match and may be the only record of a live obligation.
    #: Merging them let the report advise deleting a declaration whose subject simply
    #: is not checked here - advice that is confidently wrong, which is worse than
    #: silence: silence invites suspicion, a suggestion invites compliance.
    unexercised: list[Deviation] = field(default_factory=list)
    #: Complaints that made the runner throw the register away. Carried here so the
    #: verdict can say so: a run graded without its declarations is not a clean run
    #: that happened to have nothing declared.
    void: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.declared and not self.undeclared and not self.void

    @property
    def red(self) -> bool:
        return bool(self.undeclared) or bool(self.void)

    def verdict(self) -> str:
        # A discarded register comes first: nothing below it can be trusted while it
        # holds, because every subtraction was computed from a list thrown away.
        if self.void:
            return "REGISTER-VOID"
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
        elif age < -CLOCK_SKEW_DAYS:
            # Distinct message on purpose: a future date is readable but wrong, and
            # saying "unparseable" about it sends the reader after the wrong thing.
            # Inside CLOCK_SKEW_DAYS it is not wrong at all — see the constant.
            problems.append(f"{dev.id}: declared date {dev.declared!r} is in the future "
                            f"(by {-age} day(s), beyond the {CLOCK_SKEW_DAYS}-day clock "
                            f"skew) — the age metric would lie")
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
    # Only deviations are measured against differences - see `about_a_difference`.
    result.applicable = [d for d in deviations
                         if d.applies_to_port(port) and d.about_a_difference]
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

    # An entry that applies here and matched nothing has outlived its difference —
    # but only if it had anything to match against. A case whose sample is withdrawn
    # produces no stages, so a declaration scoped to that case alone cannot match no
    # matter how the implementation behaves, and calling it stale would invite its
    # deletion on evidence that does not exist.
    exercised = {case.slug for case in run.cases
                 if any(not stage.skipped for stage in case.stages)}
    for dev in result.applicable:
        if dev.id in matched_ids:
            continue
        if any(_match_any(dev.cases, slug) for slug in exercised):
            result.stale.append(dev)
        else:
            result.unexercised.append(dev)
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
    if cls.unexercised:
        lines.append("NOT EXERCISED — these could not have matched, and that is not "
                     "staleness:")
        for dev in cls.unexercised:
            lines.append(f"  {dev.id}: every case in its scope is absent from this "
                         f"tree, so nothing could match it")
            lines.append(f"      retires when: {dev.removed_when}")
        lines.append("  Do NOT delete these because the run says they matched nothing:"
                     " the obligation they record is still open, and the case that "
                     "would show it is not being checked here.")

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
