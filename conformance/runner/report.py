"""Rendering of conformance results.

The headline is the FIRST DIVERGENT STAGE per case, not a pass count: a count
tells you how bad things are, the stage tells you what to fix. Everything here is
plain ASCII — these reports get read in PowerShell, in Git Bash and in CI logs,
and box-drawing characters survive none of those reliably.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from conformance.runner.compare import Diff, StageResult


@dataclass
class CaseReport:
    slug: str
    doc_type: str | None
    stages: list[StageResult] = field(default_factory=list)
    error: str | None = None
    #: Set when the case is deliberately out of service because its sample was
    #: withdrawn from the public tree. Kept distinct from ``error`` on purpose: a
    #: sample that vanished without anyone declaring it is still a failure, while a
    #: declared withdrawal is a stated gap. Both are visible; only one is a defect.
    skipped: str | None = None

    @property
    def first_divergence(self) -> str | None:
        for s in self.stages:
            if not s.ok and not s.skipped:
                return s.stage
        return None

    @property
    def ok(self) -> bool:
        if self.skipped:
            return True
        return self.error is None and self.first_divergence is None

    def counts(self) -> tuple[int, int, int]:
        passed = sum(1 for s in self.stages if s.ok and not s.skipped)
        failed = sum(1 for s in self.stages if not s.ok and not s.skipped)
        skipped = sum(1 for s in self.stages if s.skipped)
        return passed, failed, skipped


@dataclass
class RunReport:
    port: str
    profile: str
    cases: list[CaseReport] = field(default_factory=list)
    info: dict | None = None
    #: Filled by the runner from conformance/deviations.py. None means the run was not
    #: classified at all (a caller that skipped the step), which is NOT the same as
    #: "classified and found clean" — kept distinguishable on purpose.
    deviations: object | None = None

    @property
    def ok(self) -> bool:
        """Did everything match, deviations aside.

        Deliberately unchanged: this stays the answer to "is the implementation
        identical to the reference". Whether the run is a red is a different
        question, answered by the classification — see `verdict`.
        """
        return bool(self.cases) and all(c.ok for c in self.cases)

    @property
    def verdict(self) -> str:
        if self.deviations is None:
            return "PASS" if self.ok else "FAIL"
        return self.deviations.verdict()


def render(run: RunReport, verbose: bool = False) -> str:
    lines: list[str] = []
    lines.append(f"port    : {run.port}")
    lines.append(f"profile : {run.profile}")
    if run.info:
        v = run.info.get("versions", {})
        lines.append(f"impl    : {run.info.get('language')} "
                     f"ort={v.get('onnxruntime')} opencv={v.get('opencv')} "
                     f"commit={run.info.get('commit')}")
    lines.append("")

    width = max((len(c.slug) for c in run.cases), default=10)
    lines.append(f"{'case'.ljust(width)}  {'pass':>5} {'fail':>5} {'skip':>5}  first divergence")
    lines.append("-" * (width + 34))
    for c in run.cases:
        p, f, s = c.counts()
        if c.skipped:
            marker = "OUT OF SERVICE: sample withdrawn"
        elif c.error:
            marker = f"ERROR: {c.error}"
        else:
            marker = c.first_divergence or "-"
        lines.append(f"{c.slug.ljust(width)}  {p:5d} {f:5d} {s:5d}  {marker}")
    lines.append("")

    # Withdrawn cases get their own block rather than a quiet dash in the table. A
    # run that verifies two cases instead of nine is not the same run, and the
    # verdict line alone cannot say so - PASS over a shrunken set reads exactly like
    # PASS over the whole set.
    withdrawn = [c for c in run.cases if c.skipped]
    if withdrawn:
        lines.append(f"out of service: {len(withdrawn)} of {len(run.cases)} case(s) "
                     f"carry no sample and were NOT verified")
        lines.append("=" * 52)
        for c in withdrawn:
            lines.append(f"  {c.slug}")
            lines.append(f"      {c.skipped}")
        lines.append("")

    failing = [c for c in run.cases if not c.ok]
    if failing:
        lines.append("details")
        lines.append("=======")
        for c in failing:
            lines.append(f"\n{c.slug}")
            if c.error:
                lines.append(f"  ERROR {c.error}")
            for st in c.stages:
                if st.ok or st.skipped:
                    continue
                lines.append(f"  [{st.stage}]")
                shown = st.diffs if verbose else st.diffs[:8]
                for d in shown:
                    lines.append(f"      {d}")
                if len(st.diffs) > len(shown):
                    lines.append(f"      ... {len(st.diffs) - len(shown)} more "
                                 f"(pass --verbose to see all)")

    # Stages that passed with a caveat: either the golden proves nothing, or only part
    # of the payload was verified. Never a failure, but a green tick here means less
    # than it looks, so they are always listed.
    vacuous = [(c.slug, s) for c in run.cases for s in c.stages if s.warn and not s.skipped]
    if vacuous:
        lines.append(f"warnings: {len(vacuous)} stage(s) passed with a caveat")
        lines.append("=" * 52)
        seen: set[str] = set()
        for slug, s in vacuous:
            if s.stage in seen:
                continue
            seen.add(s.stage)
            lines.append(f"  {s.stage}: {s.warn}")
        # The advice only fits a VACUOUS golden. Printing it under an R-02 warning --
        # where the comparison was substantive and simply not bit-exact -- reads as
        # "something is broken here" and sends the reader after a non-problem.
        if any("compares nothing" in s.warn for _, s in vacuous):
            lines.append("  (fix the emission and regenerate, or accept that the stage is"
                         " legitimately absent for these documents)")
        lines.append("")

    if run.deviations is not None:
        from conformance import deviations as deviations_mod
        section = deviations_mod.render(run.deviations)
        if section:
            lines.append("")
            lines.extend(section)
            lines.append("")

    # Three outcomes, not two: CLEAN, DECLARED (differences, all of them accounted
    # for by name), UNDECLARED (the only real red). A two-outcome verdict is what let
    # legitimate reds pile up until reading the gate became an act of memory.
    lines.append("VERDICT: " + run.verdict)
    return "\n".join(lines)


def save(run: RunReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{run.port}-{stamp}.json"

    def _stage(s: StageResult) -> dict:
        return {"stage": s.stage, "ok": s.ok, "skipped": s.skipped, "warn": s.warn,
                "diffs": [asdict(d) for d in s.diffs]}

    deviations = None
    if run.deviations is not None:
        cls = run.deviations
        deviations = {
            "verdict": cls.verdict(),
            "declared": cls.declared,
            "undeclared": cls.undeclared,
            "stale": [d.id for d in cls.stale],
            "active": len(cls.applicable),
            "oldest_days": max([a for a in (d.age_days() for d in cls.applicable)
                                if a is not None], default=0),
        }

    payload = {
        "port": run.port, "profile": run.profile, "utc": stamp,
        "ok": run.ok, "verdict": run.verdict, "deviations": deviations,
        "info": run.info,
        "cases": [{"slug": c.slug, "doc_type": c.doc_type, "error": c.error,
                   "first_divergence": c.first_divergence,
                   "stages": [_stage(s) for s in c.stages]}
                  for c in run.cases],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
