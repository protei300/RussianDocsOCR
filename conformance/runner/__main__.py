"""The conformance checker's command line.

    python -m conformance.runner list
    python -m conformance.runner run --port python
    python -m conformance.runner run --port go --profile gpu --verbose

Drives an implementation through the CLI contract in spec/cli.md, compares its
per-stage dumps and its view model against `conformance/cases/`, and reports the
first divergent stage per case.

Knows nothing about any implementation beyond `ports.json` and that contract.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from conformance import cases as cases_mod
from conformance import deviations as deviations_mod
from conformance import device_pin, models_pin
from conformance.paths import CASES, PORTS_JSON, REPO, REPORT, case_dir, stage_dir
from conformance.runner import report as report_mod
from conformance.runner.compare import (CPU, PROFILES, Diff, Profile, StageResult,
                                        compare_contours, compare_json, compare_npy,
                                        load_json, vacuous_reason)

#: Stages whose payload is an image produced by a warp, and therefore eligible for
#: relaxation R-02 (spec/tolerances.md). Strict comparison is attempted first so
#: that USE of the relaxation is visible rather than silent.
IMAGE_STAGES = frozenset({"prepare", "rotate", "borders.canvas", "deskew.canvas"})

EXIT_NOT_IMPLEMENTED = 2


def load_ports() -> dict[str, dict]:
    """Registered implementations, minus the documentation.

    JSON has no comments, so `ports.json` uses `_`-prefixed keys for prose. They
    are dropped here rather than at every use site, because forgetting once turns
    a comment into a port with no `cmd`.
    """
    if not PORTS_JSON.is_file():
        raise FileNotFoundError(f"{PORTS_JSON} is missing")
    raw = json.loads(PORTS_JSON.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def resolve_cmd(port: dict) -> list[str]:
    """Expand the registered command, mapping {python} and {repo} placeholders."""
    return [str(part).replace("{python}", sys.executable).replace("{repo}", str(REPO))
            for part in port["cmd"]]


def _port_env() -> dict[str, str]:
    """Environment for a port process, with its stdout encoding pinned.

    The contract (spec/cli.md) is that a port emits UTF-8 on stdout, and this
    checker decodes it as UTF-8 below. Nothing enforced the writing side, and on a
    Windows machine whose locale is not UTF-8 a Python implementation writes its
    stdout in the ANSI code page instead. Measured on a cp1251 machine: the
    reference failed against its own goldens with 18 differences, every one of them
    a Cyrillic OCR string turned to replacement characters, and the run then died
    with UnicodeEncodeError while printing the report — a red with no hint that the
    cause was the console.

    So the checker pins it for the child rather than trusting the machine. Harmless
    for ports that are already UTF-8 (Go, JVM): PYTHONIOENCODING means nothing to
    them.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_port(cmd: list[str], args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd + args, cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          env=_port_env())


def cmd_list(args: argparse.Namespace) -> int:
    ports = load_ports()
    print("registered ports:")
    for name, spec in ports.items():
        print(f"  {name:10s} {' '.join(str(c) for c in spec['cmd'])}")
    declared = deviations_mod.load()
    # `list` validates too. It used to read the register raw while `run` discarded it,
    # so the same broken file produced a listing full of withdrawals and a run that
    # knew of none - the two entry points disagreed about the same tree.
    complaints = deviations_mod.validate(declared)
    if complaints:
        print()
        for line in deviations_mod.void_notice(complaints):
            print(line)
        declared = []
    print("\n" + deviations_mod.state_line(declared, voided=bool(complaints)))
    print("\ncases (derived from service/seed_data/manifest.json):")
    for c in cases_mod.load_cases():
        golden = case_dir(c.slug) / "viewmodel.json"
        state = "golden" if golden.is_file() else "NO GOLDEN"
        if deviations_mod.withdrawal_for(c.slug, declared) is not None:
            note = "  [out of service: sample withdrawn]"
        elif not c.exists():
            note = "  [sample missing!]"
        else:
            note = ""
        print(f"  {c.slug:52s} {c.doc_type:22s} {state}{note}")
    return 1 if complaints else 0


def _compare_digest(stage: str, golden_file: Path, dump: Path,
                    slug: str) -> StageResult:
    """Compare an image stage.

    Three levels, in descending strictness, because measurement showed the strictest
    one is unreachable ACROSS implementations:

    1. **Full pixel comparison under R-02**, when reference pixels are present next to
       the digest (`regen --with-pixels`, gitignored). This is the real rule.
    2. **Digest equality**, which proves bit-exactness. Achievable Python-to-Python,
       and it is how the reference self-check stays honest.
    3. **Shape only**, plus a warning, when the digest differs and no pixels are
       available.

    Level 3 is not a loophole, it is the measured truth: `warpPerspective` differs by a
    few LSB on a fraction of pixels between OpenCV bindings and minor versions, so a
    digest can never match across ports. Measured for the Go port on DL_2011:
    max difference 3, mean 0.0002, 0.02 % of pixels differing, 100 % within 1 LSB —
    R-02 passes with room to spare. A SHAPE mismatch is still a hard failure, and that
    is not theoretical either: it is exactly how a one-row error in the mask
    letterbox-undo was caught (868 rows against the golden's 877).
    """
    import hashlib

    import numpy as np

    golden = load_json(golden_file)
    actual = dump / f"{stage.replace('/', '_')}.npy"
    if not actual.is_file():
        return StageResult(stage, ok=False,
                           diffs=[Diff(stage, "missing", f"{actual.name} not produced")])

    arr = np.load(actual, allow_pickle=False)
    if list(arr.shape) != golden["shape"] or str(arr.dtype) != golden["dtype"]:
        return StageResult(stage, ok=False, diffs=[Diff(
            stage, "shape",
            f"{golden['dtype']}{golden['shape']} vs {arr.dtype}{list(arr.shape)}")])

    if hashlib.sha256(arr.tobytes()).hexdigest() == golden["sha256"]:
        return StageResult(stage, ok=True)

    # Level 1: reference pixels available -> apply R-02 properly.
    golden_npy = golden_file.with_name(f"{stage.replace('/', '_')}.npy")
    if golden_npy.is_file():
        diffs = compare_npy(golden_npy, actual, CPU, relaxed_image=True)
        if diffs:
            return StageResult(stage, ok=False, diffs=diffs)
        return StageResult(stage, ok=True,
                           warn="not bit-identical; passed relaxation R-02 against "
                                "locally regenerated reference pixels")

    # Level 3: shape verified, pixel magnitude unverified here.
    return StageResult(stage, ok=True, warn=(
        "pixels differ from the golden digest and no reference pixels are present, so "
        "only the SHAPE was verified. To apply relaxation R-02 properly:\n"
        f"        python -m conformance.refcli regen --with-pixels --case {slug}\n"
        "        then re-run this check"))


def _compare_stage(stage: str, golden_file: Path, dump: Path,
                   profile: Profile, slug: str = "") -> StageResult:
    if golden_file.name.endswith(".digest.json"):
        return _compare_digest(stage, golden_file, dump, slug)

    actual = dump / golden_file.name
    if not actual.is_file():
        return StageResult(stage, ok=False,
                           diffs=[Diff(stage, "missing", f"{golden_file.name} not produced")])

    if golden_file.suffix == ".npy":
        diffs = compare_npy(golden_file, actual, profile)
        if diffs and stage in IMAGE_STAGES:
            # Strict first, R-02 second, and say so -- a relaxation that is used
            # invisibly is indistinguishable from a rule nobody checks.
            relaxed = compare_npy(golden_file, actual, profile, relaxed_image=True)
            if not relaxed:
                return StageResult(stage, ok=True,
                                   diffs=[Diff(stage, "value",
                                               "not bit-identical; passed under relaxation R-02 "
                                               f"({diffs[0].detail})")])
            return StageResult(stage, ok=False, diffs=relaxed)
        return StageResult(stage, ok=not diffs, diffs=diffs)

    golden = load_json(golden_file)
    if stage == "borders.segments":
        # Relaxation R-01: contour point counts legitimately differ across OpenCV
        # minor versions, so area, centroid and Hausdorff distance are compared
        # instead of the point list.
        diffs = compare_contours(golden, load_json(actual), profile, path=stage)
        return StageResult(stage, ok=not diffs, diffs=diffs, warn=vacuous_reason(golden))

    diffs = compare_json(golden, load_json(actual), profile, path=stage)
    # A null or empty golden compares equal to almost anything: surface it rather than
    # letting it read as a genuine pass.
    return StageResult(stage, ok=not diffs, diffs=diffs, warn=vacuous_reason(golden))


def _check_case(case, cmd: list[str], profile: Profile, extra: list[str],
                claimed: set[str] | None, timeout: int,
                withdrawal=None) -> report_mod.CaseReport:
    golden_dir = case_dir(case.slug)
    golden_stages = stage_dir(case.slug)
    rep = report_mod.CaseReport(slug=case.slug, doc_type=case.doc_type)

    # A declared withdrawal comes first: the goldens of a withdrawn case stay on
    # disk (they are the record of what the reference produced while the sample was
    # there), so checking for them before checking the declaration would report
    # "no golden" for a case whose real state is "no sample, and we said so".
    #
    # The declaration now comes from conformance/deviations.json, not from a free-text
    # `disabled` field in the seed manifest. One register, two kinds of entry: a
    # withdrawal must name its owner, its basis and the EVENT that ends it, and it is
    # validated like any other entry. An undeclared missing sample stays a failure
    # below - that is the point of moving this out of a flag anybody could write.
    if withdrawal is not None:
        rep.skipped = withdrawal.reason
        return rep
    if not (golden_dir / "viewmodel.json").is_file():
        rep.error = "no golden; run: python -m conformance.refcli regen"
        return rep
    if not case.exists():
        rep.error = f"sample image missing: {case.sample}"
        return rep

    index_path = golden_stages / "stages.json"
    if not index_path.is_file():
        rep.error = "golden stages.json missing; regenerate this case"
        return rep
    index = load_json(index_path)

    with tempfile.TemporaryDirectory(prefix=f"conf-{case.slug}-") as tmp:
        dump = Path(tmp)
        proc = run_port(cmd, ["probe", "--image", str(case.image),
                              "--dump-dir", str(dump)] + extra, timeout)
        if proc.returncode == EXIT_NOT_IMPLEMENTED:
            rep.stages.append(StageResult("probe", ok=True,
                                          skipped="implementation reports probe unimplemented"))
            return rep
        if proc.returncode != 0:
            rep.error = (f"probe exited {proc.returncode}: "
                         f"{(proc.stderr or '').strip().splitlines()[-1:] or ['no stderr']}")
            return rep

        # Golden order IS pipeline order, so 'first divergence' is meaningful.
        for entry in index["stages"]:
            stage = entry["stage"]
            if stage == "viewmodel":
                continue  # compared below, against the canonical file
            if claimed is not None and stage not in claimed and not _claims_pattern(stage, claimed):
                rep.stages.append(StageResult(stage, ok=True,
                                              skipped="not in stages_implemented"))
                continue
            rep.stages.append(_compare_stage(stage, golden_stages / entry["file"], dump,
                                             profile, case.slug))

        # The view model, from `recognize` rather than from the dump: it is the
        # contract an integrator actually consumes, and comparing the same thing
        # twice would hide a difference between the two code paths.
        proc = run_port(cmd, ["recognize", "--image", str(case.image)] + extra, timeout)
        if proc.returncode == EXIT_NOT_IMPLEMENTED:
            # A port under construction must be able to say "not yet" without being
            # scored as broken. This is what lets M1 be green on `prepare` while the
            # view model does not exist until M7.
            rep.stages.append(StageResult("viewmodel", ok=True,
                                          skipped="implementation reports recognize unimplemented"))
            return rep
        if proc.returncode != 0:
            rep.stages.append(StageResult("viewmodel", ok=False,
                                          diffs=[Diff("viewmodel", "value",
                                                      f"recognize exited {proc.returncode}")]))
            return rep
        try:
            actual_vm = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            rep.stages.append(StageResult("viewmodel", ok=False, diffs=[Diff(
                "viewmodel", "type",
                f"stdout is not JSON ({exc}); is the implementation logging to stdout?")]))
            return rep
        diffs = compare_json(load_json(golden_dir / "viewmodel.json"), actual_vm,
                             profile, path="viewmodel")
        rep.stages.append(StageResult("viewmodel", ok=not diffs, diffs=diffs))

    return rep


def _claims_pattern(stage: str, claimed: set[str]) -> bool:
    """Expand the per-field stage patterns.

    `ocr.<Field>.words` in stages_implemented covers `ocr.Last_name_ru.words`, and
    `words.<Field>.bbox` covers `words.Last_name_ru.bbox`. A port claims the pattern, not
    the fields, because which fields exist depends on the document.
    """
    if stage.startswith("ocr.") and stage.endswith(".words"):
        return "ocr.<Field>.words" in claimed
    if stage.startswith("words.") and stage.endswith(".bbox"):
        return "words.<Field>.bbox" in claimed
    return False


def cmd_run(args: argparse.Namespace) -> int:
    # Refuse before doing any work if the goldens describe a different weight set:
    # every number below is a function of the models, so comparing across sets
    # produces dozens of true-but-useless differences (see conformance/models_pin.py).
    mismatch = models_pin.mismatch_message()
    if mismatch and not args.ignore_models_pin:
        print(mismatch, file=sys.stderr)
        print("\n(--ignore-models-pin grades anyway)", file=sys.stderr)
        return 3

    # Same shape, second axis: the goldens also carry the device they were taken on.
    # Refuse a cross-device run before doing the work (conformance/device_pin.py).
    # The module arrived in this tree with the register (step 3) but nothing called
    # it, which is worse than not having it: a file found by name reads as a working
    # check. A mechanism is not present until something asks it a question.
    device = args.device or device_pin.goldens_device()
    device_mismatch = device_pin.mismatch_message(device)
    if device_mismatch and not args.ignore_device_pin:
        print(device_mismatch, file=sys.stderr)
        print("\n(--ignore-device-pin grades anyway; expect geometric differences "
              "no profile can absorb)", file=sys.stderr)
        return 3

    ports = load_ports()
    if args.port not in ports:
        print(f"unknown port {args.port!r}; registered: {', '.join(ports)}", file=sys.stderr)
        return 3
    cmd = resolve_cmd(ports[args.port])
    # The profile follows the device unless it was named explicitly. Two
    # independent defaults are how a GPU run gets graded by CPU allowances with
    # nothing said about it: the device pin answers "should this run at all", the
    # profile answers "measured by what, once it does" - and bypassing the first
    # must not silently answer the second.
    profile = PROFILES[args.profile or ("gpu" if device == "gpu" else "cpu")]
    if args.profile and args.profile != device:
        # Naming both, differently, is legal — and is exactly how a run gets graded
        # by allowances nobody chose for it. Say so; do not quietly obey.
        print(f"warning: --profile {args.profile} with --device {device}: "
              f"the numeric allowances do not belong to the device being graded",
              file=sys.stderr)

    # `--device` is ALWAYS passed on, even when it equals the default: otherwise the
    # port picks its own, and "which device produced these numbers" has no answer.
    extra: list[str] = ["--device", device]
    if args.ocr:
        extra += ["--ocr", args.ocr]

    info = None
    claimed = None
    proc = run_port(cmd, ["info"], args.timeout)
    if proc.returncode == 0:
        try:
            info = json.loads(proc.stdout)
            claimed = set(info.get("stages_implemented") or []) or None
        except json.JSONDecodeError:
            print("warning: `info` did not return JSON; grading every stage",
                  file=sys.stderr)

    # The declared register, read once. A malformed list absorbs nothing: it is
    # reported and then treated as empty, so a broken declaration can never turn a
    # missing sample into a silent pass.
    declared = deviations_mod.load()
    complaints = deviations_mod.validate(declared)
    # Printed to stdout, with the report, and not only to stderr: the reader who has
    # to understand a red is looking at the report, and stderr is where a piped or
    # logged run loses it. Six withdrawn cases go red the moment this happens, and
    # nothing else in the output would say why.
    if complaints:
        for line in deviations_mod.void_notice(complaints):
            print(line)
        declared = []
    print(deviations_mod.state_line(declared, voided=bool(complaints)))

    run = report_mod.RunReport(port=args.port, profile=profile.name, info=info)
    for case in cases_mod.select(args.case, limit=args.limit):
        print(f"  {case.slug} ...", file=sys.stderr, flush=True)
        run.cases.append(_check_case(case, cmd, profile, extra, claimed, args.timeout,
                                     deviations_mod.withdrawal_for(case.slug, declared)))

    # The second half of the register: the differences a run DID produce are split
    # into declared and undeclared, and the verdict gains its third outcome. Until
    # this call existed the file could absorb a withdrawal (a case that never runs)
    # and nothing else - a declared DIFFERENCE was validated, then graded as an
    # ordinary red, which is the failure the register was built to remove.
    run.deviations = deviations_mod.classify(run, declared, args.port)
    run.deviations.void = tuple(complaints)

    print(report_mod.render(run, verbose=args.verbose))
    saved = report_mod.save(run, REPORT)
    print(f"\nreport: {saved.relative_to(REPO)}")
    # `ok` answers "identical to the reference"; `red` answers "is this a failure",
    # and a declared difference is not one. Three things still fail regardless of the
    # register: an undeclared difference, a case that errored (a crashed port is not
    # a difference to declare), and an empty run - nothing verified is not a pass.
    hard = any(c.error for c in run.cases) or not run.cases
    return 1 if (run.deviations.red or hard or complaints) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m conformance.runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="show registered ports and cases")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("run", help="grade an implementation")
    sp.add_argument("--port", required=True)
    sp.add_argument("--profile", default=None, choices=sorted(PROFILES),
                    help="numeric allowances; defaults to the profile matching "
                         "--device, so the two cannot drift apart silently")
    sp.add_argument("--case", action="append", default=None, help="slug substring; repeatable")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--device", default=None, choices=["cpu", "gpu"],
                    help="defaults to the device the goldens were recorded on; a "
                         "different one is refused, see conformance/device_pin.py")
    sp.add_argument("--ocr", default=None, choices=["accurate", "fast"])
    sp.add_argument("--timeout", type=int, default=600)
    sp.add_argument("--verbose", action="store_true")
    sp.add_argument("--ignore-device-pin", action="store_true", dest="ignore_device_pin",
                    help="grade even when the goldens were recorded on another device "
                         "(the differences will be dominated by provider geometry)")
    sp.add_argument("--ignore-models-pin", action="store_true", dest="ignore_models_pin",
                    help="grade even when the goldens name a different weight set "
                         "(the differences will be dominated by the model change)")
    sp.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    # A report is a diagnosis, and a diagnosis that cannot be printed is worse than
    # a red one: on a cp1251 console the whole run used to end in UnicodeEncodeError
    # AFTER every case had been graded, losing the result to the terminal's code
    # page. Escaping a character the console cannot show costs one unreadable token;
    # crashing costs the run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):   # not a TextIOWrapper (redirected, tests)
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
