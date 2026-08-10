"""Per-stage timing benchmark for the pipeline.

Runs Pipeline.process_img on all samples in a folder, collects
PipelineResults.timings per stage, aggregates by doctype.
Prints a table: stage | mean_ms | share_of_total_%.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from document_processing import Pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", default="samples")
    ap.add_argument("-c", "--cycles", type=int, default=2)
    ap.add_argument("-f", "--format", default="ONNX")
    ap.add_argument("-d", "--device", default="cpu")
    args = ap.parse_args()

    images_folder = Path(args.images)
    pipe = Pipeline(model_format=args.format, device=args.device)

    # preheat
    first = next(iter(images_folder.glob("**/*.jpg")), None)
    if first:
        pipe.process_img(str(first))

    # collect: doctype -> stage -> [timings]
    per_doctype: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    totals: dict[str, list[float]] = defaultdict(list)

    images = [p for p in images_folder.glob("**/*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"[+] Found {len(images)} images, {args.cycles} cycles")

    for cyc in range(args.cycles):
        for img in images:
            try:
                r = pipe.process_img(str(img))
            except Exception as e:
                print(f"  [!] {img.name}: {e}")
                continue
            dt = r.doctype
            if not dt or dt == "NONE":
                continue
            base = dt.rsplit("_", 1)[0]
            for stage, t in r._timings.items():
                per_doctype[base][stage].append(t * 1000)
            totals[base].append(r.timings["total"] * 1000)

    print()
    print("=" * 100)
    for base in sorted(per_doctype.keys()):
        stages = per_doctype[base]
        total_mean = mean(totals[base])
        n_runs = len(totals[base])
        print(f"\n{base}  (n_runs={n_runs}, mean_total={total_mean:.1f} ms)")
        print(f"  {'stage':<22}  {'mean_ms':>10}  {'share_%':>10}")
        # sort by mean desc
        items = [(s, mean(ts)) for s, ts in stages.items()]
        items.sort(key=lambda x: -x[1])
        for stage, m in items:
            share = m / total_mean * 100
            print(f"  {stage:<22}  {m:>10.1f}  {share:>9.1f}%")

    # Overall summary across doctypes
    print()
    print("=" * 100)
    print("OVERALL (aggregated across doctypes, weighted by n_runs):")
    agg_stages: dict[str, list[float]] = defaultdict(list)
    all_totals: list[float] = []
    for base in per_doctype:
        for s, ts in per_doctype[base].items():
            agg_stages[s].extend(ts)
        all_totals.extend(totals[base])
    if all_totals:
        total_mean = mean(all_totals)
        items = [(s, mean(ts)) for s, ts in agg_stages.items()]
        items.sort(key=lambda x: -x[1])
        print(f"  total mean: {total_mean:.1f} ms, n_runs: {len(all_totals)}")
        print(f"  {'stage':<22}  {'mean_ms':>10}  {'share_%':>10}")
        for stage, m in items:
            share = m / total_mean * 100
            print(f"  {stage:<22}  {m:>10.1f}  {share:>9.1f}%")


if __name__ == "__main__":
    main()
