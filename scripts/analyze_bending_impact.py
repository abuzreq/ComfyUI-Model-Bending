#!/usr/bin/env python3
"""
Analyze the impact of bending by different demarcations (block_region, container_type,
bend_module_type, etc.). For each demarcation, groups results by that field and computes
statistics (mean, std, count, min, max) for each metric (cosine_distance, lpips_distance,
dinov2_distance, clip_distance).

Usage:
  python analyze_bending_impact.py --dir /path/to/export_folder
  python analyze_bending_impact.py --batch-dir /path/to/web_bend_demo_experiments/batch_id
  python analyze_bending_impact.py --dir ./my_export --output analysis.json --csv analysis.csv
  python analyze_bending_impact.py --dir ./my_export --demarcations block_region bend_module_type --metrics lpips_distance clip_distance
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


def _natural_sort_key(s) -> tuple:
    """Key for natural sort so e.g. '10' comes after '9'."""
    if s is None:
        s = ""
    s = str(s)
    parts = []
    for m in re.finditer(r"(\d+)|(\D+)", s):
        if m.group(1):
            parts.append((0, int(m.group(1))))
        else:
            parts.append((1, m.group(2)))
    return tuple(parts)

# Metric keys present in results (after compute_metrics.py)
METRIC_KEYS = ("cosine_distance", "lpips_distance", "dinov2_distance", "clip_distance")

# Demarcations: field name or special key. Use "steps_range" for steps_min-steps_max.
DEFAULT_DEMARCATIONS = (
    "block_region",
    "block_name",
    "container_type",
    "layer_type",
    "bend_module_type",
    "steps_range",
    "type_path",
)


def load_results_from_export(export_dir: Path) -> list:
    """Load results from export folder (data.json)."""
    data_path = export_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"No data.json in {export_dir}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def load_results_from_batch(batch_dir: Path) -> list:
    """Load results from batch folder (results.jsonl)."""
    results_path = batch_dir / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"No results.jsonl in {batch_dir}")
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def get_group_key(r: dict, demarcation: str) -> str:
    """Return the value to group by for this result and demarcation."""
    if demarcation == "steps_range":
        sm = r.get("steps_min")
        sx = r.get("steps_max")
        if sm is None and sx is None:
            return "(not set)"
        return f"{sm}-{sx}"
    return str(r.get(demarcation) if r.get(demarcation) is not None else "(empty)")


def compute_stats(values: list) -> dict:
    """Compute mean, std, min, max, count. Ignores None/NaN."""
    clean = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not clean:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    n = len(clean)
    mean = sum(clean) / n
    variance = sum((x - mean) ** 2 for x in clean) / n
    std = math.sqrt(variance) if variance else 0.0
    return {
        "count": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
    }


def run_analysis(
    results: list,
    demarcations: tuple,
    metric_keys: tuple,
) -> dict:
    """Group by each demarcation and compute per-metric stats. Excludes is_default."""
    bent = [r for r in results if not r.get("is_default")]
    out = {
        "n_bent_results": len(bent),
        "demarcations": {},
    }
    for dem in demarcations:
        groups = defaultdict(list)
        for r in bent:
            key = get_group_key(r, dem)
            groups[key].append(r)
        out["demarcations"][dem] = {}
        for group_value, group_results in sorted(groups.items(), key=lambda x: _natural_sort_key(x[0])):
            out["demarcations"][dem][group_value] = {"n_results": len(group_results)}
            for metric in metric_keys:
                values = [r.get(metric) for r in group_results]
                out["demarcations"][dem][group_value][metric] = compute_stats(values)
    return out


def print_report(analysis: dict, metric_keys: tuple) -> None:
    """Print a readable text report."""
    print(f"Bent results: {analysis['n_bent_results']}")
    print()
    for dem, groups in analysis["demarcations"].items():
        print(f"=== By {dem} ===")
        for group_value in sorted(groups.keys(), key=_natural_sort_key):
            row = groups[group_value]
            n_results = row.get("n_results", 0)
            line_parts = [f"  {group_value!r}: n={n_results}"]
            for m in metric_keys:
                s = row.get(m, {})
                if s.get("count"):
                    line_parts.append(f"    {m}: mean={s['mean']:.4f} std={s['std']:.4f}")
            print("\n".join(line_parts))
        print()
    return


def write_csv(analysis: dict, metric_keys: tuple, path: Path) -> None:
    """Write a flat CSV: demarcation, group_value, metric, count, mean, std, min, max."""
    rows = []
    for dem, groups in analysis["demarcations"].items():
        for group_value, stats_by_metric in groups.items():
            for metric in metric_keys:
                s = stats_by_metric.get(metric, {})
                rows.append({
                    "demarcation": dem,
                    "group_value": group_value,
                    "metric": metric,
                    "count": s.get("count", 0),
                    "mean": s.get("mean") if s.get("mean") is not None else "",
                    "std": s.get("std") if s.get("std") is not None else "",
                    "min": s.get("min") if s.get("min") is not None else "",
                    "max": s.get("max") if s.get("max") is not None else "",
                })
    # Sort rows so e.g. group_value "10" comes after "9" (natural order)
    rows.sort(key=lambda r: (_natural_sort_key(r["demarcation"]), _natural_sort_key(r["group_value"]), r["metric"]))
    if not rows:
        path.write_text("demarcation,group_value,metric,count,mean,std,min,max\n", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("demarcation,group_value,metric,count,mean,std,min,max\n")
        for r in rows:
            f.write(
                f"{r['demarcation']},{_csv_escape(r['group_value'])},{r['metric']},"
                f"{r['count']},{_csv_num(r['mean'])},{_csv_num(r['std'])},{_csv_num(r['min'])},{_csv_num(r['max'])}\n"
            )


def _csv_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _csv_num(x) -> str:
    if x is None or x == "":
        return ""
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze bending impact by grouping results by demarcation and computing metric statistics"
    )
    ap.add_argument("--dir", help="Export folder containing data.json (and optionally metrics already computed)")
    ap.add_argument("--batch-dir", help="Batch folder containing results.jsonl")
    ap.add_argument(
        "--demarcations",
        nargs="+",
        default=list(DEFAULT_DEMARCATIONS),
        help=f"Fields to group by (default: {list(DEFAULT_DEMARCATIONS)})",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRIC_KEYS),
        help=f"Metrics to aggregate (default: {list(METRIC_KEYS)})",
    )
    ap.add_argument("--output", "-o", help="Write full analysis JSON to this path (relative to --dir/--batch-dir if not absolute)")
    ap.add_argument("--csv", help="Write flat CSV to this path (relative to --dir/--batch-dir if not absolute)")
    ap.add_argument("--quiet", "-q", action="store_true", help="Do not print report to stdout")
    args = ap.parse_args()

    if args.dir:
        data_dir = Path(args.dir).resolve()
        results = load_results_from_export(data_dir)
    elif args.batch_dir:
        data_dir = Path(args.batch_dir).resolve()
        results = load_results_from_batch(data_dir)
    else:
        print("Specify --dir (export folder) or --batch-dir (batch folder)", file=sys.stderr)
        sys.exit(1)

    demarcations = tuple(args.demarcations)
    metric_keys = tuple(m for m in args.metrics if m in METRIC_KEYS) or METRIC_KEYS

    analysis = run_analysis(results, demarcations, metric_keys)

    if not args.quiet:
        print_report(analysis, metric_keys)

    if args.output:
        p = Path(args.output)
        out_path = (data_dir / p) if not p.is_absolute() else p.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}")

    if args.csv:
        p = Path(args.csv)
        csv_path = (data_dir / p) if not p.is_absolute() else p.resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(analysis, metric_keys, csv_path)
        print(f"Wrote {csv_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
