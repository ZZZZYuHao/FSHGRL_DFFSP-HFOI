"""Aggregate the evaluation CSVs into the summary tables of the experiment contract (new code).

    python -m result.aggregate [--eval result/eval_results.csv] [--out-dir result] [--reference ours]

Inputs: ``result/eval_results.csv`` (all methods: ours, ablations, learned baselines, rules; one row per
instance x run x episode) and the per-instance run directories ``result/<method>/<instance>/run<k>/curve.csv``.
Point estimates keep the greedy / incumbent / rule rows (sampled episodes are averaged only for methods
without them) and add ``decision_time_ms`` = episode wall-clock / number of decisions, the per-decision time
the paper reports as "time (s)" (``infer_time_ms`` stays the whole-episode wall-clock).
Outputs (all in ``--out-dir``): ``stats_summary.csv`` (mean +- std over runs per method / tier for the
fulfilment rate and the gap), ``paired_tests.csv`` (Table 11 layout against ``--reference``),
``ablation_results.csv``, ``advantage_table.csv``, ``friedman.csv`` and ``conv_<method>.csv`` (convergence
curves aggregated over instances and runs). Figures are drawn by ``paper_assets/figures/src/*.py``.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result.plot import aggregate_curves  # noqa: E402
from result.stats import (ABLATION_METHODS, DRL_METHODS, RULE_METHODS, ablation_table, advantage_table,  # noqa: E402
                          friedman, paired_table, summarize)


def load_eval_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("fulfillment_rate", "gap", "obj_model", "infer_time_ms", "ddt", "scale"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def point_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (instance, method, run): greedy / incumbent rows only, sampled episodes averaged."""
    data = frame.copy()
    if "decode" in data.columns:
        preferred = data[data["decode"].isin(["greedy", "incumbent", "rule"])]
        others = data[~data["method"].isin(preferred["method"].unique())]
        data = pd.concat([preferred, others], ignore_index=True)
    if "infer_time_ms" in data.columns and "decision_count" in data.columns:   # wall-clock per decision
        counts = pd.to_numeric(data["decision_count"], errors="coerce")
        data["decision_time_ms"] = pd.to_numeric(data["infer_time_ms"], errors="coerce") / counts.where(counts > 0)
    keys = ["instance_id", "method", "run_id"]
    numeric = ["fulfillment_rate", "gap", "obj_model", "infer_time_ms", "decision_time_ms"]
    numeric = [c for c in numeric if c in data.columns]
    meta = data.drop_duplicates(keys).set_index(keys)[[c for c in ("tier", "scale", "ddt") if c in data.columns]]
    agg = data.groupby(keys)[numeric].mean()
    return meta.join(agg).reset_index()


def aggregate(eval_path: Path, out_dir: Path, reference: str = "ours") -> dict:
    frame = point_estimates(load_eval_results(eval_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    summary = pd.concat([summarize(frame, metric="fulfillment_rate"), summarize(frame, metric="gap")],
                        ignore_index=True)
    summary.to_csv(out_dir / "stats_summary.csv", index=False)
    written["stats_summary"] = out_dir / "stats_summary.csv"

    comparators = [m for m in list(ABLATION_METHODS) + list(DRL_METHODS) + list(RULE_METHODS)
                   if (frame["method"] == m).any()]
    if (frame["method"] == reference).any() and comparators:
        paired = paired_table(frame, reference, comparators, metric="fulfillment_rate", decimals=4)
        paired.to_csv(out_dir / "paired_tests.csv", index=False)
        written["paired_tests"] = out_dir / "paired_tests.csv"
        ablation = ablation_table(frame, methods=(reference,) + tuple(ABLATION_METHODS))
        ablation.to_csv(out_dir / "ablation_results.csv", index=False)
        written["ablation_results"] = out_dir / "ablation_results.csv"
        advantage = advantage_table(frame, reference)
        advantage.to_csv(out_dir / "advantage_table.csv", index=False)
        written["advantage_table"] = out_dir / "advantage_table.csv"
        methods = [reference] + comparators
        matrix = frame.pivot_table(index="instance_id", columns="method", values="fulfillment_rate", aggfunc="mean")
        matrix = matrix[[m for m in methods if m in matrix.columns]].dropna()
        if matrix.shape[1] >= 3 and matrix.shape[0] >= 2:
            result = friedman(-matrix.values)       # ranks: 1 = highest fulfilment
            rows = [{"method": m, "mean_rank": r} for m, r in zip(matrix.columns, result["mean_ranks"])]
            table = pd.DataFrame(rows)
            table["statistic"] = result["statistic"]
            table["p_value"] = result["p_value"]
            table["n_instances"] = result["n_blocks"]
            table.to_csv(out_dir / "friedman.csv", index=False)
            written["friedman"] = out_dir / "friedman.csv"

    for method_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        run_dirs = sorted(glob.glob(str(method_dir / "*" / "run*" / "curve.csv")))
        if not run_dirs:
            continue
        conv = aggregate_curves([Path(p).parent for p in run_dirs], key="obj_best")
        conv.to_csv(out_dir / f"conv_{method_dir.name}.csv", index=False)
        written[f"conv_{method_dir.name}"] = out_dir / f"conv_{method_dir.name}.csv"
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", default="result/eval_results.csv")
    parser.add_argument("--out-dir", default="result")
    parser.add_argument("--reference", default="ours")
    args = parser.parse_args()
    eval_path = Path(args.eval)
    eval_path = eval_path if eval_path.is_absolute() else ROOT / eval_path
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    if not eval_path.exists():
        raise SystemExit(f"[FAIL] {eval_path} not found: run scripts/run_07_eval_main.py first")
    written = aggregate(eval_path, out_dir, args.reference)
    for name, path in written.items():
        print(f"[AGGREGATE] {name}: {path}")


if __name__ == "__main__":
    main()
