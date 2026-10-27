"""result/plot.py and result/aggregate.py on synthetic data (Agg backend; files written to tmp_path)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result.aggregate import aggregate, point_estimates  # noqa: E402
from result.plot import (aggregate_curves, plot_advantage_heatmap, plot_convergence, plot_decision_time,  # noqa: E402
                         plot_method_boxplot, plot_paired_gains, plot_quality_time_tradeoff, plot_training_trace)
from result.stats import advantage_table  # noqa: E402


def _eval_frame(methods, instances=6, runs=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(instances):
        ddt, scale = [500, 900, 1600][i % 3], [50, 100][i // 3]
        for method in methods:
            for run in range(runs if method == "ours" else 1):
                for decode in (["greedy", "sample"] if method in ("ours", "dqn") else ["rule"]):
                    rate = min(1.0, max(0.0, {"ours": 0.6, "nf": 0.4, "dqn": 0.45, "spt": 0.5}[method] + rng.normal(0, 0.05)))
                    rows.append({"instance_id": f"DDT{ddt}_num{scale}", "method": method, "run_id": f"run{run + 1}",
                                 "tier": "benchmark", "scale": scale, "ddt": ddt, "decode": decode, "episode": 0,
                                 "fulfillment_rate": rate, "obj_model": 1 - rate, "obj_ref": 0.5, "gap": (0.5 - rate) * 200,
                                 "infer_time_ms": 500 + 5 * scale + rng.normal(0, 20), "decision_count": 3 * scale})
    return pd.DataFrame(rows)


def _write_curves(root: Path, method: str, instance: str, runs: int):
    for run in range(1, runs + 1):
        run_dir = root / method / instance / f"run{run}"
        run_dir.mkdir(parents=True)
        epochs = np.arange(5)
        obj = 0.9 - 0.05 * epochs + 0.01 * run
        pd.DataFrame({"epoch": epochs, "instance_id": instance, "obj": obj, "obj_best": np.minimum.accumulate(obj),
                      "gap": np.nan, "feasible": True, "wall_time_s": epochs * 10.0}).to_csv(run_dir / "curve.csv", index=False)


def test_point_estimates_prefer_greedy_rows():
    frame = _eval_frame(["ours", "nf", "dqn", "spt"])
    points = point_estimates(frame)
    assert set(points["method"]) == {"ours", "nf", "dqn", "spt"}
    assert "decision_time_ms" in points.columns and points["decision_time_ms"].between(1.0, 20.0).all()
    assert len(points[points["method"] == "ours"]) == 12          # 6 instances x 2 runs, sampled rows dropped


def test_aggregate_curves_and_figures(tmp_path):
    frame = _eval_frame(["ours", "nf", "dqn", "spt"])
    for instance in frame["instance_id"].unique()[:2]:
        _write_curves(tmp_path, "ours", instance, runs=2)
    conv = aggregate_curves(sorted((tmp_path / "ours").glob("*/run*")), key="obj_best")
    assert list(conv.columns) == ["epoch", "mean", "std", "n_instances", "n_runs"] and conv["n_runs"].iloc[0] == 4
    assert conv["std"].notna().all()
    single = aggregate_curves([next((tmp_path / "ours").glob("*/run1"))], key="obj")
    assert single["std"].isna().all()                            # one run -> empty std, not 0

    out = tmp_path / "fig"
    plot_convergence([(conv, "ours")], out / "conv", reference=0.5)
    train_csv = tmp_path / "train_metrics.csv"
    pd.DataFrame({"iteration": range(40), "fulfillment_rate_mean": np.linspace(0.1, 0.6, 40)}).to_csv(train_csv, index=False)
    plot_training_trace(train_csv, out / "trace", smooth_window=7)
    points = point_estimates(frame)
    advantage = advantage_table(points, "ours")
    plot_paired_gains(advantage, ["nf", "dqn"], out / "gains")
    plot_advantage_heatmap(advantage, ["nf", "dqn", "spt"], out / "heat")
    plot_decision_time(points, "ours", out_stem=out / "time")
    plot_quality_time_tradeoff(points, [("rules", ["ours", "spt"]), ("drl", ["ours", "dqn"])], out / "tradeoff")
    plot_method_boxplot(points, ["ours", "nf", "dqn", "spt"], out_stem=out / "box")
    for name in ("conv", "trace", "gains", "heat", "time", "tradeoff", "box"):
        assert (out / f"{name}.pdf").exists() and (out / f"{name}.png").exists()


def test_aggregate_writes_summary_tables(tmp_path):
    frame = _eval_frame(["ours", "nf", "dqn", "spt"])
    eval_path = tmp_path / "eval_results.csv"
    frame.to_csv(eval_path, index=False)
    for instance in frame["instance_id"].unique()[:2]:
        _write_curves(tmp_path, "ours", instance, runs=2)
    written = aggregate(eval_path, tmp_path, "ours")
    for name in ("stats_summary", "paired_tests", "ablation_results", "advantage_table", "friedman", "conv_ours"):
        assert name in written and written[name].exists(), name
    summary = pd.read_csv(written["stats_summary"])
    assert set(summary["metric"]) == {"fulfillment_rate", "gap"}
    paired = pd.read_csv(written["paired_tests"])
    assert list(paired["comparator"]) == ["nf", "dqn", "spt"] and paired["p_holm"].between(0, 1).all()
