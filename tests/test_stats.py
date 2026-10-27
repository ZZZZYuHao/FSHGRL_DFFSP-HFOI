"""result/stats.py: the paper's Table 11 numbers, Holm, Friedman and the run-level summaries."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result.stats import (ablation_table, advantage_table, friedman, holm, paired_table, summarize,  # noqa: E402
                          wilcoxon_exact)


def test_table_11_all_positive_and_one_tie():
    fifteen = wilcoxon_exact([0.1 * (i + 1) for i in range(15)])
    assert fifteen["r_plus"] == 120 and fifteen["r_minus"] == 0 and fifteen["n"] == 15
    assert abs(fifteen["p_value"] - 2.0 / 2 ** 15) < 1e-15 and abs(fifteen["p_value"] - 6.104e-5) < 5e-8
    fourteen = wilcoxon_exact([0.1 * (i + 1) for i in range(14)] + [0.0])
    assert fourteen["r_plus"] == 105 and fourteen["n"] == 14 and fourteen["n_zero"] == 1
    assert abs(fourteen["p_value"] - 1.221e-4) < 5e-8
    family = holm([fifteen["p_value"]] * 4 + [fourteen["p_value"]])
    assert all(abs(p - 3.052e-4) < 5e-8 for p in family)


def test_wilcoxon_ties_and_rounding():
    mixed = wilcoxon_exact([0.5, -0.5, 1.0, 1.0, 2.0])
    assert mixed["r_plus"] == 1.5 + 3.5 + 3.5 + 5 and mixed["r_minus"] == 1.5
    from scipy.stats import wilcoxon

    assert abs(mixed["p_value"] - wilcoxon([0.5, -0.5, 1.0, 1.0, 2.0], method="exact").pvalue) < 1e-12
    rounded = wilcoxon_exact([0.00004, 0.3, -0.2], decimals=4)
    assert rounded["n"] == 2 and rounded["n_zero"] == 1
    assert wilcoxon_exact([0.0, 0.0])["p_value"] == 1.0


def test_holm_is_monotone_and_bounded():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.04, 0.06] or np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == [1.0, 1.0] and holm([]) == []


def test_friedman_matches_scipy():
    rng = np.random.default_rng(0)
    matrix = rng.random((12, 4))
    from scipy.stats import friedmanchisquare

    expected = friedmanchisquare(*[matrix[:, j] for j in range(4)])
    out = friedman(matrix)
    assert abs(out["statistic"] - expected.statistic) < 1e-12 and abs(out["p_value"] - expected.pvalue) < 1e-12
    assert len(out["mean_ranks"]) == 4 and abs(sum(out["mean_ranks"]) - 10.0) < 1e-12


def _frame():
    rows = []
    for i, iid in enumerate(["A", "B", "C"]):
        for method, base in (("ours", 0.8), ("nf", 0.6), ("ng", 0.75)):
            runs = 2 if method == "ours" else 1
            for run in range(runs):
                rows.append({"instance_id": iid, "method": method, "run_id": f"run{run + 1}", "tier": "benchmark",
                             "scale": 50 * (i + 1), "ddt": 500, "fulfillment_rate": base - 0.01 * i + 0.02 * run,
                             "gap": (1 - base) * 10})
    return pd.DataFrame(rows)


def test_summaries_and_tables():
    frame = _frame()
    summary = summarize(frame, metric="fulfillment_rate")
    ours = summary[summary["method"] == "ours"].iloc[0]
    assert ours["n_runs"] == 2 and ours["n_instances"] == 3 and not math.isnan(ours["std"])
    nf = summary[summary["method"] == "nf"].iloc[0]
    assert nf["n_runs"] == 1 and math.isnan(nf["std"])          # single run -> empty std, not 0
    paired = paired_table(frame, "ours", ["nf", "ng", "missing"])
    assert list(paired["comparator"]) == ["nf", "ng"] and (paired["wins"] == 3).all()
    assert abs(paired.loc[0, "mean_delta"] - 21.0) < 1e-9       # ours run mean 0.81 vs nf 0.60 -> 21 pp
    assert "p_holm" in paired.columns and paired["p_holm"].max() <= 1.0
    table = ablation_table(frame)
    assert list(table.columns)[:5] == ["instance_id", "ddt", "scale", "ours", "nf"]
    assert abs(table.loc[0, "delta_nf"] - 0.21) < 1e-9
    advantage = advantage_table(frame, "ours")
    assert set(advantage.columns) >= {"instance_id", "ddt", "scale", "nf", "ng"}
    assert abs(advantage.loc[0, "ng"] - 6.0) < 1e-9
