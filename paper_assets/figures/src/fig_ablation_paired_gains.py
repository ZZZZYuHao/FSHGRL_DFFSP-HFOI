"""Paired fulfilment differences of FSHGRL over its ablations and the DRL baselines (paper figure
``ablation_paired_gains`` and the advantage heat map ``baseline_advantage_heatmap``)."""
import pandas as pd
from _common import OUT_DIR, RESULT, require

from result.plot import plot_advantage_heatmap, plot_paired_gains
from result.stats import ABLATION_METHODS, DRL_METHODS, METHOD_LABELS, RULE_METHODS


def main() -> None:
    path = RESULT / "advantage_table.csv"
    if not require(path):
        return
    advantage = pd.read_csv(path)
    ablations = [m for m in ABLATION_METHODS if m in advantage.columns]
    if ablations:
        print("[FIG]", plot_paired_gains(advantage, ablations, OUT_DIR / "ablation_paired_gains", METHOD_LABELS))
    methods = [m for m in list(RULE_METHODS) + list(DRL_METHODS) + list(ABLATION_METHODS) if m in advantage.columns]
    if methods:
        print("[FIG]", plot_advantage_heatmap(advantage, methods, OUT_DIR / "baseline_advantage_heatmap", METHOD_LABELS))


if __name__ == "__main__":
    main()
