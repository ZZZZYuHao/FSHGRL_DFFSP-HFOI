"""FSHGRL decision time per episode grouped by DDT and order count (paper figure
``decision_time_configurations``) and the quality-time trade-off panels (``quality_time_tradeoff``)."""
from _common import OUT_DIR, RESULT, require

from result.aggregate import load_eval_results, point_estimates
from result.plot import plot_decision_time, plot_quality_time_tradeoff
from result.stats import DRL_METHODS, METHOD_LABELS, RULE_METHODS


def main() -> None:
    path = RESULT / "eval_results.csv"
    if not require(path):
        return
    frame = point_estimates(load_eval_results(path))
    if (frame["method"] == "ours").any():
        print("[FIG]", plot_decision_time(frame, "ours", out_stem=OUT_DIR / "decision_time_configurations"))
    panels = [("(a) Dispatching rules", ["ours"] + list(RULE_METHODS)), ("(b) DRL methods", ["ours"] + list(DRL_METHODS))]
    print("[FIG]", plot_quality_time_tradeoff(frame, panels, OUT_DIR / "quality_time_tradeoff", METHOD_LABELS))


if __name__ == "__main__":
    main()
