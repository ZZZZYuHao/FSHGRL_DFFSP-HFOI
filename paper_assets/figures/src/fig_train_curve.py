"""Training traces of two representative configurations (paper figure ``train_curve``).

Reads result/ours/<instance>/run1/curve.csv (greedy fulfilment after every epoch) and draws the raw trace
plus its moving average, as the former train_curve.py did on train_metrics.csv. Zero-argument: edit INSTANCES.
"""
from _common import OUT_DIR, RESULT, require

from result.plot import plot_training_trace

INSTANCES = ["DDT500_num50", "DDT1200_num100"]
RUN = "run1"


def main() -> None:
    for instance in INSTANCES:
        curve = RESULT / "ours" / instance / RUN / "curve.csv"
        if not require(curve):
            continue
        files = plot_training_trace(curve, OUT_DIR / f"train_curve_{instance}", column="obj", x_column="epoch",
                                    title=instance.replace("_", ", "))
        print(f"[FIG] {files}")


if __name__ == "__main__":
    main()
