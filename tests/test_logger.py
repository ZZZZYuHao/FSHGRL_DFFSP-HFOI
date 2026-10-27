"""Unit tests for result/logger.py and result/metrics.py (no training involved)."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result.logger import RunLogger, write_metrics  # noqa: E402
from result.metrics import convergence_metrics, gap_pct, load_reference, objective_from_fulfillment  # noqa: E402

NO_VIZ = {"viz": False}


def _rows(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_curve_incumbent_and_resume(tmp_path):
    run_dir = tmp_path / "run"
    logger = RunLogger(run_dir, NO_VIZ, resume=False)
    out = logger.curve(0, [{"instance_id": "A", "obj": 0.5, "gap": math.nan, "feasible": True, "wall_time_s": 1.0}])
    assert out[0]["obj_best"] == 0.5
    out = logger.curve(1, [{"instance_id": "A", "obj": 0.6, "obj_best": 0.4, "gap": math.nan, "feasible": True,
                            "wall_time_s": 2.0}])
    assert out[0]["obj_best"] == 0.4            # sampled incumbent passed in by the caller
    out = logger.curve(2, [{"instance_id": "A", "obj": 0.45, "gap": 12.5, "feasible": True, "wall_time_s": 3.0}])
    assert out[0]["obj_best"] == 0.4            # running minimum
    logger.env_steps = 300
    logger.log(2, {"env_steps": 300, "wall_time_s": 3.0, "train_obj_mean": 0.55, "fixed_obj_mean": 0.45})
    logger.write_final(epochs=3, wall_time_s=3.0)
    logger.close()

    curve = _rows(run_dir / "curve.csv")
    assert [row["epoch"] for row in curve] == ["0", "1", "2"]
    assert list(curve[0].keys()) == RunLogger.CURVE_COLS
    log_rows = _rows(run_dir / "log.csv")
    assert list(log_rows[0].keys()) == RunLogger.LOG_COLS and log_rows[0]["fixed_gap_mean"] == ""
    final = _rows(run_dir / "fixed_set_final.csv")
    assert final[0]["instance_id"] == "A" and float(final[0]["obj_best"]) == 0.4
    assert float(final[0]["obj_greedy_final"]) == 0.45

    resumed = RunLogger(run_dir, NO_VIZ, resume=True)
    assert resumed.incumbent == {"A": 0.4} and resumed.env_steps == 300
    assert resumed.wall_time() >= 3.0
    resumed.close()

    fresh = RunLogger(run_dir, NO_VIZ, resume=False)
    assert fresh.incumbent == {} and not (run_dir / "curve.csv").exists()
    fresh.close()


def test_write_metrics_legacy_append(tmp_path):
    path = tmp_path / "train_metrics.csv"
    write_metrics(path, {"iteration": 0, "reward": 1.5})
    write_metrics(path, {"iteration": 1, "reward": 2.5})
    rows = _rows(path)
    assert [row["iteration"] for row in rows] == ["0", "1"]
    write_metrics(path, {"iteration": 2, "reward": 3.5, "extra": "x"})   # header reconciliation
    rows = _rows(path)
    assert list(rows[0].keys()) == ["iteration", "reward", "extra"] and rows[0]["extra"] == ""


def test_snapshot_and_meta_are_yaml(tmp_path):
    import yaml

    logger = RunLogger(tmp_path / "run", NO_VIZ)
    logger.write_snapshot({"seed": 42, "path": Path("x/y"), "nested": {"a": (1, 2)}})
    logger.write_meta({"stop_reason": "max_iterations", "best": float("inf")})
    snapshot = yaml.safe_load((tmp_path / "run" / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot["path"] == "x/y" and snapshot["nested"]["a"] == [1, 2]
    meta = yaml.safe_load((tmp_path / "run" / "train_meta.yaml").read_text(encoding="utf-8"))
    assert meta["stop_reason"] == "max_iterations" and meta["best"] == float("inf")
    logger.close()


def test_gap_and_reference(tmp_path):
    assert math.isnan(gap_pct(0.5, None)) and math.isnan(gap_pct(0.5, 0.0))
    assert abs(gap_pct(0.5, 0.4) - 25.0) < 1e-12
    assert objective_from_fulfillment(0.75) == 0.25
    ref_path = tmp_path / "reference.csv"
    ref_path.write_text("instance_id,obj_ref,source,time_limit_s,status\nA,0.2,rule,0,ok\nB,0.3,rule,0,failed\n",
                        encoding="utf-8")
    assert load_reference(ref_path) == {"A": 0.2}
    assert load_reference(tmp_path / "missing.csv") == {}


def test_convergence_metrics():
    import pandas as pd

    curve = pd.DataFrame({
        "epoch": [0, 1, 2, 3], "instance_id": ["A"] * 4, "obj": [0.5, 0.4, 0.42, 0.405],
        "obj_best": [0.5, 0.4, 0.4, 0.4], "gap": [math.nan] * 4, "feasible": [True] * 4,
        "wall_time_s": [1.0, 2.0, 3.0, 4.0],
    })
    table = convergence_metrics(curve, key="obj", tol=0.05)
    row = table.iloc[0]
    assert row["final"] == 0.405 and row["best"] == 0.4
    assert row["epochs_to_tol"] == 1 and row["wall_to_tol_s"] == 2.0 and row["plateau_epoch"] == 1
    assert bool(row["feasible_final"]) and row["epochs"] == 3
