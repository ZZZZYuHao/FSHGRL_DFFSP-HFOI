"""Objective convention, reference solutions, optimality gap and convergence indicators.

Objective convention
--------------------
The trainer maximises the order fulfilment rate ``eta = completed / total_orders``. The experiment contract
works with a *minimised* objective so that ``gap``, ``obj_best`` and the convergence indicators follow the
usual "smaller is better" reading:

    obj = 1 - eta        (the unfulfilled fraction of the orders)

``obj_ref`` per instance comes from ``data/instances/reference.csv`` (columns ``instance_id, obj_ref, source,
time_limit_s, status``; written by ``baselines/run_rules.py`` from the best deterministic dispatching rule).
Without a reference the gap is NaN, never 0. Note that ``gap`` is undefined when ``obj_ref == 0`` (a reference
that fulfils every order); ``eval_results.csv`` keeps ``obj_ref`` so the absolute difference is always available.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "data" / "instances" / "reference.csv"


def objective_from_fulfillment(rate: float) -> float:
    return 1.0 - float(rate)


def fulfillment_from_objective(obj: float) -> float:
    return 1.0 - float(obj)


def load_reference(path: str | Path = REFERENCE_PATH) -> dict:
    """``{instance_id: obj_ref}`` for rows with ``status == ok``; empty when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    reference = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if str(row.get("status", "ok")).lower() != "ok":
                continue
            try:
                reference[str(row["instance_id"])] = float(row["obj_ref"])
            except (KeyError, ValueError):
                continue
    return reference


def gap_pct(obj, ref) -> float:
    """Relative optimality gap in percent, ``(obj - ref) / |ref| * 100``; NaN without a usable reference."""
    if obj is None or ref is None:
        return math.nan
    obj, ref = float(obj), float(ref)
    if ref == 0 or math.isnan(ref) or math.isnan(obj):
        return math.nan
    return (obj - ref) / abs(ref) * 100.0


def gpu_utilization(device=None):
    """Instantaneous GPU utilisation in percent (needs pynvml); empty string on CPU or when unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        if device is not None and torch.device(device).type != "cuda":
            return ""
        return float(torch.cuda.utilization(device))
    except Exception:
        return ""


def efficiency_metrics(n_env_steps: int, wall_s: float, device=None) -> dict:
    """``sps`` (environment steps per second over the epoch wall-clock) and GPU usage; CPU runs leave the GPU
    columns empty."""
    metrics = {"sps": n_env_steps / max(float(wall_s), 1e-9), "gpu_util": "", "gpu_mem_gb": ""}
    try:
        import torch

        if torch.cuda.is_available() and (device is None or torch.device(device).type == "cuda"):
            metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            metrics["gpu_util"] = gpu_utilization(device)
    except Exception:
        pass
    return metrics


def convergence_metrics(curve, key: str = "obj", tol: float = 0.01):
    """Per-instance convergence indicators from a ``curve.csv`` DataFrame (minimised objective).

    ``final`` = last value, ``best`` = minimum of ``obj_best``, ``epochs_to_tol`` / ``wall_to_tol_s`` = first
    epoch / wall-clock within ``(1 + tol)`` of the final value, ``plateau_epoch`` = last epoch at which
    ``obj_best`` improved.
    """
    import pandas as pd

    out = []
    for instance_id, group in curve.sort_values("epoch").groupby("instance_id"):
        values = group[key].astype(float).values
        final = float(values[-1])
        threshold = final + tol * abs(final)
        hit = int((values <= threshold).argmax())
        improved = group["obj_best"].astype(float).diff().fillna(-1.0) < 0
        out.append({
            "instance_id": instance_id,
            "final": final,
            "best": float(group["obj_best"].astype(float).min()),
            "epochs_to_tol": int(group["epoch"].values[hit]),
            "wall_to_tol_s": float(group["wall_time_s"].values[hit]),
            "plateau_epoch": int(group["epoch"].values[improved.values][-1]) if improved.any() else int(group["epoch"].values[0]),
            "feasible_final": str(group["feasible"].values[-1]) == "True",
            "epochs": int(group["epoch"].max()),
        })
    return pd.DataFrame(out)
