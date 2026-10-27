"""Rollout throughput of the multi-process collector versus the in-process path (zero-argument tool).

For every entry of WORKERS the same trajectory is collected (same seed before each collection; the tool
asserts that rewards, actions and log-probabilities equal the in-process run), the collection is timed
(pool start-up separately) and the time split and peak memory are recorded in result/profile_collector.csv.
The verdict follows the skill rule: recommend ``collector_workers: k`` only when the rollout speed-up is at
least 1.5x, otherwise keep 0. Edit the constants to profile another instance or environment count.
"""
from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from agent.checkpoint import build_scheduler  # noqa: E402
from data.dataset import benchmark_instance_files  # noqa: E402
from environment.collector import CollectorPool  # noqa: E402
from rollout import collect_vectorized_rollouts, collect_vectorized_rollouts_parallel  # noqa: E402
from train import load_config  # noqa: E402

INSTANCE = "DDT500_num50"
NUM_ENVS = 8
WORKERS = [0, 1, 2, 3]
REPEATS = 2                  # the best of REPEATS collections counts
MODEL_TYPE = "hgnn"
DEVICE = "cpu"
SEED = 0
THRESHOLD = 1.5
OUT = "result/profile_collector.csv"


def main() -> None:
    torch.set_num_threads(1)
    config = load_config(None)
    train_paras = dict(config["train_paras"])
    train_paras.update({"model_type": MODEL_TYPE, "viz": False, "num_envs": NUM_ENVS, "num_workers": NUM_ENVS})
    torch.manual_seed(SEED)
    policy = build_scheduler(config, train_paras, DEVICE)
    policy.eval()
    path = benchmark_instance_files([INSTANCE])[0]
    rows, reference = [], None
    for workers in WORKERS:
        pool, startup = None, 0.0
        if workers > 0:
            started = time.perf_counter()
            pool = CollectorPool(NUM_ENVS, workers)
            pool.load(path, train_paras)                      # warm-up: the workers import torch on first use
            startup = time.perf_counter() - started
        walls, buffer = [], None
        for _ in range(REPEATS):
            torch.manual_seed(SEED + 1)                       # identical trajectory in every configuration
            started = time.perf_counter()
            if pool is None:
                buffer = collect_vectorized_rollouts(policy, path, train_paras)
            else:
                buffer = collect_vectorized_rollouts_parallel(policy, path, train_paras, pool=pool)
            walls.append(time.perf_counter() - started)
        identical = reference is None or (buffer.rewards == reference.rewards
                                          and buffer.env_actions == reference.env_actions
                                          and buffer.log_probs == reference.log_probs)
        if reference is None:
            reference = buffer
        children_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0 if pool else 0.0
        if pool is not None:
            pool.close()
        row = {
            "workers": workers,
            "transitions": len(buffer),
            "wall_s": round(min(walls), 3),
            "transitions_per_s": round(len(buffer) / max(min(walls), 1e-9), 1),
            "policy_forward_s": round(buffer.policy_forward_seconds, 3),
            "obs_build_s": round(buffer.obs_build_seconds, 3),
            "simulator_step_s": round(buffer.simulator_step_seconds, 3),
            "pool_startup_s": round(startup, 2),               # process start + first load (torch import)
            "identical_to_inprocess": bool(identical),
            "parent_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
            "worker_rss_mb": round(children_rss, 1),           # peak RSS of the largest worker so far
        }
        rows.append(row)
        print(f"[PROFILE] workers={workers}: {row['transitions']} transitions in {row['wall_s']} s "
              f"({row['transitions_per_s']} /s), forward {row['policy_forward_s']} s, obs {row['obs_build_s']} s, "
              f"step {row['simulator_step_s']} s, start-up {row['pool_startup_s']} s, "
              f"identical={row['identical_to_inprocess']}", flush=True)
    baseline = rows[0]["wall_s"]
    for row in rows:
        row["speedup"] = round(baseline / max(row["wall_s"], 1e-9), 2)
    frame = pd.DataFrame(rows)
    out = ROOT / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    candidates = [row for row in rows if row["workers"] > 0 and row["identical_to_inprocess"]]
    best = max(candidates, key=lambda row: row["speedup"]) if candidates else None
    if best is not None and best["speedup"] >= THRESHOLD:
        print(f"[PROFILE] speed-up {best['speedup']}x at workers={best['workers']} -> recommend "
              f"env.collector_workers: {best['workers']} on this machine")
    else:
        reached = f"{best['speedup']}x at workers={best['workers']}" if best else "n/a"
        print(f"[PROFILE] best rollout speed-up {reached} < {THRESHOLD}x -> keep env.collector_workers: 0")
    if not all(row["identical_to_inprocess"] for row in rows):
        print("[PROFILE] WARNING: a configuration produced a different trajectory; do not use it")
    print(f"[PROFILE] -> {out}")


if __name__ == "__main__":
    main()
