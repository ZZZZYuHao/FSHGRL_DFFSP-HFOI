"""Rollout throughput versus num_envs (the environment-batch sizing evidence for the README).

Zero-argument tool: edit the constants below. Collects one iteration of complete episodes with a freshly
initialised policy for each num_envs value and reports transitions per second, the time split and peak
memory. Writes result/profile_num_envs.csv.
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
from rollout import collect_vectorized_rollouts  # noqa: E402
from train import load_config  # noqa: E402

INSTANCE = "DDT500_num50"
NUM_ENVS = [1, 2, 4, 8]
MODEL_TYPE = "hgnn"
DEVICE = "cpu"
SEED = 0
OUT = "result/profile_num_envs.csv"


def main() -> None:
    config = load_config(None)
    train_paras = dict(config["train_paras"])
    train_paras.update({"model_type": MODEL_TYPE, "viz": False})
    torch.manual_seed(SEED)
    policy = build_scheduler(config, train_paras, DEVICE)
    policy.eval()
    path = benchmark_instance_files([INSTANCE])[0]
    rows = []
    for num_envs in NUM_ENVS:
        paras = dict(train_paras, num_envs=num_envs, num_workers=num_envs)
        started = time.perf_counter()
        buffer = collect_vectorized_rollouts(policy, path, paras)
        wall = time.perf_counter() - started
        row = {
            "num_envs": num_envs,
            "transitions": len(buffer),
            "wall_s": round(wall, 3),
            "transitions_per_s": round(len(buffer) / max(wall, 1e-9), 1),
            "policy_forward_count": buffer.policy_forward_count,
            "actor_batch_mean": round(buffer.actor_batch_size_mean, 2),
            "policy_forward_s": round(buffer.policy_forward_seconds, 3),
            "obs_build_s": round(buffer.obs_build_seconds, 3),
            "simulator_step_s": round(buffer.simulator_step_seconds, 3),
            "episode_seconds_mean": round(sum(buffer.episode_seconds) / max(len(buffer.episode_seconds), 1), 3),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        }
        rows.append(row)
        print(f"[PROFILE] num_envs={num_envs}: {row['transitions']} transitions in {row['wall_s']} s "
              f"({row['transitions_per_s']} /s), policy {row['policy_forward_s']} s, obs {row['obs_build_s']} s, "
              f"step {row['simulator_step_s']} s, RSS {row['max_rss_mb']} MB", flush=True)
    out = ROOT / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[PROFILE] -> {out}")


if __name__ == "__main__":
    main()
