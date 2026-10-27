"""Synthetic instance generator in the benchmark file format (new code; the original repository had none).

Follows the paper's instance-generation table: 5 product types, 5 stages, 5 machines per stage, processing
times uniform in [25, 450] s, order inter-arrival times uniform in [1, 200] s, due-date tightness uniform in
[300, 2500] s (``due = arrival + tightness``; the exact due-date rule of the shipped benchmark files is not
documented, see docs/known-issues.md), S orders. Every value is drawn from a private ``numpy`` generator
seeded per instance, so an instance id reproduces the same file.

    python -m data.generator --n 5 --out data/instances/synthetic --tier synthetic
    python -m data.generator --n-orders 80 --ddt 900 1200 --seed 7 --out data/instances/synthetic

File format (what data/instance_io.py parses): line 1 ``R J M``; line 2 the machine count per stage; R lines
with M processing times; then the order ids, product types, arrival times and due dates (one line each).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import GeneratorCfg, load_config  # noqa: E402


def generate_instance(seed: int, n_orders: int, cfg: GeneratorCfg | None = None,
                      ddt_range: Sequence[float] | None = None) -> dict:
    """Instance data as arrays (``process_times`` is ``[products, machines]``)."""
    cfg = cfg or GeneratorCfg()
    rng = np.random.default_rng(int(seed))
    machines = cfg.n_stages * cfg.machines_per_stage
    low, high = cfg.process_time_range
    process_times = rng.integers(int(low), int(high) + 1, size=(cfg.n_products, machines))
    low, high = cfg.inter_arrival_range
    gaps = rng.integers(int(low), int(high) + 1, size=n_orders)
    arrivals = np.cumsum(gaps) - gaps[0] + 1          # the first order arrives at t = 1
    low, high = ddt_range or cfg.ddt_range
    tightness = rng.integers(int(low), int(high) + 1, size=n_orders)
    return {
        "n_products": int(cfg.n_products), "n_stages": int(cfg.n_stages),
        "machine_division": [int(cfg.machines_per_stage)] * int(cfg.n_stages),
        "process_times": process_times, "order_ids": [str(i + 1) for i in range(n_orders)],
        "product_types": rng.integers(0, int(cfg.n_products), size=n_orders),
        "arrival_times": arrivals, "due_dates": arrivals + tightness, "seed": int(seed),
        "ddt_range": [int(low), int(high)],
    }


def format_instance(data: dict) -> str:
    lines = [
        f"{data['n_products']} {data['n_stages']} {sum(data['machine_division'])}",
        " ".join(str(v) for v in data["machine_division"]),
    ]
    lines += [" ".join(str(int(v)) for v in row) for row in data["process_times"]]
    lines.append(" ".join(data["order_ids"]))
    lines.append(" ".join(str(int(v)) for v in data["product_types"]))
    lines.append(" ".join(str(int(v)) for v in data["arrival_times"]))
    lines.append(" ".join(str(int(v)) for v in data["due_dates"]))
    return "\n".join(lines) + "\n"


def write_instance(data: dict, instance_id: str, out_dir: Path, tier: str = "synthetic") -> Path:
    from data.dataset import describe_instance, write_params_yaml

    folder = Path(out_dir) / instance_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{instance_id}.txt"
    path.write_text(format_instance(data), encoding="utf-8")
    params = describe_instance(path, tier=tier, source="generated")
    params.update({"seed": data["seed"], "ddt_range": data["ddt_range"]})
    write_params_yaml(folder, params)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1, help="number of instances")
    parser.add_argument("--n-orders", type=int, default=None, help="orders per instance (default: random in the table range)")
    parser.add_argument("--ddt", type=float, nargs=2, default=None, help="due-date tightness range (default: table)")
    parser.add_argument("--seed", type=int, default=0, help="base seed; instance k uses seed + k")
    parser.add_argument("--out", default="data/instances/synthetic")
    parser.add_argument("--tier", default="synthetic")
    parser.add_argument("--prefix", default="SYN")
    args = parser.parse_args()
    cfg = load_config().instance.generator
    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    for k in range(args.n):
        seed = args.seed + k
        rng = np.random.default_rng(seed + 10_000)
        n_orders = args.n_orders or int(rng.integers(int(cfg.n_orders_range[0]), int(cfg.n_orders_range[1]) + 1))
        data = generate_instance(seed, n_orders, cfg, args.ddt)
        instance_id = f"{args.prefix}{seed:04d}_num{n_orders}"
        path = write_instance(data, instance_id, out_dir, args.tier)
        print(f"[GEN] {path}")


if __name__ == "__main__":
    main()
