"""Instance discovery and selection (moved verbatim from agent/sci_rollout.py)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence


def scan_instance_files(data_dir: str, patterns: Iterable[str]) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        if "isntance" in str(root):
            raise FileNotFoundError(
                f"Training data directory '{root}' does not exist. "
                "Use 'data/instance/competition' instead of 'data/isntance/competition'."
            )
        raise FileNotFoundError(f"Training data directory '{root}' does not exist.")

    files = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    files = sorted({
        path.resolve()
        for path in files
        if not path.name.startswith("._") and path.name != "Introduction.txt"
    })
    if not files:
        raise FileNotFoundError(f"No training instance files matched {list(patterns)} under '{root}'.")
    return files


def resolve_instance_files(data_dir: str, patterns: Iterable[str],
                           instance_names: Sequence[str] | None = None) -> List[Path]:
    all_files = scan_instance_files(data_dir, ["*.txt"] if instance_names else patterns)
    if not instance_names:
        return all_files

    by_name = {path.name: path for path in all_files}
    selected = []
    missing = []
    for name in instance_names:
        key = Path(name).name
        if key in by_name:
            selected.append(by_name[key])
        else:
            missing.append(name)
    if missing:
        available = ", ".join(sorted(by_name.keys()))
        raise FileNotFoundError(
            f"Training instances not found under '{data_dir}': {missing}. "
            f"Available instances: {available}"
        )
    return selected


def select_instance(instance_files: List[Path], strategy: str, index: int) -> tuple[int, Path]:
    if not instance_files:
        raise ValueError("No training instances are available.")
    strategy = (strategy or "fixed").lower()
    if strategy == "random":
        selected_index = random.randrange(len(instance_files))
        return selected_index, instance_files[selected_index]
    if strategy in {"fixed", "round_robin"}:
        selected_index = index % len(instance_files)
        return selected_index, instance_files[selected_index]
    raise ValueError("instance_selection must be one of: fixed, round_robin, random")


# ------------------------------------------------------------- new code: instance folders and index.csv
# One folder per instance: data/instances/<tier>/<instance_id>/<instance_id>.txt + params.yaml. The .txt file
# keeps its original name because train_metrics.csv records it in the instance_name column.
import csv  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from data.instance_io import parse_instance_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INSTANCES_DIR = ROOT / "data" / "instances"
BENCHMARK_DIR = INSTANCES_DIR / "benchmark"
INDEX_PATH = INSTANCES_DIR / "index.csv"
REFERENCE_PATH = INSTANCES_DIR / "reference.csv"
INDEX_COLS = ["instance_id", "tier", "scale", "ddt", "n_orders", "n_products", "n_stages", "n_machines", "source", "path"]
_DDT_PATTERN = re.compile(r"DDT(\d+)")


def instance_id_of(path: str | Path) -> str:
    return Path(path).stem


def parse_ddt(instance_id: str):
    """Due-date tightness encoded in the benchmark file names (DDT500_num50 -> 500); None when absent."""
    match = _DDT_PATTERN.search(str(instance_id))
    return int(match.group(1)) if match else None


def describe_instance(path: str | Path, tier: str = "benchmark", source: str = "external") -> dict:
    """Instance-level parameters for params.yaml (derived from the file; nothing is re-sampled)."""
    path = Path(path)
    spec = parse_instance_file(path)
    arrivals = spec.arrival_times.astype(np.float64)
    due_dates = spec.due_dates.astype(np.float64)
    slack = due_dates - arrivals
    inter_arrival = np.diff(np.sort(arrivals)) if arrivals.size > 1 else np.zeros(0)
    positive_times = spec.process_times[spec.process_times > 0].astype(np.float64)
    orders_per_product = np.bincount(spec.product_types, minlength=spec.product_count)
    return {
        "instance_id": path.stem,
        "tier": tier,
        "source": source,
        "file": path.name,
        "n_products": int(spec.product_count),
        "n_stages": int(spec.stage_count),
        "n_machines": int(spec.machine_count),
        "machine_division": [int(v) for v in spec.machine_division],
        "n_orders": int(spec.total_orders),
        "ddt": parse_ddt(path.stem),
        "orders_per_product": [int(v) for v in orders_per_product],
        "arrival_first": float(arrivals.min()) if arrivals.size else 0.0,
        "arrival_last": float(arrivals.max()) if arrivals.size else 0.0,
        "inter_arrival_mean": float(inter_arrival.mean()) if inter_arrival.size else 0.0,
        "due_slack_min": float(slack.min()) if slack.size else 0.0,
        "due_slack_mean": float(slack.mean()) if slack.size else 0.0,
        "due_slack_max": float(slack.max()) if slack.size else 0.0,
        "process_time_min": float(positive_times.min()) if positive_times.size else 0.0,
        "process_time_mean": float(positive_times.mean()) if positive_times.size else 0.0,
        "process_time_max": float(positive_times.max()) if positive_times.size else 0.0,
    }


def write_params_yaml(folder: str | Path, params: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "params.yaml"
    with target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(params, file, sort_keys=False, allow_unicode=True)
    return target


def prepare_benchmark(benchmark_dir: str | Path = BENCHMARK_DIR, index_path: str | Path = INDEX_PATH,
                      tier: str = "benchmark", source: str = "external") -> List[dict]:
    """Write params.yaml into every instance folder and the tier index (data/instances/index.csv).

    Idempotent: re-running rewrites identical files. ``scale`` is the number of orders (the paper's S) and
    ``ddt`` the due-date tightness from the file name; ``path`` is relative to the repository root.
    """
    benchmark_dir = Path(benchmark_dir)
    files = sorted(
        path for path in benchmark_dir.glob("*/*.txt")
        if path.is_file() and not path.name.startswith("._") and path.stem == path.parent.name
    )
    if not files:
        raise FileNotFoundError(f"No instance folders found under '{benchmark_dir}'.")
    rows = []
    for path in files:
        params = describe_instance(path, tier=tier, source=source)
        write_params_yaml(path.parent, params)
        rows.append({
            "instance_id": params["instance_id"],
            "tier": tier,
            "scale": params["n_orders"],
            "ddt": params["ddt"] if params["ddt"] is not None else "",
            "n_orders": params["n_orders"],
            "n_products": params["n_products"],
            "n_stages": params["n_stages"],
            "n_machines": params["n_machines"],
            "source": source,
            "path": path.relative_to(ROOT).as_posix(),
        })
    rows.sort(key=lambda row: (row["ddt"] if row["ddt"] != "" else -1, row["scale"], row["instance_id"]))
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_COLS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_index(index_path: str | Path = INDEX_PATH) -> dict:
    """``{instance_id: row}`` of index.csv (typed scale/ddt); empty when the index has not been written."""
    index_path = Path(index_path)
    if not index_path.exists():
        return {}
    index = {}
    with index_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            row = dict(row)
            for key in ("scale", "n_orders", "n_products", "n_stages", "n_machines"):
                if row.get(key, "") != "":
                    row[key] = int(float(row[key]))
            row["ddt"] = int(float(row["ddt"])) if row.get("ddt", "") != "" else None
            index[row["instance_id"]] = row
    return index


def benchmark_instance_files(instance_ids: Sequence[str] | None = None,
                             benchmark_dir: str | Path = BENCHMARK_DIR) -> List[Path]:
    """Instance files of the benchmark tier (all, or the given ids / file names), in index order."""
    benchmark_dir = Path(benchmark_dir)
    by_id = {path.stem: path for path in benchmark_dir.glob("*/*.txt")
             if path.is_file() and not path.name.startswith("._") and path.stem == path.parent.name}
    if not by_id:
        raise FileNotFoundError(f"No instance folders found under '{benchmark_dir}'.")
    if instance_ids:
        missing = [name for name in instance_ids if Path(name).stem not in by_id]
        if missing:
            raise FileNotFoundError(f"Unknown benchmark instances {missing}; available: {sorted(by_id)}")
        return [by_id[Path(name).stem] for name in instance_ids]
    index = load_index()
    ordered = [iid for iid in index if iid in by_id] + sorted(iid for iid in by_id if iid not in index)
    return [by_id[iid] for iid in ordered]


if __name__ == "__main__":   # python -m data.dataset  (scripts/run_01_prepare_data.py)
    _rows = prepare_benchmark()
    print(f"[DATA] {len(_rows)} benchmark instances indexed -> {INDEX_PATH}")
