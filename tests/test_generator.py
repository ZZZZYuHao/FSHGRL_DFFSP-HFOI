"""data/generator.py: reproducible files in the benchmark format that the parser and simulator accept."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.baselines.rules import RulePolicy, run_rule_episode  # noqa: E402
from data.generator import format_instance, generate_instance, write_instance  # noqa: E402
from data.instance_io import parse_instance_file  # noqa: E402


def test_generated_instance_is_reproducible_and_parsable(tmp_path):
    first = generate_instance(seed=3, n_orders=30)
    second = generate_instance(seed=3, n_orders=30)
    assert format_instance(first) == format_instance(second)
    path = write_instance(first, "SYN0003_num30", tmp_path, tier="synthetic")
    spec = parse_instance_file(path)
    assert spec.total_orders == 30 and spec.product_count == 5 and spec.machine_count == 25
    assert (spec.due_dates - spec.arrival_times).min() >= 300 and (spec.due_dates - spec.arrival_times).max() <= 2500
    assert (path.parent / "params.yaml").exists()
    record = run_rule_episode(RulePolicy("spt"), spec, {"order_top_k": 5, "fluid_mode": "off"})
    assert record["completed"] + record["discarded"] == 30
