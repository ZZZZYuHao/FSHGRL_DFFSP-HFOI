"""Pilot experiment (training gate). Disabled for this project (configs/pilot.yaml enabled: false): the
paradigm is per-instance training on the 15 benchmark instances themselves, so a pilot tier would train on
instances outside every reported result; the reason is recorded in docs/experiment-spec.md and
docs/pilot-log.md. When enabled, the listed methods are trained for pilot.max_iterations on pilot.instances,
result/pilot/pilot-report.md is written and verdict.txt is PASS when the proposed method reaches the best
rule reference (data/instances/reference.csv) on a majority of the pilot instances."""
import sys
import time

from _bootstrap import (CONFIGS, ROOT, Job, done, pilot_cfg, pilot_enabled, require_train_confirmed,
                        resolve_workers, run_jobs, step)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
METHOD_CONFIGS = {"ours": [], "a2c": ["configs/baselines/a2c.yaml"], "dqn": ["configs/baselines/dqn.yaml"],
                  "ddqn": ["configs/baselines/ddqn.yaml"]}
PILOT_DIR = ROOT / "result" / "pilot"
VERDICT = PILOT_DIR / "verdict.txt"

if not pilot_enabled():
    print("[SKIP] configs/pilot.yaml enabled: false (per-instance paradigm; reason in docs/experiment-spec.md)", flush=True)
    sys.exit(0)
if VERDICT.exists() and VERDICT.read_text(encoding="utf-8").startswith("PASS"):
    print(f"[SKIP] pilot already passed: {VERDICT}", flush=True)
    sys.exit(0)

t0 = time.time()
cfg = pilot_cfg()
instances = [str(i) for i in cfg.get("instances", [])]
methods = [str(m) for m in cfg.get("methods", ["ours"])]
max_iterations = int(cfg.get("max_iterations", 200))
require_train_confirmed()
jobs = []
for method in methods:
    for iid in instances:
        name = f"pilot/{method}/{iid}/run1"
        jobs.append(Job(name, [sys.executable, "train.py", "--config", *CONFIGS, *METHOD_CONFIGS.get(method, []),
                               "--instances", f"{iid}.txt", "--run-name", name, "--max-iterations", str(max_iterations),
                               "--seed", "1"],
                        output=ROOT / "result" / name / "train_meta.yaml"))
step(f"pilot: {len(methods)} methods x {len(instances)} instances x {max_iterations} iterations")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))

step("pilot: report and verdict")
import pandas as pd  # noqa: E402

from result.metrics import load_reference  # noqa: E402

reference = load_reference()
rows = []
for method in methods:
    for iid in instances:
        final = ROOT / "result" / f"pilot/{method}/{iid}/run1" / "fixed_set_final.csv"
        if final.exists():
            record = pd.read_csv(final).iloc[0]
            rows.append({"method": method, "instance_id": iid, "obj_best": float(record["obj_best"]),
                         "obj_greedy_final": float(record["obj_greedy_final"]), "obj_ref": reference.get(iid)})
table = pd.DataFrame(rows)
PILOT_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(PILOT_DIR / "pilot_results.csv", index=False)
ours = table[table["method"] == "ours"]
wins = int(((ours["obj_best"] <= ours["obj_ref"]) & ours["obj_ref"].notna()).sum()) if not ours.empty else 0
judged = int(ours["obj_ref"].notna().sum()) if not ours.empty else 0
verdict = "PASS" if judged and wins * 2 >= judged else "FAIL"
lines = ["# Pilot report", "", f"Verdict: **{verdict}** (ours reaches the best rule on {wins}/{judged} pilot instances)", "",
         table.to_markdown(index=False) if not table.empty else "(no results)"]
(PILOT_DIR / "pilot-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
VERDICT.write_text(f"{verdict}\n", encoding="utf-8")
done(t0, PILOT_DIR / "pilot-report.md", VERDICT)
if verdict != "PASS":
    sys.exit("[FAIL] pilot verdict FAIL: see result/pilot/pilot-report.md and docs/pilot-log.md")
