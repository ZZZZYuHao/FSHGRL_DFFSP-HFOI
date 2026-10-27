"""Evaluate every trained run on its instance and build result/eval_results.csv.

For each result/<method>/<instance_id>/run<k>/ of ours, nf, ng, a2c, dqn, ddqn: one job runs eval.py on the
checkpoint named in configs/runs.yaml (eval.checkpoint, default best_greedy_checkpoint.pt) with greedy
decoding plus eval.sampled_episodes sampled episodes, writing <run_dir>/eval_results.csv (one file per job,
merged afterwards). The rule rows of result/rules.csv are appended, and the training incumbents
(fixed_set_final.csv) are collected into result/incumbents.csv as the alternative estimate."""
import sys
import time

from _bootstrap import ROOT, Job, done, merge_csv, resolve_workers, run_jobs, run_py, runs_cfg, step

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 3, None
METHODS = ["ours", "nf", "ng", "a2c", "dqn", "ddqn"]
EVAL = runs_cfg().get("eval", {}) or {}
CHECKPOINT = str(EVAL.get("checkpoint", "best_greedy_checkpoint.pt"))
EPISODES = int(EVAL.get("sampled_episodes", 10))
BATCH = int(EVAL.get("batch_episodes", 1))
MERGED = ROOT / "result" / "eval_results.csv"

t0 = time.time()
jobs, parts = [], []
for method in METHODS:
    for ckpt in sorted((ROOT / "result" / method).glob(f"*/run*/{CHECKPOINT}")):
        run_dir = ckpt.parent
        out = run_dir / "eval_results.csv"
        parts.append(out)
        jobs.append(Job(f"{method}/{run_dir.parent.name}/{run_dir.name}/eval",
                        [sys.executable, "eval.py", "--ckpt", ckpt.relative_to(ROOT), "--decode", "both",
                         "--episodes", EPISODES, "--batch-episodes", BATCH, "--threads", "1", "--method", method,
                         "--run-id", run_dir.name, "--out", out.relative_to(ROOT)],
                        output=out))
if not jobs:
    sys.exit("[FAIL] no trained runs under result/<method>/<instance>/run*/: run scripts 03-05 first")
step(f"evaluate {len(jobs)} checkpoints ({CHECKPOINT}; greedy + {EPISODES} sampled episodes each)")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))

step("merge evaluation rows with the rule results; collect training incumbents")
rules = ROOT / "result" / "rules.csv"
if not rules.exists():
    print("[WARN] result/rules.csv missing (run scripts/run_06_run_rules.py); eval_results.csv has no rule rows")
merge_csv(parts + [rules], MERGED)
incumbent_parts = []
for method in METHODS:
    if list((ROOT / "result" / method).glob("*/run*/fixed_set_final.csv")):
        out = ROOT / "result" / f"{method}_incumbents.csv"
        run_py("eval.py", "--collect", f"result/{method}/*/run*/fixed_set_final.csv", "--method", method,
               "--out", out.relative_to(ROOT))
        incumbent_parts.append(out)
merge_csv(incumbent_parts, ROOT / "result" / "incumbents.csv")
done(t0, MERGED, ROOT / "result" / "incumbents.csv")
