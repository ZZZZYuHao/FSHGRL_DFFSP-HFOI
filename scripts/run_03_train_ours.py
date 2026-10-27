"""Train the proposed method (FSHGRL: PPO + fluid features + sparse HGNN policy), per-instance paradigm:
every benchmark instance x N independent runs (N = ours in configs/runs.yaml; a single run gets a fixed seed).
Products: result/ours/<instance_id>/run<k>/{train_metrics.csv,curve.csv,log.csv,best_greedy_checkpoint.pt,
best_checkpoint.pt,last_checkpoint.pt,config_snapshot.yaml,train_meta.yaml,fixed_set_final.csv}.
Interrupted runs resume from last_checkpoint.pt; finished runs (train_meta.yaml present) are skipped."""
from _bootstrap import (instance_ids, n_runs, require_train_confirmed, resolve_workers, run_jobs, step,
                        training_jobs)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
TIERS = ("benchmark",)
N_RUNS = n_runs("ours", 3)

require_train_confirmed()
jobs = training_jobs("ours", [], instance_ids(*TIERS), N_RUNS)
step(f"train ours (per_instance): {len(jobs)} jobs = {len(instance_ids(*TIERS))} instances x {N_RUNS} runs")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))
