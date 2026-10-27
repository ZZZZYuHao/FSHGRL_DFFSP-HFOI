"""Train the ablation variants (FSHGRL-NF: fluid features off; FSHGRL-NG: point-wise MLP encoders instead of
the graph), per-instance paradigm, N = ablations in configs/runs.yaml.
Products: result/nf/<instance_id>/run<k>/ and result/ng/<instance_id>/run<k>/ (same files as run_03).
Adding a variant = one more line in VARIANTS; the README command does not change."""
from _bootstrap import (instance_ids, n_runs, require_train_confirmed, resolve_workers, run_jobs, step,
                        training_jobs)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
VARIANTS = {"nf": ["configs/ablation/nf.yaml"], "ng": ["configs/ablation/ng.yaml"]}
TIERS = ("benchmark",)
N_RUNS = n_runs("ablations", 3)

require_train_confirmed()
jobs = []
for tag, extra in VARIANTS.items():
    jobs += training_jobs(tag, extra, instance_ids(*TIERS), N_RUNS)
step(f"train {len(VARIANTS)} ablation variants x {N_RUNS} runs (per_instance): {len(jobs)} jobs")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))
