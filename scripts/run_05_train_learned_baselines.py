"""Train the learned baselines (A2C, DQN, DDQN; new implementations on the training simulator, see
docs/migration-map.md), per-instance paradigm with the same num_envs / max_iterations budget as the proposed
method, N = learned_baselines in configs/runs.yaml. Products: result/<method>/<instance_id>/run<k>/."""
from _bootstrap import (instance_ids, n_runs, require_train_confirmed, resolve_workers, run_jobs, step,
                        training_jobs)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
BASELINES = {"a2c": ["configs/baselines/a2c.yaml"], "dqn": ["configs/baselines/dqn.yaml"],
             "ddqn": ["configs/baselines/ddqn.yaml"]}
TIERS = ("benchmark",)
N_RUNS = n_runs("learned_baselines", 1)

require_train_confirmed()
jobs = []
for tag, extra in BASELINES.items():
    jobs += training_jobs(tag, extra, instance_ids(*TIERS), N_RUNS)
step(f"train {len(BASELINES)} learned baselines x {N_RUNS} run(s) (per_instance): {len(jobs)} jobs")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))
