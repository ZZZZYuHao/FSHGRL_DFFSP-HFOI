"""Minute-scale end-to-end smoke test: train 2 iterations (PPO, hgnn) on one instance with configs/smoke.yaml,
evaluate the best greedy checkpoint, run the dispatching rules once, aggregate, draw the figures and build the
tables, then run the unit tests of the reward identity and the statistics. --smoke bypasses the
training-parameter gate; nothing here is paper data (everything goes to result/smoke/)."""
import os
import shutil
import time

from _bootstrap import ROOT, done, merge_csv, run_py, step

CONFIGS = ["configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml", "configs/smoke.yaml"]
INSTANCE = "DDT500_num50"
RUN = f"smoke/ours/{INSTANCE}/run1"              # per-instance layout, as the real matrix
SMOKE_DIR = ROOT / "result" / "smoke"
FIGURE_SCRIPTS = ["paper_assets/figures/src/fig_train_curve.py", "paper_assets/figures/src/fig_convergence.py",
                  "paper_assets/figures/src/fig_ablation_paired_gains.py", "paper_assets/figures/src/fig_decision_time.py"]

t0 = time.time()
step("smoke: remove the previous smoke outputs")
shutil.rmtree(SMOKE_DIR, ignore_errors=True)

step("smoke: train 2 iterations x 2 environments (PPO, hgnn, fluid features on)")
run_py("train.py", "--config", *CONFIGS, "--instances", f"{INSTANCE}.txt", "--run-name", RUN, "--smoke")

step("smoke: evaluate the best greedy checkpoint (greedy + 2 sampled episodes, timing split)")
run_py("eval.py", "--ckpt", f"result/{RUN}/best_greedy_checkpoint.pt", "--decode", "both", "--episodes", "2",
       "--threads", "1", "--method", "ours", "--run-id", "run1", "--out", f"result/{RUN}/eval_results.csv")

step("smoke: dispatching rules on the same instance (one run each)")
run_py("baselines/run_rules.py", "--instances", INSTANCE, "--runs-random", "1", "--out-dir", "result/smoke/rules",
       "--merged-out", "result/smoke/rules.csv", "--reference-out", "none")

step("smoke: aggregate, figures and tables")
merge_csv([ROOT / "result" / RUN / "eval_results.csv", SMOKE_DIR / "rules.csv"], SMOKE_DIR / "eval_results.csv")
run_py("result.aggregate", "--eval", "result/smoke/eval_results.csv", "--out-dir", "result/smoke")
env = dict(os.environ, FSHGRL_RESULT_DIR=str(SMOKE_DIR), FSHGRL_FIG_DIR=str(SMOKE_DIR / "figures"),
           FSHGRL_TABLE_DIR=str(SMOKE_DIR / "tables"))
for script in FIGURE_SCRIPTS:
    run_py(script, env=env)
run_py("paper_assets/scripts/make_tables.py", env=env)

step("smoke: unit tests (reward identity, rule baselines, statistics)")
run_py("pytest", "-q", "tests/test_rules.py", "tests/test_stats.py", "tests/test_config_bridge.py")

done(t0, ROOT / "result" / RUN / "curve.csv", ROOT / "result" / RUN / "train_metrics.csv",
     SMOKE_DIR / "eval_results.csv", SMOKE_DIR / "stats_summary.csv", SMOKE_DIR / "figures",
     SMOKE_DIR / "tables" / "comparison_pdr.tex")
