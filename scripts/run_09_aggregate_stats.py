"""Aggregate statistics, figures and LaTeX tables from result/eval_results.csv and the run directories:
result/stats_summary.csv, paired_tests.csv (exact Wilcoxon + Holm, the paper's Table 11 layout),
ablation_results.csv, advantage_table.csv, friedman.csv, conv_<method>.csv, the figures under
paper_assets/figures/generated/ and the tables under paper_assets/tables/."""
import time

from _bootstrap import ROOT, done, run_py, step

FIGURE_SCRIPTS = ["paper_assets/figures/src/fig_train_curve.py", "paper_assets/figures/src/fig_convergence.py",
                  "paper_assets/figures/src/fig_ablation_paired_gains.py", "paper_assets/figures/src/fig_decision_time.py"]

t0 = time.time()
step("aggregate: summaries and paired tests")
run_py("result.aggregate")
step("figures")
for script in FIGURE_SCRIPTS:
    run_py(script)
step("LaTeX tables")
run_py("paper_assets/scripts/make_tables.py")
done(t0, ROOT / "result" / "stats_summary.csv", ROOT / "result" / "paired_tests.csv",
     ROOT / "result" / "ablation_results.csv", ROOT / "paper_assets" / "figures" / "generated",
     ROOT / "paper_assets" / "tables" / "wilcoxon.tex")
