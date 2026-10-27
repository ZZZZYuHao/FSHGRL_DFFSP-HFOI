"""Functions whose bodies must stay byte-identical (checked by AST dump).

These paths cannot be exercised in this environment (no Gurobi licence, no
checkpoint resume in the reference run), so the harness compares their AST
against the golden checkout instead. Each entry is
(golden_file, qualname, candidate_file, qualname). Update the candidate side
when a function moves.
"""
TARGETS = [
    ("agent/training_simulator.py", "TrainingSimulator._solve_cached_lp",
     "environment/simulator.py", "TrainingSimulator._solve_cached_lp"),
    ("agent/training_simulator.py", "TrainingSimulator._fluid_cache_key",
     "environment/simulator.py", "TrainingSimulator._fluid_cache_key"),
    ("agent/training_simulator.py", "TrainingSimulator._fluid_features",
     "environment/simulator.py", "TrainingSimulator._fluid_features"),
    ("agent/training_simulator.py", "TrainingSimulator._load_gurobi",
     "environment/simulator.py", "TrainingSimulator._load_gurobi"),
    ("agent/train.py", "apply_lr_schedule", "train.py", "apply_lr_schedule"),
    ("agent/train.py", "load_checkpoint", "agent/checkpoint.py", "load_checkpoint"),
    ("agent/train.py", "prepare_rollout_policy", "train.py", "prepare_rollout_policy"),
    ("agent/sci_rollout.py", "select_instance", "data/dataset.py", "select_instance"),
    ("agent/train.py", "write_metrics", "result/logger.py", "write_metrics"),
    ("agent/visualization.py", "TrainingVisualizer", "result/logger.py", "TrainingVisualizer"),
]
