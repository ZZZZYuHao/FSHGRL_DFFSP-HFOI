"""Orchestrates the equivalence harness. Run from the repository root:

    python -m tests.equivalence            # L1-L7 (about 3 minutes on 4 CPUs)
    python -m tests.equivalence --quick    # L1-L5 and L7 only
    python -m tests.equivalence --determinism   # also re-run golden and compare golden vs golden

Golden artefacts are produced from the untouched original commit (c030dfa),
checked out automatically under tests/golden/_work/src.
"""
from __future__ import annotations

import argparse
import ast
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import candidate
from .ast_targets import TARGETS
from .common import (GOLDEN_COMMIT, GOLDEN_DIR, INSTANCE_NAME, LEGACY_CONFIG, MODEL_TYPES, ROOT, WORK_DIR,
                     compare_checkpoint, compare_csv, compare_nested, pinned_env, run_legacy_train, run_subprocess)

CHECKPOINT_FILES = ("best_checkpoint.pt", "last_checkpoint.pt", "ppo_policy_model.pt")


def ensure_golden_source() -> Path:
    src = WORK_DIR / "src"
    if (src / "agent" / "train.py").exists():
        head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == GOLDEN_COMMIT:
            return src
        shutil.rmtree(src)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(ROOT), str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", GOLDEN_COMMIT], check=True)
    return src


def load_base_config(src: Path) -> dict:
    if not LEGACY_CONFIG.exists():
        shutil.copyfile(src / "data" / "config.json", LEGACY_CONFIG)
    with LEGACY_CONFIG.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_manifest() -> None:
    import numpy
    import torch
    try:
        import gurobipy  # noqa: F401
        gurobi = True
    except Exception:
        gurobi = False
    manifest = {
        "golden_commit": GOLDEN_COMMIT,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "gurobipy_available": gurobi,
        "instance": INSTANCE_NAME,
        "note": "generated on CPU with OMP/MKL threads pinned to 1; fluid LP falls back to the heuristic path",
    }
    (GOLDEN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def ensure_golden_run(src: Path, base_config: dict, model_type: str, regenerate: bool) -> tuple[Path, Path]:
    csv_path = GOLDEN_DIR / model_type / "train_metrics.csv"
    pt_dir = WORK_DIR / f"golden_{model_type}"
    complete = csv_path.exists() and all((pt_dir / name).exists() for name in CHECKPOINT_FILES)
    if complete and not regenerate:
        return csv_path, pt_dir
    print(f"[golden] generating reference run for model_type={model_type} ...")
    result_dir = run_legacy_train(src, base_config, model_type, f"golden_{model_type}",
                                  WORK_DIR / f"golden_{model_type}.log")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result_dir / "train_metrics.csv", csv_path)
    pt_dir.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINT_FILES:
        shutil.copyfile(result_dir / name, pt_dir / name)
    write_manifest()
    return csv_path, pt_dir


def run_probe(tree_root: Path, instance_path: Path, out_path: Path) -> dict:
    import torch

    cmd = [sys.executable, str(ROOT / "tests" / "equivalence" / "probe.py"), str(out_path), str(LEGACY_CONFIG),
           str(instance_path), *MODEL_TYPES]
    run_subprocess(cmd, tree_root, out_path.with_suffix(".log"))
    return torch.load(out_path, map_location="cpu", weights_only=False)


def _function_node(tree: ast.Module, qualname: str):
    parts = qualname.split(".")
    scope = tree.body
    node = None
    for part in parts:
        node = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            return None
        scope = getattr(node, "body", [])
    return node


def _strip_docstring(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    clone = ast.copy_location(type(node)(**{**node.__dict__, "body": body}), node)
    return ast.dump(clone, include_attributes=False)


def ast_identity(src: Path) -> list[str]:
    mismatches = []
    for golden_file, golden_name, cand_file, cand_name in TARGETS:
        g_tree = ast.parse((src / golden_file).read_text(encoding="utf-8"))
        c_path = ROOT / cand_file
        if not c_path.exists():
            mismatches.append(f"L7 {cand_file} missing")
            continue
        c_tree = ast.parse(c_path.read_text(encoding="utf-8"))
        g_node, c_node = _function_node(g_tree, golden_name), _function_node(c_tree, cand_name)
        if g_node is None or c_node is None:
            mismatches.append(f"L7 {golden_name} / {cand_name}: function not found")
            continue
        if _strip_docstring(g_node) != _strip_docstring(c_node):
            mismatches.append(f"L7 {cand_file}:{cand_name} differs from {golden_file}:{golden_name}")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the end-to-end run (L6)")
    parser.add_argument("--regenerate-golden", action="store_true")
    parser.add_argument("--determinism", action="store_true", help="re-run golden and compare golden vs golden")
    parser.add_argument("--model-types", nargs="*", default=list(MODEL_TYPES))
    parser.add_argument("--collector-workers", type=int, default=0,
                        help="also run the L6 candidate with N environment worker processes (must match golden too)")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    src = ensure_golden_source()
    base_config = load_base_config(src)
    failures: list[str] = []
    report: list[str] = []

    # L7 ----------------------------------------------------------------
    l7 = ast_identity(src)
    failures.extend(l7)
    report.append(f"L7 AST identity: {len(TARGETS) - len(l7)}/{len(TARGETS)} functions identical")

    # L1-L5 -------------------------------------------------------------
    golden_probe_path = WORK_DIR / "probe_golden.pt"
    if args.regenerate_golden or not golden_probe_path.exists():
        golden_probe = run_probe(src, src / "data" / "instance" / "competition" / INSTANCE_NAME, golden_probe_path)
    else:
        import torch
        golden_probe = torch.load(golden_probe_path, map_location="cpu", weights_only=False)
    candidate_probe = run_probe(ROOT, candidate.instance_path(), WORK_DIR / "probe_candidate.pt")
    for layer in golden_probe:
        layer_mismatches: list[str] = []
        compare_nested(layer, golden_probe[layer], candidate_probe.get(layer), layer_mismatches)
        report.append(f"{layer}: {'OK' if not layer_mismatches else f'{len(layer_mismatches)} mismatch(es)'}")
        failures.extend(layer_mismatches[:20])

    # L6 ----------------------------------------------------------------
    if not args.quick:
        for model_type in args.model_types:
            golden_csv, golden_pt_dir = ensure_golden_run(src, base_config, model_type, args.regenerate_golden)
            if args.determinism:
                rerun_dir = run_legacy_train(src, base_config, model_type, f"golden_{model_type}_rerun",
                                             WORK_DIR / f"golden_{model_type}_rerun.log")
                mism, n, e = compare_csv(golden_csv, rerun_dir / "train_metrics.csv")
                for name in CHECKPOINT_FILES:
                    compare_checkpoint(golden_pt_dir / name, rerun_dir / name, mism)
                report.append(f"L6 determinism [{model_type}]: {'OK' if not mism else f'{len(mism)} mismatch(es)'} "
                              f"(float columns exact {e}/{n})")
                failures.extend(f"determinism {model_type}: {m}" for m in mism[:20])
            result_dir = candidate.run_candidate(base_config, model_type, WORK_DIR / f"candidate_{model_type}.log")
            mism, n, e = compare_csv(golden_csv, result_dir / "train_metrics.csv")
            tensors = exact = 0
            for name in CHECKPOINT_FILES:
                t, x = compare_checkpoint(golden_pt_dir / name, result_dir / name, mism)
                tensors += t
                exact += x
            report.append(f"L6 end-to-end [{model_type}]: {'OK' if not mism else f'{len(mism)} mismatch(es)'} "
                          f"(float columns exact {e}/{n}; checkpoint tensors exact {exact}/{tensors})")
            failures.extend(f"L6 {model_type}: {m}" for m in mism[:30])
            side_problems, side_summary = candidate.check_contract_files(result_dir)
            report.append(f"L6 contract files [{model_type}]: {'OK' if not side_problems else 'FAIL'} ({side_summary})")
            failures.extend(f"L6 {model_type}: {m}" for m in side_problems)
            if args.collector_workers > 0:
                tag = f"{model_type}, collector_workers={args.collector_workers}"
                result_dir = candidate.run_candidate(base_config, model_type,
                                                     WORK_DIR / f"candidate_{model_type}_cw{args.collector_workers}.log",
                                                     collector_workers=args.collector_workers)
                mism, n, e = compare_csv(golden_csv, result_dir / "train_metrics.csv")
                tensors = exact = 0
                for name in CHECKPOINT_FILES:
                    t, x = compare_checkpoint(golden_pt_dir / name, result_dir / name, mism)
                    tensors += t
                    exact += x
                report.append(f"L6 end-to-end [{tag}]: {'OK' if not mism else f'{len(mism)} mismatch(es)'} "
                              f"(float columns exact {e}/{n}; checkpoint tensors exact {exact}/{tensors})")
                failures.extend(f"L6 {tag}: {m}" for m in mism[:30])
                side_problems, side_summary = candidate.check_contract_files(result_dir)
                report.append(f"L6 contract files [{tag}]: {'OK' if not side_problems else 'FAIL'} ({side_summary})")
                failures.extend(f"L6 {tag}: {m}" for m in side_problems)

    print("\n".join(report))
    if failures:
        print("\nFAILURES:")
        print("\n".join(failures))
        return 1
    print("\nEQUIVALENCE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
