"""Benchmark instances: write params.yaml into every instance folder and data/instances/index.csv.

The 15 benchmark instances (DDT in {500, 600, 900, 1200, 1600} x S in {50, 100, 150}) are external files that
ship with the repository (data/instances/benchmark/<id>/<id>.txt); nothing is generated or re-sampled. The
step is idempotent (identical files on re-run)."""
import time

from _bootstrap import ROOT, done, run_py, step

INDEX = ROOT / "data" / "instances" / "index.csv"

t0 = time.time()
step("benchmark instances: params.yaml per folder + index.csv")
run_py("data.dataset")
done(t0, INDEX, ROOT / "data" / "instances" / "benchmark" / "DDT500_num50" / "params.yaml")
