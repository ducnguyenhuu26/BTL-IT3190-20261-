"""Benchmark original vs optimized code on full local data; no model retraining."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from src.data import load_favorita
from src.evaluate import metrics, online_adapter_backtest
from src.features import make_features
from src.jump_filter import JumpAwareFilter
from src.levy_process import LevyOU
from src.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "provenance/ede2f98b9ae4"


def original(name):
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    path = "src/"+name+".py"
    expected = {**manifest["code_hashes"], **manifest["reference_hashes"]}[path]
    if file_sha256(SNAPSHOT / path) != expected:
        raise ValueError(f"Benchmark reference checksum differs: {path}")
    spec = importlib.util.spec_from_file_location("src._benchmark_"+name, SNAPSHOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measured(function, *args):
    start = time.perf_counter()
    value = function(*args)
    return value, time.perf_counter()-start


def main():
    results = []
    raw, aux = load_favorita(ROOT / "data/raw")
    previous, old_time = measured(original("features").make_features, raw, aux)
    current, new_time = measured(make_features, raw, aux)
    pd.testing.assert_frame_equal(current, previous, check_exact=True)
    results.append({"operation": "Features: all Favorita rows", "rows": len(raw),
                    "original_seconds": old_time, "optimized_seconds": new_time,
                    "speedup": old_time/new_time, "verification": "Exact equality, all columns and rows"})
    print(f"Full features: {old_time:.3f}s -> {new_time:.3f}s; identical {len(raw):,} rows", flush=True)
    del raw, aux, previous, current
    wide = pd.read_csv(ROOT / "results/usgs_ablation_exploratory/validation_predictions.csv.gz", parse_dates=["date"])
    frame = wide[["date", "store_nbr", "family", "actual"]].rename(columns={"actual": "sales"})
    base = np.log1p(wide["LightGBM"].to_numpy(float))
    process = LevyOU(rho=.9, q=.005, p=.01, q_jump=.3)
    old_eval = original("evaluate")
    previous, old_time = measured(old_eval.online_adapter_backtest, frame, base, JumpAwareFilter(process, .2), "benchmark")
    current, new_time = measured(online_adapter_backtest, frame, base, JumpAwareFilter(process, .2), "benchmark")
    pd.testing.assert_frame_equal(current, previous, check_exact=False, atol=1e-9, rtol=1e-12)
    difference = float(np.max(np.abs(current.prediction-previous.prediction)))
    results.append({"operation": "Online backtest: full validation", "rows": len(frame),
                    "original_seconds": old_time, "optimized_seconds": new_time,
                    "speedup": old_time/new_time, "verification": f"max prediction difference {difference:.3g}; states/diagnostics match"})
    print(f"Online validation: {old_time:.3f}s -> {new_time:.3f}s; max difference {difference:.3g}", flush=True)
    keys = list(zip(frame.store_nbr, frame.family))
    previous, old_time = measured(old_eval.metrics, frame.sales, wide["LightGBM"], keys)
    current, new_time = measured(metrics, frame.sales, wide["LightGBM"], keys)
    for name in previous:
        if not np.isclose(current[name], previous[name], rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError(f"Metric changed after optimization: {name}")
    results.append({"operation": "Metrics: full validation", "rows": len(frame),
                    "original_seconds": old_time, "optimized_seconds": new_time,
                    "speedup": old_time/new_time, "verification": "All metrics match within 1e-10"})
    output = ROOT / "results/optimization"
    output.mkdir(exist_ok=True)
    (output / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    text = "# Optimization regression and timing\n\nOne CPU wall-time measurement per implementation on this machine; includes function execution, excludes data loading. Speedups are local measurements, not general hardware guarantees. No model retraining or new test evaluation.\n\n"
    text += "| Operation | Rows | Before (s) | After (s) | Speedup | Verification |\n| --- | ---: | ---: | ---: | ---: | --- |\n"
    for row in results:
        text += f"| {row['operation']} | {row['rows']:,} | {row['original_seconds']:.3f} | {row['optimized_seconds']:.3f} | {row['speedup']:.2f}x | {row['verification']} |\n"
    text += "\nOriginal implementation: `provenance/ede2f98b9ae4/`; SHA-256 verified before use. Historical model/prediction/lock files are retained. Report-only audits may verify that archived source; prediction/test stages still require unchanged current code.\n"
    (ROOT / "reports/OPTIMIZATION.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
