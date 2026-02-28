#!/usr/bin/env python3
"""
Benchmark llama.cpp settings to find optimal configuration for your hardware.

Sweeps GPU layer offloading, thread count, and batch size in stages so each
subsequent stage uses the winner from the previous one.  Results are saved to
~/.cache/qwen-coder-inference/benchmark-results.json and loaded by the
activation script on every future shell.

Usage:
    pixi run benchmark              # full sweep
    pixi run benchmark-quick        # fast sweep (fewer configs)
    python benchmark.py --model /path/to/model.gguf
    python benchmark.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path.home() / ".cache" / "qwen-coder-inference"
RESULTS_FILE = RESULTS_DIR / "benchmark-results.json"

BENCH_PROMPT = (
    "Write a Python function that takes a list of integers and returns "
    "the longest increasing subsequence. Include type hints and a docstring."
)
BENCH_MAX_TOKENS = 64


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_gpus() -> list[dict]:
    """Return list of GPUs with total/free VRAM in MiB via nvidia-smi."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return []
        gpus = []
        for line in r.stdout.strip().splitlines():
            idx, name, total, free = [s.strip() for s in line.split(",")]
            gpus.append(
                {"index": int(idx), "name": name, "total_mb": int(total), "free_mb": int(free)}
            )
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def cpu_count() -> int:
    return os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

MODEL_DIR_VARS = [
    "QWEN3_CODER_NEXT_MODEL_DIR",
    "QWEN35_35B_A3B_MODEL_DIR",
    "WHISPER_MODEL_DIR",
]


def find_model(explicit: str | None = None) -> str | None:
    """Locate a .gguf (or .bin) model file."""
    if explicit and Path(explicit).is_file():
        return explicit

    for var in MODEL_DIR_VARS:
        d = os.environ.get(var)
        if not d:
            continue
        p = Path(d)
        if not p.is_dir():
            continue
        for ext in ("*.gguf", "*.bin"):
            files = sorted(p.glob(ext))
            if files:
                return str(files[0])
    return None


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------

def bench_one(
    model_path: str,
    n_gpu_layers: int,
    n_threads: int,
    n_batch: int,
) -> dict:
    """Load model with the given config, run a short generation, return stats."""
    from llama_cpp import Llama  # imported here so top-level --help is fast

    load_t0 = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_batch=n_batch,
        flash_attn=True,
        verbose=False,
    )
    load_time = time.perf_counter() - load_t0

    gen_t0 = time.perf_counter()
    out = llm(BENCH_PROMPT, max_tokens=BENCH_MAX_TOKENS, echo=False)
    gen_time = time.perf_counter() - gen_t0

    prompt_tok = out["usage"]["prompt_tokens"]
    comp_tok = out["usage"]["completion_tokens"]
    tps = comp_tok / gen_time if gen_time > 0 else 0.0

    del llm  # free VRAM before next config

    return {
        "n_gpu_layers": n_gpu_layers,
        "n_threads": n_threads,
        "n_batch": n_batch,
        "load_time_s": round(load_time, 2),
        "gen_time_s": round(gen_time, 2),
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "tokens_per_second": round(tps, 2),
    }


# ---------------------------------------------------------------------------
# Staged sweep
# ---------------------------------------------------------------------------

def run_sweep(model_path: str, quick: bool) -> dict:
    gpus = detect_gpus()
    cores = cpu_count()

    print(f"Model : {model_path}")
    print(f"CPUs  : {cores} cores")
    if gpus:
        for g in gpus:
            print(f"GPU {g['index']} : {g['name']}  ({g['free_mb']} / {g['total_mb']} MiB free)")
    else:
        print("GPU   : none detected — CPU-only sweep")
    print()

    # --- Stage 1: GPU layer offloading -----------------------------------
    if gpus:
        if quick:
            ngl_candidates = [0, 20, -1]
        else:
            ngl_candidates = [0, 10, 20, 30, 40, 60, 80, -1]
    else:
        ngl_candidates = [0]

    default_threads = max(1, cores // 2)
    default_batch = 512

    print("=== Stage 1: GPU layer offloading ===")
    best_ngl, best_tps = 0, 0.0
    for ngl in ngl_candidates:
        label = f"  ngl={ngl:>3}, threads={default_threads}, batch={default_batch}"
        try:
            r = bench_one(model_path, ngl, default_threads, default_batch)
            print(f"{label}  →  {r['tokens_per_second']:6.2f} tok/s  (load {r['load_time_s']}s)")
            if r["tokens_per_second"] > best_tps:
                best_tps = r["tokens_per_second"]
                best_ngl = ngl
        except Exception as e:
            print(f"{label}  →  FAILED: {e}")
    print(f"  ✓ best n_gpu_layers = {best_ngl}\n")

    # --- Stage 2: thread count -------------------------------------------
    if quick:
        thread_candidates = [1, max(1, cores // 2), cores]
    else:
        thread_candidates = sorted(
            {1, 2, 4, max(1, cores // 4), max(1, cores // 2), cores}
        )

    print("=== Stage 2: thread count ===")
    best_threads, best_tps = default_threads, 0.0
    for nt in thread_candidates:
        label = f"  ngl={best_ngl:>3}, threads={nt:>3}, batch={default_batch}"
        try:
            r = bench_one(model_path, best_ngl, nt, default_batch)
            print(f"{label}  →  {r['tokens_per_second']:6.2f} tok/s")
            if r["tokens_per_second"] > best_tps:
                best_tps = r["tokens_per_second"]
                best_threads = nt
        except Exception as e:
            print(f"{label}  →  FAILED: {e}")
    print(f"  ✓ best n_threads = {best_threads}\n")

    # --- Stage 3: batch size ---------------------------------------------
    if quick:
        batch_candidates = [256, 512]
    else:
        batch_candidates = [128, 256, 512, 1024, 2048]

    print("=== Stage 3: batch size ===")
    best_batch, best_tps = default_batch, 0.0
    for nb in batch_candidates:
        label = f"  ngl={best_ngl:>3}, threads={best_threads:>3}, batch={nb:>5}"
        try:
            r = bench_one(model_path, best_ngl, best_threads, nb)
            print(f"{label}  →  {r['tokens_per_second']:6.2f} tok/s")
            if r["tokens_per_second"] > best_tps:
                best_tps = r["tokens_per_second"]
                best_batch = nb
        except Exception as e:
            print(f"{label}  →  FAILED: {e}")
    print(f"  ✓ best n_batch = {best_batch}\n")

    # --- Final confirmation run ------------------------------------------
    print("=== Final run with best settings ===")
    final = bench_one(model_path, best_ngl, best_threads, best_batch)
    print(f"  {final['tokens_per_second']:.2f} tok/s  "
          f"(ngl={best_ngl}, threads={best_threads}, batch={best_batch})\n")

    result = {
        "model_path": model_path,
        "hardware": {
            "cpu_cores": cores,
            "gpus": gpus,
        },
        "optimal": {
            "n_gpu_layers": best_ngl,
            "n_threads": best_threads,
            "n_batch": best_batch,
            "tokens_per_second": final["tokens_per_second"],
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results saved to {RESULTS_FILE}")
    print("These will be loaded automatically next time you activate the environment.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Path to .gguf model file (auto-detected from env vars if omitted)")
    parser.add_argument("--quick", action="store_true", help="Fewer configs per stage (faster, less precise)")
    args = parser.parse_args()

    model_path = find_model(args.model)
    if not model_path:
        print(
            "ERROR: No model found. Either:\n"
            "  1. Install a model conda package into this environment, or\n"
            "  2. Set MODEL_PATH=/path/to/model.gguf, or\n"
            "  3. Pass --model /path/to/model.gguf\n",
            file=sys.stderr,
        )
        sys.exit(1)

    results = run_sweep(model_path, quick=args.quick)
    save_results(results)


if __name__ == "__main__":
    main()
