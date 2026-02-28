#!/usr/bin/env bash
# Pixi activation script — runs every time the environment is activated.
#
# 1. If benchmark results exist, export the optimal settings as env vars
#    so llama-cpp-python (and any wrapper scripts) pick them up automatically.
# 2. If no results exist yet, nudge the user to run the benchmark.

RESULTS_FILE="${HOME}/.cache/qwen-coder-inference/benchmark-results.json"

if [ -f "$RESULTS_FILE" ]; then
    # python is guaranteed to be in the env — use it to parse JSON
    eval "$(python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('${RESULTS_FILE}').read_text())
o = r['optimal']
print(f'export LLAMA_N_GPU_LAYERS={o[\"n_gpu_layers\"]}')
print(f'export LLAMA_N_THREADS={o[\"n_threads\"]}')
print(f'export LLAMA_N_BATCH={o[\"n_batch\"]}')
print(f'export LLAMA_BENCH_TPS={o[\"tokens_per_second\"]}')
")"
    echo "[qwen-coder-inference] Loaded tuned settings: ngl=${LLAMA_N_GPU_LAYERS} threads=${LLAMA_N_THREADS} batch=${LLAMA_N_BATCH} (${LLAMA_BENCH_TPS} tok/s)"
else
    echo ""
    echo "[qwen-coder-inference] No benchmark results found."
    echo "  Run 'pixi run benchmark' (or 'pixi run benchmark-quick') to auto-detect"
    echo "  the best llama.cpp settings for your hardware. Results are cached and"
    echo "  loaded automatically on every future activation."
    echo ""
fi
