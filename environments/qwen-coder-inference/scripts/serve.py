"""Start an OpenAI-compatible API server with tuned settings."""
from __future__ import annotations

import os
import sys


def main() -> None:
    from llama_cpp.server.__main__ import main as server_main

    model = os.environ.get("MODEL_PATH") or _find_model()
    if not model:
        print("ERROR: set MODEL_PATH or install a model package.", file=sys.stderr)
        sys.exit(1)

    ngl = os.environ.get("LLAMA_N_GPU_LAYERS", "0")
    nt = os.environ.get("LLAMA_N_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
    nb = os.environ.get("LLAMA_N_BATCH", "512")

    sys.argv = [
        "llama-cpp-server",
        "--model", model,
        "--n_gpu_layers", ngl,
        "--n_threads", nt,
        "--n_batch", nb,
    ]
    server_main()


def _find_model() -> str | None:
    from pathlib import Path

    for var in ["QWEN3_CODER_NEXT_MODEL_DIR", "QWEN35_35B_A3B_MODEL_DIR"]:
        d = os.environ.get(var)
        if not d:
            continue
        p = Path(d)
        if p.is_dir():
            files = sorted(p.glob("*.gguf"))
            if files:
                return str(files[0])
    return None
