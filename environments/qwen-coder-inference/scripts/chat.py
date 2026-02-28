"""Interactive chat using the tuned llama.cpp settings."""
from __future__ import annotations

import os
import sys


def main() -> None:
    from llama_cpp import Llama

    model = os.environ.get("MODEL_PATH") or _find_model()
    if not model:
        print("ERROR: set MODEL_PATH or install a model package.", file=sys.stderr)
        sys.exit(1)

    ngl = int(os.environ.get("LLAMA_N_GPU_LAYERS", "0"))
    nt = int(os.environ.get("LLAMA_N_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
    nb = int(os.environ.get("LLAMA_N_BATCH", "512"))

    print(f"Loading {os.path.basename(model)} (ngl={ngl}, threads={nt}, batch={nb}) ...")
    llm = Llama(
        model_path=model,
        n_gpu_layers=ngl,
        n_threads=nt,
        n_batch=nb,
        flash_attn=True,
        verbose=False,
    )
    print("Ready. Type a message (Ctrl-C to quit).\n")

    history: list[dict] = []
    while True:
        try:
            user_msg = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not user_msg:
            continue

        history.append({"role": "user", "content": user_msg})
        resp = llm.create_chat_completion(
            messages=history,
            max_tokens=1024,
            stream=True,
        )

        print("AI: ", end="", flush=True)
        full = []
        for chunk in resp:
            delta = chunk["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)
            full.append(delta)
        print()
        history.append({"role": "assistant", "content": "".join(full)})


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
