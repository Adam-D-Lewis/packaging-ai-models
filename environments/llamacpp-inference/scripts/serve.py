"""Start a llama.cpp OpenAI-compatible API server.

Downloads the model from HuggingFace on first run (cached for subsequent
runs). All configuration is via environment variables.

Environment variables (all optional — sensible defaults are used):

    HF_REPO             HuggingFace repo ID for the GGUF model
                        (default: unsloth/Qwen3-Coder-Next-GGUF)
    HF_FILE             Glob pattern for which GGUF file to download
                        (default: *Q4_K_M.gguf)
    MODEL               Local path to a .gguf file. If set, HF_REPO/HF_FILE
                        are ignored. (default: download from HF_REPO)
    N_GPU_LAYERS        Number of layers to offload to GPU. -1 = all layers.
                        (default: -1)
    N_THREADS           Number of CPU threads (default: half of CPU cores)
    N_BATCH             Batch size for prompt processing (default: 512)
    MAX_MODEL_LEN       Context length (default: 8192)
    HOST                Bind address (default: 0.0.0.0)
    PORT                Bind port (default: 8000)
"""
from __future__ import annotations

import os
import sys


def _download_model(repo_id: str, filename_pattern: str) -> str:
    """Download a GGUF file from HuggingFace Hub, return local path."""
    from huggingface_hub import hf_hub_download, HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id)
    import fnmatch
    matches = [f for f in files if fnmatch.fnmatch(f, filename_pattern)]
    if not matches:
        print(f"ERROR: No file matching '{filename_pattern}' in {repo_id}", file=sys.stderr)
        print(f"  Available files: {[f for f in files if f.endswith('.gguf')]}", file=sys.stderr)
        sys.exit(1)
    # Pick the first match
    filename = sorted(matches)[0]
    print(f"[llamacpp-inference] Downloading {repo_id}/{filename} ...")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"[llamacpp-inference] Cached at: {path}")
    return path


def main() -> None:
    model_path = os.environ.get("MODEL", "")
    hf_repo = os.environ.get("HF_REPO", "unsloth/Qwen3-Coder-Next-GGUF")
    hf_file = os.environ.get("HF_FILE", "*Q4_K_M.gguf")

    if not model_path:
        model_path = _download_model(hf_repo, hf_file)

    ngl = int(os.environ.get("N_GPU_LAYERS", "-1"))
    nt = int(os.environ.get("N_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
    nb = int(os.environ.get("N_BATCH", "512"))
    ctx = int(os.environ.get("MAX_MODEL_LEN", "8192"))
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    print(f"[llamacpp-inference] Starting llama.cpp server...")
    print(f"  Model:      {os.path.basename(model_path)}")
    print(f"  GPU layers: {ngl}")
    print(f"  Threads:    {nt}")
    print(f"  Batch:      {nb}")
    print(f"  Context:    {ctx}")
    print(f"  Endpoint:   http://{host}:{port}/v1/chat/completions")
    print()

    from llama_cpp.server.__main__ import main as server_main

    sys.argv = [
        "llama-cpp-server",
        "--model", model_path,
        "--n_gpu_layers", str(ngl),
        "--n_threads", str(nt),
        "--n_batch", str(nb),
        "--n_ctx", str(ctx),
        "--host", host,
        "--port", str(port),
    ]
    server_main()


if __name__ == "__main__":
    main()
