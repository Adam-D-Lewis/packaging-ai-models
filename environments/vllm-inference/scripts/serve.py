"""Start a vLLM OpenAI-compatible API server.

This replaces the vLLM Docker image entrypoint (`vllm serve`). All
configuration is via environment variables so it works identically
whether you run it locally, on qgpu2, or on RunPod.

Environment variables (all optional — sensible defaults are used):

    MODEL                   HuggingFace model ID, local path, or GGUF spec
                            e.g. "unsloth/Qwen3-0.6B-GGUF:Q4_K_M"
                            (default: Qwen/Qwen2.5-0.5B-Instruct)
    QUANTIZATION            Quantization method: gguf, awq, gptq, or empty
                            for auto-detect (default: auto-detect)
    TOKENIZER               Explicit tokenizer. Required for GGUF models
                            because vLLM's GGUF tokenizer conversion is
                            unreliable. (default: same as MODEL)
    MAX_MODEL_LEN           Maximum context length (default: 8192)
    TENSOR_PARALLEL_SIZE    Number of GPUs for tensor parallelism (default: 1)
    GPU_MEMORY_UTILIZATION  Fraction of GPU memory vLLM can use (default: 0.90)
    HOST                    Bind address (default: 0.0.0.0)
    PORT                    Bind port (default: 8000)
    TOOL_CALL_PARSER        Tool call parser (default: none).
                            Use "qwen3_coder" for Qwen3-Coder models.
    EXTRA_ARGS              Additional vLLM CLI arguments, space-separated
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys


def main() -> None:
    model = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    quantization = os.environ.get("QUANTIZATION", "")
    tokenizer = os.environ.get("TOKENIZER", "")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "8192")
    tensor_parallel = os.environ.get("TENSOR_PARALLEL_SIZE", "1")
    gpu_mem = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    tool_parser = os.environ.get("TOOL_CALL_PARSER", "")
    extra = os.environ.get("EXTRA_ARGS", "")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--max-model-len", max_model_len,
        "--tensor-parallel-size", tensor_parallel,
        "--gpu-memory-utilization", gpu_mem,
        "--host", host,
        "--port", port,
    ]

    if quantization:
        cmd.extend(["--quantization", quantization])

    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])

    if tool_parser:
        cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_parser])

    if extra:
        cmd.extend(shlex.split(extra))

    print("[vllm-inference] Starting vLLM server...")
    print(f"  Model:    {model}")
    if quantization:
        print(f"  Quant:    {quantization}")
    if tokenizer:
        print(f"  Tokenizer: {tokenizer}")
    print(f"  Context:  {max_model_len}")
    print(f"  TP size:  {tensor_parallel}")
    print(f"  GPU mem:  {gpu_mem}")
    if tool_parser:
        print(f"  Tools:    {tool_parser}")
    print(f"  Endpoint: http://{host}:{port}/v1/chat/completions")
    print()
    print(f"  Command:  {' '.join(cmd)}")
    print()

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
