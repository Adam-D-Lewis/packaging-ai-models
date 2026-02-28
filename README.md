# Packaging AI Models as Conda Packages

Conda package recipes for distributing AI/ML models (GGUF format) using [rattler-build](https://github.com/prefix-dev/rattler-build).

Inspired by the [prefix.dev blog post](https://prefix.dev/blog/packaging-ai-ml-models-as-conda-packages) on packaging AI/ML models as conda packages.

## Models

| Recipe | Model | Quant | Size | Source |
|--------|-------|-------|------|--------|
| `whisper-tiny-test` | whisper.cpp tiny-en | q5_1 | ~31 MB | [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) |
| `qwen3.5-35b-a3b-gguf` | Qwen3.5-35B-A3B | Q4_K_M | ~21.2 GB | [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) |
| `qwen3-coder-next-gguf` | Qwen3-Coder-Next | Q4_K_M | ~48.5 GB | [unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) |

## Quick Start

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Build the small test model to validate the approach
pixi run build-test

# Build Qwen3.5-35B-A3B (Q4_K_M, ~21.2 GB download)
pixi run build-qwen35

# Build Qwen3-Coder-Next (Q4_K_M, ~48.5 GB download)
pixi run build-qwen-coder

# Build all Qwen models
pixi run build-all
```

## Building Different Quantizations

Each model recipe supports multiple quantization variants via `variants.yaml`. Edit the file to select which quantizations to build:

```yaml
# recipes/qwen3.5-35b-a3b-gguf/variants.yaml
quant_type:
  - Q4_K_M    # ~21.2 GB (default, good balance)
  - Q2_K      # ~12.9 GB (smallest)
  - Q8_0      # ~36.9 GB (highest quality)
```

Or build with a specific variant directly:

```bash
rattler-build build \
  -r recipes/qwen3.5-35b-a3b-gguf/recipe.yaml \
  --variant-config '{"quant_type": "Q8_0"}' \
  --output-dir output
```

## How It Works

Each recipe downloads a GGUF model file from HuggingFace and packages it into a conda package that:

1. Places the model in `$PREFIX/share/<model-name>/models/`
2. Sets an environment variable (e.g., `QWEN35_35B_A3B_MODEL_DIR`) pointing to the model directory
3. Includes tests to verify the model file exists and the env var is set

After installing the package, the model location is available via environment variables:

```bash
# After installing qwen3.5-35b-a3b-gguf
echo $QWEN35_35B_A3B_MODEL_DIR
# -> /path/to/env/share/qwen3.5-35b-a3b/models

# After installing qwen3-coder-next-gguf
echo $QWEN3_CODER_NEXT_MODEL_DIR
# -> /path/to/env/share/qwen3-coder-next/models
```

## Testing

### 1. Validate the build infrastructure (no large download)

Start with the whisper tiny model (~31 MB) to confirm everything works end-to-end:

```bash
pixi run build-test
```

This downloads the whisper.cpp tiny English model, packages it, and runs the built-in tests
(file existence + env var check). You should see `all tests passed!` at the end.

### 2. Build a Qwen model package

Once the test recipe succeeds, build one of the real model packages:

```bash
# Smaller model first (~21.2 GB download)
pixi run build-qwen35

# Or the larger coding model (~48.5 GB download)
pixi run build-qwen-coder
```

The output `.conda` packages are written to `output/noarch/`.

### 3. Install and verify the package

You can install the built package into a pixi environment to verify it works:

```bash
# Create a test environment and install the local package
pixi add --pypi ./output/noarch/qwen3.5-35b-a3b-gguf-1.0.0-q4_k_m.conda

# Or install directly with rattler/conda
conda install --use-local output/noarch/qwen3.5-35b-a3b-gguf-1.0.0-q4_k_m.conda
```

After installation, verify the environment variable is set and the model file exists:

```bash
echo $QWEN35_35B_A3B_MODEL_DIR
ls -lh $QWEN35_35B_A3B_MODEL_DIR/
```

### 4. Test with an inference engine

Use the packaged model with llama.cpp or any GGUF-compatible engine:

```bash
llama-cli -m $QWEN35_35B_A3B_MODEL_DIR/Qwen3.5-35B-A3B-Q4_K_M.gguf -p "Hello, world"
```

## Inference Environments

Two pixi environments for serving models, each with trade-offs:

| | `llamacpp-inference` | `vllm-inference` |
|---|---|---|
| **Best for** | Single-user, local dev | Multi-user team serving |
| **Default model** | Qwen3-Coder-Next Q4_K_M GGUF | Intel INT4 AutoRound (safetensors) |
| **Engine** | llama.cpp (GGUF-native) | vLLM (PagedAttention, continuous batching) |
| **Multi-GPU** | Manual layer splitting | `--tensor-parallel-size 2` |
| **CPU offloading** | Yes | No |
| **Concurrency** | Fixed throughput / N users | Scales with load |

Both serve an OpenAI-compatible API at `http://host:8000/v1/chat/completions`,
so all coding agent frontends (Claude Code, Aider, etc.) work with either.

### llama.cpp Inference (`environments/llamacpp-inference/`)

GGUF-native inference with optimized quantized kernels. Downloads the model
from HuggingFace on first run (cached for subsequent runs).

```bash
cd environments/llamacpp-inference
pixi install

# Default: Qwen3-Coder-Next Q4_K_M (~48.5 GB download, all layers on GPU)
pixi run serve

# Smaller model for quick testing (~400 MB, runs on CPU)
pixi run test-serve

# Override model, GPU layers, context, etc.
HF_REPO=unsloth/Qwen3-Coder-Next-GGUF \
HF_FILE="*Q8_0.gguf" \
N_GPU_LAYERS=-1 \
MAX_MODEL_LEN=32768 \
pixi run serve

# Use a local GGUF file instead of downloading
MODEL=/path/to/model.gguf pixi run serve
```

There is also an advanced version at `environments/qwen-coder-inference/` with
auto-benchmarking that sweeps GPU layers, threads, and batch size to find
optimal settings for your hardware.

### vLLM Inference (`environments/vllm-inference/`) — multi-user GPU serving

The `environments/vllm-inference/` directory is a standalone pixi project that
replaces the `vllm/vllm-openai` Docker image with a pixi-managed environment.
Everything the Docker image provides — CUDA, PyTorch, NCCL, vLLM — comes from
conda-forge instead.

The only requirement on the host is the NVIDIA driver (`nvidia-smi`).

### Why vLLM instead of llama.cpp?

| Feature | llama.cpp | vLLM |
|---------|-----------|------|
| Continuous batching | No (fixed batch) | Yes (dynamic scheduling) |
| Multi-user throughput | Divides fixed throughput | Scales with concurrency |
| Tensor parallelism | Manual layer splitting | `--tensor-parallel-size 2` |
| Memory efficiency | Standard KV cache | PagedAttention (19-27% savings) |
| GGUF performance | Native, highly optimized | Experimental, dequantizes each pass |
| CPU offloading | Excellent | Not supported |

**Rule of thumb**: use llama.cpp for single-user inference, vLLM for multi-user
serving where continuous batching matters.

### Default model: Intel/Qwen3-Coder-Next-int4-AutoRound

The default model is the [Intel INT4 AutoRound](https://huggingface.co/Intel/Qwen3-Coder-Next-int4-AutoRound)
quantization of Qwen3-Coder-Next (~43.6 GB download, ~46 GB VRAM). This uses
vLLM's optimized INT4 kernels — much better throughput than GGUF, which
dequantizes weights every forward pass.

| Attribute | Value |
|-----------|-------|
| Total parameters | 80B (3B active per token) |
| Quantization | INT4 AutoRound (group_size=128, symmetric) |
| Download size | ~43.6 GB |
| Min VRAM | ~46 GB |
| Format | safetensors (vLLM-native) |
| Context | up to 256K tokens |

### Setup

```bash
cd environments/vllm-inference
pixi install
```

### Serve

```bash
# Default: Intel INT4 Qwen3-Coder-Next with tool calling on 1 GPU
pixi run serve

# 2x GPU with tensor parallelism
TENSOR_PARALLEL_SIZE=2 pixi run serve

# Full 131K context on RunPod (2x RTX PRO 6000)
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=131072 \
GPU_MEMORY_UTILIZATION=0.95 \
pixi run serve

# qgpu2 staging (2x TITAN RTX 24 GB — tight fit)
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=8192 \
pixi run serve

# Quick smoke test with a tiny model (~1 GB, no big GPU needed)
MODEL=Qwen/Qwen2.5-0.5B-Instruct pixi run serve
```

All coding agent frontends (Claude Code, Aider, etc.) connect to the
OpenAI-compatible API at `http://host:8000/v1/chat/completions`.

### GGUF vs INT4/AWQ/GPTQ in vLLM

vLLM's GGUF support is marked "highly experimental and under-optimized." It
reads GGUF files but dequantizes weights every forward pass via its own CUDA
kernels — it does **not** use llama.cpp internally. The Intel INT4 (default),
AWQ, and GPTQ formats use optimized Marlin kernels and give significantly
better throughput. GGUF is still available via `pixi run serve-gguf` for
convenience.

### Test the server

```bash
# In another terminal:
pixi run test-chat
```

## Project Structure

```
packaging-ai-models/
  pixi.toml                              # Pixi workspace with build tasks
  recipes/
    whisper-tiny-test/
      recipe.yaml                        # Small test recipe (~31 MB model)
    qwen3.5-35b-a3b-gguf/
      recipe.yaml                        # Qwen3.5-35B-A3B recipe
      variants.yaml                      # Quantization variants (Q4_K_M default)
    qwen3-coder-next-gguf/
      recipe.yaml                        # Qwen3-Coder-Next recipe
      variants.yaml                      # Quantization variants (Q4_K_M default)
  environments/
    llamacpp-inference/
      pixi.toml                          # llama.cpp inference (GGUF-native)
      scripts/
        serve.py                         # Downloads GGUF from HF, starts server
        test_chat.py                     # Sends test request to running server
    vllm-inference/
      pixi.toml                          # vLLM inference (INT4/AWQ, multi-user)
      scripts/
        serve.py                         # Starts vLLM OpenAI-compatible server
        test_chat.py                     # Sends test request to running server
    qwen-coder-inference/
      pixi.toml                          # Advanced: llama.cpp + auto-benchmarking
      benchmark.py                       # Sweeps settings, saves optimal config
      scripts/
        activate.sh                      # Loads tuned settings on shell activation
        chat.py                          # Interactive chat using tuned settings
        serve.py                         # OpenAI-compatible API server
  output/                                # Built .conda packages (gitignored)
```

## Benefits of Conda Packaging for Models

- **Versioning**: Track model versions alongside software versions
- **Locking**: Pin exact model versions in lockfiles for reproducibility
- **Dependencies**: Declare model + inference engine compatibility
- **Caching**: Reuse conda's proven download/cache infrastructure (hardlinks, no disk bloat)
- **Traceability**: Sign packages with attestations for supply chain security

## Hosting a Custom Channel

`rattler-build upload` supports 5 targets out of the box. There is also a newer
`rattler-build publish` command that supports `s3://` and `file://` targets
directly and auto-indexes after upload. The right choice depends on package size
and who needs access.

### Option 1: Local file channel (quickest for demos)

No server needed — just a directory. Build the package, then point pixi at the
output folder:

```bash
# Build
rattler-build build -r recipes/qwen3.5-35b-a3b-gguf/recipe.yaml --output-dir output

# Use — add the local path as a channel in any pixi.toml
# channels = ["conda-forge", "file:///absolute/path/to/output"]
```

pixi and conda both understand `file://` channels. The `rattler-build build`
command already generates the `repodata.json` index in `output/noarch/`, so
there is nothing else to do.

### Option 2: S3 bucket (best for large models, no size limit)

Works with AWS S3, MinIO, Cloudflare R2, or any S3-compatible store.
No size limit — upload 50 GB packages without issue.

```bash
# Upload (credentials via env vars or AWS config)
rattler-build upload s3 \
  --channel s3://my-bucket/my-channel \
  --region us-east-1 \
  output/noarch/*.conda

# For S3-compatible stores (MinIO, R2, etc.) add --endpoint-url:
rattler-build upload s3 \
  --channel s3://models/conda-channel \
  --endpoint-url https://minio.example.com \
  --addressing-style path \
  output/noarch/*.conda
```

Then in `pixi.toml`, point at the bucket (pixi supports `s3://` natively, or
use the HTTPS URL for public buckets):

```toml
# Native S3 URL (pixi handles auth via AWS env vars / config)
channels = ["conda-forge", "s3://my-bucket/my-channel"]

# Public bucket via HTTPS (no auth needed)
channels = ["conda-forge", "https://my-bucket.s3.amazonaws.com/my-channel"]

# MinIO / R2 — configure the endpoint in pixi
channels = ["conda-forge", "s3://models/conda-channel"]

# For non-AWS S3 providers, add endpoint config:
# [workspace.s3-options.models]
# endpoint-url = "https://minio.example.com"
# region = "us-east-1"
# force-path-style = true
```

**Quick demo with MinIO** (runs locally in Docker, no AWS account needed):

```bash
# Start MinIO
docker run -d -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Create a bucket (via mc CLI or the web console at localhost:9001)
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mb local/models

# Upload
rattler-build upload s3 \
  --channel s3://models/conda-channel \
  --endpoint-url http://localhost:9000 \
  --access-key-id minioadmin \
  --secret-access-key minioadmin \
  --addressing-style path \
  output/noarch/*.conda

# Use
# channels = ["conda-forge", "http://localhost:9000/models/conda-channel"]
```

### Option 3: prefix.dev (managed, easiest setup)

Hosted by the rattler/pixi team. Free tier available, but has a **100 MB default
file size limit** (contact them to raise it for large models — not viable for
multi-GB GGUF files without a custom arrangement).

```bash
# Authenticate
rattler-build upload prefix \
  --channel my-channel \
  --api-key $PREFIX_API_KEY \
  output/noarch/*.conda

# Use
# channels = ["conda-forge", "https://repo.prefix.dev/my-channel"]
```

### Option 4: Anaconda.org

The original conda hosting. Free for public packages.

```bash
rattler-build upload anaconda \
  --owner my-username \
  --api-key $ANACONDA_API_KEY \
  output/noarch/*.conda

# Use
# channels = ["conda-forge", "https://conda.anaconda.org/my-username"]
```

### Option 5: Quetz (self-hosted, open source — unmaintained)

A self-hosted conda channel server from the mamba-org team. **Note:** the last
release was Dec 2023 and the project appears unmaintained. Use with caution.

```bash
rattler-build upload quetz \
  --url https://quetz.example.com \
  --channel my-channel \
  output/noarch/*.conda
```

### Option 6: JFrog Artifactory

Enterprise artifact management with conda channel support.

```bash
rattler-build upload artifactory \
  --url https://myorg.jfrog.io \
  --channel conda-local \
  output/noarch/*.conda
```

### Summary

| Option | Size limit | Setup effort | Best for |
|--------|-----------|-------------|----------|
| Local `file://` | Disk space | None | Local dev, demos |
| S3 / MinIO / R2 | Unlimited | Low | Large models, teams |
| prefix.dev | 100 MB default | None | Small packages, easy sharing |
| Anaconda.org | Varies | None | Public distribution |
| Quetz | Unlimited | Medium | Self-hosted, full control |
| Artifactory | Unlimited | Medium | Enterprise |
