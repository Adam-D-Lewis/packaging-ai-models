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

## Inference Environment (model + llama.cpp + auto-tuning)

The `environments/qwen-coder-inference/` directory is a standalone pixi project
that bundles the model with llama.cpp (via `llama-cpp-python`) and an
auto-benchmarking script that finds the best settings for your hardware.

### Setup

```bash
cd environments/qwen-coder-inference

# Edit pixi.toml to enable CUDA if you have a GPU:
#   1. Uncomment the cuda-version line matching your driver
#   2. Uncomment the extra-index-urls line for CUDA wheels

pixi install
```

### Auto-tune

On first activation pixi will remind you to benchmark. Run it once — results
are cached and loaded automatically on every future shell:

```bash
pixi run benchmark          # full sweep (~10-20 min depending on model size)
pixi run benchmark-quick    # fewer configs (~5 min)
```

The benchmark sweeps GPU layer offloading, thread count, and batch size in
stages, then saves the winning config to
`~/.cache/qwen-coder-inference/benchmark-results.json`. On every subsequent
`pixi shell` the activation script exports these as env vars
(`LLAMA_N_GPU_LAYERS`, `LLAMA_N_THREADS`, `LLAMA_N_BATCH`) so all tools
pick them up automatically.

### Use

```bash
# Interactive chat (uses tuned settings automatically)
pixi run chat

# OpenAI-compatible API server
pixi run serve
# Then: curl http://localhost:8000/v1/chat/completions ...

# Or use the env vars directly with any GGUF tool
llama-cli -m $QWEN3_CODER_NEXT_MODEL_DIR/*.gguf \
  -ngl $LLAMA_N_GPU_LAYERS -t $LLAMA_N_THREADS -b $LLAMA_N_BATCH \
  -p "Hello"
```

### Pointing at a model

The scripts look for a model in this order:
1. `MODEL_PATH` env var (explicit path to a `.gguf` file)
2. `QWEN3_CODER_NEXT_MODEL_DIR` / `QWEN35_35B_A3B_MODEL_DIR` env vars
   (set automatically when a model conda package is installed)
3. Fail with an error telling you what to set

## vLLM Inference Environment (multi-user GPU serving)

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

### Setup

```bash
cd environments/vllm-inference
pixi install
```

### Serve a model

```bash
# Small model for testing (~1 GB download)
MODEL=Qwen/Qwen2.5-0.5B-Instruct pixi run serve

# Qwen3-Coder-Next on 2x GPU with tool calling
MODEL=Qwen/Qwen3-Coder-Next \
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=131072 \
GPU_MEMORY_UTILIZATION=0.95 \
TOOL_CALL_PARSER=qwen3_coder \
pixi run serve

# GGUF model (experimental — see note below)
MODEL="unsloth/Qwen3-0.6B-GGUF:Q4_K_M" \
TOKENIZER=Qwen/Qwen3-0.6B \
pixi run serve-gguf
```

All coding agent frontends (Claude Code, Aider, etc.) connect to the
OpenAI-compatible API at `http://host:8000/v1/chat/completions`.

### GGUF vs AWQ/GPTQ in vLLM

vLLM's GGUF support is marked "highly experimental and under-optimized." It
reads GGUF files but dequantizes weights every forward pass via its own CUDA
kernels — it does **not** use llama.cpp internally. For best vLLM throughput,
prefer AWQ or GPTQ quantizations. GGUF is supported for convenience since it's
the most common format on HuggingFace.

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
    qwen-coder-inference/
      pixi.toml                          # Inference env: llama.cpp + CUDA + auto-tune
      benchmark.py                       # Sweeps settings, saves optimal config
      scripts/
        activate.sh                      # Loads tuned settings on shell activation
        chat.py                          # Interactive chat using tuned settings
        serve.py                         # OpenAI-compatible API server
    vllm-inference/
      pixi.toml                          # vLLM env: replaces vllm/vllm-openai Docker
      scripts/
        serve.py                         # Starts vLLM OpenAI-compatible server
        test_chat.py                     # Sends test request to running server
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
