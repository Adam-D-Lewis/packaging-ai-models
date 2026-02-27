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
  output/                                # Built .conda packages (gitignored)
```

## Benefits of Conda Packaging for Models

- **Versioning**: Track model versions alongside software versions
- **Locking**: Pin exact model versions in lockfiles for reproducibility
- **Dependencies**: Declare model + inference engine compatibility
- **Caching**: Reuse conda's proven download/cache infrastructure (hardlinks, no disk bloat)
- **Traceability**: Sign packages with attestations for supply chain security

## Publishing

To publish packages to a prefix.dev channel:

```bash
rattler-build build -r recipes/qwen3.5-35b-a3b-gguf/recipe.yaml --output-dir output
rattler-build upload prefix -c <your-channel> output/noarch/*.conda
```

Note: prefix.dev has a 1 GB default package size limit. Contact them to increase it for large models.
