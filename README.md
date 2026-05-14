# unified-hf-model-setup

A portable toolkit for downloading, converting, and running HuggingFace language models locally - LLMs, Vision-Language Models (VLMs), and BitNet ternary models - with a single shell script.

---

## Contents

```
unified-hf-model-setup/
├── hf_model_setup.sh          # Main entry point - download + setup any HF model
├── chat_hf_vlm.py             # Generic interactive chat CLI for any HF VLM
└── README.md                  # This file
```

---

## Quick Start

```bash
# LLM - download, convert to GGUF, register with ollama, open chat
bash hf_model_setup.sh Qwen/Qwen3-4B-Instruct --quant q4_k_m

# VLM - download, install torch, open Python chat
bash hf_model_setup.sh ibm-granite/granite-vision-4.1-4b

# BitNet - download, compile CPU kernels, open inference
bash hf_model_setup.sh microsoft/bitnet-b1.58-2B-4T

# Help
bash hf_model_setup.sh --help
```

---

## Prerequisites

### External (manual, one-time)

| Tool | Required for | Install |
|---|---|---|
| `bash` ≥ 4 | Running the script | Pre-installed on Linux/macOS |
| `curl` | Downloading the `uv` installer | `sudo apt install curl` |
| `python3.12` | Creating the venv | `sudo apt install python3.12` |
| `git` | Cloning `llama.cpp` and `BitNet` repos | `sudo apt install git` |
| `ollama` | Registering and running LLM GGUFs | `curl -fsSL https://ollama.com/install.sh \| sh` |

> **Note:** `ollama` is only needed for LLM models. VLMs and BitNet do not use it.

### For Gated HuggingFace Models (optional)

Some models (Llama, Gemma) require accepting a license on HuggingFace before downloading.

1. Go to `https://huggingface.co/settings/tokens` and create a token
2. Run:
   ```bash
   hf auth login
   ```
   The token is saved to `~/.cache/huggingface/token` and read automatically by the script.

Public models (Qwen, BitNet, IBM Granite, nomic-embed-text) do **not** require a token.

### Internal (automatic - no manual action needed)

| Tool / Package | Purpose | How it's installed |
|---|---|---|
| `uv` | Fast Python package manager | Auto-installed via `curl` if not found |
| Python venv | Isolated Python environment | Created at `~/venvs/hf-vlm` by `uv` |
| `huggingface_hub` | HF model metadata and auth | Installed by the script via `uv pip` |
| `hf_transfer` | Faster HF downloads | Installed by the script |
| `transformers` ≥ 4.40 | Model loading (VLMs and conversion) | Installed by the script |
| `accelerate` | Device mapping for large models | Installed by the script |
| `safetensors` | Loading model weights efficiently | Installed by the script |
| `sentencepiece` | Tokenizer support | Installed by the script |
| `gguf` | GGUF metadata writing (conversion) | Installed by the script |
| `Pillow` | Image loading for VLMs | Installed by the script |
| `numpy` | Tensor operations | Installed by the script |
| `protobuf` | Tokenizer protobuf format | Installed by the script |
| `bitsandbytes` | 4-bit NF4 quantisation for VLMs | Installed by the script |
| `torch` (CPU or CUDA) | Model inference | Installed by the script (CPU or CUDA auto-detected) |
| `torchvision` | Image preprocessing for some VLMs | Installed by the script |
| `clang` + `cmake` | Compiling BitNet CPU kernels | Auto-installed via `sudo apt-get` (BitNet only) |
| `llama.cpp` | GGUF conversion + quantisation | Cloned from GitHub (LLM only) |
| `BitNet` (bitnet.cpp) | BitNet model setup + inference | Cloned from GitHub (BitNet only) |

---

## `hf_model_setup.sh` - Detailed Reference

### Usage

```bash
bash hf_model_setup.sh <hf_repo_id> [OPTIONS]
```

### Arguments

| Argument | Description |
|---|---|
| `hf_repo_id` | HuggingFace repository ID (e.g. `Qwen/Qwen3-4B-Instruct`). Supports LLMs, VLMs, and BitNet models. |

### Options

| Option | Default | Description |
|---|---|---|
| `--name <name>` | Lowercase repo basename | Override the ollama model name used for registration. Only affects LLM models. |
| `--quant <level>` | `f16` | GGUF quantisation level. Only affects LLM models. See table below. |
| `--help`, `-h` | - | Show help and exit. |

#### Quantisation levels (`--quant`)

| Level | Description | Approx. size (4B model) |
|---|---|---|
| `f16` | Full float16 - highest quality | ~8 GB |
| `q8_0` | 8-bit quantisation - good quality, half the size | ~4 GB |
| `q4_k_m` | 4-bit K-quant - recommended for low RAM | ~2.5 GB |

### Pipeline Steps

The script runs 4 steps for every model:

```
[1/4] Python environment
      └── installs uv if missing
      └── creates ~/venvs/hf-vlm if missing
      └── installs base Python deps (transformers, accelerate, gguf, etc.)

[2/4] Download
      └── fetches file listing from HF API
      └── skips files already downloaded (resume support)
      └── uses urllib with SSL bypass (works behind corporate MITM proxies)
      └── shows ASCII progress bar for large files

[3/4] Detect model type
      └── reads config.json → architectures, model_type
      └── classifies as: llm | vlm | bitnet

[4/4] Type-specific setup (see below)
```

### Model Type Handling

#### LLM (e.g. Qwen3, Llama, Gemma, Mistral)

```
→ Clones llama.cpp (if not present)
→ Installs llama.cpp conversion deps
→ Converts safetensors → GGUF (f16 or q8_0 base)
→ Quantises to target level (q8_0 or q4_k_m) if requested
→ Writes a Modelfile with stop tokens, temperature, top_p, num_ctx
→ Registers GGUF with ollama
→ Launches: ollama run <name>
```

#### VLM (e.g. Qwen3-VL, IBM Granite Vision, Gemma4)

```
→ Detects GPU via nvcc / nvidia-smi
    CUDA 12.x → installs torch+cu121, uses bfloat16
    CUDA 11.x → installs torch+cu118, uses bfloat16
    No GPU    → installs torch+cpu,   uses float32
→ Launches: python chat_hf_vlm.py --model <dir> --dtype <dtype>
```

> VLMs are not registered with ollama because ollama has a bug with `conv3d` operators in some VLM architectures.

#### BitNet (e.g. microsoft/bitnet-b1.58-2B-4T)

```
→ Clones microsoft/BitNet (if not present)
→ Initialises llama.cpp submodule
→ Installs clang + cmake if missing (via apt-get)
→ Installs bitnet.cpp Python deps
→ Creates symlink BitNet/models/<name> → ~/models/<slug>
→ Runs setup_env.py: converts to i2_s GGUF + compiles CPU kernels
→ Launches: python run_inference.py -m <gguf> -cnv
```

---

## `chat_hf_vlm.py` - Generic VLM Chat

Interactive chat CLI supporting any HuggingFace VLM. Handles both transformers v4 (`AutoModelForVision2Seq`) and v5 (`AutoModelForImageTextToText`), plus custom `trust_remote_code` architectures.

### Usage

```bash
# Always pass the FULL ABSOLUTE PATH to the downloaded model directory
~/venvs/hf-vlm/bin/python chat_hf_vlm.py \
    --model ~/models/Qwen-Qwen3-VL-2B-Instruct

~/venvs/hf-vlm/bin/python chat_hf_vlm.py \
    --model ~/models/ibm-granite-granite-vision-4.1-4b

# With an initial image
~/venvs/hf-vlm/bin/python chat_hf_vlm.py \
    --model ~/models/Qwen-Qwen3-VL-2B-Instruct \
    --image /path/to/img.jpg

# Force 4-bit NF4 quantisation (needs bitsandbytes, ~2-3 GB for 4B models)
~/venvs/hf-vlm/bin/python chat_hf_vlm.py \
    --model ~/models/ibm-granite-granite-vision-4.1-4b \
    --4bit
```

> **Note:** `--model` must be a full local path (e.g. `~/models/Qwen-Qwen3-VL-2B-Instruct`).
> Passing just the directory name (e.g. `Qwen-Qwen3-VL-2B-Instruct`) causes transformers to
> query HuggingFace Hub and fail with a 404 error.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | Required | **Full absolute path** to local model directory |
| `--image` | None | Path to an initial image (attached to the first message) |
| `--dtype` | Auto | Force dtype: `float32`, `float16`, `bfloat16`. Auto: `float16` on CPU, `bfloat16` on CUDA |
| `--4bit` | Off | Load weights in 4-bit NF4 via bitsandbytes (~2-3 GB for 4B models) |

### In-Chat Commands

| Command | Description |
|---|---|
| `/image <path>` | Attach an image to the next message |
| `/image <path> <question>` | Attach an image and ask a question inline |
| `/clear` | Reset conversation history |
| `/quit` or Ctrl-C | Exit |

---

## Paths Used

All paths are relative to `$HOME` and created automatically if missing.

| Path | Purpose |
|---|---|
| `~/venvs/hf-vlm/` | Python virtual environment (Python 3.12) |
| `~/models/<org>-<repo>/` | Downloaded model weights |
| `~/llama.cpp/` | llama.cpp clone (LLM conversion + quantisation) |
| `~/BitNet/` | microsoft/BitNet clone (BitNet setup) |
| `~/.local/bin/uv` | uv binary (auto-installed) |
| `~/.cache/huggingface/token` | HF auth token (written by `hf auth login`) |

---

## Manual Browser Download (Alternative to Script Download)

If the script's download step fails (e.g. severe proxy restrictions), you can download model files manually from `https://huggingface.co/<org>/<repo>/tree/main` and place them in the correct directory.

### What to download

| File | Required? | Notes |
|---|---|---|
| `config.json` | Yes | Architecture detection |
| `tokenizer.json` | Yes | Tokenizer |
| `tokenizer_config.json` | Yes | Stop tokens, chat template |
| `generation_config.json` | If present | Generation defaults |
| `special_tokens_map.json` | If present | Special tokens |
| `model.safetensors` | Yes (single-file models) | Weights |
| `model-00001-of-0000N.safetensors` … | Yes (multi-shard) | Download **all shards** |
| `model.safetensors.index.json` | Yes (multi-shard only) | Shard map — required |
| `tokenizer.model` or `vocab.json` | If present | Vocabulary |
| `preprocessor_config.json` | VLMs only | Image preprocessing |

**Skip:** `flax_model.msgpack`, `tf_model.h5`, `pytorch_model.bin`, `onnx/` subfolder

### Where to place the files

The script expects model files at:
```
~/models/<org>-<repo>/
```

The slug is derived by replacing `/` with `-` in the repo ID:

| Repo ID | Target directory |
|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | `~/models/Qwen-Qwen3-VL-2B-Instruct/` |
| `ibm-granite/granite-vision-4.1-4b` | `~/models/ibm-granite-granite-vision-4.1-4b/` |
| `microsoft/bitnet-b1.58-2B-4T` | `~/models/microsoft-bitnet-b1.58-2B-4T/` |

Place all files **directly** in the directory (no subfolders).

### Then run normally

```bash
bash hf_model_setup.sh Qwen/Qwen3-VL-2B-Instruct
```

The script detects all shards are already present and **skips the download step**, going straight to type detection and setup.

---

## Notes

- **Corporate proxy (Zscaler, etc.):** The download step uses `urllib` with SSL verification disabled to bypass MITM certificate issues. Standard `hf` CLI often stalls or fails in these environments.
- **Resume support:** Partial downloads are resumed automatically. Already-complete files are skipped.
- **Idempotent:** Re-running the script on an already-downloaded model is safe - it skips completed steps.
- **VLM + ollama:** VLMs are intentionally not registered with ollama. Ollama has a known issue with `conv3d` operations used in some VLM vision encoders.
- **BitNet double-quantisation:** If BitNet inference produces gibberish, it is a known issue with double-quantisation when the source weights are already ternary-packed. The pipeline itself is correct.
