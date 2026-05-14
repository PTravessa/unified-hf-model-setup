#!/usr/bin/env bash
# hf_model_setup.sh
# Download any HuggingFace model, auto-detect type, convert to GGUF + register
# with ollama (LLMs), launch Python chatbot (VLMs), or set up via bitnet.cpp (BitNet).
#
# Usage:
#   bash hf_model_setup.sh <hf_repo_id> [options]
#
# Options:
#   --name  <ollama_name>   Override ollama model name (default: repo basename)
#   --quant <level>         GGUF quantisation: f16 | q8_0 | q4_k_m  (default: f16)
#
# Examples:
#   bash hf_model_setup.sh Qwen/Qwen3-4B-Instruct
#   bash hf_model_setup.sh google/gemma-3-4b-it --name gemma3-4b --quant q4_k_m
#   bash hf_model_setup.sh meta-llama/Llama-3.2-3B-Instruct --quant q8_0
#   bash hf_model_setup.sh ibm-granite/granite-vision-4.1-4b   # VLM → Python chat
#   bash hf_model_setup.sh Qwen/Qwen3-VL-2B-Instruct            # VLM → Python chat
#   bash hf_model_setup.sh microsoft/bitnet-b1.58-2B-4T         # BitNet → bitnet.cpp

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
# Resolve the directory containing this script so it works from any location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UV="${HOME}/.local/bin/uv"
VENV_DIR="${HOME}/venvs/hf-vlm"
PYTHON="${VENV_DIR}/bin/python"
MODELS_DIR="${HOME}/models"
LLAMA_CPP="${HOME}/llama.cpp"
BITNET_CPP="${HOME}/BitNet"
# chat_hf_vlm.py lives alongside this script — works on any machine without
# relying on a pre-existing ~/chat_hf_vlm.py copy.
CHAT_SCRIPT="${SCRIPT_DIR}/chat_hf_vlm.py"
HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
HF_TOKEN_FILE2="${HOME}/.huggingface/token"

# ── Arg parsing ───────────────────────────────────────────────────────────────
_usage() {
    cat <<'EOF'
Usage:
  bash hf_model_setup.sh <hf_repo_id> [OPTIONS]

Arguments:
  hf_repo_id          HuggingFace repository ID, e.g. Qwen/Qwen3-4B-Instruct
                      Supports LLMs, VLMs, and BitNet models.

Options:
  --name  <name>      Override the ollama model name used for registration.
                      Default: lowercase basename of the repo ID.
                      Example: --name my-qwen3

  --quant <level>     GGUF quantisation level for LLM models (ignored for VLMs/BitNet).
                      Choices: f16 | q8_0 | q4_k_m
                      Default: f16
                      f16     — full precision float16, largest file, best quality
                      q8_0    — 8-bit quantisation, ~half the size of f16
                      q4_k_m  — 4-bit K-quant, ~quarter the size, recommended for low RAM

  --help, -h          Show this help message and exit.

Model type detection (automatic):
  LLM    → converts to GGUF, registers with ollama, opens ollama chat
  VLM    → downloads weights, installs torch (CUDA if available, else CPU),
           launches interactive Python chat (chat_hf_vlm.py)
  BitNet → sets up via bitnet.cpp (i2_s quantisation, CPU-optimised kernels)

Setup (first run on a new machine):
  - Installs uv (Python package manager) automatically if not present
  - Creates venv at ~/venvs/hf-vlm with Python 3.12
  - Installs all required Python dependencies
  - Downloads the model (resumes partial downloads, skips completed files)
  - clang + cmake installed automatically when needed (BitNet only, via apt-get)

HuggingFace login (gated models only — Llama, Gemma, etc.):
  Get a token at https://huggingface.co/settings/tokens, then run:
    hf auth login
  Token is saved to ~/.cache/huggingface/token and read automatically.
  Public models (Qwen, BitNet, Granite, nomic) do not require a token.

Examples:
  bash hf_model_setup.sh Qwen/Qwen3-4B-Instruct
  bash hf_model_setup.sh Qwen/Qwen3-4B-Instruct --quant q4_k_m
  bash hf_model_setup.sh google/gemma-3-4b-it --name gemma3-4b --quant q8_0
  bash hf_model_setup.sh meta-llama/Llama-3.2-3B-Instruct --quant q4_k_m
  bash hf_model_setup.sh ibm-granite/granite-vision-4.1-4b
  bash hf_model_setup.sh Qwen/Qwen3-VL-2B-Instruct
  bash hf_model_setup.sh microsoft/bitnet-b1.58-2B-4T
EOF
}

# Show help if requested or no arguments given
if [[ $# -lt 1 ]]; then
    _usage
    exit 1
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    _usage
    exit 0
fi

HF_REPO="$1"; shift

OLLAMA_NAME_OVERRIDE=""
QUANT="f16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            [[ $# -lt 2 ]] && { echo "ERROR: --name requires a value." >&2; exit 1; }
            OLLAMA_NAME_OVERRIDE="$2"; shift 2 ;;
        --quant)
            [[ $# -lt 2 ]] && { echo "ERROR: --quant requires a value." >&2; exit 1; }
            QUANT="$2"; shift 2 ;;
        --help|-h)
            _usage; exit 0 ;;
        *)
            echo "ERROR: Unknown option '$1'. Run with --help for usage." >&2
            exit 1 ;;
    esac
done

MODEL_SLUG="${HF_REPO//\//-}"
OLLAMA_NAME="${OLLAMA_NAME_OVERRIDE:-$(echo "${HF_REPO##*/}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')}"
MODEL_DIR="${MODELS_DIR}/${MODEL_SLUG}"
GGUF_F16="${MODELS_DIR}/${MODEL_SLUG}-f16.gguf"

# Final GGUF path depends on quant level
case "${QUANT}" in
    f16)     GGUF_OUT="${GGUF_F16}" ;;
    q8_0)    GGUF_OUT="${MODELS_DIR}/${MODEL_SLUG}-q8_0.gguf" ;;
    q4_k_m)  GGUF_OUT="${MODELS_DIR}/${MODEL_SLUG}-q4_k_m.gguf" ;;
    *)        echo "  ERROR: Unknown quant '${QUANT}'. Use f16, q8_0, or q4_k_m." >&2; exit 1 ;;
esac

echo "========================================================"
echo "  HF Model Setup"
echo "  Repo  : ${HF_REPO}"
echo "  Dir   : ${MODEL_DIR}"
echo "  Ollama: ${OLLAMA_NAME}"
echo "  Quant : ${QUANT}"
echo "========================================================"

# ── Step 1: Ensure uv + venv + base deps ─────────────────────────────────────
echo ""
echo "[1/4] Python environment..."

# Auto-install uv if not present (single static binary, no sudo required).
if [[ ! -f "${UV}" ]]; then
    echo "  uv not found — installing via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Reload PATH in case the installer added ~/.local/bin
    export PATH="${HOME}/.local/bin:${PATH}"
    if [[ ! -f "${UV}" ]]; then
        echo "  ERROR: uv install failed. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo "  uv installed: $(${UV} --version)"
fi

if [[ ! -f "${PYTHON}" ]]; then
    echo "  Creating venv..."
    "${UV}" venv "${VENV_DIR}" --python 3.12
fi

"${UV}" pip install --python "${PYTHON}" \
    "huggingface_hub[cli]" hf_transfer \
    "transformers>=4.40" accelerate safetensors \
    sentencepiece gguf Pillow numpy protobuf \
    2>/dev/null | tail -2

echo "  [OK]"

# ── Step 2: Download ──────────────────────────────────────────────────────────
echo ""
echo "[2/4] Downloading ${HF_REPO}..."
mkdir -p "${MODEL_DIR}"

HF_TOKEN="${HF_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "${HF_TOKEN}" && -f "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE}")
elif [[ -z "${HF_TOKEN}" && -f "${HF_TOKEN_FILE2}" ]]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE2}")
fi
if [[ -n "${HF_TOKEN}" ]]; then
    echo "  HF token found."
else
    echo "  No HF token — public repos only. For gated repos run: hf auth login"
fi

# Improvement 2: skip download entirely if all safetensors shards are present
_all_shards_present() {
    local index="${MODEL_DIR}/model.safetensors.index.json"
    # Single-file models: check model.safetensors directly
    if [[ ! -f "${index}" ]]; then
        [[ -f "${MODEL_DIR}/model.safetensors" ]]
        return
    fi
    # Multi-shard: every shard listed in the index must exist and be non-empty
    python3 - <<PYEOF
import json, sys
from pathlib import Path
idx = json.load(open("${index}"))
shards = set(idx["weight_map"].values())
out = Path("${MODEL_DIR}")
missing = [s for s in shards if not (out / s).exists() or (out / s).stat().st_size == 0]
sys.exit(0 if not missing else 1)
PYEOF
}

if _all_shards_present; then
    echo "  All weight shards already present — skipping download."
else
    # Primary: urllib with SSL bypass + resume support (works behind corporate MITM proxy).
    # hf CLI stalls on large files when SSL verification is enforced by proxy.
    echo "  Downloading via urllib (SSL bypass for corporate proxy)..."
    "${PYTHON}" - <<PYEOF
import ssl, json, sys
from pathlib import Path
import urllib.request

# ── SSL context (bypass corporate MITM proxy) ────────────────────────────────
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

token = "${HF_TOKEN}" or None
ignore = (".msgpack", ".h5", "flax_model", "tf_model", ".ot", "rust_model.ot")
repo = "${HF_REPO}"
out = Path("${MODEL_DIR}")
out.mkdir(parents=True, exist_ok=True)

auth_headers: dict = {"Authorization": f"Bearer {token}"} if token else {}

# ── Fetch file listing from HF API ───────────────────────────────────────────
api_url = f"https://huggingface.co/api/models/{repo}"
req = urllib.request.Request(api_url, headers=auth_headers)
try:
    with urllib.request.urlopen(req, context=ctx) as r:
        meta = json.loads(r.read())
except urllib.error.HTTPError as e:
    if e.code == 401:
        print(f"  ERROR: HTTP 401 Unauthorized for '{repo}'.", file=sys.stderr)
        print(f"  This repo is gated. To fix:", file=sys.stderr)
        print(f"    1. Accept the license at https://huggingface.co/{repo}", file=sys.stderr)
        print(f"    2. Log in: hf auth login  (or set HF_HUB_TOKEN env var)", file=sys.stderr)
    elif e.code == 403:
        print(f"  ERROR: HTTP 403 Forbidden for '{repo}'.", file=sys.stderr)
        print(f"  Accept the model license at https://huggingface.co/{repo}", file=sys.stderr)
    elif e.code == 404:
        print(f"  ERROR: Repository '{repo}' not found on HuggingFace (HTTP 404).", file=sys.stderr)
        print(f"  Check the repo ID — it may be private, gated, or misspelled.", file=sys.stderr)
    else:
        print(f"  ERROR: HF API returned HTTP {e.code}: {e.reason}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"  ERROR: Failed to reach HuggingFace API: {e}", file=sys.stderr)
    sys.exit(1)

files = [
    s["rfilename"] for s in meta.get("siblings", [])
    if not any(p in s["rfilename"] for p in ignore)
]

# ── Download each file with resume support and progress reporting ─────────────
CHUNK = 1 << 20  # 1 MB read chunks — bar refreshes every 1 MB

for f in files:
    dest = out / f
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Resume — check existing partial size
    existing = dest.stat().st_size if dest.exists() else 0

    url = f"https://huggingface.co/{repo}/resolve/main/{f}"
    headers = dict(auth_headers)

    # HEAD request to get total size
    head_req = urllib.request.Request(url, headers=headers, method="HEAD")
    try:
        with urllib.request.urlopen(head_req, context=ctx) as hr:
            total = int(hr.headers.get("Content-Length", 0))
    except Exception:
        total = 0

    if existing > 0 and total > 0 and existing > total:
        # Local file is larger than remote — corrupt (e.g. leftover from failed hf CLI).
        print(f"  corrupt {f} (local {existing // (1024**2)} MB > remote {total // (1024**2)} MB) — re-downloading", flush=True)
        dest.unlink()
        existing = 0

    if existing > 0 and existing == total:
        print(f"  skip {f} ({existing // (1024**2)} MB)", flush=True)
        continue

    if existing > 0 and total > 0:
        print(f"  resume {f} from {existing // (1024**2)} MB / {total // (1024**2)} MB", flush=True)
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    else:
        print(f"  dl {f} ({total // (1024**2)} MB)" if total else f"  dl {f}", flush=True)
        mode = "wb"

    def _print_bar(done: int, total: int, width: int = 40) -> None:
        """Overwrite the current terminal line with an ASCII progress bar."""
        pct = done / total if total else 0
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        done_mb = done // (1024 ** 2)
        total_mb = total // (1024 ** 2)
        line = f"  [{bar}] {pct * 100:5.1f}%  {done_mb} / {total_mb} MB"
        sys.stdout.write(f"\r{line}")
        sys.stdout.flush()

    req = urllib.request.Request(url, headers=headers)
    downloaded = existing

    with urllib.request.urlopen(req, context=ctx) as r, open(dest, mode) as fp:
        while chunk := r.read(CHUNK):   # 1 MB chunks — bar refreshes every 1 MB
            fp.write(chunk)
            downloaded += len(chunk)
            if total:
                _print_bar(downloaded, total)
            else:
                sys.stdout.write(f"\r    {downloaded // (1024**2)} MB")
                sys.stdout.flush()

    if total:
        _print_bar(downloaded, total)
    sys.stdout.write("\n")  # move to next line after bar completes
    print(f"  done {f} ({downloaded // (1024**2)} MB)", flush=True)

print("  [OK] urllib download complete.", flush=True)
PYEOF
fi

# ── Step 3: Detect model type + llama.cpp support ─────────────────────────────
echo ""
echo "[3/4] Detecting model type..."

# Outputs two lines: "<llm|vlm>  <arch>"
read -r MODEL_TYPE MODEL_ARCH < <("${PYTHON}" - <<PYEOF
import json, sys
from pathlib import Path

cfg_path = Path("${MODEL_DIR}/config.json")
if not cfg_path.exists():
    print("llm unknown")
    sys.exit(0)

cfg = json.load(open(cfg_path))
archs = cfg.get("architectures", [])
model_type = cfg.get("model_type", "")
arch = archs[0] if archs else "unknown"

is_bitnet = (
    any(x in str(archs) for x in ["Bitnet", "BitNet"])
    or model_type in ("bitnet",)
    or cfg.get("weight_bits") == 1
)

is_vlm = (
    not is_bitnet
    and (
        "vision_config" in cfg
        or "vision_tower" in cfg
        or any(x in str(archs) for x in ["Vision", "VL", "Llava", "Idefics", "Flamingo", "Blip"])
        or any(x in model_type for x in ["vl", "vision", "llava", "blip", "flamingo"])
    )
)

if is_bitnet:
    print("bitnet " + arch)
elif is_vlm:
    print("vlm " + arch)
else:
    print("llm " + arch)
PYEOF
)

echo "  Model type : ${MODEL_TYPE}"
echo "  Architecture: ${MODEL_ARCH}"

# ── Shared: common GGUF/ollama helper vars and functions ──────────────────────
CONVERT="${LLAMA_CPP}/convert_hf_to_gguf.py"
LLAMA_QUANTIZE="${LLAMA_CPP}/build/bin/llama-quantize"
[[ ! -f "${LLAMA_QUANTIZE}" ]] && LLAMA_QUANTIZE="${LLAMA_CPP}/llama-quantize"

# Check if llama.cpp convert script supports a given architecture class name
_llama_supports_arch() {
    grep -q "\"${1}\"" "${CONVERT}" 2>/dev/null
}

# Derive stop tokens from tokenizer_config.json and print as Modelfile PARAMETER lines
_stop_tokens() {
    "${PYTHON}" - <<PYEOF
import json
from pathlib import Path

tc_path = Path("${MODEL_DIR}/tokenizer_config.json")
stops = set()

if tc_path.exists():
    tc = json.load(open(tc_path))
    for key in ("eos_token", "bos_token"):
        val = tc.get(key)
        if isinstance(val, str) and val:
            stops.add(val)
        elif isinstance(val, dict):
            content = val.get("content", "")
            if content:
                stops.add(content)
    for tok in tc.get("added_tokens_decoder", {}).values():
        if tok.get("special") and "eos" in tok.get("content", "").lower():
            stops.add(tok["content"])

for t in ["<|end_of_text|>", "<|eot_id|>", "<eos>", "</s>", "<|im_end|>"]:
    stops.add(t)

for t in sorted(stops):
    print(f'PARAMETER stop "{t}"')
PYEOF
}

# Install llama.cpp conversion deps (shared between LLM and VLM paths)
_install_convert_deps() {
    if [[ -f "${LLAMA_CPP}/requirements/requirements-convert_hf_to_gguf.txt" ]]; then
        echo "  Installing llama.cpp conversion deps..."
        "${UV}" pip install --python "${PYTHON}" \
            --index-strategy unsafe-best-match \
            -r "${LLAMA_CPP}/requirements/requirements-convert_hf_to_gguf.txt" \
            2>/dev/null | tail -1 || true
    fi
}

# Register a GGUF with ollama, writing a Modelfile first.
# Usage: _ollama_register <gguf_path> [mmproj_gguf_path]
_ollama_register() {
    local gguf_out="$1"
    local mmproj="${2:-}"
    local stop_tokens
    stop_tokens=$(_stop_tokens)

    local modelfile="${MODELS_DIR}/${MODEL_SLUG}.Modelfile"

    if [[ -n "${mmproj}" ]]; then
        cat > "${modelfile}" <<MFEOF
FROM ${gguf_out}
ADAPTER ${mmproj}

${stop_tokens}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
MFEOF
    else
        cat > "${modelfile}" <<MFEOF
FROM ${gguf_out}

${stop_tokens}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
MFEOF
    fi

    echo "  Registering with ollama as '${OLLAMA_NAME}'..."
    ollama rm "${OLLAMA_NAME}" 2>/dev/null || true
    ollama create "${OLLAMA_NAME}" -f "${modelfile}"
}

# Convert backbone to f16 GGUF (idempotent).
# Uses streaming write (no --use-temp-file) to minimise peak RAM during the
# write phase. q8_0 is used as the conversion base when quant != f16 to
# halve output size (~4 GB vs ~8 GB for f16) and reduce write-phase RAM.
_convert_f16() {
    if [[ -f "${GGUF_F16}" ]]; then
        echo "  f16 GGUF already exists: ${GGUF_F16}"
    else
        echo "  Converting backbone to f16 GGUF..."
        "${PYTHON}" "${CONVERT}" \
            "${MODEL_DIR}" \
            --outfile "${GGUF_F16}" \
            --outtype f16
        echo "  [OK] f16 GGUF: ${GGUF_F16} ($(du -sh "${GGUF_F16}" | cut -f1))"
    fi
}

# Convert backbone directly to q8_0 GGUF (used when quant != f16 to avoid
# the large f16 intermediate and reduce peak RAM during write).
_convert_q8() {
    local q8_path="${MODELS_DIR}/${MODEL_SLUG}-q8_0.gguf"
    if [[ -f "${q8_path}" ]]; then
        echo "  q8_0 GGUF already exists: ${q8_path}"
    else
        echo "  Converting backbone directly to q8_0 GGUF (lower peak RAM)..."
        "${PYTHON}" "${CONVERT}" \
            "${MODEL_DIR}" \
            --outfile "${q8_path}" \
            --outtype q8_0
        echo "  [OK] q8_0 GGUF: ${q8_path} ($(du -sh "${q8_path}" | cut -f1))"
    fi
    echo "${q8_path}"
}

# Quantise a source GGUF → target quant (idempotent), delete source after.
# Usage: _quantise <source_gguf>
_quantise() {
    local src_gguf="$1"
    if [[ "${QUANT}" == "f16" ]]; then
        return 0
    fi
    if [[ -f "${GGUF_OUT}" ]]; then
        echo "  ${QUANT} GGUF already exists: ${GGUF_OUT}"
        return 0
    fi
    echo "  Quantising $(basename "${src_gguf}") → ${QUANT}..."
    if [[ ! -f "${LLAMA_QUANTIZE}" ]]; then
        echo "  ERROR: llama-quantize not found at ${LLAMA_QUANTIZE}"
        echo "  Build it: cd ${LLAMA_CPP} && cmake -B build && cmake --build build -j"
        exit 1
    fi
    "${LLAMA_QUANTIZE}" --allow-requantize "${src_gguf}" "${GGUF_OUT}" "${QUANT^^}"
    echo "  [OK] ${QUANT} GGUF: ${GGUF_OUT} ($(du -sh "${GGUF_OUT}" | cut -f1))"
    echo "  Removing intermediate: ${src_gguf}..."
    rm -f "${src_gguf}"
    echo "  [OK] intermediate deleted."
}

# ── Step 4a: BitNet → bitnet.cpp ─────────────────────────────────────────────
if [[ "${MODEL_TYPE}" == "bitnet" ]]; then
    echo ""
    echo "[4/4] BitNet model detected → setting up via bitnet.cpp (i2_s, CPU-optimised)..."

    # Clone bitnet.cpp if not already present
    if [[ ! -d "${BITNET_CPP}" ]]; then
        echo "  Cloning microsoft/BitNet..."
        git clone https://github.com/microsoft/BitNet "${BITNET_CPP}"
    else
        echo "  bitnet.cpp already cloned: ${BITNET_CPP}"
    fi

    # Initialise git submodules (llama.cpp is a submodule — required for requirements.txt)
    if [[ ! -f "${BITNET_CPP}/3rdparty/llama.cpp/CMakeLists.txt" ]]; then
        echo "  Initialising submodules (llama.cpp)..."
        git -C "${BITNET_CPP}" submodule update --init --recursive
    else
        echo "  Submodules already initialised."
    fi

    # Ensure clang + cmake are installed — required by setup_env.py's CMake build step
    if ! command -v clang &>/dev/null || ! command -v cmake &>/dev/null; then
        echo "  Installing clang + cmake (required by bitnet.cpp)..."
        sudo apt-get install -y clang cmake 2>&1 | tail -3
    fi

    # Ensure pip is available in the venv (uv venvs don't include pip by default,
    # but setup_env.py calls sys.executable -m pip internally)
    "${UV}" pip install --python "${PYTHON}" pip 2>/dev/null | tail -1 || true

    # Install bitnet.cpp Python deps
    echo "  Installing bitnet.cpp deps..."
    "${UV}" pip install --python "${PYTHON}" \
        -r "${BITNET_CPP}/requirements.txt" \
        2>/dev/null | tail -1 || true

    # Resolve the exact model_name that setup_env.py's SUPPORTED_HF_MODELS uses.
    # get_model_name() returns os.path.basename(model_dir) when --hf-repo is not passed,
    # so we create a symlink whose basename matches the expected key.
    # This avoids passing --hf-repo (which would trigger a redundant re-download).
    BITNET_MODEL_NAME=$("${PYTHON}" - <<PYEOF
import json, sys
from pathlib import Path

supported = {
    "1bitLLM/bitnet_b1_58-large":              "bitnet_b1_58-large",
    "1bitLLM/bitnet_b1_58-3B":                 "bitnet_b1_58-3B",
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens":   "Llama3-8B-1.58-100B-tokens",
    "tiiuae/Falcon3-7B-Instruct-1.58bit":      "Falcon3-7B-Instruct-1.58bit",
    "tiiuae/Falcon3-7B-1.58bit":               "Falcon3-7B-1.58bit",
    "tiiuae/Falcon3-10B-Instruct-1.58bit":     "Falcon3-10B-Instruct-1.58bit",
    "tiiuae/Falcon3-10B-1.58bit":              "Falcon3-10B-1.58bit",
    "tiiuae/Falcon3-3B-Instruct-1.58bit":      "Falcon3-3B-Instruct-1.58bit",
    "tiiuae/Falcon3-3B-1.58bit":               "Falcon3-3B-1.58bit",
    "tiiuae/Falcon3-1B-Instruct-1.58bit":      "Falcon3-1B-Instruct-1.58bit",
    "microsoft/BitNet-b1.58-2B-4T":            "BitNet-b1.58-2B-4T",
    "tiiuae/Falcon-E-3B-Instruct":             "Falcon-E-3B-Instruct",
    "tiiuae/Falcon-E-1B-Instruct":             "Falcon-E-1B-Instruct",
    "tiiuae/Falcon-E-3B-Base":                 "Falcon-E-3B-Base",
    "tiiuae/Falcon-E-1B-Base":                 "Falcon-E-1B-Base",
}
repo = "${HF_REPO}"
# Exact match first, then case-insensitive fallback
name = supported.get(repo)
if name is None:
    repo_lower = repo.lower()
    for k, v in supported.items():
        if k.lower() == repo_lower:
            name = v
            break
if name is None:
    # Not in the official list — fall back to repo basename (setup_env.py will error
    # with NotImplementedError if the model truly isn't supported)
    name = repo.split("/")[-1]
print(name)
PYEOF
)
    echo "  BitNet model name: ${BITNET_MODEL_NAME}"

    # Create a symlink inside BitNet/models/ whose basename = BITNET_MODEL_NAME.
    # setup_env.py uses os.path.basename(model_dir) as the model name when --hf-repo
    # is not passed, so this lets it resolve the correct codegen params without
    # triggering a redundant re-download.
    BITNET_MODELS_DIR="${BITNET_CPP}/models"
    BITNET_MODEL_LINK="${BITNET_MODELS_DIR}/${BITNET_MODEL_NAME}"
    mkdir -p "${BITNET_MODELS_DIR}"
    if [[ ! -L "${BITNET_MODEL_LINK}" && ! -d "${BITNET_MODEL_LINK}" ]]; then
        echo "  Creating symlink: ${BITNET_MODEL_LINK} → ${MODEL_DIR}"
        ln -s "${MODEL_DIR}" "${BITNET_MODEL_LINK}"
    else
        echo "  Model link already exists: ${BITNET_MODEL_LINK}"
    fi

    # The GGUF is written by prepare_model() into model_dir (the symlink target = MODEL_DIR)
    BITNET_GGUF="${MODEL_DIR}/ggml-model-i2_s.gguf"

    if [[ -f "${BITNET_GGUF}" ]]; then
        echo "  i2_s GGUF already exists: ${BITNET_GGUF}"
    else
        echo "  Running setup_env.py (converts + compiles kernels, may take several minutes)..."
        # CWD must be BITNET_CPP — setup_env.py uses relative paths for 3rdparty/, logs/, build/
        # Pass model-dir as a path relative to BITNET_CPP so basename = BITNET_MODEL_NAME
        ( cd "${BITNET_CPP}" && "${PYTHON}" setup_env.py \
            --model-dir "models/${BITNET_MODEL_NAME}" \
            -q i2_s )
    fi

    echo ""
    echo "========================================================"
    echo "  [DONE] BitNet model ready."
    echo "  GGUF : ${BITNET_GGUF}"
    echo "  Chat : cd ${BITNET_CPP} && python run_inference.py -m ${BITNET_GGUF} -p \"<prompt>\" -n 200 -cnv"
    echo "========================================================"
    # run_inference.py uses relative path 'build/bin/llama-cli' so CWD must be BITNET_CPP
    cd "${BITNET_CPP}"
    exec "${PYTHON}" run_inference.py \
        -m "${BITNET_GGUF}" \
        -p "You are a helpful assistant." \
        -n 512 \
        -cnv

# ── Step 4b: LLM → GGUF + ollama ─────────────────────────────────────────────
elif [[ "${MODEL_TYPE}" == "llm" ]]; then
    echo ""
    echo "[4/4] LLM detected → converting to GGUF (${QUANT}) and registering with ollama..."

    if [[ ! -f "${CONVERT}" ]]; then
        echo "  ERROR: ${CONVERT} not found."
        echo "  Run: git clone https://github.com/ggerganov/llama.cpp ${LLAMA_CPP}"
        exit 1
    fi

    _install_convert_deps
    if [[ "${QUANT}" == "f16" ]]; then
        _convert_f16
        _quantise "${GGUF_F16}"
    else
        Q8_GGUF=$(_convert_q8)
        _quantise "${Q8_GGUF}"
    fi
    _ollama_register "${GGUF_OUT}"

    echo ""
    echo "========================================================"
    echo "  [DONE] LLM registered with ollama."
    echo "  Chat:  ollama run ${OLLAMA_NAME}"
    echo "========================================================"
    exec ollama run "${OLLAMA_NAME}"

# ── Step 4c: VLM → Python chatbot ────────────────────────────────────────────
else
    echo ""

    # Detect CUDA availability via nvcc or nvidia-smi
    HAS_CUDA=false
    CUDA_VER=""
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
        HAS_CUDA=true
    elif command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
        HAS_CUDA=true
    fi

    if ${HAS_CUDA}; then
        echo "[4/4] VLM detected → GPU found (CUDA ${CUDA_VER}) — installing CUDA torch."
        # Map CUDA version to the nearest PyTorch wheel suffix (cu121 covers 12.x, cu118 for 11.x)
        if [[ "${CUDA_VER}" == 12* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            TORCH_TORCH="torch==2.5.1+cu121"
            TORCH_TV="torchvision==0.20.1+cu121"
        elif [[ "${CUDA_VER}" == 11* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            TORCH_TORCH="torch==2.5.1+cu118"
            TORCH_TV="torchvision==0.20.1+cu118"
        else
            # Unknown CUDA version — let pip pick the best available CUDA wheel
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            TORCH_TORCH="torch"
            TORCH_TV="torchvision"
        fi
        CHAT_DTYPE="bfloat16"
        CHAT_INFO="GPU bfloat16 — fast inference on CUDA."
    else
        echo "[4/4] VLM detected → no GPU found — installing CPU torch."
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_TORCH="torch==2.5.1+cpu"
        TORCH_TV="torchvision==0.20.1+cpu"
        CHAT_DTYPE="float32"
        CHAT_INFO="CPU float32, ~12 GB RAM (needs 16 GB swap on low-RAM machines)."
    fi

    echo "  Installing torch + torchvision (${CHAT_DTYPE})..."
    "${UV}" pip install --python "${PYTHON}" \
        "${TORCH_TORCH}" "${TORCH_TV}" \
        --index-url "${TORCH_INDEX}" \
        --quiet 2>/dev/null | tail -1 || true

    if [[ ! -f "${CHAT_SCRIPT}" ]]; then
        echo "  ERROR: ${CHAT_SCRIPT} not found at ${CHAT_SCRIPT}"
        exit 1
    fi

    echo ""
    echo "========================================================"
    echo "  [INFO] VLM chat (${CHAT_INFO})"
    echo "  Commands: /image <path> [question]  /clear  /quit"
    echo "========================================================"
    echo ""
    exec "${PYTHON}" "${CHAT_SCRIPT}" --model "${MODEL_DIR}" --dtype "${CHAT_DTYPE}"
fi
