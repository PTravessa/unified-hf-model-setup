#!/usr/bin/env python3
# Run with: /home/pcarvalt/venvs/hf-vlm/bin/python /home/pcarvalt/chat_hf_vlm.py --model <model_dir>
"""Generic interactive chat CLI for any HuggingFace VLM.

Supports:
- Standard transformers Auto classes (AutoModelForImageTextToText, AutoModelForVision2Seq)
- Custom trust_remote_code architectures (auto_map fallback)
- Models without a chat_template (plain-text prompt fallback)
- Auto-patching of known broken imports in model files (e.g. LlamaFlashAttention2)

Usage (always pass the full absolute path to the local model directory):
    python chat_hf_vlm.py --model /home/pcarvalt/models/Qwen-Qwen3-VL-2B-Instruct
    python chat_hf_vlm.py --model /home/pcarvalt/models/ibm-granite-granite-vision-4.1-4b
    python chat_hf_vlm.py --model /home/pcarvalt/models/Qwen-Qwen3-VL-2B-Instruct --image /path/to/img.jpg

In-chat commands:
    /image <path> [question]   -- attach an image (optionally with inline question)
    /clear                     -- reset conversation history
    /quit or Ctrl-C            -- exit
"""
import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Known source-level patches applied to model files before loading.
# Each entry: (filename_glob, old_text, new_text, description)
# ---------------------------------------------------------------------------
_SOURCE_PATCHES: list[tuple[str, str, str, str]] = [
    (
        "modeling_deepseekv2.py",
        "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n    LlamaFlashAttention2\n)",
        "from transformers.models.llama.modeling_llama import (\n    LlamaAttention,\n)\nLlamaFlashAttention2 = LlamaAttention  # removed in transformers>=4.48, aliased for compat",
        "DeepSeek V2: LlamaFlashAttention2 removed in transformers>=4.48",
    ),
]


def _compat_patch(model_id: str) -> None:
    """Apply known source-level compatibility patches to model files in *model_id*.

    Patches are applied to both the model directory and the HuggingFace
    transformers module cache so that ``trust_remote_code`` loads pick up
    the fix regardless of which copy is loaded first.

    Args:
        model_id: Path to the local model directory.
    """
    # Candidate directories: model dir + HF module cache
    cache_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    candidates: list[Path] = [Path(model_id)]
    if cache_root.exists():
        # Find any subdirectory whose name contains the model slug
        slug = Path(model_id).name.lower().replace("-", "_hyphen_")
        for d in cache_root.iterdir():
            if d.is_dir() and (slug in d.name.lower() or d.name.lower() in slug):
                candidates.append(d)

    for filename_glob, old_text, new_text, description in _SOURCE_PATCHES:
        for base in candidates:
            for filepath in base.rglob(filename_glob):
                src = filepath.read_text(encoding="utf-8")
                if old_text in src:
                    src = src.replace(old_text, new_text)
                    filepath.write_text(src, encoding="utf-8")
                    # Remove stale .pyc so Python re-compiles the patched source
                    for pyc in filepath.parent.rglob("*.pyc"):
                        try:
                            pyc.unlink()
                        except OSError:
                            pass
                    print(f"  [compat] Patched {filepath.name}: {description}", flush=True)


def _has_chat_template(processor: Any) -> bool:
    """Return True if *processor* has a usable chat template.

    Args:
        processor: The loaded processor / tokenizer.

    Returns:
        True if ``apply_chat_template`` will succeed, False otherwise.
    """
    # AutoProcessor wraps a tokenizer; check the tokenizer directly when present
    tok = getattr(processor, "tokenizer", processor)
    tmpl = getattr(tok, "chat_template", None)
    return bool(tmpl)


def _detect_generate_style(model: Any, processor: Any) -> str:
    """Detect the correct inference style for *model*.

    Inference styles:
    - ``"standard"``      : use processor.apply_chat_template + model.generate
    - ``"no_template"``   : model has no chat_template; build a plain prompt instead
    - ``"native_infer"``  : model ships its own infer() method (e.g. DeepSeek-OCR-2,
                             usually GPU-only); we warn and fall back to no_template

    Args:
        model: The loaded model instance.
        processor: The loaded processor instance.

    Returns:
        One of ``"standard"``, ``"no_template"``, or ``"native_infer"``.
    """
    if hasattr(model, "infer"):
        # Model has a custom infer() — check if it's CUDA-only
        import inspect
        src = inspect.getsource(model.infer)
        if ".cuda()" in src or 'autocast("cuda"' in src:
            print(
                "  [warn] This model's infer() is CUDA-only and cannot run on CPU.\n"
                "         Falling back to generic generate() — results may vary.",
                flush=True,
            )
        return "native_infer"
    if _has_chat_template(processor):
        return "standard"
    return "no_template"


def _build_plain_prompt(history: list[dict], images: list[Any]) -> str:
    """Build a minimal plain-text prompt for models without a chat template.

    Uses DeepSeek-style ``<|User|>`` / ``<|Assistant|>`` roles with an
    ``<image>`` token prepended when images are present.

    Args:
        history: List of message dicts with ``role`` and ``content`` keys.
        images: List of PIL images (used only to decide whether to insert
            the ``<image>`` placeholder).

    Returns:
        A plain-text prompt string.
    """
    lines: list[str] = []
    for i, msg in enumerate(history):
        role = msg.get("role", "user")
        # Flatten content: extract text parts from list content
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = " ".join(text_parts).strip()
        # Prepend <image> token on the first user message when images are attached
        if role in ("user", "<|User|>") and images and i == len(history) - 1:
            content = f"<image>\n{content}"
        lines.append(f"<|{role.strip('<|>')}|>{content}")
    lines.append("<|Assistant|>")
    return "\n".join(lines)


def _load_auto_model(
    model_id: str,
    dtype: torch.dtype,
    low_mem: bool = True,
    load_in_4bit: bool = False,
) -> Any:
    """Try Auto model classes in order until one succeeds.

    Order: AutoModelForImageTextToText (transformers>=5) ->
           AutoModelForVision2Seq (transformers<5) ->
           AutoModel (trust_remote_code fallback).

    Args:
        model_id: Local directory path or HuggingFace repo ID.
        dtype: Torch dtype to load the model with.
        low_mem: If True, enables low_cpu_mem_usage to load shards one at a
            time, halving peak RAM usage during loading.
        load_in_4bit: If True, loads weights in 4-bit via bitsandbytes (~2-3 GB
            for a 4B model). Requires the bitsandbytes package.

    Returns:
        Loaded model instance.

    Raises:
        RuntimeError: If no Auto class succeeds.
    """
    import transformers as _tf

    has_cuda = torch.cuda.is_available()
    extra_kwargs: dict[str, Any] = {}

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            extra_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                # Allow non-quantisable layers (e.g. lm_head) to stay in fp32 on CPU
                llm_int8_enable_fp32_cpu_offload=True,
            )
            if has_cuda:
                # GPU available: let accelerate place layers optimally
                extra_kwargs["device_map"] = "auto"
            else:
                # CPU-only: every layer on CPU; non-quantisable layers stay fp32
                extra_kwargs["device_map"] = {"": "cpu"}
        except ImportError:
            print(
                "  WARNING: bitsandbytes not installed "
                "- falling back to float16 on CPU.",
                flush=True,
            )
            load_in_4bit = False

    import logging

    base_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": low_mem,
        "trust_remote_code": True,
        **extra_kwargs,
    }
    if "device_map" not in base_kwargs:
        base_kwargs["device_map"] = "cpu"

    # Suppress transformers "Unrecognized configuration class" warnings that are
    # expected when probing Auto classes for custom-arch models (e.g. DeepSeek-OCR-2).
    _tf_logger = logging.getLogger("transformers")
    _prev_level = _tf_logger.level

    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"):
        cls = getattr(_tf, class_name, None)
        if cls is None:
            continue
        try:
            _tf_logger.setLevel(logging.ERROR)
            kwargs: dict[str, Any] = {"dtype": dtype, **base_kwargs}
            model = cls.from_pretrained(model_id, **kwargs)
            _tf_logger.setLevel(_prev_level)
            print(f"Loaded via {class_name}", flush=True)
            return model
        except TypeError:
            # Older transformers does not accept "dtype=", fall back to "torch_dtype="
            kwargs = {"torch_dtype": dtype, **base_kwargs}
            try:
                model = cls.from_pretrained(model_id, **kwargs)
                _tf_logger.setLevel(_prev_level)
                print(f"Loaded via {class_name} (torch_dtype fallback)", flush=True)
                return model
            except (ImportError, AttributeError, ValueError, OSError):
                pass  # try next Auto class silently
        except (ImportError, AttributeError, ValueError, OSError):
            pass  # try next Auto class silently
    _tf_logger.setLevel(_prev_level)

    # Final fallback: load the model class directly from the auto_map in config.json.
    # Required for models with fully custom architectures (e.g. DeepSeek-OCR-2)
    # that are not registered with any standard Auto class.
    print("Auto classes exhausted — trying direct class load via auto_map...", flush=True)
    try:
        import json
        import os

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cfg_path = os.path.join(model_id, "config.json")
        with open(cfg_path, "r") as f:
            raw_cfg = json.load(f)
        auto_map: dict[str, str] = raw_cfg.get("auto_map", {})
        # Prefer AutoModelForImageTextToText > AutoModel
        for key in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
            if key in auto_map:
                # e.g. "modeling_deepseekocr2.DeepseekOCR2ForCausalLM"
                module_dot_class = auto_map[key]
                break
        else:
            raise RuntimeError("No usable entry found in auto_map.")

        model_cls = get_class_from_dynamic_module(module_dot_class, model_id)
        kwargs = {"torch_dtype": dtype, **base_kwargs}
        model = model_cls.from_pretrained(model_id, **kwargs)
        print(f"Loaded via direct class load: {module_dot_class}", flush=True)
        return model
    except Exception as exc:
        print(f"Direct class load failed: {exc}", flush=True)

    raise RuntimeError("Could not load model with any Auto class.")


def load_model(
    model_id: str,
    dtype_override: str | None = None,
    load_in_4bit: bool = False,
) -> tuple[Any, AutoProcessor, str]:
    """Load model and processor from a local path or HuggingFace Hub ID.

    Also applies known source-level compatibility patches (e.g. removed
    transformers symbols) and detects the correct inference style.

    On CPU, defaults to float16 with low_cpu_mem_usage=True to halve peak RAM.
    On CUDA, defaults to bfloat16.
    With load_in_4bit=True, uses bitsandbytes NF4 quantisation (~2-3 GB for 4B models).

    Args:
        model_id: Local directory path or HuggingFace repo ID.
        dtype_override: Optional dtype string ('float32', 'float16', 'bfloat16').
            Overrides the automatic selection.
        load_in_4bit: If True, loads weights in 4-bit via bitsandbytes.
            Best option for 16 GB RAM systems running 4B+ models.

    Returns:
        Tuple of (model, processor, generate_style) where generate_style is
        one of ``"standard"``, ``"no_template"``, or ``"native_infer"``.
    """
    import warnings

    warnings.filterwarnings("ignore")

    # Apply known source-level patches before loading (idempotent)
    if Path(model_id).is_dir():
        _compat_patch(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_override:
        if dtype_override not in _dtype_map:
            raise ValueError(
                f"Unknown dtype '{dtype_override}'. "
                f"Choose from: {list(_dtype_map)}"
            )
        dtype = _dtype_map[dtype_override]
    elif device == "cuda":
        dtype = torch.bfloat16
    else:
        # float16 on CPU: ~8 GB for 4B model, fits in 16 GB with OS overhead.
        # low_cpu_mem_usage=True loads shards sequentially, halving peak RAM.
        dtype = torch.float16

    quant_label = " + 4-bit NF4" if load_in_4bit else ""
    print(f"\n[1/3] Loading processor from {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"      Processor ready. Device={device}, dtype={dtype}{quant_label}", flush=True)

    print("[2/3] Loading model weights (low_cpu_mem_usage=True)...", flush=True)
    model = _load_auto_model(model_id, dtype, low_mem=True, load_in_4bit=load_in_4bit)
    # Skip model.eval() — it can block with partially-offloaded weights.
    # torch.no_grad() in generate() is sufficient for inference-only use.
    gen_style = _detect_generate_style(model, processor)
    print(f"\n[3/3] Model ready: {type(model).__name__}  (generate style: {gen_style})", flush=True)
    return model, processor, gen_style


def get_max_context(model: Any) -> int:
    """Return the model's maximum context window in tokens.

    Checks common config attribute names in order of precedence.
    Falls back to 4096 if none are found.

    Args:
        model: The loaded model instance.

    Returns:
        Maximum number of tokens the model supports as input + output.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return 4096
    for attr in (
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
        "n_ctx",
    ):
        val = getattr(cfg, attr, None)
        if val is None:
            # also check nested language_config (e.g. DeepSeek VL2)
            lang = getattr(cfg, "language_config", None)
            if lang is not None:
                val = getattr(lang, attr, None)
        if val is not None and isinstance(val, int) and val > 0:
            return val
    return 4096


# ANSI escape codes for in-place terminal line updates.
_CURSOR_UP   = "\033[1A"
_ERASE_LINE  = "\033[2K"


def _context_bar_line(used_tokens: int, max_tokens: int, width: int = 44) -> str:
    """Return a formatted context bar string (no trailing newline).

    Args:
        used_tokens: Number of tokens consumed so far.
        max_tokens: Total context window size.
        width: Width of the fill portion of the bar.

    Returns:
        Formatted bar string.
    """
    pct = min(used_tokens / max_tokens, 1.0) if max_tokens else 0.0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    remaining = max(max_tokens - used_tokens, 0)
    return (
        f"\n  Context: [{bar}] {used_tokens:,}/{max_tokens:,}"
        f"  {remaining:,} left  {pct * 100:.1f}%"
    )


def print_context_bar(used_tokens: int, max_tokens: int, *, update: bool = False) -> None:
    """Print or in-place overwrite the context usage bar.

    On the first call (update=False) the bar is written as a new line.
    On subsequent calls (update=True) the cursor moves up one line and
    rewrites it in place so the bar stays at the same screen position.

    Args:
        used_tokens: Number of tokens consumed so far (prompt + history).
        max_tokens: Total context window size.
        update: If True, overwrite the previous bar line in place.
    """
    line = _context_bar_line(used_tokens, max_tokens)
    if update:
        sys.stdout.write(f"{_CURSOR_UP}{_ERASE_LINE}\r{line}\n")
    else:
        sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def generate(
    model: Any,
    processor: AutoProcessor,
    history: list[dict],
    images: list[Image.Image],
    max_new_tokens: int = 512,
    gen_style: str = "standard",
) -> tuple[str, int]:
    """Run inference given conversation history and optional images.

    Supports three inference styles controlled by *gen_style*:
    - ``"standard"``    : apply_chat_template + processor + model.generate
    - ``"no_template"`` : plain-text prompt fallback (no chat_template)
    - ``"native_infer"``: model has its own infer(); falls back to no_template
                          if not on CUDA

    Args:
        model: The loaded vision-language model.
        processor: The associated processor.
        history: List of message dicts with 'role' and 'content'.
        images: List of PIL images attached to the current user turn.
        max_new_tokens: Maximum tokens to generate.
        gen_style: Inference style detected by _detect_generate_style().

    Returns:
        Tuple of (assistant response string, total input tokens used).
    """
    device = next(model.parameters()).device
    tok = getattr(processor, "tokenizer", processor)

    # ------------------------------------------------------------------ #
    # Build inputs                                                         #
    # ------------------------------------------------------------------ #
    inputs: dict[str, Any] | None = None

    if gen_style == "standard":
        if images:
            # Sanitize history: replace PIL image objects with a plain placeholder
            # so apply_chat_template doesn't choke, then pass real images to processor()
            sanitized: list[dict] = []
            for msg in history:
                if isinstance(msg.get("content"), list):
                    new_content = []
                    for part in msg["content"]:
                        if isinstance(part, dict) and part.get("type") == "image":
                            new_content.append({"type": "image", "url": ""})
                        else:
                            new_content.append(part)
                    sanitized.append({"role": msg["role"], "content": new_content})
                else:
                    sanitized.append(msg)
            try:
                text = processor.apply_chat_template(
                    sanitized, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = _build_plain_prompt(history, images)
            if "<image>" not in text:
                text = "<image>\n" + text
            inputs = processor(text=[text], images=images, return_tensors="pt")
        else:
            try:
                inputs = processor.apply_chat_template(
                    history,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except Exception:
                try:
                    text = processor.apply_chat_template(
                        history, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(text=[text], return_tensors="pt")
                except Exception:
                    # Last resort: plain prompt
                    text = _build_plain_prompt(history, [])
                    inputs = tok(text, return_tensors="pt")

    if gen_style in ("no_template", "native_infer") or inputs is None:
        # Plain-text prompt: works for models without chat_template and as
        # CPU fallback for native_infer models (their infer() is CUDA-only)
        text = _build_plain_prompt(history, images)
        if images:
            try:
                inputs = processor(text=[text], images=images, return_tensors="pt")
            except Exception:
                inputs = tok(text, return_tensors="pt")
        else:
            inputs = tok(text, return_tensors="pt")

    assert inputs is not None, "Could not build model inputs."
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # ------------------------------------------------------------------ #
    # Generate                                                             #
    # ------------------------------------------------------------------ #
    import logging
    _tf_gen_logger = logging.getLogger("transformers.generation")
    _prev = _tf_gen_logger.level
    _tf_gen_logger.setLevel(logging.ERROR)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    _tf_gen_logger.setLevel(_prev)

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return tok.decode(generated, skip_special_tokens=True).strip(), input_len


def build_user_message(text: str, image: Image.Image | None) -> dict:
    """Build a user message dict, optionally including an image.

    Args:
        text: The user text prompt.
        image: Optional PIL image to attach.

    Returns:
        A message dict compatible with apply_chat_template.
    """
    content: list[dict] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def main() -> None:
    """Run the interactive chat loop.

    Displays a context window bar at startup and after every assistant turn
    so the user can monitor token usage against the model's max context.
    """
    parser = argparse.ArgumentParser(description="Generic HF VLM interactive chat")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Local model directory or HuggingFace repo ID",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to an initial image")
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help=(
            "Force model dtype. Defaults: float16 on CPU, bfloat16 on CUDA. "
            "Use float32 only if float16 causes errors."
        ),
    )
    parser.add_argument(
        "--4bit",
        dest="load_in_4bit",
        action="store_true",
        default=False,
        help=(
            "Load weights in 4-bit NF4 via bitsandbytes (~2-3 GB for 4B models). "
            "Recommended for 16 GB RAM systems. Requires: pip install bitsandbytes"
        ),
    )
    args = parser.parse_args()

    model, processor, gen_style = load_model(
        args.model,
        dtype_override=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    max_ctx = get_max_context(model)
    history: list[dict] = []
    pending_image: Image.Image | None = None

    if args.image:
        path = Path(args.image)
        if path.exists():
            pending_image = Image.open(path).convert("RGB")
            print(f"Image loaded: {args.image}")
        else:
            print(f"Warning: image not found: {args.image}")

    model_name = Path(args.model).name if Path(args.model).exists() else args.model
    print("=" * 60)
    print(f"Chat with {model_name}")
    print(f"Max context: {max_ctx:,} tokens")
    print("Commands: /image <path> [question]  /clear  /quit")
    print("=" * 60)

    used_tokens: int = 0  # updated after every generate() call

    while True:
        try:
            prefix = "[img] " if pending_image else ""
            bar = _context_bar_line(used_tokens, max_ctx)
            user_input = input(f"{bar}\n{prefix}You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)

        if user_input.lower() == "/clear":
            history = []
            pending_image = None
            print("Conversation cleared.")
            continue

        # /image can be standalone or inline: /image /path/file.jpg optional question
        if user_input.lower().startswith("/image "):
            remainder = user_input[7:].strip()
            parts = remainder.split(None, 1)
            img_path = Path(parts[0])
            inline_text = parts[1].strip() if len(parts) > 1 else ""
            if img_path.exists():
                pending_image = Image.open(img_path).convert("RGB")
                print(f"Image loaded: {img_path}")
            else:
                print(f"File not found: {img_path}")
                continue
            if not inline_text:
                continue
            user_input = inline_text

        images_for_call: list[Image.Image] = []
        if pending_image is not None:
            images_for_call.append(pending_image)

        msg = build_user_message(user_input, pending_image)
        history.append(msg)
        pending_image = None

        print("Assistant: ", end="", flush=True)
        response, input_tokens = generate(model, processor, history, images_for_call, gen_style=gen_style)
        print(response)

        history.append({"role": "assistant", "content": response})
        used_tokens = input_tokens  # bar updates on next loop iteration via input() prompt


if __name__ == "__main__":
    main()
