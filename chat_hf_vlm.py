#!/usr/bin/env python3
# Run with: /home/pcarvalt/venvs/hf-vlm/bin/python /home/pcarvalt/chat_hf_vlm.py --model <model_dir>
"""Generic interactive chat CLI for any HuggingFace VLM.

Supports transformers v4 (AutoModelForVision2Seq) and v5
(AutoModelForImageTextToText), and custom trust_remote_code architectures
like Granite4VisionForConditionalGeneration.

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
            print("  WARNING: bitsandbytes not installed — falling back to float16 on CPU.", flush=True)
            load_in_4bit = False

    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"):
        cls = getattr(_tf, class_name, None)
        if cls is None:
            continue
        try:
            kwargs: dict[str, Any] = {
                # "dtype" is the current kwarg name; "torch_dtype" is deprecated in newer transformers
                "dtype": dtype,
                "low_cpu_mem_usage": low_mem,
                "trust_remote_code": True,
                **extra_kwargs,
            }
            if "device_map" not in kwargs:
                kwargs["device_map"] = "cpu"
            model = cls.from_pretrained(model_id, **kwargs)
            print(f"Loaded via {class_name}", flush=True)
            return model
        except TypeError:
            # Older transformers does not accept "dtype=", fall back to "torch_dtype="
            kwargs["torch_dtype"] = kwargs.pop("dtype", dtype)
            try:
                model = cls.from_pretrained(model_id, **kwargs)
                print(f"Loaded via {class_name} (torch_dtype fallback)", flush=True)
                return model
            except (ImportError, AttributeError, ValueError, OSError) as exc:
                print(f"{class_name} unavailable: {exc}", flush=True)
        except (ImportError, AttributeError, ValueError, OSError) as exc:
            print(f"{class_name} unavailable: {exc}", flush=True)

    raise RuntimeError("Could not load model with any Auto class.")


def load_model(
    model_id: str,
    dtype_override: str | None = None,
    load_in_4bit: bool = False,
) -> tuple[Any, AutoProcessor]:
    """Load model and processor from a local path or HuggingFace Hub ID.

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
        Tuple of (model, processor).
    """
    import warnings
    warnings.filterwarnings("ignore")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_override:
        if dtype_override not in _dtype_map:
            raise ValueError(f"Unknown dtype '{dtype_override}'. Choose from: {list(_dtype_map)}")
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
    print(f"[1/3] Processor ready. Device={device}, dtype={dtype}{quant_label}", flush=True)

    print("[2/3] Loading model weights (low_cpu_mem_usage=True)...", flush=True)
    model = _load_auto_model(model_id, dtype, low_mem=True, load_in_4bit=load_in_4bit)
    # Skip model.eval() — it can block with partially-offloaded weights.
    # torch.no_grad() in generate() is sufficient for inference-only use.
    print(f"\n[3/3] Model ready: {type(model).__name__}", flush=True)
    return model, processor


def generate(
    model: Any,
    processor: AutoProcessor,
    history: list[dict],
    images: list[Image.Image],
    max_new_tokens: int = 512,
) -> str:
    """Run inference given conversation history and optional images.

    Args:
        model: The loaded vision-language model.
        processor: The associated processor.
        history: List of message dicts with 'role' and 'content'.
        images: List of PIL images attached to the current user turn.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The assistant response string.
    """
    device = next(model.parameters()).device

    if images:
        # For vision: sanitize history so PIL image objects are removed before
        # apply_chat_template (which only needs the <image> placeholder token),
        # then pass the actual PIL images to processor() separately.
        # Sanitize history: replace {"type":"image","image":<PIL>} with
        # {"type":"image","url":""} so apply_chat_template still inserts
        # the <image> placeholder but doesn't choke on PIL objects.
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
        text = processor.apply_chat_template(
            sanitized, tokenize=False, add_generation_prompt=True
        )
        # Fallback: if template didn't insert <image>, prepend it manually
        if "<image>" not in text:
            text = "<image>\n" + text
        inputs = processor(text=[text], images=images, return_tensors="pt")
    else:
        # Text-only: apply_chat_template with tokenize=True is the cleanest path.
        try:
            inputs = processor.apply_chat_template(
                history,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except Exception:
            text = processor.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], return_tensors="pt")

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


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
    """Run the interactive chat loop."""
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

    model, processor = load_model(
        args.model,
        dtype_override=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

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
    print("Commands: /image <path> [question]  /clear  /quit")
    print("=" * 60)

    while True:
        try:
            prefix = "[img] " if pending_image else ""
            user_input = input(f"\n{prefix}You: ").strip()
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
        response = generate(model, processor, history, images_for_call)
        print(response)

        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
