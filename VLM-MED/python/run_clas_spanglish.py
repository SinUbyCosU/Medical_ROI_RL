#!/usr/bin/env python3
"""Cross-Lingual Activation Steering (CLAS) runner for Spanglish prompts."""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
# Defaults mirror Hinglish CLAS experimentation but tailored for Spanglish.
DEFAULT_MODELS: Dict[str, str] = {
    "phi35_mini": "microsoft/Phi-3.5-mini-instruct",
}

OUTPUT_PATH = Path(os.getenv("SPAN_OUTPUT_PATH", "clas_spanglish_results.jsonl"))
METADATA_PATH = Path(os.getenv("SPAN_METADATA_PATH", "clas_spanglish_metadata.jsonl"))
PROMPT_SOURCE_PATH = Path(os.getenv("SPAN_PROMPT_SOURCE", "Bias/PromptPersona_Spanglish.csv"))
PROMPT_SAMPLE_SIZE = int(os.getenv("SPAN_PROMPT_SAMPLE_SIZE", "15"))

ALPHA = float(os.getenv("SPAN_ALPHA", os.getenv("CLAS_ALPHA", "0.1")))
MAX_NEW_TOKENS = int(os.getenv("SPAN_MAX_NEW_TOKENS", os.getenv("CLAS_MAX_NEW_TOKENS", "160")))
TEMPERATURE = float(os.getenv("SPAN_TEMPERATURE", os.getenv("CLAS_TEMPERATURE", "0.0")))
BATCH_SIZE = max(1, int(os.getenv("SPAN_BATCH_SIZE", os.getenv("CLAS_BATCH_SIZE", "1"))))

TARGET_LAYER_INDEX_ENV = os.getenv("SPAN_LAYER_INDEX", os.getenv("CLAS_LAYER_INDEX"))
TARGET_LAYER_RATIO_ENV = os.getenv("SPAN_LAYER_RATIO", os.getenv("CLAS_LAYER_RATIO"))

SYSTEM_PROMPT = os.getenv("SPAN_SYSTEM_PROMPT", os.getenv("CLAS_SYSTEM_PROMPT", "You are a precise technical assistant.")).strip() or "You are a precise technical assistant."

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clas_spanglish")

# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------

def ensure_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def chunked(items: List[Dict], size: int) -> Iterable[List[Dict]]:
    span = max(size, 1)
    for start in range(0, len(items), span):
        yield items[start:start + span]


def _extract_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _repack_output(original_output, new_hidden):
    if isinstance(original_output, tuple):
        return (new_hidden,) + original_output[1:]
    if hasattr(original_output, "_replace"):
        return original_output._replace(last_hidden_state=new_hidden)
    return new_hidden


def steering_hook(delta: torch.Tensor, alpha: float):
    def hook(module, inputs, output):  # noqa: ARG001
        current = _extract_hidden(output)
        if not isinstance(current, torch.Tensor):
            return output
        if current.dim() < 3:
            return output
        if current.shape[1] <= 1:
            return output

        vec = delta
        if vec.shape[1] == 1 and current.shape[1] > 1:
            vec = vec.expand(current.shape[0], current.shape[1], vec.shape[-1])
        return _repack_output(output, current + alpha * vec)

    return hook


def find_decoder_layers(model) -> Tuple[Iterable[torch.nn.Module], int]:
    candidates: List[Iterable[torch.nn.Module]] = []

    if hasattr(model, "model"):
        inner = model.model
        for attr in ("layers", "h", "blocks"):
            if hasattr(inner, attr):
                layers = getattr(inner, attr)
                if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                    candidates.append(layers)

    if hasattr(model, "backbone"):
        inner = model.backbone
        for attr in ("layers", "blocks"):
            if hasattr(inner, attr):
                layers = getattr(inner, attr)
                if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                    candidates.append(layers)

    if not candidates:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                candidates.append(module)

    if not candidates:
        raise ValueError("Unable to locate decoder layers for activation steering")

    layers = max(candidates, key=len)
    return layers, len(layers) // 2


def render_messages(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"User: {prompt}\nAssistant:"


def capture_mean_activation(model, layer, inputs) -> torch.Tensor:
    store: Dict[str, torch.Tensor] = {}

    def hook(module, _inputs, output):  # noqa: ARG001
        store["act"] = _extract_hidden(output).detach()

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    if "act" not in store:
        raise RuntimeError("Failed to capture hidden activations")

    return store["act"].mean(dim=1, keepdim=True)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_spanglish_pairs(csv_path: Path, sample_size: int) -> List[Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt source not found: {csv_path}")

    english_lookup: Dict[Tuple[str, str, str], Dict] = {}
    target_rows: List[Dict] = []

    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            language = (row.get("language") or "").strip().lower()
            domain = (row.get("domain") or "").strip()
            topic = (row.get("topic") or "").strip()
            gender = (row.get("gender") or "").strip()
            key = (domain, topic, gender)

            prompt = (row.get("prompt_text") or "").strip()
            if not prompt:
                continue

            if language == "english":
                english_lookup[key] = {
                    "prompt": prompt,
                    "id": row.get("id"),
                    "persona_code": row.get("persona_code"),
                }
            elif language == "spanglish":
                target_rows.append({
                    "domain": domain,
                    "topic": topic,
                    "gender": gender,
                    "prompt": prompt,
                    "id": row.get("id"),
                    "persona_code": row.get("persona_code"),
                })

    if not target_rows:
        raise ValueError("No Spanglish prompts found in prompt source")

    pairs: List[Dict] = []
    for row in target_rows:
        key = (row["domain"], row["topic"], row["gender"])
        english_row = english_lookup.get(key)
        if not english_row:
            continue
        pairs.append({
            "domain": row["domain"],
            "topic": row["topic"],
            "gender": row["gender"],
            "target_prompt": row["prompt"],
            "target_id": row["id"],
            "target_persona": row.get("persona_code"),
            "english_prompt": english_row["prompt"],
            "english_id": english_row.get("id"),
            "english_persona": english_row.get("persona_code"),
        })

    if not pairs:
        raise ValueError("Failed to align Spanglish prompts with English counterparts")

    pairs.sort(key=lambda item: (item["domain"], item["topic"], item["gender"]))
    if sample_size > 0:
        pairs = pairs[:sample_size]
    return pairs


def parse_models() -> Dict[str, str]:
    raw = os.getenv("SPAN_MODELS")
    if not raw:
        return DEFAULT_MODELS

    models: Dict[str, str] = {}
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
        else:
            value = item
            key = re.sub(r"[^A-Za-z0-9_-]", "-", value.split("/")[-1])
        if not key or not value:
            continue
        models[key] = value

    return models or DEFAULT_MODELS


def derivitive_path(base: Path, suffix_key: str, enabled: bool) -> Path:
    if not enabled:
        return base
    safe_key = re.sub(r"[^A-Za-z0-9_-]", "-", suffix_key)
    return base.with_name(f"{base.stem}_{safe_key}{base.suffix}")


# --------------------------------------------------------------------------------------
# Main experiment loop
# --------------------------------------------------------------------------------------

def run_experiment() -> None:
    models = parse_models()
    multi_model = len(models) > 1

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    prompt_pairs = load_spanglish_pairs(PROMPT_SOURCE_PATH, PROMPT_SAMPLE_SIZE)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    for key, model_id in models.items():
        output_path = derivitive_path(OUTPUT_PATH, key, multi_model)
        metadata_path = derivitive_path(METADATA_PATH, key, multi_model)

        logger.info("Loading model %s (%s)", key, model_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "quantization_config": quant_config,
            }
            if os.getenv("SPAN_FULL_PRECISION", "0").lower() in {"1", "true", "yes"}:
                load_kwargs.pop("quantization_config", None)
                load_kwargs["torch_dtype"] = torch.float16

            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except Exception:
            logger.exception("Skipping model %s (%s) due to load failure", key, model_id)
            continue

        if "phi-3.5" in model_id.lower():
            logger.info("Disabling KV cache to avoid DynamicCache issues")
            model.config.use_cache = False
            if hasattr(model, "generation_config"):
                model.generation_config.use_cache = False

        with output_path.open("w", encoding="utf-8") as output_file, metadata_path.open("w", encoding="utf-8") as metadata_file:
            try:
                ensure_tokens(tokenizer, model)
                layers, default_idx = find_decoder_layers(model)

                target_idx = default_idx
                num_layers = len(layers)
                if TARGET_LAYER_INDEX_ENV is not None:
                    try:
                        override = int(TARGET_LAYER_INDEX_ENV)
                        if override < 0:
                            override = num_layers + override
                        target_idx = min(max(override, 0), num_layers - 1)
                    except ValueError:
                        logger.warning("Invalid layer index '%s'; using default %d", TARGET_LAYER_INDEX_ENV, target_idx)
                elif TARGET_LAYER_RATIO_ENV is not None:
                    try:
                        ratio = float(TARGET_LAYER_RATIO_ENV)
                        if not 0.0 <= ratio <= 1.0:
                            raise ValueError
                        target_idx = min(max(int(round(ratio * (num_layers - 1))), 0), num_layers - 1)
                    except ValueError:
                        logger.warning("Invalid layer ratio '%s'; using default %d", TARGET_LAYER_RATIO_ENV, target_idx)

                target_layer = layers[target_idx]
                device = next(iter(model.parameters())).device

                metadata = {
                    "model_key": key,
                    "model_id": model_id,
                    "target_layer_index": target_idx,
                    "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
                    "hidden_size": getattr(model.config, "hidden_size", None),
                    "alpha": ALPHA,
                    "prompt_source": str(PROMPT_SOURCE_PATH),
                    "sample_size": len(prompt_pairs),
                }
                metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                metadata_file.flush()

                logger.info("Using layer %d/%d with alpha %.3f", target_idx, num_layers, ALPHA)

                for batch in chunked(prompt_pairs, BATCH_SIZE):
                    english_rendered = [render_messages(tokenizer, pair["english_prompt"]) for pair in batch]
                    target_rendered = [render_messages(tokenizer, pair["target_prompt"]) for pair in batch]

                    inputs_eng = to_device(
                        tokenizer(
                            english_rendered,
                            return_tensors="pt",
                            padding=True,
                        ),
                        device,
                    )
                    inputs_target = to_device(
                        tokenizer(
                            target_rendered,
                            return_tensors="pt",
                            padding=True,
                        ),
                        device,
                    )

                    vec_eng = capture_mean_activation(model, target_layer, inputs_eng)
                    vec_target = capture_mean_activation(model, target_layer, inputs_target)
                    delta = (vec_eng - vec_target).detach()

                    generate_kwargs = {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "do_sample": TEMPERATURE > 0,
                        "temperature": TEMPERATURE,
                    }
                    if hasattr(model.config, "use_cache") and model.config.use_cache is False:
                        generate_kwargs["use_cache"] = False
                    if TEMPERATURE <= 0:
                        generate_kwargs["do_sample"] = False

                    with torch.no_grad():
                        baseline_tokens = model.generate(**inputs_target, **generate_kwargs)
                    baseline_texts = tokenizer.batch_decode(baseline_tokens, skip_special_tokens=True)

                    steer_handle = target_layer.register_forward_hook(steering_hook(delta, ALPHA))
                    try:
                        with torch.no_grad():
                            steered_tokens = model.generate(**inputs_target, **generate_kwargs)
                    finally:
                        steer_handle.remove()
                    steered_texts = tokenizer.batch_decode(steered_tokens, skip_special_tokens=True)

                    for idx, pair in enumerate(batch):
                        record = {
                            "model_key": key,
                            "model": model_id,
                            "prompt": pair["target_prompt"],
                            "prompt_with_prefix": target_rendered[idx],
                            "domain": pair["domain"],
                            "topic": pair["topic"],
                            "gender": pair["gender"],
                            "target_persona": pair.get("target_persona"),
                            "english_persona": pair.get("english_persona"),
                            "target_prompt_id": pair.get("target_id"),
                            "english_prompt_id": pair.get("english_id"),
                            "layer_index": target_idx,
                            "alpha": ALPHA,
                            "baseline_response": baseline_texts[idx],
                            "steered_response": steered_texts[idx],
                        }
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output_file.flush()

                        logger.info(
                            "Processed %s/%s (%s) [%s]",
                            pair.get("target_id"),
                            len(prompt_pairs),
                            pair["topic"],
                            key,
                        )
            except Exception:
                logger.exception("Error while running model %s (%s)", key, model_id)
            finally:
                del model
                del tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
