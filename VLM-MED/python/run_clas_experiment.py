#!/usr/bin/env python3
"""Cross-Lingual Activation Steering (CLAS) experiment runner."""

import csv
import gc
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import torch
from huggingface_hub.errors import GatedRepoError
from requests import HTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DEFAULT_MODELS: Dict[str, str] = {
    "llama31_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2_7b": "Qwen/Qwen2-7B-Instruct",
    "zephyr_7b": "HuggingFaceH4/zephyr-7b-beta",
    "openhermes_mistral": "teknium/OpenHermes-2.5-Mistral-7B",
    "nous_hermes_mistral": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "phi35_mini": "microsoft/Phi-3.5-mini-instruct",
}


def resolve_models() -> Dict[str, str]:
    requested = os.getenv("CLAS_MODELS")
    if not requested:
        return DEFAULT_MODELS

    keys = [key.strip() for key in requested.split(",") if key.strip()]
    unknown = [key for key in keys if key not in DEFAULT_MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys requested via CLAS_MODELS: {unknown}")
    return {key: DEFAULT_MODELS[key] for key in keys}


MODELS = resolve_models()
OUTPUT_PATH = Path(os.getenv("CLAS_OUTPUT_PATH", "clas_multi_model_results.jsonl"))
METADATA_PATH = Path(os.getenv("CLAS_METADATA_PATH", "clas_model_metadata.jsonl"))
PROMPT_SOURCE_PATH = Path(os.getenv("CLAS_PROMPT_SOURCE", "Bias/PromptPersona_Full_600.csv"))
PROMPT_SAMPLE_SIZE = int(os.getenv("CLAS_PROMPT_SAMPLE_SIZE", "75"))

stop_after_env = os.getenv("CLAS_STOP_AFTER")
try:
    STOP_AFTER = int(stop_after_env) if stop_after_env is not None else None
except ValueError:
    STOP_AFTER = None

PROMPT_PREFIX = (os.getenv("CLAS_PROMPT_PREFIX", "") or "").strip()
PROMPT_PREFIX_POS = os.getenv("CLAS_PROMPT_PREFIX_POS", "prepend").strip().lower()
if PROMPT_PREFIX_POS not in {"prepend", "append"}:
    PROMPT_PREFIX_POS = "prepend"

SYSTEM_PROMPT = (os.getenv("CLAS_SYSTEM_PROMPT", "You are a precise technical assistant.") or "").strip() or "You are a precise technical assistant."

FORCE_HINGLISH = str(os.getenv("CLAS_FORCE_HINGLISH", "0")).lower() in {"1", "true", "yes"}
HINGLISH_THRESHOLD = float(os.getenv("CLAS_HINGLISH_THRESHOLD", "0.08"))
HINGLISH_MIN_KEYWORDS = int(os.getenv("CLAS_HINGLISH_MIN_KEYWORDS", "3"))
REWRITE_TEMPERATURE = float(os.getenv("CLAS_REWRITE_TEMPERATURE", "0.5"))
HINGLISH_REWRITE_SYSTEM = (
    os.getenv(
        "CLAS_REWRITE_SYSTEM_PROMPT",
        "You rewrite answers into natural, colloquial Hinglish (Hindi words transliterated).",
    )
    or "You rewrite answers into natural, colloquial Hinglish (Hindi words transliterated)."
)
FORCE_REWRITE_ALWAYS = str(os.getenv("CLAS_FORCE_REWRITE_ALWAYS", "0")).lower() in {"1", "true", "yes"}

TARGET_LAYER_INDEX_ENV = os.getenv("CLAS_LAYER_INDEX")
TARGET_LAYER_RATIO_ENV = os.getenv("CLAS_LAYER_RATIO")

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
HINGLISH_KEYWORDS = {
    "bhai",
    "yaar",
    "nahi",
    "haan",
    "karna",
    "karoon",
    "karo",
    "mat",
    "bahut",
    "bohot",
    "hai",
    "hun",
    "hoon",
    "kya",
    "kaise",
    "kyunki",
    "par",
    "se",
    "mein",
    "meri",
    "tera",
    "tum",
    "tumhe",
    "tumhare",
    "apni",
    "apna",
    "samadhan",
    "karoge",
    "matlab",
    "garam",
    "thoda",
    "zyada",
    "thik",
    "sahi",
    "batau",
    "dhyaan",
    "dhyan",
    "jaldi",
    "chalega",
    "karlo",
    "krlo",
    "lekin",
    "magar",
    "bilkul",
    "sambhal",
    "repair",
    "service",
    "thandi",
    "garam",
    "saans",
}

ALPHA = float(os.getenv("CLAS_ALPHA", "1.5"))
MAX_NEW_TOKENS = int(os.getenv("CLAS_MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("CLAS_TEMPERATURE", "0.6"))
BATCH_SIZE = max(1, int(os.getenv("CLAS_BATCH_SIZE", "1")))
FULL_PRECISION_MODELS = {
    key.strip()
    for key in os.getenv("CLAS_FULL_PRECISION_MODELS", "").split(",")
    if key.strip()
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clas")


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


def tokenize_latin_words(text: str) -> List[str]:
    return LATIN_WORD_RE.findall(text.lower())


def hinglish_keyword_ratio(text: str) -> float:
    tokens = tokenize_latin_words(text)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in HINGLISH_KEYWORDS)
    return hits / max(len(tokens), 1)


def needs_hinglish_enforcement(text: str) -> bool:
    if not text.strip():
        return True
    if DEVANAGARI_RE.search(text):
        return False
    ratio = hinglish_keyword_ratio(text)
    tokens = tokenize_latin_words(text)
    hits = sum(1 for token in tokens if token in HINGLISH_KEYWORDS)
    if hits >= HINGLISH_MIN_KEYWORDS:
        return False
    return ratio < HINGLISH_THRESHOLD


def rewrite_to_hinglish(model, tokenizer, device, text: str) -> str:
    messages = [
        {"role": "system", "content": HINGLISH_REWRITE_SYSTEM},
        {
            "role": "user",
            "content": (
                "Rewrite the following answer entirely in natural, colloquial Hinglish (Hindi words transliterated). "
                "Preserve the meaning and structure, keep it concise, and avoid English sentences except necessary tech terms.\n\n"
                f"Answer:\n{text}"
            ),
        },
    ]
    chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_template, return_tensors="pt").to(device)

    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": REWRITE_TEMPERATURE,
    }
    if hasattr(model.config, "use_cache") and model.config.use_cache is False:
        generate_kwargs["use_cache"] = False

    with torch.no_grad():
        output_tokens = model.generate(**inputs, **generate_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    generated = output_tokens[0][prompt_len:]
    rewritten = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return rewritten or text


def enforce_hinglish(model, tokenizer, device, text: str) -> Tuple[str, str]:
    status = "original"
    if not FORCE_HINGLISH:
        return text, status

    trigger_rewrite = FORCE_REWRITE_ALWAYS or needs_hinglish_enforcement(text)
    if trigger_rewrite:
        rewritten = rewrite_to_hinglish(model, tokenizer, device, text)
        if rewritten != text:
            text = rewritten
            status = "rewritten"
        else:
            status = "rewrite_attempted"

    if needs_hinglish_enforcement(text):
        status = "needs_manual_fix"

    return text, status


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

        seq_len = current.shape[1]
        if seq_len <= 1:
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


def load_processed_keys(path: Path) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    if not path.exists():
        return keys

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = record.get("model")
            prompt = record.get("prompt")
            if model and prompt:
                keys.add((model, prompt))
    return keys


def load_hinglish_pairs(csv_path: Path, sample_size: int) -> List[Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt source not found: {csv_path}")

    eng_lookup: Dict[Tuple[str, str, str], Dict] = {}
    hing_rows: List[Dict] = []

    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            language = (row.get("language") or "").strip().lower()
            domain = (row.get("domain") or "").strip()
            topic = (row.get("topic") or "").strip()
            gender = (row.get("gender") or "").strip()
            key = (domain, topic, gender)

            if language == "english":
                eng_lookup[key] = {
                    "prompt": (row.get("prompt_text") or "").strip(),
                    "id": row.get("id"),
                    "persona_code": row.get("persona_code"),
                }
            elif language == "hinglish":
                hing_rows.append({
                    "domain": domain,
                    "topic": topic,
                    "gender": gender,
                    "prompt": (row.get("prompt_text") or "").strip(),
                    "id": row.get("id"),
                    "persona_code": row.get("persona_code"),
                })

    if not hing_rows:
        raise ValueError("No Hinglish prompts found in prompt source")

    pairs: List[Dict] = []
    for row in hing_rows:
        key = (row["domain"], row["topic"], row["gender"])
        english_row = eng_lookup.get(key)
        if not english_row:
            continue
        pairs.append(
            {
                "domain": row["domain"],
                "topic": row["topic"],
                "gender": row["gender"],
                "hinglish_prompt": row["prompt"],
                "hinglish_id": row["id"],
                "hinglish_persona": row.get("persona_code"),
                "english_prompt": english_row["prompt"],
                "english_id": english_row.get("id"),
                "english_persona": english_row.get("persona_code"),
            }
        )

    if not pairs:
        raise ValueError("Failed to align Hinglish prompts with English counterparts")

    pairs.sort(key=lambda item: (item["domain"], item["topic"], item["gender"]))

    shuffle_env = str(os.getenv("CLAS_SHUFFLE", "0")).lower()
    if shuffle_env in {"1", "true", "yes"}:
        seed_env = os.getenv("CLAS_SEED")
        if seed_env is not None:
            try:
                random.seed(int(seed_env))
            except ValueError:
                pass
        random.shuffle(pairs)

    if sample_size > 0:
        pairs = pairs[:sample_size]

    return pairs


def collect_model_metadata(model_key: str, model_id: str, model, target_idx: int) -> Dict:
    config = getattr(model, "config", None)

    def cfg_field(name: str, default=None):
        return getattr(config, name, default) if config is not None else default

    metadata = {
        "model": model_key,
        "model_id": model_id,
        "target_layer_index": target_idx,
        "num_hidden_layers": cfg_field("num_hidden_layers"),
        "hidden_size": cfg_field("hidden_size"),
        "intermediate_size": cfg_field("intermediate_size"),
        "num_attention_heads": cfg_field("num_attention_heads"),
        "rope_theta": cfg_field("rope_theta"),
        "max_position_embeddings": cfg_field("max_position_embeddings"),
        "vocab_size": cfg_field("vocab_size"),
        "model_type": cfg_field("model_type"),
    }

    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype):
        metadata["model_dtype"] = str(model_dtype)

    return metadata


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


def run_experiment() -> None:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    prompt_pairs = load_hinglish_pairs(PROMPT_SOURCE_PATH, PROMPT_SAMPLE_SIZE)
    processed_keys = load_processed_keys(OUTPUT_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("a", encoding="utf-8") as output_file, METADATA_PATH.open("a", encoding="utf-8") as metadata_file:
        for model_key, model_id in MODELS.items():
            logger.info("Processing %s (%s)", model_key, model_id)

            tokenizer = None
            model = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

                load_kwargs = {
                    "device_map": "auto",
                    "trust_remote_code": True,
                }
                if model_key in FULL_PRECISION_MODELS:
                    load_kwargs["torch_dtype"] = torch.float16
                else:
                    load_kwargs["quantization_config"] = quant_config

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **load_kwargs,
                )

                if "phi-3.5" in model_id.lower():
                    logger.info("Disabling KV cache for %s to avoid DynamicCache issues", model_key)
                    model.config.use_cache = False
                    if hasattr(model, "generation_config"):
                        model.generation_config.use_cache = False
            except (GatedRepoError, HTTPError) as err:
                logger.warning("Skipping %s due to access issue: %s", model_key, err)
                continue
            except Exception as err:  # pylint: disable=broad-except
                logger.exception("Failed to load %s", model_key)
                continue

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
                        logger.warning("Invalid CLAS_LAYER_INDEX=%s; using default index %d", TARGET_LAYER_INDEX_ENV, target_idx)
                elif TARGET_LAYER_RATIO_ENV is not None:
                    try:
                        ratio = float(TARGET_LAYER_RATIO_ENV)
                        if not 0.0 <= ratio <= 1.0:
                            raise ValueError
                        target_idx = min(max(int(round(ratio * (num_layers - 1))), 0), num_layers - 1)
                    except ValueError:
                        logger.warning("Invalid CLAS_LAYER_RATIO=%s; using default index %d", TARGET_LAYER_RATIO_ENV, target_idx)

                target_layer = layers[target_idx]
                device = next(iter(model.parameters())).device

                metadata_file.write(json.dumps(collect_model_metadata(model_key, model_id, model, target_idx), ensure_ascii=False) + "\n")
                metadata_file.flush()

                limit = STOP_AFTER or len(prompt_pairs)
                processed_count = sum(
                    1 for pair in prompt_pairs if (model_key, pair["hinglish_prompt"]) in processed_keys
                )

                if processed_count:
                    logger.info(
                        "Resuming %s with %d/%s prompts already completed",
                        model_key,
                        processed_count,
                        limit,
                    )

                if STOP_AFTER is not None and processed_count >= STOP_AFTER:
                    logger.info("Reached STOP_AFTER=%d for %s", STOP_AFTER, model_key)
                else:
                    remaining_pairs: List[Dict] = []
                    max_needed = STOP_AFTER - processed_count if STOP_AFTER is not None else None
                    for pair in prompt_pairs:
                        key = (model_key, pair["hinglish_prompt"])
                        if key in processed_keys:
                            continue
                        remaining_pairs.append(pair)
                        if max_needed is not None and len(remaining_pairs) >= max_needed:
                            break

                    for batch in chunked(remaining_pairs, BATCH_SIZE):
                        if not batch:
                            continue
                        try:
                            original_prompts: List[str] = []
                            prefixed_prompts: List[str] = []
                            prompt_keys: List[Tuple[str, str]] = []
                            english_rendered: List[str] = []
                            hinglish_rendered: List[str] = []

                            for pair in batch:
                                original_prompt = pair["hinglish_prompt"]
                                prompt_key = (model_key, original_prompt)

                                hinglish_prompt = original_prompt
                                if PROMPT_PREFIX:
                                    if PROMPT_PREFIX_POS == "append":
                                        hinglish_prompt = f"{hinglish_prompt}\n\n{PROMPT_PREFIX}"
                                    else:
                                        hinglish_prompt = f"{PROMPT_PREFIX}\n\n{hinglish_prompt}"

                                original_prompts.append(original_prompt)
                                prefixed_prompts.append(hinglish_prompt)
                                prompt_keys.append(prompt_key)
                                english_rendered.append(render_messages(tokenizer, pair["english_prompt"]))
                                hinglish_rendered.append(render_messages(tokenizer, hinglish_prompt))

                            inputs_eng = to_device(
                                tokenizer(
                                    english_rendered,
                                    return_tensors="pt",
                                    padding=True,
                                ),
                                device,
                            )
                            inputs_hin = to_device(
                                tokenizer(
                                    hinglish_rendered,
                                    return_tensors="pt",
                                    padding=True,
                                ),
                                device,
                            )

                            vec_eng = capture_mean_activation(model, target_layer, inputs_eng)
                            vec_hin = capture_mean_activation(model, target_layer, inputs_hin)
                            delta = (vec_eng - vec_hin).detach()

                            generate_kwargs = {
                                "max_new_tokens": MAX_NEW_TOKENS,
                                "do_sample": True,
                                "temperature": TEMPERATURE,
                            }
                            if hasattr(model.config, "use_cache") and model.config.use_cache is False:
                                generate_kwargs["use_cache"] = False

                            with torch.no_grad():
                                baseline_tokens = model.generate(**inputs_hin, **generate_kwargs)
                            baseline_texts = tokenizer.batch_decode(baseline_tokens, skip_special_tokens=True)

                            steer_handle = target_layer.register_forward_hook(steering_hook(delta, ALPHA))
                            try:
                                with torch.no_grad():
                                    steered_tokens = model.generate(**inputs_hin, **generate_kwargs)
                            finally:
                                steer_handle.remove()
                            steered_texts = tokenizer.batch_decode(steered_tokens, skip_special_tokens=True)

                            for idx, pair in enumerate(batch):
                                baseline_text, baseline_status = enforce_hinglish(
                                    model,
                                    tokenizer,
                                    device,
                                    baseline_texts[idx],
                                )
                                steered_text, steered_status = enforce_hinglish(
                                    model,
                                    tokenizer,
                                    device,
                                    steered_texts[idx],
                                )

                                record = {
                                    "model": model_key,
                                    "prompt": original_prompts[idx],
                                    "prompt_with_prefix": prefixed_prompts[idx],
                                    "domain": pair["domain"],
                                    "topic": pair["topic"],
                                    "gender": pair["gender"],
                                    "hinglish_persona": pair.get("hinglish_persona"),
                                    "english_persona": pair.get("english_persona"),
                                    "hinglish_prompt_id": pair.get("hinglish_id"),
                                    "english_prompt_id": pair.get("english_id"),
                                    "baseline_response": baseline_text,
                                    "steered_response": steered_text,
                                    "baseline_hinglish_status": baseline_status,
                                    "steered_hinglish_status": steered_status,
                                }

                                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                                output_file.flush()

                                processed_keys.add(prompt_keys[idx])
                                processed_count += 1

                                logger.info(
                                    "Completed prompt %d/%s for %s (hinglish_id=%s)",
                                    processed_count,
                                    limit,
                                    model_key,
                                    pair.get("hinglish_id"),
                                )
                        except Exception as err:  # pylint: disable=broad-except
                            logger.exception("Error while processing batch for model=%s", model_key)
                            for pair in batch:
                                error_record = {
                                    "model": model_key,
                                    "prompt": pair.get("hinglish_prompt"),
                                    "error": str(err),
                                }
                                output_file.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                                output_file.flush()

                logger.info("Finished %s with %d prompts", model_key, processed_count)
            except Exception:  # pylint: disable=broad-except
                logger.exception("Aborting %s due to unrecoverable error", model_key)
            finally:
                if model is not None:
                    del model
                if tokenizer is not None:
                    del tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def main():
    run_experiment()


if __name__ == "__main__":
    main()
