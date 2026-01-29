#!/usr/bin/env python3
"""Comprehensive mitigation experiment runner.

This script compares two mitigation strategies against baseline Hinglish responses
for a list of open-weight language models using Cross-Lingual Activation Steering
(CLASS) and a reasoning-first prompting technique.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("mitigation")


# --------------------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, str] = {
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "phi35_mini": "microsoft/Phi-3.5-mini-instruct",
    "zephyr_7b": "HuggingFaceH4/zephyr-7b-beta",
}

DEFAULT_RESULTS_CSV = Path("audited_results.csv")
DEFAULT_PERSONA_CSV = Path("Bias/PromptPersona_Full_600.csv")
DEFAULT_OUTPUT_PATH = Path("comprehensive_mitigation_results.jsonl")
DEFAULT_SAMPLE_SIZE = 50
ALPHA = 1.5
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.6


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
@dataclass
class PromptRecord:
    prompt_id: str
    model_key: str
    hinglish_prompt: str
    english_prompt: str


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comprehensive mitigation experiment.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="CSV containing per-model scores including instructional_density",
    )
    parser.add_argument(
        "--persona-csv",
        type=Path,
        default=DEFAULT_PERSONA_CSV,
        help="Persona CSV with Hinglish / English prompt pairs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSONL for experiment results",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of prompts per model to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_REGISTRY.keys()),
        help="Optional subset of model keys to run",
    )
    args = parser.parse_args()
    return args


def normalise_column(df: pd.DataFrame, *candidates: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Required column missing. Expected one of: {candidates}")


def load_persona_prompts(persona_csv: Path) -> Tuple[pd.DataFrame, Dict[str, str], Dict[Tuple[str, str, str], str]]:
    if not persona_csv.exists():
        raise FileNotFoundError(f"Persona CSV not found: {persona_csv}")

    df = pd.read_csv(persona_csv)
    required_cols = {"id", "language", "prompt_text", "domain", "topic", "gender"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Persona CSV missing required columns: {missing}")

    df["language_norm"] = df["language"].str.strip().str.lower()

    hinglish_lookup = {}
    english_lookup = {}

    for _, row in df.iterrows():
        lang = row["language_norm"]
        if lang == "hinglish":
            hinglish_lookup[str(row["id"])] = row["prompt_text"]
        elif lang == "english":
            key = (row["domain"], row["topic"], row["gender"])
            english_lookup[key] = row["prompt_text"]

    return df, hinglish_lookup, english_lookup


def select_worst_prompts(
    results_csv: Path,
    persona_df: pd.DataFrame,
    hinglish_lookup: Dict[str, str],
    english_lookup: Dict[Tuple[str, str, str], str],
    sample_size: int,
    model_keys: List[str],
) -> Dict[str, List[PromptRecord]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)

    prompt_col = normalise_column(df, "prompt_id", "id", "promptId")
    model_col = normalise_column(df, "model_name", "model", "model_id", "model_key")
    lang_col = normalise_column(df, "language")
    density_col = normalise_column(df, "instructional_density", "density", "instruction_density")

    supplementary_cols = {}
    for col in ("domain", "topic", "gender"):
        if col in df.columns:
            supplementary_cols[col] = col

    df[lang_col] = df[lang_col].str.strip().str.lower()

    filtered = df[(df[lang_col] == "hinglish") & (df[density_col] < 5)]
    if filtered.empty:
        raise ValueError("No Hinglish prompts with instructional_density < 5 were found.")

    grouped_records: Dict[str, List[PromptRecord]] = {key: [] for key in model_keys}

    for model_key in model_keys:
        df_model = filtered[filtered[model_col] == model_key]
        if df_model.empty:
            logger.warning("No entries for model %s after filtering; skipping.", model_key)
            continue

        df_sorted = df_model.sort_values(by=density_col, ascending=True)
        df_sample = df_sorted.head(sample_size)

        for _, row in df_sample.iterrows():
            prompt_id = str(row[prompt_col])
            hinglish_prompt = hinglish_lookup.get(prompt_id)
            if hinglish_prompt is None and "prompt_text" in df.columns:
                hinglish_prompt = row["prompt_text"]

            if not hinglish_prompt:
                logger.warning("Skipping prompt %s for model %s: Hinglish text not found", prompt_id, model_key)
                continue

            english_prompt = None

            if {"domain", "topic", "gender"}.issubset(supplementary_cols):
                key = (
                    row[supplementary_cols["domain"]],
                    row[supplementary_cols["topic"]],
                    row[supplementary_cols["gender"]],
                )
                english_prompt = english_lookup.get(key)

            if english_prompt is None:
                # fall back: try persona df matching by domain/topic/gender for the prompt id
                persona_row = persona_df[persona_df["id"].astype(str) == prompt_id]
                if not persona_row.empty:
                    row0 = persona_row.iloc[0]
                    key = (row0["domain"], row0["topic"], row0["gender"])
                    english_prompt = english_lookup.get(key)

            if english_prompt is None:
                logger.warning(
                    "English counterpart missing for prompt %s (model %s); using Hinglish prompt as fallback",
                    prompt_id,
                    model_key,
                )
                english_prompt = hinglish_prompt

            grouped_records[model_key].append(
                PromptRecord(
                    prompt_id=prompt_id,
                    model_key=model_key,
                    hinglish_prompt=hinglish_prompt,
                    english_prompt=english_prompt,
                )
            )

        logger.info("Selected %d prompts for model %s", len(grouped_records[model_key]), model_key)

    return grouped_records


def ensure_token_settings(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def extract_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def repack_output(original_output, new_hidden):
    if isinstance(original_output, tuple):
        return (new_hidden,) + original_output[1:]
    if hasattr(original_output, "_replace"):
        return original_output._replace(last_hidden_state=new_hidden)
    return new_hidden


def capture_hook(store: Dict[str, torch.Tensor], name: str):
    def hook(module, inputs, output):  # noqa: ARG001
        store[name] = extract_hidden(output).detach()

    return hook


def steering_hook(delta: torch.Tensor, alpha: float):
    def hook(module, inputs, output):  # noqa: ARG001
        current = extract_hidden(output)
        steer_vec = delta
        if steer_vec.shape[1] == 1 and current.shape[1] > 1:
            steer_vec = delta.expand(current.shape[0], current.shape[1], delta.shape[-1])
        steered = current + alpha * steer_vec
        return repack_output(output, steered)

    return hook


def find_decoder_layers(model) -> Tuple[Iterable[torch.nn.Module], int]:
    candidates = []

    if hasattr(model, "model"):
        inner = model.model
        for attr in ("layers", "blocks", "h"):
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
        raise ValueError("Unable to locate decoder layers for activation steering.")

    layers = max(candidates, key=len)
    midpoint = len(layers) // 2
    return layers, midpoint


def apply_chat_template(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise technical assistant."},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"User: {user_prompt}\nAssistant:"


def compute_density_score(text: str) -> int:
    if not text:
        return 0

    bullet_matches = re.findall(r"(?:^|\n)\s*[-*]\s", text)
    numbered_matches = re.findall(r"(?:^|\n)\s*\d+[.)]\s", text)
    code_blocks = text.count("```") // 2

    return int(len(bullet_matches) + len(numbered_matches) + code_blocks)


def generate_text(model, tokenizer, prompt: str, device: torch.device, generate_kwargs: Dict) -> str:
    formatted = apply_chat_template(tokenizer, prompt)
    inputs = to_device(tokenizer(formatted, return_tensors="pt"), device)
    with torch.no_grad():
        tokens = model.generate(**inputs, **generate_kwargs)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def run_experiment(
    prompts_by_model: Dict[str, List[PromptRecord]],
    output_path: Path,
    model_keys: List[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for model_key in model_keys:
            if model_key not in MODEL_REGISTRY:
                logger.warning("Unknown model key %s, skipping", model_key)
                continue

            prompt_list = prompts_by_model.get(model_key, [])
            if not prompt_list:
                logger.warning("No prompts selected for model %s, skipping", model_key)
                continue

            model_id = MODEL_REGISTRY[model_key]
            logger.info("Loading model %s (%s)", model_key, model_id)

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=quant_config,
                    trust_remote_code=True,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to load %s (%s): %s", model_key, model_id, exc)
                continue

            try:
                ensure_token_settings(tokenizer, model)
                layers, midpoint = find_decoder_layers(model)
                target_layer = layers[midpoint]
                device = next(model.parameters()).device

                model_lower_id = model_id.lower()
                if "phi-3.5" in model_lower_id or "phi3.5" in model_lower_id:
                    logger.info("Disabling KV cache for %s to avoid DynamicCache issues", model_key)
                    model.config.use_cache = False
                    if hasattr(model, "generation_config"):
                        model.generation_config.use_cache = False

                generate_kwargs = {
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "do_sample": True,
                    "temperature": TEMPERATURE,
                }
                if hasattr(model.config, "use_cache") and model.config.use_cache is False:
                    generate_kwargs["use_cache"] = False

                for record in prompt_list:
                    activations: Dict[str, torch.Tensor] = {}

                    # Step 1: baseline Hinglish generation
                    baseline_text = generate_text(model, tokenizer, record.hinglish_prompt, device, generate_kwargs)

                    # Step 2: CLAS generation
                    # 2a. capture English activations
                    formatted_eng = apply_chat_template(tokenizer, record.english_prompt)
                    inputs_eng = to_device(tokenizer(formatted_eng, return_tensors="pt"), device)
                    handle = target_layer.register_forward_hook(capture_hook(activations, "eng"))
                    with torch.no_grad():
                        model(**inputs_eng)
                    handle.remove()

                    if "eng" not in activations:
                        logger.warning("Failed to capture English activations for prompt %s", record.prompt_id)
                        continue

                    vec_eng = activations["eng"].mean(dim=1, keepdim=True)

                    # 2b. capture Hinglish activations
                    formatted_hin = apply_chat_template(tokenizer, record.hinglish_prompt)
                    inputs_hin = to_device(tokenizer(formatted_hin, return_tensors="pt"), device)
                    handle = target_layer.register_forward_hook(capture_hook(activations, "hin"))
                    with torch.no_grad():
                        model(**inputs_hin)
                    handle.remove()

                    if "hin" not in activations:
                        logger.warning("Failed to capture Hinglish activations for prompt %s", record.prompt_id)
                        continue

                    vec_hin = activations["hin"].mean(dim=1, keepdim=True)
                    steering_vec = (vec_eng - vec_hin).detach()

                    steer_handle = target_layer.register_forward_hook(steering_hook(steering_vec, ALPHA))
                    steered_tokens = model.generate(**inputs_hin, **generate_kwargs)
                    steer_handle.remove()
                    clas_text = tokenizer.decode(steered_tokens[0], skip_special_tokens=True)

                    # Step 3: reasoning-first prompting
                    reasoning_prompt = (
                        "User Query: {query}\n\nInstructions: 1. Think silently in English about the technical "
                        "solution. 2. Translate that solution into detailed Hinglish. Output format: "
                        "[THOUGHT]: ... [RESPONSE]: ..."
                    ).format(query=record.hinglish_prompt)
                    reasoning_text = generate_text(model, tokenizer, reasoning_prompt, device, generate_kwargs)

                    baseline_density = compute_density_score(baseline_text)
                    clas_density = compute_density_score(clas_text)
                    reasoning_density = compute_density_score(reasoning_text)

                    output_file.write(
                        json.dumps(
                            {
                                "model": model_key,
                                "model_id": model_id,
                                "prompt_id": record.prompt_id,
                                "original_prompt": record.hinglish_prompt,
                                "baseline_density": baseline_density,
                                "clas_density": clas_density,
                                "reasoning_density": reasoning_density,
                                "baseline_text": baseline_text,
                                "clas_text": clas_text,
                                "reasoning_text": reasoning_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    output_file.flush()

            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args()

    persona_df, hinglish_lookup, english_lookup = load_persona_prompts(args.persona_csv)
    prompts_by_model = select_worst_prompts(
        args.results_csv,
        persona_df,
        hinglish_lookup,
        english_lookup,
        args.sample_size,
        args.models,
    )

    run_experiment(prompts_by_model, args.output, args.models)


if __name__ == "__main__":
    main()
