#!/usr/bin/env python3
"""Measure layer-wise cosine similarity between English and Hinglish prompts."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_SOURCE_DEFAULT = Path("Bias/PromptPersona_Full_600.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure layer-wise cosine similarity across English/Hinglish pairs")
    parser.add_argument("--prompt-csv", type=Path, default=PROMPT_SOURCE_DEFAULT, help="CSV containing English/Hinglish prompt rows")
    parser.add_argument("--model-id", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--num-pairs", type=int, default=20, help="Number of aligned prompt pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pair sampling")
    parser.add_argument("--output", type=Path, default=Path("layer_divergence.csv"), help="Where to write the CSV results")
    return parser.parse_args()


def load_prompt_pairs(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {path}")

    english_lookup: Dict[Tuple[str, str, str], str] = {}
    hinglish_rows: List[Tuple[Tuple[str, str, str], str]] = []

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            language = (row.get("language") or "").strip().lower()
            domain = (row.get("domain") or "").strip()
            topic = (row.get("topic") or "").strip()
            gender = (row.get("gender") or "").strip()
            prompt = (row.get("prompt_text") or "").strip()
            if not prompt:
                continue
            key = (domain, topic, gender)
            if language == "english":
                english_lookup[key] = prompt
            elif language == "hinglish":
                hinglish_rows.append((key, prompt))

    pairs: List[Tuple[str, str]] = []
    for key, hinglish_prompt in hinglish_rows:
        english_prompt = english_lookup.get(key)
        if english_prompt:
            pairs.append((english_prompt, hinglish_prompt))

    if not pairs:
        raise ValueError("Failed to align English/Hinglish prompt pairs. Check the CSV contents.")

    return pairs


def pick_pairs(pairs: Sequence[Tuple[str, str]], k: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    pool = list(pairs)
    rng.shuffle(pool)
    if k <= 0 or k >= len(pool):
        return pool[:k or len(pool)]
    return pool[:k]


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>"


def capture_hidden_states(model, tokenizer, prompt: str) -> Iterable[torch.Tensor]:
    formatted = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states  # type: ignore[attr-defined]


def cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(vec_a.float(), vec_b.float(), dim=0).item()


def main() -> None:
    args = parse_args()

    all_pairs = load_prompt_pairs(args.prompt_csv)
    sampled_pairs = pick_pairs(all_pairs, args.num_pairs, args.seed)
    if len(sampled_pairs) < args.num_pairs:
        print(f"Warning: only {len(sampled_pairs)} pairs available (requested {args.num_pairs})")

    print(f"Measuring Layer-wise Divergence on {len(sampled_pairs)} pairs...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise AttributeError("Model config missing num_hidden_layers; cannot proceed")

    layer_sims = {layer_idx: [] for layer_idx in range(num_layers + 1)}  # +1 for embedding layer

    for english_prompt, hinglish_prompt in tqdm(sampled_pairs, desc="Comparing pairs"):
        eng_states = capture_hidden_states(model, tokenizer, english_prompt)
        hin_states = capture_hidden_states(model, tokenizer, hinglish_prompt)
        for layer_idx, (eng_hidden, hin_hidden) in enumerate(zip(eng_states, hin_states)):
            vec_eng = eng_hidden[0, -1, :]
            vec_hin = hin_hidden[0, -1, :]
            layer_sims[layer_idx].append(cosine_similarity(vec_eng, vec_hin))

    rows: List[Dict[str, float]] = []
    for layer_idx in sorted(layer_sims.keys()):
        sims = layer_sims[layer_idx]
        avg_sim = float(np.mean(sims)) if sims else float("nan")
        rows.append({"layer": layer_idx, "cosine_similarity": avg_sim})
        print(f"Layer {layer_idx:02d}: Sim = {avg_sim:.4f}")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
