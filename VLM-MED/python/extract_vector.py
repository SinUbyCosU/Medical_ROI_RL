#!/usr/bin/env python3
"""Extract a COMI-LINGUA steering vector from matched English/Hinglish prompt pairs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT_SOURCE = Path("Bias/PromptPersona_Full_600.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a COMI-LINGUA steering vector")
    parser.add_argument(
        "--prompt-csv",
        type=Path,
        default=DEFAULT_PROMPT_SOURCE,
        help="Prompt catalog containing paired English/Hinglish rows",
    )
    parser.add_argument("--output", type=Path, default=Path("vectors/comilingua_vector.pt"))
    parser.add_argument("--model-id", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--layer-id", type=int, default=16, help="Transformer layer index to sample")
    parser.add_argument("--num-pairs", type=int, default=20, help="Number of English/Hinglish prompt pairs to use")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for pair sampling")
    return parser.parse_args()


def load_prompt_pairs(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {path}")

    english_lookup = {}
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

    if not hinglish_rows:
        raise ValueError("No Hinglish prompts discovered in prompt catalog")

    pairs: List[Tuple[str, str]] = []
    for key, hinglish_prompt in hinglish_rows:
        english_prompt = english_lookup.get(key)
        if english_prompt:
            pairs.append((english_prompt, hinglish_prompt))

    if not pairs:
        raise ValueError("Failed to align English/Hinglish prompts")

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


def capture_hidden_state(model, tokenizer, prompt: str, layer_idx: int) -> torch.Tensor:
    formatted = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states: Iterable[torch.Tensor] = outputs.hidden_states  # type: ignore[attr-defined]
    layer_states = hidden_states[layer_idx]
    return layer_states[0, -1, :].detach()


def main() -> None:
    args = parse_args()

    all_pairs = load_prompt_pairs(args.prompt_csv)
    sampled_pairs = pick_pairs(all_pairs, args.num_pairs, args.seed)
    if len(sampled_pairs) < args.num_pairs:
        print(f"Warning: only {len(sampled_pairs)} pairs available (requested {args.num_pairs})")

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    diffs: List[torch.Tensor] = []
    for idx, (english_prompt, hinglish_prompt) in enumerate(sampled_pairs, start=1):
        eng_state = capture_hidden_state(model, tokenizer, english_prompt, args.layer_id)
        hin_state = capture_hidden_state(model, tokenizer, hinglish_prompt, args.layer_id)
        diffs.append(eng_state - hin_state)
        if idx % 5 == 0 or idx == len(sampled_pairs):
            print(f"Processed {idx}/{len(sampled_pairs)} pairs")

    if not diffs:
        raise RuntimeError("No hidden state differences collected")

    vector = torch.stack(diffs).mean(dim=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, args.output)
    print(f"Saved steering vector to {args.output}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
