#!/usr/bin/env python3
"""Quickly compare baseline vs steered outputs using the Spanglish vector."""

from __future__ import annotations

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
VECTOR_PATH = "vectors/spanglish_vector.pt"
LAYER_ID = 16
ALPHAS = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5]
OUTPUT_PATH = "results_spanglish_alpha_sweep.jsonl"

PROMPTS = [
    "El server esta down, como lo restarteo?",
    "Necesito un script de python para scrapear data.",
    "Mi laptop esta over heating mucho.",
    "Como cancelo mi subscription de Netflix?",
    "Explicame como funciona el blockchain.",
]

INSTRUCTION = "Piensa en inglés y responde en español."


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{prompt}\n<|assistant|>"


def prompt_bound_hook(vector: torch.Tensor, alpha: float):
    unit = vector / vector.norm()

    def _hook(module, inputs, output):  # noqa: ARG001
        if isinstance(output, tuple):
            hidden_states = output[0]
            remainder = output[1:]
        else:
            hidden_states = output
            remainder = None
        if hidden_states.dim() == 3 and hidden_states.size(1) > 1:
            current_norm = hidden_states.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            injection = unit.view(1, 1, -1)
            hidden_states = hidden_states + alpha * current_norm * injection
        if remainder is None:
            return hidden_states
        return (hidden_states,) + remainder

    return _hook


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    vector = torch.load(VECTOR_PATH, map_location=model.device).to(model.dtype)

    output_path = Path(OUTPUT_PATH)

    with output_path.open("w", encoding="utf-8") as fout:
        print("--- SPANGLISH EVALUATION (Layer 16) ---")
        for prompt in PROMPTS:
            augmented_prompt = f"{INSTRUCTION}\n\n{prompt}"
            formatted = format_prompt(tokenizer, augmented_prompt)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            with torch.no_grad():
                baseline_ids = model.generate(**inputs, max_new_tokens=120, do_sample=False)
            baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)

            print(f"\nPrompt: {prompt}")
            print(f"Instruction: {INSTRUCTION}")
            print(f"[Baseline]: {baseline_text[-160:].strip()}")

            for alpha in ALPHAS:
                layer_module = model.model.layers[LAYER_ID]
                handle = layer_module.register_forward_hook(prompt_bound_hook(vector, alpha))
                try:
                    with torch.no_grad():
                        steered_ids = model.generate(**inputs, max_new_tokens=120, do_sample=False)
                finally:
                    handle.remove()
                steered_text = tokenizer.decode(steered_ids[0], skip_special_tokens=True)
                print(f"[Steered α={alpha:.2f}]: {steered_text[-160:].strip()}")

                record = {
                    "model": MODEL_ID,
                    "model_id": MODEL_ID,
                    "prompt": prompt,
                    "prompt_with_prefix": augmented_prompt,
                    "variant_label": f"alpha_{alpha:.2f}",
                    "layer_index": LAYER_ID,
                    "alpha": alpha,
                    "instruction": INSTRUCTION,
                    "baseline_response": baseline_text,
                    "steered_response": steered_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
