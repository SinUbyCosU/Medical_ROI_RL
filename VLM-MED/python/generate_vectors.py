#!/usr/bin/env python3
"""Generate steering vector (mu_eng - mu_hin) and save to vectors/tech_vector.pt

Usage: python generate_vectors.py
"""

import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
LAYER_IDX = 16
SAVE_DIR = Path("vectors")
SAVE_PATH = SAVE_DIR / "tech_vector.pt"


def ensure_tokens(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def get_hidden_states(prompts, model, tokenizer, layer_idx):
    accumulated = []
    model.config.use_cache = False
    model.eval()

    for text in tqdm(prompts, desc=f"Extract L{layer_idx}"):
        msgs = [{"role": "user", "content": text}]
        try:
            inp_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            inp_text = text

        inputs = tokenizer(inp_text, return_tensors="pt", padding=True).to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]  # [B, S, D]
        last = hidden[:, -1, :].detach()  # [B, D]
        accumulated.append(last)

    all_states = torch.cat(accumulated, dim=0)
    mean = all_states.mean(dim=0, keepdim=True)
    return mean


def main():
    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    ensure_tokens(tokenizer, model)
    # disable cache globally to avoid DynamicCache/get_usable_length issues
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # calibration prompts (5 shown; replace with full calibration set if available)
    calibration_data = [
        {"eng": "My wifi is not connecting, how do I reset the router?", "hin": "Mera wifi connect nahi ho raha, router reset kaise karun?"},
        {"eng": "How do I flush the DNS cache in Windows?", "hin": "Windows mein DNS flush kaise karte hain?"},
        {"eng": "My laptop is overheating when I play games.", "hin": "Laptop overheat ho raha hai gaming karte waqt."},
        {"eng": "Explain how to use a list comprehension in Python.", "hin": "Python mein list comprehension samjhao example ke saath."},
        {"eng": "My phone battery drains very quickly, give me tips.", "hin": "Phone ki battery jaldi drain hoti hai, tips do."},
    ]

    eng_prompts = [d["eng"] for d in calibration_data]
    hin_prompts = [d["hin"] for d in calibration_data]

    mu_eng = get_hidden_states(eng_prompts, model, tokenizer, LAYER_IDX)
    mu_hin = get_hidden_states(hin_prompts, model, tokenizer, LAYER_IDX)

    steering = mu_eng - mu_hin

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(steering.cpu(), str(SAVE_PATH))

    print(f"Saved vector to {SAVE_PATH}")
    print(f"Vector shape: {tuple(steering.shape)}; norm: {steering.norm().item():.4f}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
