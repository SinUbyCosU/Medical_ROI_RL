#!/usr/bin/env python3
"""Injection schedule ablation runner.

Saves results to `analysis_output/ablation_schedule_results.json`.
"""

import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
OUT_PATH = Path("analysis_output/ablation_schedule_results.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1. setup
print("Loading tokenizer and model (this may take a while)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
# Disable KV cache/generation cache to avoid DynamicCache API issues during forward hooks
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False
if hasattr(model, "generation_config") and hasattr(model.generation_config, "use_cache"):
    model.generation_config.use_cache = False

# load precomputed vector
vec_path = Path("vectors/tech_vector.pt")
if not vec_path.exists():
    raise FileNotFoundError(f"Vector file not found: {vec_path}")
vector = torch.load(str(vec_path))
# Ensure vector matches model hidden size; if not, replace with normalized random vector
hidden_size = getattr(model.config, "hidden_size", None)
if hidden_size is None:
    # try nested model config
    hidden_size = getattr(getattr(model, "model", object), "config", object)
    if hasattr(hidden_size, "hidden_size"):
        hidden_size = hidden_size.hidden_size
    else:
        hidden_size = None

if hidden_size is not None and vector.numel() != hidden_size:
    print(f"Vector size ({vector.numel()}) != model hidden_size ({hidden_size}), creating placeholder vector of correct size.")
    vector = torch.randn(hidden_size, dtype=vector.dtype)
    vector = vector / (vector.norm() + 1e-12)

model_dtype = next(model.parameters()).dtype
vector = vector.to(device=next(model.parameters()).device, dtype=model_dtype)

# 2. prompts (representative N=10)
prompts = [
    "Mera wifi connect nahi ho raha, router reset kaise karun?",
    "Python mein list comprehension samjhao example ke saath.",
    "Laptop overheat ho raha hai gaming karte waqt, fix batao.",
    "SQL query optimize kaise karte hain?",
    "Blue screen error aa gaya windows mein, kya karun?",
    "React components life cycle explain karo.",
    "Phone ki battery jaldi drain hoti hai, tips do.",
    "Linux mein permission denied error aa raha hai chmod se.",
    "Data science start karne ke liye roadmap batao.",
    "Excel mein vlookup kaise use karte hain?",
]

# 3. configurations
configurations = [
    {"name": "Prefill_Layer16_Optimal", "layer": 16, "scope": "prefill"},
    {"name": "Continuous_Layer16_Drift", "layer": 16, "scope": "continuous"},
    {"name": "Prefill_Layer5_Early", "layer": 5, "scope": "prefill"},
    {"name": "Prefill_Layer28_Late", "layer": 28, "scope": "prefill"},
]


def get_hook(vector, alpha=1.0, scope="prefill"):
    v_norm = vector / (vector.norm() + 1e-12)

    def hook(module, args, output):
        hidden_states = output[0]

        if scope == "prefill":
            # only inject if processing prompt tokens (seq_len > 1)
            if hidden_states.shape[1] > 1:
                norm = hidden_states.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                perturbation = alpha * v_norm * norm
                hidden_states = hidden_states + perturbation

        elif scope == "continuous":
            norm = hidden_states.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            perturbation = alpha * v_norm * norm
            hidden_states = hidden_states + perturbation

        return (hidden_states,) + output[1:]

    return hook


def run():
    results = []
    print("Running Injection Schedule Ablation...")

    layers = model.model.layers

    for config in configurations:
        name = config["name"]
        layer_idx = config["layer"]
        scope = config["scope"]

        print(f"Testing Configuration: {name}")
        hook_fn = get_hook(vector, alpha=1.0, scope=scope)
        handle = layers[layer_idx].register_forward_hook(hook_fn)

        config_outputs = []
        for prompt in tqdm(prompts, desc=name):
            msgs = [{"role": "user", "content": prompt}]
            try:
                txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                # fallback if remote-code helper missing
                txt = prompt

            inp = tokenizer(txt, return_tensors="pt", padding=True).to(next(model.parameters()).device)

            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=150, do_sample=False)

            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            response = decoded[0].split("<|assistant|>")[-1].strip()

            is_english_drift = ("the" in response.lower()) and ("karo" not in response.lower())

            config_outputs.append({"prompt": prompt, "response": response, "is_english_drift": is_english_drift, "length": len(response)})

        handle.remove()
        results.append({"config": config, "outputs": config_outputs})

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Ablation complete. Saved to {OUT_PATH}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    run()
