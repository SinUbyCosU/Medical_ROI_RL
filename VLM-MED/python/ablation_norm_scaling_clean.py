import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# SETUP
model_id = "microsoft/Phi-3.5-mini-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load your vector
vector = torch.load("vectors/tech_vector.pt").to(device)  # Ensure you have this file

prompts = [
    "Mera wifi connect nahi ho raha, router reset kaise karun?",
    "Laptop overheat ho raha hai heavy gaming ke waqt.",
    "SQL query slow chal raha hai, optimization tips do."
]

def run_generate(prompt, hook_fn):
    msgs = [{"role": "user", "content": prompt}]
    inp = tokenizer.apply_chat_template(msgs, tokenize=True, return_tensors="pt", add_generation_prompt=True).to(device)
    handle = model.model.layers[16].register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=100, do_sample=False)
    handle.remove()
    return tokenizer.decode(out[0], skip_special_tokens=True)

# HOOK 1: YOUR METHOD (Normalized / Rotation)
def hook_normalized(module, args, output):
    h = output[0]
    if h.shape[1] > 1:  # Prefill only
        # Scale vector to match current token's norm (Rotation)
        current_norm = h.norm(dim=-1, keepdim=True)
        v_unit = vector / vector.norm()
        h += 1.0 * v_unit * current_norm
    return (h,) + output[1:]

# HOOK 2: NAIVE METHOD (Unnormalized / Addition)
def hook_unnormalized(module, args, output):
    h = output[0]
    if h.shape[1] > 1:
        # Just add the raw vector (Magnitude Injection)
        # This usually "blows up" the logits
        h += 1.0 * vector
    return (h,) + output[1:]

results = []
for p in prompts:
    res_norm = run_generate(p, hook_normalized)
    res_raw = run_generate(p, hook_unnormalized)
    results.append({
        "prompt": p,
        "normalized_response": res_norm.split("assistant")[-1].strip(),
        "unnormalized_response": res_raw.split("assistant")[-1].strip()
    })

# Print comparison
for r in results:
    print(f"\n--- Prompt: {r['prompt']} ---")
    print(f"[Normalized (CLAS)]: {r['normalized_response'][:100]}...")
    print(f"[Unnormalized (Raw)]: {r['unnormalized_response'][:100]}...")

# Save
with open("ablation_norm_results.json", "w") as f:
    json.dump(results, f, indent=2)
