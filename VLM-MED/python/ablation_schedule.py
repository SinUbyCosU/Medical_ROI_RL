import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
LAYER_SETUPS = [
    {"name": "Layer 16 (Prefill - Optimal)", "layer": 16, "scope": "prefill"},
    {"name": "Layer 16 (Continuous - Drift)", "layer": 16, "scope": "continuous"},
    {"name": "Layer 5 (Too Early)", "layer": 5, "scope": "prefill"},
    {"name": "Layer 28 (Too Late)", "layer": 28, "scope": "prefill"},
]
PROMPT = "Explain strictly in Hindi how to secure a router."

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
vector = torch.load("vectors/tech_vector.pt").to(model.device)

results = []

for cfg in LAYER_SETUPS:

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        inject = cfg["scope"] == "continuous" or hidden.shape[1] > 1
        if inject:
            hidden_norm = hidden.norm(dim=-1, keepdim=True)
            v_unit = vector / vector.norm()
            hidden = hidden + v_unit * hidden_norm

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = model.model.layers[cfg["layer"]].register_forward_hook(hook)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=100, do_sample=False)

    decoded = tokenizer.decode(output[0]).split("<|assistant|>")[-1].strip()
    english_drift = any(word in decoded.lower().split() for word in ["the", "and", "is", "to"])

    results.append(
        {
            "config": cfg["name"],
            "response_preview": decoded[:200] + ("..." if len(decoded) > 200 else ""),
            "drift_detected": bool(english_drift),
        }
    )

    handle.remove()

print(json.dumps(results, indent=2, ensure_ascii=False))
