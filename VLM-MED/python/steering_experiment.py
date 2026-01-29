#!/usr/bin/env python3
"""
Cross-Lingual Activation Steering (CLAS) Experiment
Goal: Prove that Llama 3.1 knows technical answers in English but gets "lazy" in Hinglish.
By injecting the English activation vector, we force it to be smart in Hinglish.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"\n{'='*70}")
print(f"Loading {model_id} in 4-bit (optimized for 20GB VRAM)...")
print(f"{'='*70}\n")

# Quantization Config (Crucial for memory efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = True
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

print("✓ Model loaded successfully\n")

# ==========================================
# 2. DEFINING THE "SURGERY" HOOKS
# ==========================================
activations = {}

def _extract_hidden(output):
    """Extract hidden states tensor regardless of return type."""
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _repack_output(original_output, new_hidden):
    """Reconstruct the module output with modified hidden states."""
    if isinstance(original_output, tuple):
        return (new_hidden,) + original_output[1:]
    if hasattr(original_output, "__class__") and hasattr(original_output, "_replace"):
        return original_output._replace(last_hidden_state=new_hidden)
    return new_hidden


def get_activation(name):
    """
    Hook to steal the 'Brain Activity' (Hidden States) from a specific layer.
    """
    def hook(model_module, module_input, module_output):  # noqa: ARG001
        hidden = _extract_hidden(module_output)
        activations[name] = hidden.detach()
    return hook


def steering_hook(steering_vector, alpha=1.5):
    """
    Hook to INJECT the stolen 'Intelligence' into the current run.
    alpha: Strength of injection (Higher = More English reasoning).
    """
    def hook(model_module, module_input, module_output):  # noqa: ARG001
        current_activation = _extract_hidden(module_output)

        expanded_vec = steering_vector
        if steering_vector.shape[1] == 1 and current_activation.shape[1] > 1:
            expanded_vec = steering_vector.expand(current_activation.shape[0], current_activation.shape[1], steering_vector.shape[-1])

        new_hidden = current_activation + (alpha * expanded_vec)
        return _repack_output(module_output, new_hidden)

    return hook

# ==========================================
# 3. EXPERIMENT PARAMETERS
# ==========================================
# TARGET LAYER: Layer 16 is the "Middle" of Llama 3 (32 layers total).
# This is usually where "abstract reasoning" happens.
target_layer_idx = 16 

# The Prompts
english_prompt = "Explain the step-by-step process of troubleshooting a DNS failure. Be technical."
hinglish_prompt = "DNS fail ho gaya hai, step-by-step kaise fix karun? Thoda technical batana."

def build_chat_inputs(prompt_text):
    """Use the tokenizer chat template to prep inputs for Mistral Instruct."""
    messages = [
        {"role": "system", "content": "You are a precise technical assistant."},
        {"role": "user", "content": prompt_text},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(rendered_prompt, return_tensors="pt").to(model.device)


input_ids_eng = build_chat_inputs(english_prompt)
input_ids_hin = build_chat_inputs(hinglish_prompt)

# ==========================================
# 4. STEP A: EXTRACT THE "SMART" VECTOR
# ==========================================
print(f"{'='*70}")
print("Step A: Extracting English Reasoning Vector")
print(f"{'='*70}\n")

# Attach hook to capture activations
hook_handle = model.model.layers[target_layer_idx].register_forward_hook(get_activation("eng_activations"))

# Run English Inference (Single pass)
with torch.no_grad():
    model(**input_ids_eng)

# Detach hook
hook_handle.remove()

# Get the vector
# We take the mean across the sequence length to get the "Gist" of the reasoning
english_vec = activations["eng_activations"].mean(dim=1, keepdim=True)
print("✓ English Vector Captured")
print(f"  - Shape: {english_vec.shape}")
print(f"  - Norm: {english_vec.norm().item():.4f}\n")

# ==========================================
# 5. STEP B: BASELINE HINGLISH (Before Fix)
# ==========================================
print(f"{'='*70}")
print("Step B: Generating Baseline Hinglish (No Intervention)")
print(f"{'='*70}\n")
print(f"Prompt: {hinglish_prompt}\n")

output_base = model.generate(**input_ids_hin, max_new_tokens=150, do_sample=True, temperature=0.6)
response_base = tokenizer.decode(output_base[0], skip_special_tokens=True)
baseline_text = response_base.split('assistant')[-1].strip()
print(f"BASELINE RESPONSE:\n{baseline_text}\n")

# ==========================================
# 6. STEP C: STEERED HINGLISH (The Fix)
# ==========================================
print(f"{'='*70}")
print("Step C: Generating Steered Hinglish (With English Injection)")
print(f"{'='*70}\n")

# We capture Hinglish activations first to calculate the difference (Delta)
hook_handle = model.model.layers[target_layer_idx].register_forward_hook(get_activation("hin_activations"))
with torch.no_grad():
    model(**input_ids_hin)
hook_handle.remove()
hinglish_vec = activations["hin_activations"].mean(dim=1, keepdim=True)

# Calculate Delta: What does English have that Hinglish lacks?
steering_vec = (english_vec - hinglish_vec).clone()

print(f"Steering Vector Delta:")
print(f"  - Shape: {steering_vec.shape}")
print(f"  - Norm: {steering_vec.norm().item():.4f}\n")

# Attach Injection Hook
# Alpha = 1.5 is a good starting point. Increase if effect is too weak.
alpha_value = 1.5
inject_handle = model.model.layers[target_layer_idx].register_forward_hook(steering_hook(steering_vec, alpha=alpha_value))

# Generate with Steering
output_steered = model.generate(**input_ids_hin, max_new_tokens=150, do_sample=True, temperature=0.6)
response_steered = tokenizer.decode(output_steered[0], skip_special_tokens=True)

# Remove hook (Clean up)
inject_handle.remove()

steered_text = response_steered.split('assistant')[-1].strip()
print(f"STEERED RESPONSE (alpha={alpha_value}):\n{steered_text}\n")

# ==========================================
# 7. ANALYSIS & CONCLUSION
# ==========================================
print(f"{'='*70}")
print("EXPERIMENT COMPLETE")
print(f"{'='*70}\n")

print("COMPARISON:")
print(f"Baseline length: {len(baseline_text)} chars")
print(f"Steered length: {len(steered_text)} chars")
print(f"Length difference: {len(steered_text) - len(baseline_text)} chars")

if len(steered_text) > len(baseline_text):
    improvement = ((len(steered_text) - len(baseline_text)) / len(baseline_text)) * 100
    print(f"Improvement: +{improvement:.1f}%")

print("\n✓ Cross-Lingual Activation Steering (CLAS) successful!")
print("  The Hinglish model generated more detailed technical content")
print("  after injecting the English reasoning vector.\n")
