import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import pandas as pd
# ...existing code...
# Load calibration pairs directly from eng_hin_pairs.csv (persona_code, prompt_text_eng, prompt_text_hin)
def load_calibration_pairs(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Calibration pairs CSV is empty.")
        raise RuntimeError("No calibration pairs found! Check your CSV.")
    return df[['persona_code','prompt_text_eng','prompt_text_hin']].values.tolist()

# ...existing code...

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, default=None, help='CSV file with calibration pairs')
    args = parser.parse_args()

    if args.pairs:
        calibration_pairs = load_calibration_pairs(args.pairs)
    else:
        raise ValueError('Please provide --pairs with a CSV of calibration pairs.')
    # ...existing code...
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ================= STEP 1: COMPUTE CAA VECTOR =================
print("Computing Global CAA Vector...")
diff_vectors = []

def get_activations(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Mean across all prompt tokens at TARGET_LAYER
    # outputs.hidden_states: tuple of (layer_0, ..., layer_N), each (B, S, H)
    h = outputs.hidden_states[TARGET_LAYER][0]  # (seq, hidden)
    return h.mean(dim=0)  # (hidden,)

for pos_prompt, neg_prompt in tqdm(calibration_pairs, desc="Calib pairs"):
    pos_act = get_activations(f"<|user|>\n{pos_prompt}<|end|>\n<|assistant|>\n")
    neg_act = get_activations(f"<|user|>\n{neg_prompt}<|end|>\n<|assistant|>\n")
    diff_vectors.append(pos_act - neg_act)

caa_vector = torch.stack(diff_vectors).mean(dim=0)
print(f"Vector computed. Norm: {caa_vector.norm().item():.4f}")

# ================= STEP 2: DEFINE HOOK =================
def steering_hook(module, input, output):
    if isinstance(output, tuple):
        hidden_state = output[0]
        hidden_state += COEFF * caa_vector
        return (hidden_state,) + output[1:]
    else:
        return output + COEFF * caa_vector

# ================= STEP 3: RUN INFERENCE =================
print(f"\nRunning Inference with CAA (Layer {TARGET_LAYER}, Alpha {COEFF})...")
layer_module = model.model.layers[TARGET_LAYER]
handle = layer_module.register_forward_hook(steering_hook)

test_prompts = calib_df[calib_df['language'] == 'Hinglish']['prompt_text'].tolist()[:5]  # Use 5 for demo

try:
    for prompt in tqdm(test_prompts, desc="Test prompts"):
        print(f"\nInput: {prompt}")
        inputs = tokenizer(f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n", return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response:\n{response.split('<|assistant|>')[-1]}")
finally:
    handle.remove()
    print("\nHook removed.")
