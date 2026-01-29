import pandas as pd
import json
from tqdm import tqdm
from clas_model import CLAS_Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=1.0, help='Steering alpha value (default: 1.0)')
parser.add_argument('--output', type=str, default='layer_sweep_results_persona.csv', help='Output CSV file')
parser.add_argument('--num-prompts', type=int, default=50, help='Number of prompts to use (default: 50)')
args = parser.parse_args()

# Load prompts from the CSV
prompts_df = pd.read_csv("Bias/PromptPersona_Full_600.csv").head(args.num_prompts)

# Layers to sweep (same as before)
layers = [10, 12, 13, 14, 16, 18, 20, 24, 28]

results = []
model = CLAS_Model()

for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
    prompt_id = row['id']
    prompt_text = row['prompt_text']
    for layer in layers:
        response = model.generate(prompt_text, injection_layer=layer, alpha=args.alpha)
        results.append({
            "prompt_id": str(prompt_id),
            "layer": layer,
            "response_text": response,
            "utility_score": None  # Placeholder for future scoring
        })

# Save results to CSV
pd.DataFrame(results).to_csv(args.output, index=False)
print(f"Wrote results to {args.output}")
