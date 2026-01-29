
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from caa_model import CAA_Model
from tqdm import tqdm

# Load evaluation prompts
prompts = pd.read_csv('prompts/eval_prompts.csv').head(50)  # Use 50 for quick test
layers = [10, 12, 13, 14, 16, 18, 20, 24, 28]

model = CAA_Model()
results = []

for _, row in tqdm(prompts.iterrows(), total=len(prompts)):
    prompt_id = row['id']
    prompt_text = str(row['prompt_text']) if not pd.isna(row['prompt_text']) else ""
    for layer in layers:
        response = model.generate(prompt_text, injection_layer=layer, alpha=1.0)
        results.append({
            'prompt_id': prompt_id,
            'layer': layer,
            'response_text': response
        })

out_path = 'results/caa_outputs.csv'
pd.DataFrame(results).to_csv(out_path, index=False)
print(f'Wrote CAA experiment results to {out_path}')
