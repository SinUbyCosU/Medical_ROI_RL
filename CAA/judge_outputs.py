
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import json
from Bias.judge_outputs_llamaguard import judge_batch

# Load CAA outputs
outputs = pd.read_csv('results/caa_outputs.csv')

results = []
for _, row in outputs.iterrows():
    response = row['response_text']
    score = judge_batch([response])[0]
    results.append({
        'prompt_id': row['prompt_id'],
        'layer': row['layer'],
        'score': score
    })

with open('results/caa_outputs_judged.jsonl', 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print('Judged results saved to results/caa_outputs_judged.jsonl')
