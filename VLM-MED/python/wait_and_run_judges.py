#!/usr/bin/env python3
"""Wait for ablation generation outputs, then run judge_clas_results.py on each, and finally aggregate.

Expected output files (generation): results_ablation_layer{ell}_alpha_{sign}{mag}.jsonl
Judge outputs will be written to analysis_output/{input_basename}_judged.jsonl
"""
from pathlib import Path
import time
import subprocess
import json

GENERATED = []
LAYERS = [2,16,32]
ALPHAS = [0.05, -0.05]
for ell in LAYERS:
    for a in ALPHAS:
        sign = 'pos' if a>=0 else 'neg'
        mag = str(abs(a)).replace('.','')
        name = f'results_ablation_layer{ell}_alpha_{sign}{mag}.jsonl'
        GENERATED.append(Path(name))

JUDGE_SCRIPT = Path('judge_clas_results.py')
AGG_SCRIPT = Path('aggregate_ablation_results.py')

print('Waiting for generation outputs...')
# wait until all files exist and are non-empty
while True:
    ready = [p.exists() and p.stat().st_size>0 for p in GENERATED]
    print(f"{sum(ready)}/{len(GENERATED)} ready")
    if all(ready):
        break
    time.sleep(10)

print('All generation outputs present; running judges...')
for p in GENERATED:
    out = Path('analysis_output')/(p.stem + '_judged.jsonl')
    cmd = [str(Path('.venv/bin/python')), str(JUDGE_SCRIPT), '--input', str(p), '--output', str(out)]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd)

print('Running aggregation/plot...')
subprocess.run([str(Path('.venv/bin/python')), str(AGG_SCRIPT)])
print('Done.')
