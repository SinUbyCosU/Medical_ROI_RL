#!/usr/bin/env bash
set -euo pipefail
. /root/.venv/bin/activate
python - <<'PY'
import json, subprocess, sys
models = json.load(open('models_diffusers_10.json'))[1:]
cmd = ['python', 'scripts/run_vlm_diffusers.py']
for m in models:
    cmd += ['--model-id', m]
cmd += [
    '--device','cuda',
    '--fp16',
    '--attention-slicing',
    '--num-inference-steps','30',
    '--guidance-scale','7.5',
    '--tag-overlay',
    '--csv','vlm_hallucination_dataset.csv'
]
print('Running:', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
PY
