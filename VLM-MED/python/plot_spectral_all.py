#!/usr/bin/env python3
"""Plot EVR1, mean_sim, and mean_abs_proj across layers using JSON inputs.

Reads: analysis_output/spectral_stats.json, analysis_output/pc1_summary.json
Writes: analysis_output/evr_per_layer.png, analysis_output/mean_sim_per_layer.png, analysis_output/combined_spectral.png
Also writes a plot data JSON: analysis_output/plot_inputs.json
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SPEC = Path('analysis_output/spectral_stats.json')
PC1 = Path('analysis_output/pc1_summary.json')
OUT_DIR = Path('analysis_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not SPEC.exists():
    raise SystemExit(f'Missing {SPEC}')
spec = json.loads(SPEC.read_text(encoding='utf-8'))
pc1 = json.loads(PC1.read_text(encoding='utf-8')) if PC1.exists() else {}

layers = sorted(int(k) for k in spec.keys())
evr1 = [spec[str(k)]['evr1'] if spec[str(k)]['evr1'] is not None else np.nan for k in layers]
mean_sim = [spec[str(k)]['mean_sim'] if spec[str(k)]['mean_sim'] is not None else np.nan for k in layers]
mean_proj = [pc1.get(str(k),{}).get('mean_abs_proj', np.nan) for k in layers]

# write plot inputs JSON
plot_inputs = {'layers':layers,'evr1':evr1,'mean_sim':mean_sim,'mean_abs_proj':mean_proj}
(OUT_DIR/'plot_inputs.json').write_text(json.dumps(plot_inputs, indent=2))

# EVR plot
fig,ax=plt.subplots(figsize=(8,3.5))
ax.plot(layers, evr1, marker='o')
ax.set_xlabel('Layer')
ax.set_ylabel('EVR of PC1')
ax.set_title('EVR1 per layer')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR/'evr_per_layer.png', dpi=150)
plt.close(fig)

# mean sim plot
fig,ax=plt.subplots(figsize=(8,3.5))
ax.plot(layers, mean_sim, marker='o', color='C1')
ax.set_xlabel('Layer')
ax.set_ylabel('Mean Cosine Similarity')
ax.set_title('Mean Cosine Similarity per layer')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR/'mean_sim_per_layer.png', dpi=150)
plt.close(fig)

# combined
fig,ax1 = plt.subplots(figsize=(8,4))
ax1.plot(layers, evr1, marker='o', label='EVR1')
ax1.set_xlabel('Layer')
ax1.set_ylabel('EVR1')
ax2 = ax1.twinx()
ax2.plot(layers, mean_sim, marker='s', color='C1', label='Mean Cosine Sim')
ax2.set_ylabel('Mean Cosine Sim')
fig.suptitle('Spectral Fingerprint Metrics by Layer')
ax1.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT_DIR/'combined_spectral.png', dpi=150)
plt.close(fig)

print('Wrote plots to', OUT_DIR)
