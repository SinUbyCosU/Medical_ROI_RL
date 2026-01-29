#!/usr/bin/env python3
"""Aggregate judged ablation results and plot mean deltas per metric across runs.

Reads judged JSONLs in analysis_output matching *_judged.jsonl
Writes analysis_output/ablation_summary.json and analysis_output/ablation_deltas.png
"""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

JDIR = Path('analysis_output')
judged_files = sorted(p for p in JDIR.glob('*_judged.jsonl'))
if not judged_files:
    print('No judged files found in analysis_output'); raise SystemExit(1)

summary = {}
metrics = ['assertiveness','complexity','emotional_distance','instructional_density']
for p in judged_files:
    vals = {m:[] for m in metrics}
    with p.open(encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            delta = rec.get('delta') or {}
            for m in metrics:
                v = delta.get(m)
                if v is not None:
                    try:
                        vals[m].append(float(v))
                    except Exception:
                        pass
    means = {m: (statistics.mean(vals[m]) if vals[m] else None) for m in metrics}
    summary[p.name] = {'n': sum(len(vals[m]) for m in metrics)//len(metrics) if any(vals.values()) else 0, 'means':means}

OUT_JSON = JDIR/'ablation_summary.json'
OUT_JSON.write_text(json.dumps(summary, indent=2))
print('Wrote', OUT_JSON)

# plot
labels = list(summary.keys())
means_by_metric = {m:[summary[l]['means'][m] if summary[l]['means'][m] is not None else 0.0 for l in labels] for m in metrics}

x = range(len(labels))
fig,ax = plt.subplots(figsize=(max(6,len(labels)*0.8),4))
width = 0.18
for i,m in enumerate(metrics):
    ax.bar([xi + i*width for xi in x], means_by_metric[m], width=width, label=m.replace('_',' ').title())
ax.set_xticks([xi + width*(len(metrics)-1)/2 for xi in x])
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Mean Delta (steered - baseline)')
ax.legend()
fig.tight_layout()
plot_path = JDIR/'ablation_deltas.png'
fig.savefig(plot_path, dpi=150)
print('Wrote', plot_path)
