#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

IN_PATH = Path("analysis_output/ablation_schedule_results.json")
OUT_PNG = Path("analysis_output/ablation_schedule_drift.png")

with IN_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

names = []
drift_fracs = []
mean_lens = []

for entry in data:
    cfg = entry["config"]["name"]
    outputs = entry["outputs"]
    names.append(cfg)
    if not outputs:
        drift_fracs.append(0.0)
        mean_lens.append(0.0)
        continue
    drifts = [1 if o.get("is_english_drift") else 0 for o in outputs]
    lengths = [o.get("length", 0) for o in outputs]
    drift_fracs.append(float(sum(drifts)) / len(drifts))
    mean_lens.append(float(sum(lengths)) / len(lengths))

x = np.arange(len(names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()

ax1.bar(x - width/2, drift_fracs, width, label="Drift Fraction", color="#1f77b4")
ax2.bar(x + width/2, mean_lens, width, label="Mean Length", color="#ff7f0e", alpha=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=20, ha="right")
ax1.set_ylabel("Drift Fraction")
ax2.set_ylabel("Mean Response Length")
ax1.set_ylim(0, 1.0)

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.title("Injection Schedule Ablation — Drift vs Length")
plt.tight_layout()
plt.savefig(OUT_PNG)
print(f"Saved plot to {OUT_PNG}")
