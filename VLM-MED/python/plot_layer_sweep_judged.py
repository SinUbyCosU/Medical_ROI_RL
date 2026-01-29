import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load judged results
with open("outputs_local/judged_outputs_llamaguard.jsonl", "r", encoding="utf-8") as f:
    judged = [json.loads(line) for line in f]

# Load sweep results
sweep = pd.read_csv("layer_sweep_results.csv")


# Build judged_df with both prompt_id and layer
judged_df = pd.DataFrame([
    {"prompt_id": str(j["prompt_id"]), "layer": int(j["layer"]), **j["score"]}
    for j in judged if "score" in j and j.get("prompt_id") is not None and j.get("layer") is not None
])

sweep["prompt_id"] = sweep["prompt_id"].astype(str)
sweep["layer"] = sweep["layer"].astype(int)
merged = pd.merge(sweep, judged_df, on=["prompt_id", "layer"], how="inner")

# Plot differences for each score dimension by layer
score_cols = ["assertiveness", "complexity", "emotional_distance", "instructional_density"]


plt.figure(figsize=(14, 8))
for i, col in enumerate(score_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x="layer", y=col, data=merged)
    plt.title(f"{col.title()} by Layer")
plt.tight_layout()
plt.savefig("outputs_local/layer_sweep_judge_boxplots_fixed.png")
plt.show()

# Optionally: print summary stats
grouped = merged.groupby("layer")[score_cols].mean()
print("Mean scores by layer:\n", grouped)
