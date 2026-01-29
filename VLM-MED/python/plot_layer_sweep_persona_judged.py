import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load judged results
with open("outputs_local/judged_outputs_llamaguard.jsonl", "r", encoding="utf-8") as f:
    judged = [json.loads(line) for line in f]

# Build DataFrame
judged_df = pd.DataFrame([
    {"prompt_id": str(j["prompt_id"]), "layer": int(j["layer"]), "model": j["model"], **j["score"]}
    for j in judged if "score" in j and j.get("prompt_id") is not None and j.get("layer") is not None
])

# Map model file to alpha value for labeling
model_to_alpha = {
    "layer_sweep_results_persona.csv": "alpha=1.0",
    "layer_sweep_results_persona_alpha005.csv": "alpha=0.05"
}
judged_df["alpha"] = judged_df["model"].map(model_to_alpha)

score_cols = ["assertiveness", "complexity", "emotional_distance", "instructional_density"]

# Plot boxplots for each score dimension by layer and alpha
plt.figure(figsize=(16, 10))
for i, col in enumerate(score_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x="layer", y=col, hue="alpha", data=judged_df)
    plt.title(f"{col.title()} by Layer and Alpha")
plt.tight_layout()
plt.savefig("outputs_local/persona_layer_sweep_judge_boxplots.png")
plt.show()

# Print mean scores by layer and alpha
grouped = judged_df.groupby(["alpha", "layer"])[score_cols].mean()
print("Mean scores by layer and alpha:\n", grouped)
