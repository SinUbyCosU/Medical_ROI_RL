import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load judged results
with open('../results/caa_outputs_judged.jsonl', 'r', encoding='utf-8') as f:
    judged = [json.loads(line) for line in f]

df = pd.DataFrame(judged)
score_cols = ['assertiveness', 'complexity', 'emotional_distance', 'instructional_density']

# Expand score dict
scores = pd.json_normalize(df['score'])
df = pd.concat([df.drop(columns=['score']), scores], axis=1)

plt.figure(figsize=(14, 8))
for i, col in enumerate(score_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='layer', y=col, data=df)
    plt.title(f'{col.title()} by Layer (CAA)')
plt.tight_layout()
plt.savefig('../plots/caa_layer_boxplots.png')
plt.show()

# Print means
grouped = df.groupby('layer')[score_cols].mean()
print('Mean scores by layer:\n', grouped)
