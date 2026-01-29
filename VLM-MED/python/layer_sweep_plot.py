#!/usr/bin/env python3
"""Plot layer sweep results.

Reads layer_sweep_results.csv and writes fig_layer_sweep.png.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('layer_sweep_results.csv')
    if 'layer' not in df.columns or 'utility_score' not in df.columns:
        raise ValueError('CSV must have columns: layer, utility_score')

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='layer', y='utility_score', errorbar=('ci', 95))
    plt.axvline(16, linestyle='--', color='red', label='Intervention Point')
    plt.axvline(13, linestyle='--', color='gray', label='Linearity Peak')
    plt.title('Instructional Density vs. Injection Layer')
    plt.xlabel('Layer')
    plt.ylabel('Utility Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_layer_sweep.png', dpi=300)


if __name__ == '__main__':
    main()
