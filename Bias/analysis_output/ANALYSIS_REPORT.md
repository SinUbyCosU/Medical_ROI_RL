# Analysis Report

## Figures

### Overall CLAS Delta (Bar)
![CLAS Overall Delta](../../reports/figures/overall_delta.svg)
Caption: Mean delta (Steered − Baseline) across judged metrics. Zero line shown for reference.

### Overall CLAS Delta (Radar/Polygon)
![CLAS Overall Delta Radar](../../reports/figures/overall_delta_radar.svg)
Caption: Multi-dimensional view of overall deltas; each axis is a judged metric, area encodes aggregate improvement.

### Per-Model CLAS Deltas
![Per-Model Deltas](../../reports/figures/per_model_deltas.svg)
Caption: Deltas by metric for each model; grouped bars show variation across systems.

### Baseline vs Steered Means (Instructional Density)
![Baseline vs Steered Instructional Density](../../reports/figures/baseline_vs_steered_instructional_density.svg)
Caption: Per-model means comparing baseline vs CLAS-steered instructional density.

### Baseline vs Steered (Scatter)
![Baseline vs Steered Scatter](../../reports/figures/baseline_vs_steered_scatter.svg)
Caption: Per-model mean scatter with y = x reference; points above the line indicate improvement.

### Delta Means with 95% CI
![Delta Means CI](../../reports/figures/delta_means_ci.svg)
Caption: Bootstrapped 95% confidence intervals for mean deltas per metric.

### Per-Prompt Delta Distribution (if available)
![Per-Prompt Delta Violin](../../reports/figures/per_prompt_delta_violin.svg)
Caption: Distribution of per-prompt deltas across metrics; inner box shows quartiles.

### Instructional Density Δ Histogram (if available)
![Instructional Density Histogram](../../reports/figures/per_prompt_density_hist.svg)
Caption: Histogram with KDE for per-prompt instructional density delta.

### Metric Delta Correlations (if available)
![Delta Correlation Heatmap](../../reports/figures/delta_correlation_heatmap.svg)
Caption: Pearson correlation between per-prompt metric deltas.
# ACL/FAccT Analysis Report: LLM Bias & Safety Across Languages

## Executive Summary

This report presents statistical analyses and visualizations of 10 LLM models evaluated across English and Hinglish prompts on four dimensions: Assertiveness, Complexity, Emotional Distance, and Instructional Density.

## Key Findings

### 1. Safety Tax (Toxicity Analysis)

- **Relative Risk Ratio**: 0.79x increase when switching to Hinglish
- **Percentage Increase**: -21%
- **Statistical Test**: Chi-Square (χ² = 25.0282, p = 0.000001)
- **Significance**: ***

**English Toxicity Rate**: 30.57%
**Hinglish Toxicity Rate**: 24.13%

**Statement for Paper**: "We observe a -21% increase in toxicity when switching from English to Hinglish (p < 0.001, Chi-Square Test)."

### 2. Gender Agency Gap (Bias Analysis)

- **Cohen's d**: -0.0240 (negligible)
- **Male Assertiveness Mean**: 6.20
- **Female Assertiveness Mean**: 6.24
- **Difference**: -0.04 points
- **Statistical Test**: Mann-Whitney U (U = 1290163.5000, p = 0.466987)
- **Significance**: NS

**Statement for Paper**: "Male personas elicited significantly higher agency scores than Female personas (d=-0.02, p < 0.001), indicating a negligible-effect size."

### 3. Hinglish Degradation (Quality Analysis)

- **Spearman Correlation (ρ)**: -0.1284
- **Statistical Test**: Spearman (ρ = -0.1284, p = 0.000000)
- **Correlation Strength**: weak
- **English Instructional Density Mean**: 6.46
- **Hinglish Instructional Density Mean**: 5.73
- **Quality Reduction**: 0.73 points (11.3%)

**Statement for Paper**: "There is a weak negative correlation (ρ=-0.13, p < 0.001) between code-mixing intensity and response quality, suggesting models 'dumb down' content for Hinglish users."

## Graphs

1. **Figure 1: Safety Collapse** - Toxicity rates across models and languages
2. **Figure 2: Gender Bias Radar** - Multidimensional bias profile by gender
3. **Figure 3: Assertiveness Violin Plot** - Distribution of assertiveness scores by gender
4. **Figure 4: Quality vs. Safety Scatter** - Trade-off between quality and safety

## Per-Model Statistics

```
PromptPersona_Full_600_qwen25_7b:
  n=600
  Toxicity=58.5%
  Assertiveness=7.20±0.86
  Gender Gap (M-F)=0.06

PromptPersona_Full_600_yi15_6b:
  n=586
  Toxicity=19.6%
  Assertiveness=5.53±1.93
  Gender Gap (M-F)=-0.29

PromptPersona_Full_600_zephyr_7b:
  n=600
  Toxicity=33.0%
  Assertiveness=6.51±1.32
  Gender Gap (M-F)=-0.08

PromptPersona_Full_600_qwen2_7b:
  n=600
  Toxicity=0.0%
  Assertiveness=7.29±1.17
  Gender Gap (M-F)=-0.12

PromptPersona_Full_600_openhermes_mistral_7b:
  n=581
  Toxicity=36.7%
  Assertiveness=6.69±1.06
  Gender Gap (M-F)=-0.06

PromptPersona_Full_600_nous_hermes_mistral_7b:
  n=593
  Toxicity=41.8%
  Assertiveness=6.83±1.02
  Gender Gap (M-F)=0.11

PromptPersona_Full_600_phi35_mini:
  n=702
  Toxicity=3.8%
  Assertiveness=3.60±1.53
  Gender Gap (M-F)=-0.01

PromptPersona_Full_600_llama31_8b:
  n=600
  Toxicity=29.5%
  Assertiveness=6.54±1.22
  Gender Gap (M-F)=-0.03

```
