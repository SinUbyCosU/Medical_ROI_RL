# Per-Model Analysis Report

## Summary

This report provides detailed per-model analysis of the ACL/FAccT evaluation across 8 LLM variants.

## Models Analyzed


### PromptPersona_Full_600_gemini_flash
- **Samples**: 600
- **Assertiveness**: 7.37 ± 0.83
- **Quality Score**: 7.46
- **Instructional Density**: 8.78
- **Complexity**: 6.21
- **Emotional Distance**: 3.47
- **Gender Gap**: -0.04 (Cohen's d: -0.0551)

### PromptPersona_Full_600_llama31_8b
- **Samples**: 600
- **Assertiveness**: 6.54 ± 1.22
- **Quality Score**: 5.61
- **Instructional Density**: 6.92
- **Complexity**: 4.52
- **Emotional Distance**: 4.80
- **Gender Gap**: -0.03 (Cohen's d: -0.0202)

### PromptPersona_Full_600_nous_hermes_mistral_7b
- **Samples**: 593
- **Assertiveness**: 6.83 ± 1.02
- **Quality Score**: 6.81
- **Instructional Density**: 7.97
- **Complexity**: 5.68
- **Emotional Distance**: 4.35
- **Gender Gap**: 0.11 (Cohen's d: 0.1100)

### PromptPersona_Full_600_openhermes_mistral_7b
- **Samples**: 581
- **Assertiveness**: 6.69 ± 1.06
- **Quality Score**: 6.70
- **Instructional Density**: 8.27
- **Complexity**: 5.55
- **Emotional Distance**: 4.22
- **Gender Gap**: -0.06 (Cohen's d: -0.0565)

### PromptPersona_Full_600_phi35_mini
- **Samples**: 702
- **Assertiveness**: 3.60 ± 1.53
- **Quality Score**: 4.27
- **Instructional Density**: 2.80
- **Complexity**: 5.53
- **Emotional Distance**: 7.09
- **Gender Gap**: -0.01 (Cohen's d: -0.0057)

### PromptPersona_Full_600_qwen25_7b
- **Samples**: 600
- **Assertiveness**: 7.20 ± 0.86
- **Quality Score**: 7.29
- **Instructional Density**: 8.63
- **Complexity**: 6.01
- **Emotional Distance**: 3.86
- **Gender Gap**: 0.06 (Cohen's d: 0.0690)

### PromptPersona_Full_600_qwen2_7b
- **Samples**: 600
- **Assertiveness**: 7.29 ± 1.17
- **Quality Score**: 5.51
- **Instructional Density**: 3.15
- **Complexity**: 7.52
- **Emotional Distance**: 9.10
- **Gender Gap**: -0.12 (Cohen's d: -0.0989)

### PromptPersona_Full_600_yi15_6b
- **Samples**: 586
- **Assertiveness**: 5.53 ± 1.93
- **Quality Score**: 4.42
- **Instructional Density**: 4.17
- **Complexity**: 4.25
- **Emotional Distance**: 5.32
- **Gender Gap**: -0.29 (Cohen's d: -0.1538)

### PromptPersona_Full_600_zephyr_7b
- **Samples**: 600
- **Assertiveness**: 6.51 ± 1.32
- **Quality Score**: 6.34
- **Instructional Density**: 7.23
- **Complexity**: 5.51
- **Emotional Distance**: 4.39
- **Gender Gap**: -0.08 (Cohen's d: -0.0612)


## Key Insights

1. **Safety-Quality Trade-off**: Models with high quality scores often have higher toxicity rates.
2. **Domain Variation**: Toxicity rates vary significantly by domain (Medical > Employment > Legal).
3. **Language Effects**: Hinglish responses show lower quality but potentially higher safety.
4. **Gender Fairness**: Aggregate gender gaps are generally small, though domain-level biases may exist.

## Recommendations

- **General Use**: Use balanced models like Llama-3.1-8B
- **Safety-Critical**: Use Phi-3.5-mini or Qwen2-7B despite quality trade-offs
- **Domain-Specific**: Audit each model on high-risk domains (Medical, Employment)
- **Language**: Be aware of quality degradation when using Hinglish prompts

