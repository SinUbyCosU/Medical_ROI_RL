# Contrastive Activation Addition (CAA) Baseline

This directory contains the implementation and experiments for the CAA (Contrastive Activation Addition) baseline for cross-lingual steering with Phi-3.5 Mini.

## Structure
- `extract_caa_vector.py`: Extracts CAA steering vectors from calibration prompts (mean across all prompt tokens).
- `caa_model.py`: Model wrapper for CAA steering (injection throughout generation, norm-preserving scaling).
- `run_caa_experiment.py`: Runs CAA steering on evaluation prompts, saves outputs.
- `judge_outputs.py`: Judges outputs for Instructional Density (ΔD) and Jailbreak Rate.
- `plots/`: Stores analysis plots.
- `results/`: Stores experiment outputs and judged results.
- `prompts/`: Stores prompt files (calibration/evaluation splits).

## Workflow
1. Extract CAA vector from calibration prompts (English vs. Hinglish).
2. Run CAA steering on evaluation prompts (inject throughout generation).
3. Judge outputs for ΔD and Jailbreak Rate.
4. Plot and analyze results.
