# Medical ROI RL: Faithful and Bias-Aware Medical Image Analysis

A comprehensive research project focused on developing trustworthy vision-language models for medical imaging, with emphasis on Region of Interest (ROI) consistency, bias mitigation, and interpretability through Contrastive Activation Addition (CAA).

## 🔬 Project Overview

This repository contains research code for training and analyzing medical imaging models with built-in faithfulness guarantees, bias detection, and mitigation strategies. The work spans multiple critical areas of AI safety in medical imaging:

1. **ROI Consistency Training**: Models that provide interpretable evidence for their predictions
2. **Bias Analysis & Mitigation**: Comprehensive testing and correction of demographic biases
3. **Contrastive Activation Addition (CAA)**: Steering vectors for controlled model behavior
4. **Hallucination Detection**: Testing and validation on CheXpert benchmark

## 📁 Repository Structure

### VLM-MED/
Vision-Language Model for Medical imaging with ROI consistency training.

- `train_roi_consistency.py`: Training with mask/crop/re-predict losses
- `infer_roi_consistency.py`: Inference with interpretable saliency masks
- `train_chexmultimodal.csv`: Training data manifest
- `checkpoints/`: Model checkpoints
- `logs/`: Training logs and metrics

**Key Features:**
- Grad-CAM style saliency masks
- Consistency loss: predictions should remain stable when using only the highlighted ROI
- Drop loss: confidence should decrease when ROI is removed
- Sparsity regularization for tight, focused masks

### Bias/
Comprehensive bias detection, analysis, and mitigation pipeline.

**Core Scripts:**
- `multi_model_generation.py`: Generate outputs from multiple LLMs
- `judge_gemini_flash.py`: Quality evaluation using Gemini
- `judge_outputs_llamaguard.py`: Safety evaluation with LlamaGuard
- `bias_nlp_analysis.py`: NLP-based bias detection
- `smart_fix_*.py`: Automated bias mitigation strategies
- `adversarial_testing.py`: Robustness testing under adversarial prompts

**Analysis:**
- `analysis_acl_facct.py`: Analysis for ACL/FAccT publication
- `compare_score_improvements.py`: Track mitigation effectiveness
- `tone_polygon_analysis.py`: Multi-dimensional tone analysis
- `token_analysis.py`: Token-level bias patterns

**Pipelines:**
- `run_regen_pipeline.py`: Regenerate biased outputs
- `run_smart_fix_pipeline.py`: Automated bias correction
- `launch_regen_pipeline.sh`: Batch processing launcher

### CAA/
Contrastive Activation Addition for model steering.

- `extract_caa_vector.py`: Extract steering vectors from activation differences
- `caa_model.py`: Model wrapper with CAA injection
- `run_caa_experiment.py`: Run steering experiments
- `judge_outputs.py`: Evaluate steered outputs

**Applications:**
- Cross-lingual steering (English ↔ Hinglish)
- Behavioral control (instructional density, jailbreak resistance)
- Interpretable model intervention

### chexpert-test-set-labels/
Official CheXpert test set with ground truth labels and radiologist annotations.

- `groundtruth.csv`: Gold standard labels
- `labeler/`: Automated labeling tools
- `radiologists/`: Human expert annotations

### datasets/
- `hallucination/`: Hallucination detection benchmarks
- `nih_roi/`: NIH Chest X-ray dataset with ROI annotations

### data/
- `probes_364.json`: Evaluation probes
- `probes_unique.json`: Deduplicated probe set

### figures/
Analysis outputs and visualizations:
- Ablation studies by model, layer, and alpha parameters
- Delta statistics summaries
- Diagnostic visualizations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SinUbyCosU/Medical_ROI_RL.git
cd Medical_ROI_RL

# Install VLM-MED dependencies
cd VLM-MED
pip install -r requirements.txt
```

### Training ROI Consistency Model

```bash
cd VLM-MED
python train_roi_consistency.py \
  --train_csv train_chexmultimodal.csv \
  --image_root /path/to/images \
  --classes PNEUMONIA,CARDIOMEGALY,PLEURAL_EFFUSION,ATELECTASIS \
  --backbone convnext_base \
  --batch_size 8 \
  --epochs 10 \
  --lr 3e-4 \
  --keep_ratio 0.1 \
  --margin 0.2 \
  --alpha 1.0 --beta 0.5 --gamma 1e-3 --delta 1e-4 \
  --save checkpoints/roi_consistency.pt
```

### Running Bias Analysis

```bash
cd Bias

# Generate outputs from multiple models
python multi_model_generation.py

# Run quality evaluation
python quality_eval_runner.py

# Analyze bias patterns
python bias_nlp_analysis.py

# Apply smart fix mitigation
python smart_fix_final.py
```

### CAA Steering Experiment

```bash
cd CAA

# Extract steering vectors
python extract_caa_vector.py

# Run steering experiment
python run_caa_experiment.py

# Evaluate results
python judge_outputs.py
```

## 📊 Key Components

### ROI Consistency Training
The model learns to:
1. Predict pathologies from full chest X-rays
2. Generate saliency masks highlighting relevant regions
3. Maintain consistent predictions when shown only the highlighted ROI
4. Decrease confidence when the ROI is removed (masked out)

**Loss Function:**
$$\mathcal{L} = \alpha \mathcal{L}_{BCE} + \beta \mathcal{L}_{consistency} + \gamma \mathcal{L}_{drop} + \delta \mathcal{L}_{sparsity}$$

### Bias Mitigation Pipeline
1. **Detection**: Multi-model generation with demographic variations
2. **Analysis**: NLP-based bias metrics, tone analysis, token distributions
3. **Mitigation**: Smart fix strategies targeting low-quality, biased outputs
4. **Validation**: Adversarial testing and quality re-evaluation

### CAA Steering
Contrastive steering vectors extracted from:
$$v_{CAA} = \mathbb{E}[\text{activations}(x_{positive})] - \mathbb{E}[\text{activations}(x_{negative})]$$

Injected during generation with norm-preserving scaling.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{medical_roi_rl_2026,
  title={Medical ROI RL: Faithful and Bias-Aware Medical Image Analysis},
  author={SinUbyCosU},
  year={2026},
  url={https://github.com/SinUbyCosU/Medical_ROI_RL}
}
```

## 📄 License

This project is released for research purposes. Please ensure compliance with medical data regulations and institutional review board (IRB) requirements when using this code with clinical data.

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request for:
- Bug fixes
- New bias detection metrics
- Additional steering methods
- Model architecture improvements

## 📧 Contact

For questions and collaborations, please open an issue on GitHub.

## 🔗 Related Work

- CheXpert: [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)
- NIH Chest X-rays: [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- Contrastive Activation Addition: [CAA Paper](https://arxiv.org/abs/2312.06681)

---

**Status**: Active Research Project | Last Updated: January 2026
