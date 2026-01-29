# Medical ROI RL: Faithful and Bias-Aware Medical Image Analysis

A research project focused on developing trustworthy and interpretable deep learning models for medical imaging. This work addresses critical challenges in AI safety for healthcare applications through ROI-based faithfulness constraints and comprehensive bias detection and mitigation.

## 🔬 Project Overview

This repository contains research implementations for training and evaluating medical imaging models with built-in interpretability and fairness guarantees. The work addresses three key areas:

1. **ROI Consistency Training**: Deep learning models that provide visual evidence for their diagnostic predictions through interpretable saliency maps
2. **Bias Detection & Mitigation**: Systematic identification and correction of demographic biases in model outputs
3. **Benchmarking & Validation**: Rigorous evaluation on CheXpert and NIH Chest X-ray datasets

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

### Inference on Test Images

```bash
cd VLM-MED
python infer_roi_consistency.py \
  --checkpoint checkpoints/roi_consistency.pt \
  --image /path/to/chest_xray.jpg \
  --keep_ratio 0.1 \
  --save_mask output/saliency_mask.png \
  --save_crop output/roi_crop.png
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
A systematic approach to ensuring fairness in medical AI:

1. **Detection Phase**: Generate model outputs across diverse demographic variations
2. **Analysis Phase**: Compute NLP-based bias metrics, tone analysis, and token distribution patterns
3. **Mitigation Phase**: Apply targeted correction strategies to low-quality or biased outputs
4. **Validation Phase**: Adversarial testing and quality re-evaluation to verify improvements

The pipeline supports multiple evaluation frameworks including quality scoring, safety assessment, and demographic parity analysis.

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

## 🔗 Datasets & Resources

- **CheXpert Dataset**: [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)
- **NIH Chest X-rays**: [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)

---

**Status**: Active Research Project | Last Updated: January 2026