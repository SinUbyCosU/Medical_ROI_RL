# Stochastic Straight-Through Mask Training

This document describes the updated ROI-consistency training recipe that replaces the earlier PPO-based mask learner with a more stable, straight-through stochastic mask strategy. It captures the motivation, the architectural changes, and the exact commands required to reproduce the latest experiments.

## Motivation

- **Baseline drift** – the original dynamic-loss run converged, but the learned mask collapsed (mean coverage → 0, auxiliary losses ≈ 0).
- **PPO instability** – replacing the mask with a PPO policy introduced high variance and produced overly dense masks (coverage > 0.6) that failed to beat the standard classifier.
- **Research-backed alternative** – binary-concrete (Gumbel-sigmoid) masks with straight-through gradients are widely used in saliency/ROI literature (e.g., Supervised Grad-CAM, L2X, Straight-Through Gumbel estimators). They deliver stable convergence while enforcing sparsity.

## Key Changes in `train_roi_consistency.py`

1. **Mask generator network**
   - `MaskGeneratorNet` (3×conv + 1×1 head) outputs logits only.
   - Removed PPO value head and action log-prob tracking.

2. **Stochastic mask sampling**
   - `sample_stochastic_mask` applies optional Gumbel noise and temperature scaling.
   - Straight-through estimator: during the forward pass, masks are binarised (`>=0.5`) but gradients pass through the sigmoid probabilities.
   - Optional top-k projection (`--mask_project`) enforces the desired keep ratio while keeping gradients via a straight-through correction.

3. **Temperature scheduling**
   - Controlled with `--mask_temperature`, `--mask_temperature_min`, and `--mask_temperature_decay` (applied each epoch).
   - Helps anneal exploration → deterministic masks near convergence.

4. **Loss shaping**
   - Retained multi-task losses (BCE, consistency, drop, sparsity, TV).
   - Added stronger mask volume penalty (`--mask_volume_lambda`, default 5.0) targeting `--mask_volume_target` (default 0.1) to keep masks sparse.
   - Dynamic GradNorm weights still active; they play nicely with the stochastic mask when the auxiliary signals stay non-zero.

5. **Logging updates**
   - Training CSV now records: epoch, steps, loss components, dynamic weights, mask mean/TV.
   - Console logs include the current temperature for quick sanity checks.

## Reproduction

From a clean workspace (assuming the dataset is already downloaded to `/root/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3`):

```bash
cd /root/roi-consistency-new
source .venv/bin/activate

# Fresh run with stochastic straight-through masks
python train_roi_consistency.py \
  --train_csv nih_roi_train.csv \
  --image_root /root/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3 \
  --batch_size 24 \
  --accum_steps 1 \
  --epochs 20 \
  --channels_last \
  --autocast_dtype bf16 \
  --workers 6 \
  --prefetch_factor 4 \
  --persistent_workers \
  --save checkpoints/nih_roi_consistency_opt.pt \
  --img_size 224 \
  --lr 1e-4 \
  --alpha 1.0 \
  --beta 0.5 \
  --gamma 5e-3 \
  --delta 5e-4 \
  --keep_ratio 0.1 \
  --log_csv train_metrics_opt_maskst.csv \
  --mask_volume_target 0.1 \
  --mask_volume_lambda 5.0 \
  --mask_temperature 0.7 \
  --mask_temperature_min 0.2 \
  --mask_temperature_decay 0.98 \
  --mask_hard \
  --mask_gumbel \
  --mask_project | tee -a train_run_opt_maskst.log
```

> Tip: run this inside tmux (`tmux new -s roi-train-opt '... | tee ...'`) so you can detach safely.

## Interpreting Metrics

- **`mask_mean`** – should trend toward `mask_volume_target` (≈ 0.1). If it drifts high, increase `--mask_volume_lambda` or lower the temperature floor.
- **`loss_volume`** – near-zero indicates the mask is sticking to the target coverage.
- **Dynamic weights (`w_full`…`w_tv`)** – should remain non-zero so each loss term keeps contributing. If any weight collapses, consider raising `--dynamic_weight_min` (currently 1e-3).
- **Console temperature** – confirms the annealing schedule. Expect values to drop from 0.7 toward 0.2 over epochs.

## Comparison Against Baselines

Use the existing CSVs for reference:

- `train_metrics_standard.csv` – Plain ConvNeXt BCE baseline.
- `train_metrics.csv` – ROI consistency with deterministic masks (original setup).
- `train_metrics_opt_maskst.csv` – New stochastic straight-through mask run (current work in progress).

Plot `loss_full` or run validation to confirm the new strategy outperforms the standard baseline. For quick checks:

```bash
python - <<'PY'
import pandas as pd
opt = pd.read_csv('train_metrics_opt_maskst.csv')
base = pd.read_csv('train_metrics_standard.csv')
print('Opt masked final loss_full:', opt['loss_full'].tail().mean())
print('Standard final loss_full:', base['loss_full'].tail().mean())
PY
```

## Next Steps

- **Stability tuning** – adjust temperature decay, volume penalty, or dynamic weight floor if masks under/over cover.
- **Evaluation** – refresh validation metrics (AUC, F1) to verify improvements carry to the test split.
- **Extension** – once mask learning is solid, reintroduce text-conditioning or auxiliary objectives incrementally.

For questions or hyperparameter sweeps, reuse this document as the canonical reference for the new mask training regime.
