# ROI Consistency Loop (Prediction → Mask → Crop → Re-predict)

This repo provides a minimal training and inference setup for a reviewer-friendly, faithfulness-oriented imaging model:

1. Predict on full image.
2. Derive a saliency mask (Grad-CAM style).
3. Crop to the salient region (or mask-out the region).
4. Re-predict on the crop and on the masked-out image.
5. Train with consistency and drop losses so the model depends on the evidence it highlights.

## Files
- `train_roi_consistency.py`: Training script with mask/crop/re-predict losses.
- `infer_roi_consistency.py`: Inference script that outputs full vs. crop probabilities and saves mask/crop images.
- `requirements.txt`: Python dependencies.

## Data format
CSV with columns:
- `image`: path relative to `--image_root`
- `labels`: comma-separated class names (e.g., `PNEUMONIA,CARDIOMEGALY`)

Provide class names as a comma-separated list to `--classes`, in the same canonical spelling used in the CSV.

## Training example
```bash
python train_roi_consistency.py \
  --train_csv data/train.csv \
  --image_root data/images \
  --classes PNEUMONIA,CARDIOMEGALY,PLEURAL_EFFUSION \
  --backbone convnext_base \
  --batch_size 8 --epochs 10 --lr 3e-4 \
  --keep_ratio 0.1 --margin 0.2 \
  --alpha 1.0 --beta 0.5 --gamma 1e-3 --delta 1e-4 \
  --save checkpoints/roi_consistency.pt
```

## Inference example
```bash
python infer_roi_consistency.py \
  --checkpoint checkpoints/roi_consistency.pt \
  --image sample.jpg \
  --keep_ratio 0.1 \
  --save_mask mask.png \
  --save_crop crop.png
```

## Loss terms
- `BCE`: main classification loss on full image.
- `Consistency`: classification loss on the crop (ROI alone should suffice).
- `Drop`: encourages confidence to drop when the ROI is removed (`masked-out` image).
- `Sparsity`: keeps the mask tight (mean pixel value penalty).
- `TV`: smoothness on the mask.

## Notes
- Backbone defaults to `convnext_base` from timm; you can swap any features_only backbone.
- Saliency uses Grad-CAM-style gradients on the last feature map.
- Mask is top-k% of saliency; adaptive Otsu or percentile can be substituted.
- Crops are resized back to the model input size for re-prediction.

## Safety
For medical research use only. Not for clinical deployment.
