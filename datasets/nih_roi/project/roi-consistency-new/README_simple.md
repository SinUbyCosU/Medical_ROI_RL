# ROI Consistency Walkthrough (Unofficial, Plain-English)

This guide explains what the code does, why each step exists, and what to expect, in beginner-friendly terms.

## What we are trying to do
- Train a chest X-ray classifier on the NIH ChestX-ray14 dataset.
- At the same time, learn a mask that highlights the most important image regions (ROI consistency). The model should keep similar predictions when we keep only the important parts and should get less confident when we remove them.
- Output: a trained model checkpoint, prediction scores, and optional visualizations of the masks/crops.

## Big picture pipeline
1) Get the dataset (images + CSV metadata).
2) Make train/test CSVs that pair each image filename with its labels.
3) Train the model (ConvNeXt-Base + small mask network) with a combined loss.
4) Run inference to get predictions and mask visualizations.

## Step-by-step
### 1) Download the dataset
- We use KaggleHub to download NIH ChestX-ray14: `kagglehub.dataset_download("nih-chest-xrays/data")`.
- The download path looks like `/root/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3`.
- Images are split across many folders (`images_001`, `images_002`, ...). We symlink all PNGs into one folder called `images/` for simplicity.

### 2) Build train/test CSVs
- Script: `make_chexmultimodal_csv.py`.
- Train CSV command (simplified):
  - Input: `Data_Entry_2017.csv` (metadata) + unified `images/` folder.
  - Output: `nih_roi_train.csv` with two columns: `image` and `labels`.
  - Empty labels are filled with `Normal` so every row has at least one label.
- Test CSV command: same, but filtered by NIH's official `test_list.txt` to create `nih_roi_test.csv`.
- Why: The model reads these CSVs to know which image file goes with which labels.

### 3) Model architecture
- Backbone: ConvNeXt-Base (pretrained on ImageNet) from torchvision. We replace the classifier layer to match the 14 NIH labels.
- Mask network: a tiny 3-layer conv net (3→16→32→1) that predicts a soft mask over the image.
- Why: The mask lets us test "faithfulness"—does the model rely on the highlighted regions?

### 4) Forward pass logic
- Compute logits on the full image.
- Turn the mask into a binary top-k mask (keep only the top ~10% pixels).
- Create two variants:
  - `img_crop` = image * mask (kept regions only).
  - `img_drop` = image * (1 - mask) (removed regions only).
- Run the backbone on all three (full, crop, drop) and get probabilities.

### 5) Losses (all added together)
- `loss_full`: BCE on full image vs true labels (normal multi-label loss).
- `loss_consist`: L1 between full probs and crop probs (encourage consistency when we keep important parts).
- `loss_drop`: ReLU(prob_drop - prob_full) (penalize being confident when evidence is removed).
- `loss_sparse`: mean(mask) (keep mask small/sparse).
- `loss_tv`: total variation (smooth mask, avoid noisy blobs).
- Why: Together they push the model to highlight useful regions and be less confident when those regions are gone.

### 6) Training settings (current defaults)
- Batch size 8, gradient accumulation 2, epochs 8, image size 224.
- Optimizer: AdamW, lr 1e-4.
- Mixed precision: bf16; channels-last; TF32 enabled on CUDA; cuDNN benchmark on.
- Dataloader: 8 workers, pinned memory, persistent workers, prefetch_factor 6.
- Checkpoint saved each epoch to `checkpoints/nih_roi_consistency.pt` (includes model, mask_net, classes, args).
- Why: These settings balance speed and memory on a single modern NVIDIA GPU (~12–24 GB VRAM).

### 7) Inference and outputs
- Script: `infer_roi_consistency.py`.
- Inputs: checkpoint, CSV (or single image), image_root, keep_ratio.
- Outputs:
  - `preds.json`: per-image `prob_full`, `prob_crop`, `prob_drop`, and `faithfulness = prob_full - prob_drop`.
  - Optional visuals (`roi_vis/`): saved masks and crops.
- Why: To inspect model confidence and see if masks align with meaningful regions.

### 8) One-command run (tmux-friendly)
- Run everything end-to-end:
  ```bash
  cd /root/roi-consistency-new
  tmux new -s roi-run './run_all.sh | tee roi-run.log'
  ```
- What it does: creates venv, installs deps, downloads data, unifies images, builds CSVs, trains 8 epochs, runs inference, writes logs to `roi-run.log`.

### 9) Expected artifacts
- `checkpoints/nih_roi_consistency.pt` — trained weights.
- `nih_roi_train.csv`, `nih_roi_test.csv` — data splits.
- `preds.json` — predictions + faithfulness scores.
- `roi_vis/` — optional masks/crops when enabled.
- `roi-run.log` — full log if run via `run_all.sh`.

### 10) Interpreting faithfulness
- High `prob_full`, similar `prob_crop`, and lower `prob_drop` means the mask kept key evidence and dropping it reduced confidence.
- If `prob_drop` stays high, the mask may be missing important regions or the model is overconfident.

## FAQs (quick)
- **Why symlink images?** The dataset is sharded; a flat `images/` folder makes lookups simple.
- **Why ConvNeXt-Base?** Strong ImageNet backbone with good transfer for medical images after fine-tuning.
- **Why bf16/channels-last/TF32?** To speed up training on NVIDIA GPUs while keeping quality.
- **What if GPU is small?** Lower `--batch_size` or `--workers`, or increase `--accum_steps`.
- **How to stop/resume?** Stop with Ctrl+C in tmux; restart `run_all.sh` or run `train_roi_consistency.py` pointing to existing CSVs.
