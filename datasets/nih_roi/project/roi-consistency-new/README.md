# ROI Consistency on NIH ChestX-ray14

End-to-end, reproducible pipeline to download NIH ChestX-ray14, prepare splits, train a ROI-consistency classifier with ConvNeXt-Base, and run inference/faithfulness visualizations. Suitable for paper appendix/methods.

## Quickstart (one command, tmux-safe)

```bash
cd /root/roi-consistency-new
./run_all.sh  # creates venv, downloads dataset, builds CSVs, trains 8 epochs, runs inference, logs to roi-run.log
# run under tmux: tmux new -s roi-run './run_all.sh | tee roi-run.log'
```

## Dataset details
- Source: NIH ChestX-ray14 via KaggleHub (`kagglehub.dataset_download("nih-chest-xrays/data")`), version path `/root/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3`.
- Contents: image shards `images_001`… with PNGs inside `images/`, metadata `Data_Entry_2017.csv`, split lists `train_val_list.txt`, `test_list.txt`, bounding boxes `BBox_List_2017.csv` (not used here).
- Image unification: all shard PNGs are symlinked into `$DATA_PATH/images` for flat access.
- Labels: 14 NIH findings — Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia.
- Splits: train CSV uses full list; test CSV filtered by official `test_list.txt`.

## Preprocessing & CSV construction
- Script: `make_chexmultimodal_csv.py`.
- Train CSV:
  - Input: `$DATA_PATH/Data_Entry_2017.csv`, images at `$DATA_PATH/images`.
  - Command: `--out_csv nih_roi_train.csv --label_from "Finding Labels" --fillna "Normal"`.
- Test CSV:
  - Adds `--filter_list $DATA_PATH/test_list.txt` → `nih_roi_test.csv`.
- Each row: `image` filename, comma-separated `labels`. Empty labels filled with `Normal`.

## Model architecture
- Backbone: `torchvision.models.convnext_base` pretrained on ImageNet; classifier head replaced with linear layer sized to the inferred class count.
- Mask network: 3-layer conv stack (3→16→32→1) producing a spatial mask.
- ROI-consistency formulation: generate top-k binary mask (keep_ratio), create a kept image (`img_crop = img * mask_bin`) and a dropped image (`img_drop = img * (1 - mask_bin)`), and enforce consistency/faithfulness between logits.

## Losses (summed)
- `loss_full`: BCEWithLogits on full image logits vs multi-label targets.
- `loss_consist`: L1 between sigmoid probs of full vs crop.
- `loss_drop`: ReLU on (prob_drop − prob_full) to penalize high confidence when evidence is removed.
- `loss_sparse`: mean(mask) for sparsity.
- `loss_tv`: total variation on mask for smoothness.
- Weights: `alpha=1.0`, `beta=0.5`, `gamma=1e-3`, `delta=1e-4`.

## Training configuration (current default)
- Script: `train_roi_consistency.py` (full masked model).
- Hyperparameters: batch size 8, grad accumulation 2, epochs 8, image size 224, lr 1e-4 (AdamW), keep_ratio 0.1.
- AMP: `bf16`; channels-last; TF32 enabled on CUDA; cuDNN benchmark on.
- Dataloader: 8 workers, pinned memory, persistent workers, prefetch_factor 6.
- Checkpoint: `checkpoints/nih_roi_consistency.pt` saved every epoch (includes model, mask_net, classes, args).
- Metrics logging: `--log_csv train_metrics.csv` appends per-epoch averages; console can be tee’d.
- Dynamic loss weights (optional): `--dynamic_weights` enables GradNorm-style balancing of `loss_full`, `loss_consist`, `loss_drop`, `loss_sparse`, `loss_tv` with initial weights `w_full=1.0, w_consist=0.5, w_drop=1e-3, w_sparse=1e-4, w_tv=1e-5`.

### Baseline & ablation runner
- Script: `train_roi_baselines.py` supports `--mode` in `{full, standard, random_mask, random_cutout, no_consist, no_sparse, no_tv}`.
- Example (standard ConvNeXt, no mask):
  ```bash
  tmux new -d -s roi-standard "cd /root/roi-consistency-new && source .venv/bin/activate && \
    python train_roi_baselines.py --mode standard --train_csv nih_roi_train.csv \
    --image_root /root/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3 \
    --batch_size 8 --accum_steps 2 --epochs 8 --channels_last --autocast_dtype bf16 \
    --workers 8 --prefetch_factor 6 --persistent_workers \
    --save checkpoints/nih_roi_standard.pt --log_csv train_metrics_standard.csv | tee train_run_standard.log"
  ```
- Dynamic run (full masked with auto-balanced losses):
  ```bash
  tmux new -d -s roi-train-dyn "cd /root/roi-consistency-new && source .venv/bin/activate && \
    python train_roi_consistency.py --train_csv nih_roi_train.csv \
    --image_root /root/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3 \
    --batch_size 8 --accum_steps 2 --epochs 8 --channels_last --autocast_dtype bf16 \
    --workers 8 --prefetch_factor 6 --persistent_workers \
    --save checkpoints/nih_roi_consistency_dyn.pt --log_csv train_metrics_dyn.csv \
    --keep_ratio 0.1 --lr 1e-4 --alpha 1.0 --beta 0.5 --gamma 1e-3 --delta 1e-4 \
    --dynamic_weights --w_full_init 1.0 --w_consist_init 0.5 --w_drop_init 1e-3 \
    --w_sparse_init 1e-4 --w_tv_init 1e-5 | tee train_run_dyn.log"
  ```

### Monitoring runs (tmux)
- List sessions: `tmux ls`
- Tail progress: `tmux capture-pane -pt <session> | tail -n 40`
- Live attach/detach: `tmux attach -t <session>` (Ctrl+B, D to detach)
- Logs: `tail -f train_run_*.log`
- Metrics CSVs: `tail -n 5 train_metrics*.csv`

## Inference & outputs
- Script: `infer_roi_consistency.py`.
- Batch inference: `--csv nih_roi_test.csv --image_root $DATA_PATH --checkpoint checkpoints/nih_roi_consistency.pt --keep_ratio 0.1 --save_vis_dir roi_vis --save_json preds.json`.
- Single image: `--image $DATA_PATH/images/<file>.png --save_mask mask.png --save_crop crop.png`.
- Outputs: `preds.json` with per-image `prob_full`, `prob_crop`, `prob_drop`, `faithfulness = p_full - p_drop`; optional visualizations in `roi_vis/`.

## Reproduction steps (manual)
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
python - <<'PY'
import kagglehub, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    path = kagglehub.dataset_download("nih-chest-xrays/data")
print(path)
PY
DATA_PATH=<printed path>
# unify images if needed
mkdir -p "$DATA_PATH/images" && find "$DATA_PATH" -maxdepth 1 -type d -name 'images_*' -print0 | while IFS= read -r -d '' d; do find "$d" -type f -name '*.png' -print0 | xargs -0 -I {} ln -sf {} "$DATA_PATH/images/"; done
# build CSVs
python make_chexmultimodal_csv.py --src_csv "$DATA_PATH/Data_Entry_2017.csv" --image_dir "$DATA_PATH/images" --out_csv nih_roi_train.csv --label_from "Finding Labels" --fillna "Normal"
python make_chexmultimodal_csv.py --src_csv "$DATA_PATH/Data_Entry_2017.csv" --image_dir "$DATA_PATH/images" --filter_list "$DATA_PATH/test_list.txt" --out_csv nih_roi_test.csv --label_from "Finding Labels"
# train
python train_roi_consistency.py --train_csv nih_roi_train.csv --image_root "$DATA_PATH" --batch_size 8 --accum_steps 2 --epochs 8 --channels_last --autocast_dtype bf16 --workers 8 --prefetch_factor 6 --persistent_workers --save checkpoints/nih_roi_consistency.pt --img_size 224 --lr 1e-4 --alpha 1.0 --beta 0.5 --gamma 1e-3 --delta 1e-4 --keep_ratio 0.1
# infer
python infer_roi_consistency.py --checkpoint checkpoints/nih_roi_consistency.pt --csv nih_roi_test.csv --image_root "$DATA_PATH" --keep_ratio 0.1 --save_vis_dir roi_vis --save_json preds.json
```

## Reporting notes (for paper)
- Data: NIH ChestX-ray14, official test split (`test_list.txt`), 112,120 train rows / 25,596 test rows after CSV construction.
- Preprocessing: RGB conversion, resize to 224×224, ImageNet mean/std normalization; label strings split on commas, empty labels → "Normal".
- Model: ConvNeXt-Base backbone (ImageNet-pretrained), custom mask net, multi-label BCE head.
- Training: 8 epochs, AdamW lr 1e-4, batch 8, accum 2, bf16 AMP, channels-last, TF32, cuDNN benchmark. Loss weights as above.
- Hardware assumption: single NVIDIA GPU (bf16/TF32 capable), ~12–24 GB VRAM target. DataLoader warnings may suggest fewer workers if constrained.
- Outputs retained: `checkpoints/nih_roi_consistency.pt`, `preds.json`, optional `roi_vis/` masks/crops, logs in `roi-run.log` when using `run_all.sh` under tmux.

## Acknowledgement

This work was supported by the Intramural Research Program of the NIH Clinical Center (clinicalcenter.nih.gov) and National Library of Medicine (www.nlm.nih.gov). We thank NVIDIA Corporation for the GPU donations.
