# ROI Consistency - NIH ChestX-ray14

Minimal runnable pipeline to download NIH ChestX-ray14 (via KaggleHub), build CSVs, train the ROI-consistency classifier, and run inference/visualization.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Download dataset (KaggleHub)

```python
import kagglehub
path = kagglehub.dataset_download("nih-chest-xrays/data")
print("dataset path:", path)
# expect: path/{images,Data_Entry_2017.csv,train_val_list.txt,test_list.txt,BBox_List_2017.csv}
```

## 2) Build train / test CSVs

```bash
# train CSV
python make_chexmultimodal_csv.py \ 
  --src_csv $PATH_FROM_KAGGLE/Data_Entry_2017.csv \ 
  --image_dir $PATH_FROM_KAGGLE/images \ 
  --out_csv nih_roi_train.csv \ 
  --label_from "Finding Labels" \ 
  --fillna "Normal"

# test CSV using NIH official list
python make_chexmultimodal_csv.py \ 
  --src_csv $PATH_FROM_KAGGLE/Data_Entry_2017.csv \ 
  --image_dir $PATH_FROM_KAGGLE/images \ 
  --filter_list $PATH_FROM_KAGGLE/test_list.txt \ 
  --out_csv nih_roi_test.csv \ 
  --label_from "Finding Labels"
```

## 3) Train ROI-consistency model

```bash
python train_roi_consistency.py \ 
  --train_csv nih_roi_train.csv \ 
  --image_root $PATH_FROM_KAGGLE \ 
  --classes "" \ 
  --backbone convnext_base \ 
  --batch_size 4 \ 
  --epochs 8 \ 
  --accum_steps 4 \ 
  --channels_last \ 
  --autocast_dtype bf16 \ 
  --workers 4 \ 
  --save checkpoints/nih_roi_consistency.pt \ 
  --img_size 224 \ 
  --lr 1e-4 \ 
  --alpha 1.0 \ 
  --beta 0.5 \ 
  --gamma 1e-3 \ 
  --delta 1e-4 \ 
  --keep_ratio 0.1
```

Notes:
- 14 classes auto-inferred if `--classes` is empty.
- Grad accumulation + channels_last + bf16 target 12–24 GB VRAM.

## 4) Evaluate / faithfulness

```bash
python infer_roi_consistency.py \ 
  --checkpoint checkpoints/nih_roi_consistency.pt \ 
  --csv nih_roi_test.csv \ 
  --image_root $PATH_FROM_KAGGLE \ 
  --keep_ratio 0.1 \ 
  --save_vis_dir roi_vis \ 
  --save_json preds.json
```

Outputs: `preds.json` with per-image `prob_full`, `prob_crop`, `prob_drop`, `faithfulness = p_full - p_drop`. When `--save_vis_dir` is set, saves mask and crop PNGs per image.

## 5) Single-image inference

```bash
python infer_roi_consistency.py \ 
  --checkpoint checkpoints/nih_roi_consistency.pt \ 
  --image $PATH_FROM_KAGGLE/images/00000001_000.png \ 
  --keep_ratio 0.1 \ 
  --save_mask mask_1.png \ 
  --save_crop crop_1.png
```

## 6) Labels

Model works with the 14 NIH labels: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia.

## Acknowledgement

This work was supported by the Intramural Research Program of the NIH Clinical Center (clinicalcenter.nih.gov) and National Library of Medicine (www.nlm.nih.gov). We thank NVIDIA Corporation for the GPU donations.
