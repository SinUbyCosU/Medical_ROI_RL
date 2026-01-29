#!/usr/bin/env bash
set -uo pipefail
cd /root/roi-consistency-new
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
DATA_PATH=$(python3 - <<'PY'
import io, contextlib, kagglehub
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
	path = kagglehub.dataset_download("nih-chest-xrays/data")
print(path)
PY
)
echo "DATA_PATH=$DATA_PATH"
# Create unified images directory with symlinks to shard folders if needed
if [ ! -d "$DATA_PATH/images" ]; then
	mkdir -p "$DATA_PATH/images"
	find "$DATA_PATH" -maxdepth 1 -type d -name 'images_*' -print0 | while IFS= read -r -d '' d; do
		find "$d" -type f -name '*.png' -print0 | xargs -0 -I {} ln -sf {} "$DATA_PATH/images/"
	done
fi
python3 make_chexmultimodal_csv.py --src_csv "$DATA_PATH/Data_Entry_2017.csv" --image_dir "$DATA_PATH/images" --out_csv nih_roi_train.csv --label_from "Finding Labels" --fillna "Normal"
python3 make_chexmultimodal_csv.py --src_csv "$DATA_PATH/Data_Entry_2017.csv" --image_dir "$DATA_PATH/images" --filter_list "$DATA_PATH/test_list.txt" --out_csv nih_roi_test.csv --label_from "Finding Labels"
python3 train_roi_consistency.py \
  --train_csv nih_roi_train.csv \
  --image_root "$DATA_PATH" \
  --classes "" \
  --backbone convnext_base \
  --batch_size 8 \
  --epochs 8 \
  --accum_steps 2 \
  --channels_last \
  --autocast_dtype bf16 \
  --workers 8 \
  --prefetch_factor 6 \
  --persistent_workers \
  --save checkpoints/nih_roi_consistency.pt \
  --img_size 224 \
  --lr 1e-4 \
  --alpha 1.0 \
  --beta 0.5 \
  --gamma 1e-3 \
  --delta 1e-4 \
  --keep_ratio 0.1
python3 infer_roi_consistency.py --checkpoint checkpoints/nih_roi_consistency.pt --csv nih_roi_test.csv --image_root "$DATA_PATH" --keep_ratio 0.1 --save_vis_dir roi_vis --save_json preds.json
