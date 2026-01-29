#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision import models

from train_roi_consistency import IMAGENET_MEAN, IMAGENET_STD, build_model, topk_mask


def load_checkpoint(path: Path, device: torch.device):
    # Torch 2.6+ defaults to weights_only=True; the saved file includes non-tensor fields.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    classes: List[str] = ckpt["classes"]
    model, mask_net = build_model(len(classes))
    model.load_state_dict(ckpt["model"])
    mask_net.load_state_dict(ckpt["mask_net"])
    model.to(device).eval()
    mask_net.to(device).eval()
    return model, mask_net, classes, ckpt


def build_transform(img_size: int):
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def to_pil(t: torch.Tensor) -> Image.Image:
    # t: C,H,W tensor in [0,1] roughly
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def infer_image(model, mask_net, image: torch.Tensor, keep_ratio: float):
    with torch.no_grad():
        logits_full = model(image.unsqueeze(0))
        mask = torch.sigmoid(mask_net(image.unsqueeze(0)))
        mask_bin = topk_mask(mask, keep_ratio)
        img_crop = image.unsqueeze(0) * mask_bin
        img_drop = image.unsqueeze(0) * (1.0 - mask_bin)
        logits_crop = model(img_crop)
        logits_drop = model(img_drop)
        return (
            torch.sigmoid(logits_full)[0],
            torch.sigmoid(logits_crop)[0],
            torch.sigmoid(logits_drop)[0],
            mask_bin[0],
        )


def run_csv(args, model, mask_net, classes):
    df = pd.read_csv(args.csv)
    transform = build_transform(args.img_size)
    device = next(model.parameters()).device
    save_dir = Path(args.save_vis_dir) if args.save_vis_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for _, row in df.iterrows():
        img_path = Path(args.image_root) / "images" / row["image"]
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image = transform(img).to(device)
        p_full, p_crop, p_drop, mask = infer_image(model, mask_net, image, args.keep_ratio)
        faithfulness = (p_full - p_drop).cpu().tolist()
        record = {
            "image": row["image"],
            "labels": row["labels"],
            "prob_full": p_full.cpu().tolist(),
            "prob_crop": p_crop.cpu().tolist(),
            "prob_drop": p_drop.cpu().tolist(),
            "faithfulness": faithfulness,
        }
        outputs.append(record)

        if save_dir:
            base = Path(row["image"]).stem
            mask_img = to_pil(mask.repeat(3, 1, 1))
            crop_img = to_pil((image * mask).cpu())
            mask_img.save(save_dir / f"{base}_mask.png")
            crop_img.save(save_dir / f"{base}_crop.png")

    out_json = Path(args.save_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"classes": classes, "outputs": outputs}, indent=2))
    print(f"Saved predictions to {out_json}")


def run_single(args, model, mask_net, classes):
    transform = build_transform(args.img_size)
    device = next(model.parameters()).device
    img_path = Path(args.image)
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        image = transform(img).to(device)

    p_full, p_crop, p_drop, mask = infer_image(model, mask_net, image, args.keep_ratio)
    print("prob_full:", dict(zip(classes, [float(x) for x in p_full.cpu()])))
    print("prob_crop:", dict(zip(classes, [float(x) for x in p_crop.cpu()])))
    print("prob_drop:", dict(zip(classes, [float(x) for x in p_drop.cpu()])))

    if args.save_mask:
        Path(args.save_mask).parent.mkdir(parents=True, exist_ok=True)
        to_pil(mask.repeat(3, 1, 1)).save(args.save_mask)
        print(f"Saved mask to {args.save_mask}")
    if args.save_crop:
        Path(args.save_crop).parent.mkdir(parents=True, exist_ok=True)
        to_pil((image * mask).cpu()).save(args.save_crop)
        print(f"Saved crop to {args.save_crop}")


def main():
    parser = argparse.ArgumentParser(description="Infer ROI consistency model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--image_root", type=Path, help="Root containing images/ when using --csv")
    parser.add_argument("--image", type=Path, help="Single image path")
    parser.add_argument("--keep_ratio", type=float, default=0.1)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--save_vis_dir", type=Path, help="Dir to save mask/crop for CSV mode")
    parser.add_argument("--save_json", type=Path, default=Path("preds.json"))
    parser.add_argument("--save_mask", type=Path, help="Path to save mask for single image")
    parser.add_argument("--save_crop", type=Path, help="Path to save crop for single image")
    args = parser.parse_args()

    if not args.csv and not args.image:
        raise SystemExit("Provide either --csv or --image")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mask_net, classes, ckpt = load_checkpoint(args.checkpoint, device)

    if args.csv:
        run_csv(args, model, mask_net, classes)
    else:
        run_single(args, model, mask_net, classes)


if __name__ == "__main__":
    main()
