#!/usr/bin/env python3
"""Train ROI consistency and baseline variants without touching current run.

Modes:
- full: original ROI-consistency (mask_net + consistency/drop/sparse/TV)
- standard: ConvNeXt classifier only (no mask net, BCE only)
- random_mask: random mask with same keep_ratio (uses consistency/drop/sparse/TV with random mask)
- random_cutout: cutout-style dropout mask, no top-k (uses consistency/drop/sparse/TV with random cutout)
- no_consist: full model but beta=0 (disables consistency loss)
- no_sparse: full model but gamma=0 (disables sparsity)
- no_tv: full model but delta=0 (disables TV)

Logging: per-epoch averages appended to CSV; console logs are tee-friendly.
"""

import argparse
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

autocast_dtypes = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_classes(classes_arg: str, df: pd.DataFrame) -> List[str]:
    if classes_arg.strip():
        return [c.strip() for c in classes_arg.split(",") if c.strip()]
    labels = set()
    for entry in df["labels"]:
        if pd.isna(entry):
            continue
        labels.update([x.strip() for x in str(entry).split(",") if x.strip()])
    return sorted(labels)


class ChexDataset(Dataset):
    def __init__(self, csv_path: Path, image_root: Path, classes: Sequence[str], img_size: int = 224):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / "images" / row["image"]
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image = self.transform(img)
        labels = torch.zeros(len(self.classes), dtype=torch.float32)
        for lbl in str(row["labels"]).split(","):
            lbl = lbl.strip()
            if lbl in self.class_to_idx:
                labels[self.class_to_idx[lbl]] = 1.0
        return image, labels


def build_model(num_classes: int):
    backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    in_features = backbone.classifier[2].in_features
    backbone.classifier[2] = nn.Linear(in_features, num_classes)
    mask_net = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 1, kernel_size=1),
    )
    return backbone, mask_net


def topk_mask(mask: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    b, _, h, w = mask.shape
    flat = mask.view(b, -1)
    k = max(1, int(keep_ratio * h * w))
    vals, _ = torch.topk(flat, k=k, dim=1)
    thresh = vals[:, -1].unsqueeze(1)
    return (flat >= thresh).float().view_as(mask)


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
    dw = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
    return dh + dw


def random_mask_like(images: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    b, _, h, w = images.shape
    noise = torch.rand((b, 1, h, w), device=images.device)
    return topk_mask(noise, keep_ratio)


def random_cutout_mask(images: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    b, _, h, w = images.shape
    mask = torch.ones((b, 1, h, w), device=images.device)
    cut_area = max(1, int((1 - keep_ratio) * h * w))
    side = int(cut_area ** 0.5)
    side = max(1, min(side, h, w))
    for i in range(b):
        top = torch.randint(0, max(1, h - side + 1), (1,), device=images.device).item()
        left = torch.randint(0, max(1, w - side + 1), (1,), device=images.device).item()
        mask[i, :, top : top + side, left : left + side] = 0.0
    return mask


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    df = pd.read_csv(args.train_csv)
    classes = parse_classes(args.classes, df)
    dataset = ChexDataset(args.train_csv, args.image_root, classes, img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        pin_memory_device="cuda" if device.type == "cuda" else "",
        persistent_workers=args.persistent_workers if args.workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    model, mask_net = build_model(len(classes))
    model.to(device)

    # mode-specific loss weights
    alpha = args.alpha
    beta = 0.0 if args.mode == "no_consist" else args.beta
    gamma = 0.0 if args.mode == "no_sparse" else args.gamma
    delta = 0.0 if args.mode == "no_tv" else args.delta

    if args.mode in {"full", "no_consist", "no_sparse", "no_tv"}:
        mask_net.to(device)
        params = list(model.parameters()) + list(mask_net.parameters())
    else:
        params = list(model.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.autocast_dtype != "fp32")
    bce = nn.BCEWithLogitsLoss()
    accum = args.accum_steps

    # metrics log
    if args.log_csv:
        log_path = Path(args.log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():
            log_path.write_text("epoch,steps,loss,loss_full,loss_consist,loss_drop,loss_sparse,loss_tv\n")

    for epoch in range(args.epochs):
        model.train()
        if args.mode in {"full", "no_consist", "no_sparse", "no_tv"}:
            mask_net.train()
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        sum_loss = sum_full = sum_consist = sum_drop = sum_sparse = sum_tv = 0.0
        step_count = 0

        for step, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=autocast_dtypes[args.autocast_dtype]):
                if args.mode == "standard":
                    logits_full = model(images)
                    loss_full = bce(logits_full, targets)
                    loss_consist = torch.tensor(0.0, device=device)
                    loss_drop = torch.tensor(0.0, device=device)
                    loss_sparse = torch.tensor(0.0, device=device)
                    loss_tv = torch.tensor(0.0, device=device)
                    loss = loss_full / accum
                else:
                    # choose mask
                    if args.mode in {"full", "no_consist", "no_sparse", "no_tv"}:
                        mask = torch.sigmoid(mask_net(images))
                        mask_bin = topk_mask(mask, args.keep_ratio)
                    elif args.mode == "random_mask":
                        mask = random_mask_like(images, args.keep_ratio)
                        mask_bin = mask
                    elif args.mode == "random_cutout":
                        mask = random_cutout_mask(images, args.keep_ratio)
                        mask_bin = mask
                    else:
                        raise ValueError(f"Unknown mode: {args.mode}")

                    img_crop = images * mask_bin
                    img_drop = images * (1.0 - mask_bin)

                    logits_full = model(images)
                    logits_crop = model(img_crop)
                    logits_drop = model(img_drop)

                    prob_full = torch.sigmoid(logits_full)
                    prob_crop = torch.sigmoid(logits_crop)
                    prob_drop = torch.sigmoid(logits_drop)

                    loss_full = bce(logits_full, targets)
                    loss_consist = F.l1_loss(prob_full, prob_crop)
                    loss_drop = F.relu(prob_drop - prob_full.detach()).mean()
                    loss_sparse = mask.mean()
                    loss_tv = total_variation(mask)

                    loss = (
                        loss_full
                        + alpha * loss_consist
                        + beta * loss_drop
                        + gamma * loss_sparse
                        + delta * loss_tv
                    )
                    loss = loss / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            effective_loss = float(loss.item() * accum)
            sum_loss += effective_loss
            sum_full += float(loss_full.item())
            sum_consist += float(loss_consist.item())
            sum_drop += float(loss_drop.item())
            sum_sparse += float(loss_sparse.item())
            sum_tv += float(loss_tv.item())
            step_count += 1

            pbar.set_postfix({"loss": effective_loss})

        # save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "mask_net": mask_net.state_dict() if args.mode in {"full", "no_consist", "no_sparse", "no_tv"} else None,
            "classes": classes,
            "args": vars(args),
            "epoch": epoch + 1,
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.save)
        print(f"Saved checkpoint to {args.save}")

        if args.log_csv and step_count:
            avg_loss = sum_loss / step_count
            avg_full = sum_full / step_count
            avg_consist = sum_consist / step_count
            avg_drop = sum_drop / step_count
            avg_sparse = sum_sparse / step_count
            avg_tv = sum_tv / step_count
            with Path(args.log_csv).open("a") as f:
                f.write(
                    f"{epoch+1},{step_count},{avg_loss},{avg_full},{avg_consist},{avg_drop},{avg_sparse},{avg_tv}\n"
                )
            print(
                f"epoch {epoch+1}: loss={avg_loss:.4f} full={avg_full:.4f} "
                f"consist={avg_consist:.4f} drop={avg_drop:.4f} "
                f"sparse={avg_sparse:.4f} tv={avg_tv:.4f}"
            )


def build_args():
    parser = argparse.ArgumentParser(description="Train ROI consistency baselines on NIH ChestX-ray14")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--classes", default="", help="Comma-separated class list; empty infers from CSV")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--autocast_dtype", choices=list(autocast_dtypes.keys()), default="bf16")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--save", type=Path, default=Path("checkpoints/nih_roi_consistency.pt"))
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--keep_ratio", type=float, default=0.1)
    parser.add_argument("--log_csv", type=str, default="train_metrics.csv")
    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "standard",
            "random_mask",
            "random_cutout",
            "no_consist",
            "no_sparse",
            "no_tv",
        ],
        default="full",
        help="Baseline variant",
    )
    return parser


if __name__ == "__main__":
    args = build_args().parse_args()
    if args.channels_last and hasattr(torch, "set_default_memory_format"):
        torch.set_default_memory_format(torch.channels_last)
    train(args)
