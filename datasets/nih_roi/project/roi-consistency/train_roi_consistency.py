#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm
from torch.distributions import Bernoulli, kl_divergence


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
    # infer from CSV labels
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
    # mask: [B,1,H,W] in [0,1]
    b, _, h, w = mask.shape
    flat = mask.view(b, -1)
    k = max(1, int(keep_ratio * h * w))
    vals, _ = torch.topk(flat, k=k, dim=1)
    thresh = vals[:, -1].unsqueeze(1)
    binary = (flat >= thresh).float().view_as(mask)
    return binary


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
    dw = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
    return dh + dw


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.train_csv)
    classes = parse_classes(args.classes, df)
    dataset = ChexDataset(args.train_csv, args.image_root, classes, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    model, mask_net = build_model(len(classes))
    model.to(device)
    mask_net.to(device)

    mask_net_old = copy.deepcopy(mask_net)
    mask_net_old.to(device)
    mask_net_old.eval()
    for param in mask_net_old.parameters():
        param.requires_grad = False

    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.lr_model)
    optimizer_mask = torch.optim.AdamW(mask_net.parameters(), lr=args.lr_mask)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.autocast_dtype != "fp32")
    bce = nn.BCEWithLogitsLoss()
    accum = args.accum_steps
    ema_reward = None

    for epoch in range(args.epochs):
        model.train()
        mask_net.train()
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        optimizer_model.zero_grad(set_to_none=True)
        optimizer_mask.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=autocast_dtypes[args.autocast_dtype]):
                logits_full = model(images)
                mask_logits = mask_net(images)
                mask_probs = torch.sigmoid(mask_logits)
                mask_dist = Bernoulli(logits=mask_logits)
                mask_sample = mask_dist.sample()
                mask_sample = mask_sample.to(images.dtype)

                img_crop = images * mask_sample
                img_drop = images * (1.0 - mask_sample)

                logits_crop = model(img_crop)
                logits_drop = model(img_drop)

                prob_full = torch.sigmoid(logits_full)
                prob_crop = torch.sigmoid(logits_crop)
                prob_drop = torch.sigmoid(logits_drop)

                loss_full = bce(logits_full, targets)
                loss_consist = F.l1_loss(prob_full, prob_crop)
                loss_drop = F.relu(prob_drop - prob_full.detach()).mean()
                loss_sparse = mask_probs.mean()
                loss_tv = total_variation(mask_probs)

                reward = (
                    args.alpha * (-loss_drop)
                    + args.beta * (-loss_consist)
                    + args.gamma * (-loss_full)
                    + args.delta * (-loss_sparse)
                    + args.eta * (-loss_tv)
                )
                reward = (reward - reward.mean()) / (reward.std() + 1e-6)
                reward_mean = reward.mean().detach()
                if ema_reward is None:
                    ema_reward = reward_mean
                else:
                    ema_reward = args.ema_decay * ema_reward + (1 - args.ema_decay) * reward_mean
                advantage = (reward - ema_reward).detach()

                with torch.no_grad():
                    mask_logits_old = mask_net_old(images)
                    dist_old = Bernoulli(logits=mask_logits_old)
                logp_old = dist_old.log_prob(mask_sample).sum(dim=[1, 2, 3])
                logp = mask_dist.log_prob(mask_sample).sum(dim=[1, 2, 3])
                ratio = torch.exp(logp - logp_old.detach())
                clip_ratio = ratio.clamp(1 - args.ppo_eps, 1 + args.ppo_eps)
                loss_ppo = -torch.min(ratio * advantage, clip_ratio * advantage).mean()
                kl = kl_divergence(dist_old, mask_dist).sum(dim=[1, 2, 3]).mean()
                mask_entropy = mask_dist.entropy().sum(dim=[1, 2, 3]).mean()
                mask_mean = mask_sample.mean()

                loss_supervised = (
                    loss_full
                    + args.alpha * loss_consist
                    + args.beta * loss_drop
                    + args.gamma * loss_sparse
                    + args.delta * loss_tv
                )
                loss_mask = loss_supervised + args.lambda_ppo * loss_ppo + args.lambda_kl * kl
                loss_mask = loss_mask / accum

            scaler.scale(loss_mask).backward()

            if (step + 1) % accum == 0:
                for optim in (optimizer_model, optimizer_mask):
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(mask_net.parameters(), args.max_grad_norm)
                scaler.step(optimizer_model)
                scaler.step(optimizer_mask)
                scaler.update()
                optimizer_model.zero_grad(set_to_none=True)
                optimizer_mask.zero_grad(set_to_none=True)
                with torch.no_grad():
                    mask_net_old.load_state_dict(mask_net.state_dict())

            clip_frac = (torch.abs(ratio - 1.0) > args.ppo_eps).float().mean()
            pbar.set_postfix(
                {
                    "loss": float(loss_supervised.item()),
                    "reward": float(reward.mean().item()),
                    "adv": float(advantage.mean().item()),
                    "ratio": float(ratio.mean().item()),
                    "clip": float(clip_frac.item()),
                    "mask_mean": float(mask_mean.item()),
                    "entropy": float(mask_entropy.item()),
                }
            )

        ckpt = {
            "model": model.state_dict(),
            "mask_net": mask_net.state_dict(),
            "classes": classes,
            "args": vars(args),
            "epoch": epoch + 1,
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.save)
        print(f"Saved checkpoint to {args.save}")


def build_args():
    parser = argparse.ArgumentParser(description="Train ROI consistency model on NIH ChestX-ray14")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--classes", default="", help="Comma-separated class list; empty infers from CSV")
    parser.add_argument("--backbone", default="convnext_base")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--autocast_dtype", choices=list(autocast_dtypes.keys()), default="bf16")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save", type=Path, default=Path("checkpoints/nih_roi_consistency.pt"))
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--keep_ratio", type=float, default=0.1, help="Fraction of pixels kept in mask")
    return parser


if __name__ == "main__":
    # typo guard
    raise SystemExit("Use python train_roi_consistency.py --help")


if __name__ == "__main__":
    args = build_args().parse_args()
    if args.channels_last:
        torch.set_default_memory_format(torch.channels_last)
    train(args)
