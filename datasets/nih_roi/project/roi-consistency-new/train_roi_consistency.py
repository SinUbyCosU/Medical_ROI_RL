#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models


class MaskGeneratorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

autocast_dtypes = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def sample_stochastic_mask(logits: torch.Tensor, temperature: float, hard: bool, use_gumbel: bool) -> torch.Tensor:
    if use_gumbel:
        eps = 1e-8
        uniform = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(uniform + eps) + eps)
        noisy_logits = (logits + gumbel) / max(temperature, eps)
        mask_prob = torch.sigmoid(noisy_logits)
    else:
        mask_prob = torch.sigmoid(logits / max(temperature, 1e-8))

    if hard:
        mask_hard = (mask_prob >= 0.5).float()
        mask = mask_hard + (mask_prob - mask_hard).detach()
    else:
        mask = mask_prob
    return mask


def pick_anchor_params(model: nn.Module):
    # Use classifier weight as representative params; fallback to first param.
    for name, param in model.named_parameters():
        if "classifier" in name and param.requires_grad:
            return [param]
    for param in model.parameters():
        if param.requires_grad:
            return [param]
    return []


def grad_norm(loss: torch.Tensor, params) -> torch.Tensor:
    if not params:
        return torch.tensor(0.0, device=loss.device)
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = torch.tensor(0.0, device=loss.device)
    for g in grads:
        if g is not None:
            total = total + g.norm(2)
    return total


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
    mask_net = MaskGeneratorNet()
    return backbone, mask_net


def random_mask_like(images: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Generate a random binary mask with the same keep_ratio."""
    b, _, h, w = images.shape
    noise = torch.rand((b, 1, h, w), device=images.device)
    return topk_mask(noise, keep_ratio)


def random_cutout_mask(images: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Generate a cutout-style binary mask (1=keep, 0=cut) with area ~ keep_ratio."""
    b, _, h, w = images.shape
    mask = torch.ones((b, 1, h, w), device=images.device)
    cut_area = max(1, int((1 - keep_ratio) * h * w))
    side = int((cut_area) ** 0.5)
    side = max(1, min(side, h, w))
    for i in range(b):
        top = torch.randint(0, max(1, h - side + 1), (1,), device=images.device).item()
        left = torch.randint(0, max(1, w - side + 1), (1,), device=images.device).item()
        mask[i, :, top : top + side, left : left + side] = 0.0
    return mask


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
    mask_net.to(device)
    params = list(model.parameters()) + list(mask_net.parameters())

    use_dynamic = args.dynamic_weights
    w_full = nn.Parameter(torch.tensor(args.w_full_init, device=device), requires_grad=False)
    w_consist = nn.Parameter(torch.tensor(args.w_consist_init, device=device), requires_grad=False)
    w_drop = nn.Parameter(torch.tensor(args.w_drop_init, device=device), requires_grad=False)
    w_sparse = nn.Parameter(torch.tensor(args.w_sparse_init, device=device), requires_grad=False)
    w_tv = nn.Parameter(torch.tensor(args.w_tv_init, device=device), requires_grad=False)
    dynamic_weight_params = [w_full, w_consist, w_drop, w_sparse, w_tv]

    if use_dynamic:
        params += dynamic_weight_params

    anchor_params = list(model.parameters()) + list(mask_net.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.autocast_dtype != "fp32")
    bce = nn.BCEWithLogitsLoss()
    accum = args.accum_steps

    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            load_kwargs = {"map_location": device}
            try:
                checkpoint = torch.load(resume_path, weights_only=False, **load_kwargs)
            except TypeError:  # torch<2.6 compatibility
                checkpoint = torch.load(resume_path, **load_kwargs)
            model.load_state_dict(checkpoint["model"])
            mask_state = checkpoint.get("mask_net")
            if mask_state is not None:
                missing = mask_net.load_state_dict(mask_state, strict=False)
                if missing.missing_keys or missing.unexpected_keys:
                    print(
                        "Warning: mask_net checkpoint keys mismatched; proceeding with available weights."
                    )
            weight_state = checkpoint.get("weights")
            if use_dynamic and weight_state:
                with torch.no_grad():
                    w_full.data.fill_(weight_state.get("w_full", float(w_full.item())))
                    w_consist.data.fill_(weight_state.get("w_consist", float(w_consist.item())))
                    w_drop.data.fill_(weight_state.get("w_drop", float(w_drop.item())))
                    w_sparse.data.fill_(weight_state.get("w_sparse", float(w_sparse.item())))
                    w_tv.data.fill_(weight_state.get("w_tv", float(w_tv.item())))
            opt_state = checkpoint.get("optimizer")
            if opt_state:
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception as exc:  # pragma: no cover - resume fallback
                    print(f"Warning: failed to load optimizer state ({exc}); continuing with fresh optimizer.")
            start_epoch = int(checkpoint.get("epoch", 0))
            ckpt_classes = checkpoint.get("classes")
            if ckpt_classes and list(ckpt_classes) != list(classes):
                print("Warning: checkpoint classes differ from current CSV; continuing regardless.")
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Warning: resume path {resume_path} not found; starting from scratch.")

    # metrics log setup (includes weight snapshots and mask stats)
    if args.log_csv:
        log_path = Path(args.log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():
            log_path.write_text(
                "epoch,steps,loss,loss_full,loss_consist,loss_drop,loss_sparse,loss_tv,loss_volume,"
                "w_full,w_consist,w_drop,w_sparse,w_tv,mask_mean,mask_tv\n"
            )

    current_temp = args.mask_temperature

    for epoch in range(start_epoch, args.epochs):
        model.train()
        mask_net.train()
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        # running sums for averages per epoch
        sum_loss = sum_full = sum_consist = sum_drop = sum_sparse = sum_tv = sum_volume = 0.0
        sum_w_full = sum_w_consist = sum_w_drop = sum_w_sparse = sum_w_tv = 0.0
        sum_mask_mean = sum_mask_tv = 0.0
        step_count = 0

        for step, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=autocast_dtypes[args.autocast_dtype]):
                logits_full = model(images)
                mask_logits = mask_net(images)
                mask_logits = mask_logits.to(dtype=torch.float32)
                mask = sample_stochastic_mask(
                    mask_logits,
                    temperature=current_temp,
                    hard=args.mask_hard,
                    use_gumbel=args.mask_gumbel,
                ).to(images.dtype)
                if args.mask_project:
                    mask_hard = topk_mask(mask, args.keep_ratio)
                    mask = mask_hard + (mask - mask_hard).detach()
                mask_mean_val = mask.mean()
                mask_tv_val = total_variation(mask)
                loss_volume = (mask_mean_val - args.mask_volume_target) ** 2

                img_crop = images * mask
                img_drop = images * (1.0 - mask)

                logits_crop = model(img_crop)
                logits_drop = model(img_drop)

                prob_full = torch.sigmoid(logits_full)
                prob_crop = torch.sigmoid(logits_crop)
                prob_drop = torch.sigmoid(logits_drop)

                loss_full = bce(logits_full, targets)
                loss_consist = F.l1_loss(prob_full, prob_crop)
                loss_drop = F.relu(prob_drop - prob_full.detach()).mean()
                loss_sparse = mask_mean_val
                loss_tv = mask_tv_val

                if use_dynamic:
                    eps = 1e-6
                    g_full = grad_norm(loss_full, anchor_params) + eps
                    g_consist = grad_norm(loss_consist, anchor_params) + eps
                    g_drop = grad_norm(loss_drop, anchor_params) + eps
                    g_sparse = grad_norm(loss_sparse, anchor_params) + eps
                    g_tv = grad_norm(loss_tv, anchor_params) + eps

                    g_target = (g_full + g_consist + g_drop + g_sparse + g_tv) / 5.0

                    ratio_full = (g_full / g_target).detach()
                    ratio_consist = (g_consist / g_target).detach()
                    ratio_drop = (g_drop / g_target).detach()
                    ratio_sparse = (g_sparse / g_target).detach()
                    ratio_tv = (g_tv / g_target).detach()

                    with torch.no_grad():
                        # Keep weights in a useful band so auxiliary terms continue to learn.
                        w_full.mul_(ratio_full).clamp_(min=args.dynamic_weight_min, max=args.dynamic_weight_max)
                        w_consist.mul_(ratio_consist).clamp_(min=args.dynamic_weight_min, max=args.dynamic_weight_max)
                        w_drop.mul_(ratio_drop).clamp_(min=args.dynamic_weight_min, max=args.dynamic_weight_max)
                        w_sparse.mul_(ratio_sparse).clamp_(min=args.dynamic_weight_min, max=args.dynamic_weight_max)
                        w_tv.mul_(ratio_tv).clamp_(min=args.dynamic_weight_min, max=args.dynamic_weight_max)

                    loss = (
                        w_full * loss_full
                        + w_consist * loss_consist
                        + w_drop * loss_drop
                        + w_sparse * loss_sparse
                        + w_tv * loss_tv
                    )
                else:
                    loss = (
                        loss_full
                        + args.alpha * loss_consist
                        + args.beta * loss_drop
                        + args.gamma * loss_sparse
                        + args.delta * loss_tv
                    )

                loss = loss + args.mask_volume_lambda * loss_volume

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
            sum_volume += float(loss_volume.item())

            # track weights and mask stats (mean and TV)
            sum_w_full += float(w_full.item())
            sum_w_consist += float(w_consist.item())
            sum_w_drop += float(w_drop.item())
            sum_w_sparse += float(w_sparse.item())
            sum_w_tv += float(w_tv.item())
            sum_mask_mean += float(mask_mean_val.item())
            sum_mask_tv += float(mask_tv_val.item())

            step_count += 1

            pbar.set_postfix({"loss": effective_loss})

        ckpt = {
            "model": model.state_dict(),
            "mask_net": mask_net.state_dict(),
            "classes": classes,
            "args": vars(args),
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
        }
        if use_dynamic:
            ckpt["weights"] = {
                "w_full": float(w_full.item()),
                "w_consist": float(w_consist.item()),
                "w_drop": float(w_drop.item()),
                "w_sparse": float(w_sparse.item()),
                "w_tv": float(w_tv.item()),
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
            avg_volume = sum_volume / step_count
            avg_w_full = sum_w_full / step_count
            avg_w_consist = sum_w_consist / step_count
            avg_w_drop = sum_w_drop / step_count
            avg_w_sparse = sum_w_sparse / step_count
            avg_w_tv = sum_w_tv / step_count
            avg_mask_mean = sum_mask_mean / step_count
            avg_mask_tv = sum_mask_tv / step_count
            with Path(args.log_csv).open("a") as f:
                f.write(
                    f"{epoch+1},{step_count},{avg_loss},{avg_full},{avg_consist},{avg_drop},{avg_sparse},{avg_tv},{avg_volume},"
                    f"{avg_w_full},{avg_w_consist},{avg_w_drop},{avg_w_sparse},{avg_w_tv},{avg_mask_mean},{avg_mask_tv}\n"
                )
            print(
                f"epoch {epoch+1}: loss={avg_loss:.4f} full={avg_full:.4f} "
                f"consist={avg_consist:.4f} drop={avg_drop:.4f} "
                f"sparse={avg_sparse:.4f} tv={avg_tv:.4f} volume={avg_volume:.4f} "
                f"w=[{avg_w_full:.3f},{avg_w_consist:.3f},{avg_w_drop:.3f},{avg_w_sparse:.3f},{avg_w_tv:.3f}] "
                f"mask_mean={avg_mask_mean:.4f} mask_tv={avg_mask_tv:.4f} temp={current_temp:.3f}"
            )

        current_temp = max(args.mask_temperature_min, current_temp * args.mask_temperature_decay)


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
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True, help="Keep dataloader workers alive between epochs")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--save", type=Path, default=Path("checkpoints/nih_roi_consistency.pt"))
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint path to resume training")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--keep_ratio", type=float, default=0.1, help="Fraction of pixels kept in mask")
    parser.add_argument("--log_csv", type=str, default="train_metrics.csv", help="Path to append training metrics CSV")
    parser.add_argument("--dynamic_weights", action="store_true", help="Enable GradNorm-style dynamic loss weights")
    parser.add_argument("--w_full_init", type=float, default=1.0)
    parser.add_argument("--w_consist_init", type=float, default=0.5)
    parser.add_argument("--w_drop_init", type=float, default=1e-3)
    parser.add_argument("--w_sparse_init", type=float, default=1e-4)
    parser.add_argument("--w_tv_init", type=float, default=1e-5)
    parser.add_argument("--dynamic_weight_min", type=float, default=1e-3, help="Minimum dynamic weight value")
    parser.add_argument("--dynamic_weight_max", type=float, default=1e3, help="Maximum dynamic weight value")
    parser.add_argument("--mask_volume_target", type=float, default=0.1, help="Desired average mask value")
    parser.add_argument("--mask_volume_lambda", type=float, default=5.0, help="Weight for mask volume penalty")
    parser.add_argument("--mask_temperature", type=float, default=0.5, help="Temperature for mask sampling")
    parser.add_argument("--mask_temperature_min", type=float, default=0.2, help="Minimum temperature for annealing")
    parser.add_argument("--mask_temperature_decay", type=float, default=0.99, help="Per-epoch temperature decay factor")
    parser.add_argument("--mask_hard", action="store_true", help="Use straight-through hard binarization of mask")
    parser.add_argument("--mask_gumbel", action="store_true", help="Enable Gumbel noise for stochastic mask sampling")
    parser.add_argument("--mask_project", action="store_true", help="Project sampled mask to keep-ratio via top-k")
    return parser


if __name__ == "main__":
    # typo guard
    raise SystemExit("Use python train_roi_consistency.py --help")


if __name__ == "__main__":
    args = build_args().parse_args()
    if args.channels_last and hasattr(torch, "set_default_memory_format"):
        torch.set_default_memory_format(torch.channels_last)
    train(args)
