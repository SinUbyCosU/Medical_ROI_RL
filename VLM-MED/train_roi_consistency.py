import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import timm
from tqdm import tqdm


class ImageCSV(Dataset):
    def __init__(self, csv_path: str, image_root: str, class_names, img_size: int = 224):
        df = pd.read_csv(csv_path)
        if "image" not in df.columns or "labels" not in df.columns:
            raise ValueError("CSV must have columns: image, labels (comma-separated)")
        self.paths = df["image"].tolist()
        self.labels = df["labels"].tolist()
        self.root = image_root
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.img_size = img_size
        self.tx = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.paths[idx])
        img = Image.open(path).convert("RGB")
        y = torch.zeros(len(self.class_names), dtype=torch.float32)
        for cls in str(self.labels[idx]).split(','):
            cls = cls.strip()
            if cls in self.class_to_idx:
                y[self.class_to_idx[cls]] = 1.0
        return self.tx(img), y


class ClsWithFeat(nn.Module):
    def __init__(self, backbone: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        feat_ch = self.backbone.feature_info.channels()[-1]
        self.head = nn.Linear(feat_ch, num_classes)

    def forward(self, x):
        feats = self.backbone(x)[-1]
        pooled = feats.mean(dim=(2, 3))
        logits = self.head(pooled)
        return logits, feats


def compute_saliency(feats: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_logit = (logits * targets).sum()
    grad = torch.autograd.grad(target_logit, feats, retain_graph=True, create_graph=False)[0]
    sal = (grad * feats).mean(dim=1, keepdim=True)
    sal = F.relu(sal)
    sal = sal - sal.amin(dim=(2, 3), keepdim=True)
    sal = sal / (sal.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return sal


def mask_topk(sal: torch.Tensor, keep_ratio: float = 0.1) -> torch.Tensor:
    b, _, h, w = sal.shape
    flat = sal.view(b, -1)
    k = max(1, int(flat.shape[1] * keep_ratio))
    vals, _ = torch.topk(flat, k, dim=1)
    thresh = vals[:, -1].unsqueeze(-1)
    mask = (flat >= thresh).float().view(b, 1, h, w)
    return mask


def mask_to_box(mask: torch.Tensor, pad: int = 4):
    mask_np = mask.squeeze(0).cpu().numpy()
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return [(0, 0, mask_np.shape[1] - 1, mask_np.shape[0] - 1)]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(mask_np.shape[1] - 1, x2 + pad)
    y2 = min(mask_np.shape[0] - 1, y2 + pad)
    return [(x1, y1, x2, y2)]


def crop_and_resize(img: torch.Tensor, box, size: int) -> torch.Tensor:
    c, h, w = img.shape
    x1, y1, x2, y2 = box
    crop = img[:, y1:y2+1, x1:x2+1]
    crop = T.functional.resize(crop, (size, size))
    return crop


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    tv_h = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
    tv_w = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
    return tv_h + tv_w


def infer_classes_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    if "labels" not in df.columns:
        raise ValueError("CSV must have a 'labels' column")
    seen = []
    for lbl in df["labels"].fillna(""):
        parts = [p.strip() for p in str(lbl).split(',') if p.strip()]
        for p in parts:
            if p not in seen:
                seen.append(p)
    if not seen:
        raise ValueError("No labels found to infer classes")
    return seen


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('medium')

    if args.classes:
        class_names = [c.strip() for c in args.classes.split(',') if c.strip()]
    else:
        class_names = infer_classes_from_csv(args.train_csv)

    # Optional: auto-create CSV for CheXpert layout if requested and missing
    if args.auto_make_csv and not os.path.exists(args.train_csv):
        chex_train = os.path.join(args.image_root, "train.csv")
        if not os.path.exists(chex_train):
            raise FileNotFoundError(f"auto_make_csv enabled but {chex_train} not found; set --image_root to CheXpert folder or disable --auto_make_csv")
        df = pd.read_csv(chex_train)
        rows = []
        for _, r in df.iterrows():
            labs = []
            for c in class_names:
                v = r.get(c)
                if pd.notna(v) and v == 1:
                    labs.append(c)
            rows.append({"image": r["Path"], "labels": ",".join(labs)})
        pd.DataFrame(rows).to_csv(args.train_csv, index=False)
        print(f"[auto_make_csv] wrote {len(rows)} rows to {args.train_csv}")

    ds = ImageCSV(args.train_csv, args.image_root, class_names, img_size=args.image_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
    )

    model = ClsWithFeat(args.backbone, num_classes=len(class_names)).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }
    amp_dtype = dtype_map[args.autocast_dtype] if device.type == 'cuda' else torch.float32
    autocast_enabled = device.type == 'cuda' and args.autocast_dtype != 'fp32'
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=autocast_enabled)
    else:
        class _NoopScaler:
            def scale(self, x):
                return x
            def step(self, opt):
                opt.step()
            def update(self):
                return None
            def unscale_(self, opt):
                return None
        scaler = _NoopScaler()

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        opt.zero_grad(set_to_none=True)
        for step, (img, y) in enumerate(pbar):
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.channels_last:
                img = img.to(memory_format=torch.channels_last)

            autocast_ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if autocast_enabled else nullcontext()
            with autocast_ctx:
                logits_full, feats = model(img)
                preds_full = torch.sigmoid(logits_full)
                targets = torch.where(y > 0, y, preds_full.detach())
                sal = compute_saliency(feats, logits_full, targets)
                mask = mask_topk(sal, keep_ratio=args.keep_ratio)
                # Upsample mask to input resolution before masking out the image
                mask_up = F.interpolate(mask, size=img.shape[2:], mode='bilinear', align_corners=False)

                crops = []
                for i in range(img.shape[0]):
                    box = mask_to_box(mask[i])[0]
                    crop = crop_and_resize(img[i], box, args.image_size)
                    crops.append(crop)
                crops = torch.stack(crops, dim=0)
                if args.channels_last:
                    crops = crops.to(memory_format=torch.channels_last)

                logits_crop, _ = model(crops)
                preds_crop = torch.sigmoid(logits_crop)

                mask_for_img = mask_up.expand(-1, img.shape[1], -1, -1)
                img_out = img * (1 - mask_for_img)
                logits_out, _ = model(img_out)
                preds_out = torch.sigmoid(logits_out)

                bce = F.binary_cross_entropy_with_logits(logits_full, y)
                consist = F.binary_cross_entropy_with_logits(logits_crop, y)
                drop = torch.relu(preds_out - preds_full + args.margin).mean()
                sparse = mask.mean()
                smooth = total_variation(mask)

                loss = bce + args.alpha * consist + args.beta * drop + args.gamma * sparse + args.delta * smooth
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": loss.item() * args.accum_steps, "bce": bce.item(), "cons": consist.item()})

        # Flush any remaining gradients if the last step did not align with accum_steps
        if (step + 1) % args.accum_steps != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        if args.save and (epoch + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "classes": class_names,
                "backbone": args.backbone,
            }, args.save)


def parse_args():
    ap = argparse.ArgumentParser(description="ROI-consistency training: Prediction->Mask->Crop->Re-predict")
    ap.add_argument('--train_csv', type=str, required=True, help='CSV with columns: image, labels')
    ap.add_argument('--image_root', type=str, required=True, help='Root directory for images')
    ap.add_argument('--classes', type=str, default='', help='Comma-separated class names; if empty, inferred from labels in CSV')
    ap.add_argument('--backbone', type=str, default='convnext_base', help='timm backbone name')
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--keep_ratio', type=float, default=0.1, help='Top-k saliency fraction to keep')
    ap.add_argument('--margin', type=float, default=0.2, help='Margin for drop loss')
    ap.add_argument('--alpha', type=float, default=1.0, help='Weight for crop consistency')
    ap.add_argument('--beta', type=float, default=0.5, help='Weight for drop loss')
    ap.add_argument('--gamma', type=float, default=1e-3, help='Weight for sparsity')
    ap.add_argument('--delta', type=float, default=1e-4, help='Weight for TV smoothness')
    ap.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
    ap.add_argument('--channels_last', action='store_true', help='Use channels_last memory format for lower memory')
    ap.add_argument('--autocast_dtype', type=str, choices=['fp16', 'bf16', 'fp32'], default='bf16', help='Autocast dtype to control memory/throughput')
    ap.add_argument('--auto_make_csv', action='store_true', help='Auto-create train_csv from CheXpert train.csv using provided classes')
    ap.add_argument('--save', type=str, default='checkpoints/roi_consistency.pt')
    ap.add_argument('--save_every', type=int, default=1)
    ap.add_argument('--cpu', action='store_true', help='Force CPU')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
