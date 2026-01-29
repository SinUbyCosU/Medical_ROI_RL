import argparse
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm

from train_roi_consistency import mask_topk, mask_to_box, crop_and_resize, compute_saliency


class ClsWithFeat(nn.Module):
    def __init__(self, backbone: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, features_only=True)
        feat_ch = self.backbone.feature_info.channels()[-1]
        self.head = nn.Linear(feat_ch, num_classes)

    def forward(self, x):
        feats = self.backbone(x)[-1]
        pooled = feats.mean(dim=(2, 3))
        logits = self.head(pooled)
        return logits, feats


def load_image(path: str, size: int = 224) -> torch.Tensor:
    tx = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])
    img = Image.open(path).convert('RGB')
    return tx(img)


def infer(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    class_names = ckpt.get('classes')
    backbone = ckpt.get('backbone', args.backbone)
    model = ClsWithFeat(backbone, num_classes=len(class_names))
    model.load_state_dict(ckpt['model'])
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.to(device).eval()

    img = load_image(args.image, size=args.image_size).unsqueeze(0).to(device)
    img.requires_grad_(True)

    logits_full, feats = model(img)
    preds_full = torch.sigmoid(logits_full)
    targets = (preds_full > args.target_thresh).float()
    if targets.sum() == 0:
        targets = preds_full

    sal = compute_saliency(feats, logits_full, targets)
    mask = mask_topk(sal, keep_ratio=args.keep_ratio)
    box = mask_to_box(mask[0])[0]
    crop = crop_and_resize(img[0], box, args.image_size).unsqueeze(0)

    logits_crop, _ = model(crop.to(device))
    preds_crop = torch.sigmoid(logits_crop)

    print("Classes:", class_names)
    print("Full-image probs:", preds_full.detach().cpu().numpy())
    print("Crop probs:", preds_crop.detach().cpu().numpy())
    print("Box (x1,y1,x2,y2):", box)

    if args.save_mask:
        mask_np = (mask[0,0].detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask_np).save(args.save_mask)
    if args.save_crop:
        crop_img = crop[0].detach().cpu()
        crop_img = (crop_img * 0.25 + 0.5).clamp(0,1)
        T.ToPILImage()(crop_img).save(args.save_crop)


def parse_args():
    ap = argparse.ArgumentParser(description='Inference for ROI-consistency model')
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--backbone', type=str, default='convnext_base')
    ap.add_argument('--keep_ratio', type=float, default=0.1)
    ap.add_argument('--target_thresh', type=float, default=0.5)
    ap.add_argument('--save_mask', type=str, default=None)
    ap.add_argument('--save_crop', type=str, default=None)
    ap.add_argument('--cpu', action='store_true')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    infer(args)
