"""
Segmentation Validation Script — v2
Matches train_v2.py exactly:
  - FPNDecoder with ResidualBlock (stochastic depth)
  - 11 classes (Background/Trees/LushBushes/DryGrass/DryBushes/GroundClutter/Flowers/Logs/Rocks/Landscape/Sky)
  - Intermediate feature extraction via hooks at blocks [2, 5, 8, 11]
  - Loads v2 checkpoint format: {model, backbone_partial, feat_indices, config}
  - Optional TTA (horizontal flip)
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.switch_backend('Agg')


# ============================================================================
# Class Definitions  (must match train_v2.py exactly)
# ============================================================================

VALUE_MAP = {
    0:     0,
    100:   1,
    200:   2,
    300:   3,
    500:   4,
    550:   5,
    600:   6,
    700:   7,
    800:   8,
    7100:  9,
    10000: 10,
}
N_CLASSES = len(VALUE_MAP)  # 11

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

# 11 visually distinct colours (RGB)
COLOR_PALETTE = np.array([
    [0,   0,   0  ],  # Background     — black
    [34,  139, 34 ],  # Trees          — forest green
    [0,   255, 0  ],  # Lush Bushes    — lime
    [210, 180, 140],  # Dry Grass      — tan
    [139, 90,  43 ],  # Dry Bushes     — brown
    [128, 128, 0  ],  # Ground Clutter — olive
    [255, 105, 180],  # Flowers        — hot pink
    [139, 69,  19 ],  # Logs           — saddle brown
    [128, 128, 128],  # Rocks          — gray
    [160, 82,  45 ],  # Landscape      — sienna
    [135, 206, 235],  # Sky            — sky blue
], dtype=np.uint8)


def convert_mask(mask: Image.Image) -> Image.Image:
    arr     = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls_id in VALUE_MAP.items():
        new_arr[arr == raw] = cls_id
    return Image.fromarray(new_arr)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """(H, W) class-id array → (H, W, 3) RGB."""
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        color[mask == c] = COLOR_PALETTE[c]
    return color


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, size):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.size      = size

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        name  = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask  = Image.open(os.path.join(self.masks_dir, name))
        mask  = convert_mask(mask)

        image = TF.resize(image, self.size, interpolation=Image.BILINEAR)
        mask  = TF.resize(mask,  self.size, interpolation=Image.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask, name


# ============================================================================
# Model — must mirror train_v2.py exactly
# ============================================================================

class ResidualBlock(nn.Module):
    """ConvNeXt-style block with stochastic depth (drop_path disabled at eval)."""
    def __init__(self, channels, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.dw        = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm      = nn.GroupNorm(8, channels)
        self.pw1       = nn.Conv2d(channels, channels * 4, 1)
        self.pw2       = nn.Conv2d(channels * 4, channels, 1)
        self.drop      = nn.Dropout2d(dropout)
        self.act       = nn.GELU()
        self.drop_path = drop_path

    def forward(self, x):
        residual = x
        h = self.dw(x)
        h = self.norm(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.pw2(h)
        if self.training and self.drop_path > 0:
            keep_prob = 1 - self.drop_path
            mask = torch.rand(h.shape[0], 1, 1, 1, device=h.device) < keep_prob
            h = h * mask.float() / keep_prob
        return h + residual


class FPNDecoder(nn.Module):
    def __init__(self, embed_dim, num_classes, token_h, token_w, num_blocks=12):
        super().__init__()
        self.H, self.W   = token_h, token_w
        self.num_blocks  = num_blocks
        self.feat_indices = [2, 5, 8, 11]

        hidden = 256
        fpn_ch = 128

        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, fpn_ch, 1),
                nn.GroupNorm(8, fpn_ch),
                nn.GELU()
            ) for _ in self.feat_indices
        ])

        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1),
                nn.GroupNorm(8, fpn_ch),
                nn.GELU()
            ) for _ in range(len(self.feat_indices) - 1)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(fpn_ch, hidden, 1),
            nn.GroupNorm(16, hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden, dropout=0.1, drop_path=0.1),
            ResidualBlock(hidden, dropout=0.1, drop_path=0.1),
            ResidualBlock(hidden, dropout=0.05, drop_path=0.05),
        )
        self.classifier = nn.Conv2d(hidden, num_classes, 1)

        self.aux_head = nn.Sequential(
            nn.Conv2d(fpn_ch, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, num_classes, 1)
        )

    def tokens_to_spatial(self, tokens):
        B, N, C = tokens.shape
        return tokens.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

    def forward(self, intermediate_features):
        fmaps = [
            self.laterals[i](self.tokens_to_spatial(intermediate_features[i]))
            for i in range(len(self.feat_indices))
        ]
        top = fmaps[-1]
        for i in range(len(fmaps) - 2, -1, -1):
            top = F.interpolate(top, size=fmaps[i].shape[2:], mode='bilinear', align_corners=False)
            top = self.td_convs[i](top + fmaps[i])

        aux_out  = self.aux_head(fmaps[1])
        x        = self.proj(top)
        x        = self.blocks(x)
        main_out = self.classifier(x)
        return main_out, aux_out


# ============================================================================
# Intermediate feature extraction (identical to train_v2.py)
# ============================================================================

def extract_intermediate_features(backbone, imgs, feat_indices, expected_n_patches=None):
    captured = {}

    def make_hook(idx):
        def hook(module, input, output):
            captured[idx] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = []
    for idx in feat_indices:
        h = backbone.blocks[idx].register_forward_hook(make_hook(idx))
        handles.append(h)

    with torch.set_grad_enabled(torch.is_grad_enabled()):
        _ = backbone.forward_features(imgs)

    for h in handles:
        h.remove()

    if expected_n_patches is None:
        expected_n_patches = backbone.patch_embed.num_patches

    feats = []
    for idx in feat_indices:
        f = captured[idx]
        patch_tokens = f[:, 1 : 1 + expected_n_patches, :]
        feats.append(patch_tokens)
    return feats


# ============================================================================
# TTA (matches train_v2.py)
# ============================================================================

@torch.no_grad()
def predict_with_tta(imgs, backbone, model, feat_indices):
    preds = []
    for flip in [False, True]:
        x = imgs
        if flip:
            x = torch.flip(x, dims=[3])
        feats     = extract_intermediate_features(backbone, x, feat_indices)
        logits, _ = model(feats)
        out = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
        if flip:
            out = torch.flip(out, dims=[3])
        preds.append(out)
    return torch.mean(torch.stack(preds), dim=0)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou_per_class(pred_logits, target, num_classes=N_CLASSES):
    pred = torch.argmax(pred_logits, dim=1).view(-1)
    tgt  = target.view(-1)
    ious = []
    for c in range(num_classes):
        p            = pred == c
        t            = tgt  == c
        intersection = (p & t).sum().float()
        union        = (p | t).sum().float()
        ious.append(float('nan') if union == 0 else (intersection / union).item())
    return np.array(ious)


def compute_mean_iou(pred_logits, target, num_classes=N_CLASSES):
    return float(np.nanmean(compute_iou_per_class(pred_logits, target, num_classes)))


def compute_dice(pred_logits, target, num_classes=N_CLASSES, smooth=1e-6):
    pred   = torch.argmax(pred_logits, dim=1).view(-1)
    tgt    = target.view(-1)
    scores = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (tgt  == c).float()
        scores.append(((2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)).item())
    return float(np.mean(scores))


def compute_pixel_accuracy(pred_logits, target):
    return (torch.argmax(pred_logits, 1) == target).float().mean().item()


# ============================================================================
# Visualisation
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    img  = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.clip(np.moveaxis(img, 0, -1) * std + mean, 0, 1)

    gt_color   = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);       axes[0].set_title('Input Image');   axes[0].axis('off')
    axes[1].imshow(gt_color);  axes[1].set_title('Ground Truth');  axes[1].axis('off')
    axes[2].imshow(pred_color);axes[2].set_title('Prediction');    axes[2].axis('off')
    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(mean_iou, mean_dice, mean_acc, per_class_iou, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(txt_path, 'w') as f:
        f.write("EVALUATION RESULTS (v2)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:        {mean_iou:.4f}\n")
        f.write(f"Mean Dice:       {mean_dice:.4f}\n")
        f.write(f"Pixel Accuracy:  {mean_acc:.4f}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(CLASS_NAMES, per_class_iou):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A (absent)"
            f.write(f"  {name:<20}: {iou_str}\n")
    print(f"\nMetrics saved → {txt_path}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ['#e74c3c' if v < 0.4 else '#f39c12' if v < 0.6 else '#2ecc71'
              for v in np.nan_to_num(per_class_iou)]
    bars = ax.bar(CLASS_NAMES, np.nan_to_num(per_class_iou), color=colors)
    ax.axhline(mean_iou, color='navy', linestyle='--',
               label=f'Mean IoU = {mean_iou:.3f}')
    ax.set(title='Per-Class IoU (v2)', ylabel='IoU', ylim=(0, 1))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=9)
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    for bar, v in zip(bars, per_class_iou):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.2f}' if not np.isnan(v) else 'N/A',
                ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved → {os.path.join(output_dir, 'per_class_iou.png')}")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Segmentation validation — v2')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'segmentation_head_best_v2.pth'),
                        help='Path to v2 checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val'),
                        help='Path to validation dataset (expects Color_Images/ and Segmentation/ sub-dirs)')
    parser.add_argument('--output_dir', type=str, default='./predictions_v2',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of side-by-side comparisons to save')
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation (horizontal flip)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP (use if running on CPU)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and not args.no_amp
    print(f"Device: {device}  |  AMP: {use_amp}  |  TTA: {args.tta}")

    # ── Image size (must match train_v2.py)
    H = int(((540 // 2) // 14) * 14)   # 378
    W = int(((960 // 2) // 14) * 14)   # 476
    SIZE = (H, W)

    # ── Dataset
    print(f"Loading dataset from {args.data_dir} …")
    valset     = MaskDataset(args.data_dir, SIZE)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=min(4, os.cpu_count() or 1),
                            pin_memory=(device.type == 'cuda'))
    print(f"Loaded {len(valset)} samples")

    # ── Backbone
    print("Loading DINOv2-small backbone …")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)

    # ── Load v2 checkpoint
    print(f"Loading v2 checkpoint from {args.model_path} …")
    ckpt = torch.load(args.model_path, map_location=device)

    # Restore the partially fine-tuned backbone weights
    if 'backbone_partial' in ckpt:
        missing, unexpected = backbone.load_state_dict(ckpt['backbone_partial'], strict=False)
        print(f"  Backbone partial restore — missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print("  Warning: no backbone_partial key found; using pretrained weights only.")

    # Read config saved at training time
    cfg          = ckpt.get('config', {})
    embed_dim    = cfg.get('embed_dim', 384)
    token_h      = cfg.get('token_h',  H // 14)
    token_w      = cfg.get('token_w',  W // 14)
    n_blocks     = cfg.get('n_blocks', 12)
    feat_indices = ckpt.get('feat_indices', [2, 5, 8, 11])
    print(f"  Config — embed_dim={embed_dim}  token_grid={token_h}×{token_w}  "
          f"n_blocks={n_blocks}  feat_indices={feat_indices}")

    # Infer expected patch count
    try:
        patch_size = backbone.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        patch_size = int(patch_size)
    except Exception:
        patch_size = 14
    expected_n_patches = token_h * token_w

    # ── Decoder
    model = FPNDecoder(embed_dim, N_CLASSES, token_h, token_w, num_blocks=n_blocks).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("Model loaded successfully.")

    # ── Output sub-dirs
    masks_dir      = os.path.join(args.output_dir, 'masks')
    masks_color_dir= os.path.join(args.output_dir, 'masks_color')
    comparisons_dir= os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Evaluation loop
    print(f"\nRunning evaluation (TTA={'on' if args.tta else 'off'}) …")
    all_ious       = []
    all_dices      = []
    all_accs       = []
    all_class_ious = np.zeros(N_CLASSES)
    n_batches      = 0
    sample_count   = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating", unit="batch")
        for imgs, labels, data_ids in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast(device_type='cuda', enabled=use_amp):
                if args.tta:
                    outputs = predict_with_tta(imgs, backbone, model, feat_indices)
                else:
                    feats     = extract_intermediate_features(
                        backbone, imgs, feat_indices, expected_n_patches)
                    logits, _ = model(feats)
                    outputs   = F.interpolate(logits, size=imgs.shape[2:],
                                              mode="bilinear", align_corners=False)

            all_ious.append(compute_mean_iou(outputs, labels))
            all_dices.append(compute_dice(outputs, labels))
            all_accs.append(compute_pixel_accuracy(outputs, labels))
            all_class_ious += np.nan_to_num(
                compute_iou_per_class(outputs, labels))
            n_batches += 1

            predicted = torch.argmax(outputs, dim=1)

            # Save per-image outputs
            for i in range(imgs.shape[0]):
                name      = data_ids[i]
                base_name = os.path.splitext(name)[0]

                pred_np = predicted[i].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_np).save(
                    os.path.join(masks_dir, f'{base_name}_pred.png'))

                pred_color = mask_to_color(pred_np)
                cv2.imwrite(
                    os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels[i], predicted[i],
                        os.path.join(comparisons_dir,
                                     f'sample_{sample_count:03d}_comparison.png'),
                        name)

                sample_count += 1

            pbar.set_postfix(mIoU=f"{np.mean(all_ious):.3f}")

    # ── Aggregate
    mean_iou       = float(np.mean(all_ious))
    mean_dice      = float(np.mean(all_dices))
    mean_acc       = float(np.mean(all_accs))
    per_class_iou  = all_class_ious / n_batches

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS (v2)")
    print("=" * 55)
    print(f"Mean IoU:        {mean_iou:.4f}")
    print(f"Mean Dice:       {mean_dice:.4f}")
    print(f"Pixel Accuracy:  {mean_acc:.4f}")
    print("-" * 55)
    print(f"{'Class':<22} {'IoU':>6}")
    print("-" * 55)
    for name, iou in zip(CLASS_NAMES, per_class_iou):
        bar = '█' * int(iou * 20)
        print(f"  {name:<20} {bar:<20} {iou:.3f}")
    print("=" * 55)

    save_metrics_summary(mean_iou, mean_dice, mean_acc, per_class_iou, args.output_dir)

    print(f"\nDone! {sample_count} images processed.")
    print(f"Outputs in {args.output_dir}/")
    print(f"  masks/           raw class-id masks (0–10)")
    print(f"  masks_color/     RGB visualisations")
    print(f"  comparisons/     side-by-side ({args.num_samples} samples)")
    print(f"  evaluation_metrics.txt + per_class_iou.png")


if __name__ == "__main__":
    main()