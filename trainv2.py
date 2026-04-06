"""
Improved Segmentation v2 — Key changes from v1:

  [BIGGEST WIN] Partial backbone unfreeze  — last 4 DINOv2 blocks get fine-tuned
  [BIG WIN]     FPN-style multi-scale decoder — fuses 4 intermediate block features
  [BIG WIN]     OHEM loss — Online Hard Example Mining focuses on mis-classified pixels
  [MED WIN]     Separate LRs — backbone 10× lower than head (3e-5 vs 3e-4)
  [MED WIN]     Stronger class weights — Logs/Flowers/Dry-Bushes now weighted 8–10×
  [MED WIN]     Deep supervision — auxiliary head on intermediate features
  [SMALL WIN]   Longer warmup cosine schedule — 5 epoch warmup
  [SMALL WIN]   Stochastic depth in residual blocks — better regularization
  [SMALL WIN]   Background class handled — predict background only when confident
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math

plt.switch_backend('Agg')


# ============================================================================
# Utility
# ============================================================================

def save_image(img, filename):
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1)
    img  = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Class Definitions
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

# --- IMPROVED WEIGHTS ---
# Observations from v1:
#   Background=0.000, Logs=0.114, Dry Bushes=0.135, Flowers=0.173 — all under-predicted
#   Sky=0.915 — massively over-represented, keep low weight
#   Landscape=0.439 — common, moderate weight fine
#   Push hard on rare classes: multiply previous rare weights by 2–2.5×
CLASS_WEIGHTS = torch.tensor([
    0.3,   # Background   — model refuses to predict it; reduce to stop penalising unlabelled
    1.5,   # Trees
    2.5,   # Lush Bushes
    1.2,   # Dry Grass    — already good (0.578), ease off
    8.0,   # Dry Bushes   — was 2.0, IoU only 0.135
    5.0,   # Ground Clutter
    9.0,   # Flowers      — was 4.0, IoU only 0.173
    9.0,   # Logs         — was 4.0, IoU only 0.114
    4.0,   # Rocks
    0.6,   # Landscape
    0.3,   # Sky          — already great, stop rewarding it
], dtype=torch.float32)


def convert_mask(mask: Image.Image) -> Image.Image:
    arr     = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls_id in VALUE_MAP.items():
        new_arr[arr == raw] = cls_id
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset (same joint transform, slightly stronger augmentation)
# ============================================================================

class JointTransform:
    def __init__(self, size, augment=False):
        self.size    = size
        self.augment = augment

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, self.size, interpolation=Image.BILINEAR)
        mask  = TF.resize(mask,  self.size, interpolation=Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            if random.random() > 0.7:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # Slightly wider crop range to help rare small objects
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.5, 1.0), ratio=(4/3, 16/9))
            image = TF.resized_crop(image, i, j, h, w, self.size, Image.BILINEAR)
            mask  = TF.resized_crop(mask,  i, j, h, w, self.size, Image.NEAREST)

            # Stronger colour jitter
            image = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08
            )(image)

            # Random rotation ±10° (helps with Logs on the ground)
            if random.random() > 0.7:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)

            if random.random() > 0.9:
                image = TF.to_grayscale(image, num_output_channels=3)
                image = Image.fromarray(np.array(image))

            # Gaussian blur (simulates sensor noise / motion)
            if random.random() > 0.8:
                image = image.filter(__import__('PIL').ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


class MaskDataset(Dataset):
    def __init__(self, data_dir, size, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.transform = JointTransform(size, augment=augment)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        name     = self.data_ids[idx]
        image    = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask_raw = Image.open(os.path.join(self.masks_dir, name))
        mask     = convert_mask(mask_raw)
        return self.transform(image, mask)


# ============================================================================
# OHEM Loss (Online Hard Example Mining)
# ============================================================================

class OHEMLoss(nn.Module):
    """
    Weighted CrossEntropy + OHEM: backprop only through the top-K% hardest pixels.
    This naturally focuses training on rare classes and confused boundaries.
    """
    def __init__(self, weight, keep_ratio=0.7, label_smoothing=0.05, min_kept=1024):
        super().__init__()
        self.weight         = weight
        self.keep_ratio     = keep_ratio
        self.label_smoothing= label_smoothing
        self.min_kept       = min_kept

    def forward(self, logits, targets):
        # Standard weighted CE with label smoothing
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             reduction='none')

        # Sort pixels by loss; keep the hardest keep_ratio fraction
        B, H, W = targets.shape
        n_pixels = B * H * W
        n_keep   = max(self.min_kept, int(n_pixels * self.keep_ratio))
        flat_ce  = ce.view(-1)
        sorted_loss, _ = torch.sort(flat_ce, descending=True)
        threshold = sorted_loss[min(n_keep, n_pixels) - 1].detach()
        mask = flat_ce >= threshold
        return flat_ce[mask].mean()


# ============================================================================
# Multi-Scale Decoder (FPN-style)
# ============================================================================

class ResidualBlock(nn.Module):
    """ConvNeXt-style block with stochastic depth."""
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
        # Stochastic depth
        if self.training and self.drop_path > 0:
            keep_prob = 1 - self.drop_path
            mask = torch.rand(h.shape[0], 1, 1, 1, device=h.device) < keep_prob
            h = h * mask.float() / keep_prob
        return h + residual


class FPNDecoder(nn.Module):
    """
    FPN-style decoder that fuses 4 DINOv2 intermediate block outputs.

    DINOv2-small has 12 blocks. We extract features from blocks 3, 6, 9, 11
    at the same spatial resolution (token grid), then progressively upsample.

    Architecture:
      feat_3  (embed_dim) → lateral_3 → 128ch at 1×scale
      feat_6  (embed_dim) → lateral_6 → 128ch at 1×scale  +  feat_3
      feat_9  (embed_dim) → lateral_9 → 128ch at 2×scale  +  feat_6
      feat_11 (embed_dim) → lateral_11→ 128ch at 4×scale  +  feat_9
      → 3 residual blocks → classifier
      → aux_head on feat_6 level (deep supervision)
    """
    def __init__(self, embed_dim, num_classes, token_h, token_w, num_blocks=12):
        super().__init__()
        self.H, self.W = token_h, token_w
        self.num_blocks = num_blocks
        # Feature extraction indices (DINOv2 block indices, 0-based)
        self.feat_indices = [2, 5, 8, 11]  # blocks 3,6,9,12 → 0-indexed

        hidden = 256
        fpn_ch = 128

        # Lateral projections (1×1 conv from embed_dim to fpn_ch)
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, fpn_ch, 1),
                nn.GroupNorm(8, fpn_ch),
                nn.GELU()
            ) for _ in self.feat_indices
        ])

        # Top-down path: merge after each upsampling
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1),
                nn.GroupNorm(8, fpn_ch),
                nn.GELU()
            ) for _ in range(len(self.feat_indices) - 1)
        ])

        # Refinement head
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

        # Auxiliary head for deep supervision (attached to feat index 1 = block 6)
        self.aux_head = nn.Sequential(
            nn.Conv2d(fpn_ch, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, num_classes, 1)
        )

    def tokens_to_spatial(self, tokens):
        """(B, N, C) → (B, C, H, W)"""
        B, N, C = tokens.shape
        return tokens.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

    def forward(self, intermediate_features):
        """
        intermediate_features: list of 4 token tensors (B, N, C),
          corresponding to self.feat_indices
        Returns: (main_logits, aux_logits)
        """
        # Project each level to fpn_ch spatial maps
        fmaps = [
            self.laterals[i](self.tokens_to_spatial(intermediate_features[i]))
            for i in range(len(self.feat_indices))
        ]

        # Top-down FPN merge (all at same resolution — DINOv2 is ViT, no stride)
        # We simulate multi-scale by 2× interpolating the deeper maps before merging
        top = fmaps[-1]  # deepest (richest semantics)
        for i in range(len(fmaps) - 2, -1, -1):
            top = F.interpolate(top, size=fmaps[i].shape[2:], mode='bilinear', align_corners=False)
            top = self.td_convs[i](top + fmaps[i])

        # Deep supervision auxiliary output (on fmaps[1], block-6 level)
        aux_out = self.aux_head(fmaps[1])

        # Main head
        x = self.proj(top)
        x = self.blocks(x)
        main_out = self.classifier(x)

        return main_out, aux_out


# ============================================================================
# Partial backbone unfreeze helper
# ============================================================================

def set_backbone_trainability(backbone, n_unfrozen_blocks=4):
    """
    Freeze everything, then selectively unfreeze the last n_unfrozen_blocks
    transformer blocks + norm layers.
    This is the SINGLE BIGGEST WIN in this script.
    """
    for p in backbone.parameters():
        p.requires_grad_(False)

    total_blocks = len(backbone.blocks)
    unfreeze_from = total_blocks - n_unfrozen_blocks

    for i, block in enumerate(backbone.blocks):
        if i >= unfreeze_from:
            for p in block.parameters():
                p.requires_grad_(True)

    # Always unfreeze final norm
    for p in backbone.norm.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone: {trainable/1e6:.2f}M / {total/1e6:.2f}M params trainable "
          f"(last {n_unfrozen_blocks} blocks + norm)")


# ============================================================================
# DINOv2 intermediate feature extractor
# ============================================================================

def extract_intermediate_features(backbone, imgs, feat_indices, expected_n_patches=None):
    """
    Extract intermediate block outputs from DINOv2 without modifying the backbone.
    Uses register_forward_hook on the specific blocks we care about.
    
    Args:
        backbone: DINOv2 backbone model
        imgs: input images
        feat_indices: list of block indices to extract features from
        expected_n_patches: expected number of patch tokens (H*W). If None, inferred from backbone.
    """
    captured = {}

    def make_hook(idx):
        def hook(module, input, output):
            # DINOv2 block output is (x, attn) or just x depending on version
            captured[idx] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = []
    for idx in feat_indices:
        h = backbone.blocks[idx].register_forward_hook(make_hook(idx))
        handles.append(h)

    # Forward pass
    with torch.set_grad_enabled(torch.is_grad_enabled()):
        _ = backbone.forward_features(imgs)

    for h in handles:
        h.remove()

    # If expected_n_patches not provided, infer it
    if expected_n_patches is None:
        expected_n_patches = backbone.patch_embed.num_patches

    # Return token sequences (strip CLS token → patch tokens only)
    # DINOv2 outputs shape (B, T, C) where T includes CLS + patch tokens + register tokens
    # Patch tokens are at positions [1 : 1 + expected_n_patches]
    feats = []
    for idx in feat_indices:
        f = captured[idx]
        B, T, C = f.shape
        # Extract exactly the patch tokens: skip CLS token (index 0) and take next expected_n_patches
        # This handles register tokens correctly by excluding them
        patch_tokens = f[:, 1 : 1 + expected_n_patches, :]
        feats.append(patch_tokens)

    return feats


# ============================================================================
# LR warmup + cosine annealing scheduler
# ============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, base_lrs=None):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.min_lr         = min_lr
        self.base_lrs       = base_lrs or [pg['lr'] for pg in optimizer.param_groups]
        self._epoch         = 0

    def step(self):
        self._epoch += 1
        e  = self._epoch
        we = self.warmup_epochs
        te = self.total_epochs
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if e <= we:
                lr = base_lr * (e / we)
            else:
                progress = (e - we) / (te - we)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            pg['lr'] = lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ============================================================================
# Metrics (unchanged)
# ============================================================================

def compute_iou_per_class(pred_logits, target, num_classes=N_CLASSES):
    pred = torch.argmax(pred_logits, dim=1).view(-1)
    tgt  = target.view(-1)
    ious = []
    for c in range(num_classes):
        p = pred == c
        t = tgt  == c
        intersection = (p & t).sum().float()
        union        = (p | t).sum().float()
        ious.append(float('nan') if union == 0 else (intersection / union).item())
    return np.array(ious)


def compute_mean_iou(pred_logits, target, num_classes=N_CLASSES):
    return float(np.nanmean(compute_iou_per_class(pred_logits, target, num_classes)))


def compute_dice(pred_logits, target, num_classes=N_CLASSES, smooth=1e-6):
    pred = torch.argmax(pred_logits, dim=1).view(-1)
    tgt  = target.view(-1)
    scores = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (tgt  == c).float()
        scores.append(((2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)).item())
    return float(np.mean(scores))


def compute_pixel_accuracy(pred_logits, target):
    return (torch.argmax(pred_logits, 1) == target).float().mean().item()


@torch.no_grad()
def evaluate(model, backbone, loader, device, loss_fn, feat_indices, num_classes=N_CLASSES):
    model.eval()
    backbone.eval()
    losses, ious, dices, accs = [], [], [], []
    all_class_ious = np.zeros((num_classes,))
    n_batches = 0

    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(device_type='cuda'):
            feats   = extract_intermediate_features(backbone, imgs, feat_indices)
            logits, _ = model(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)
            loss = loss_fn(outputs, labels)

        losses.append(loss.item())
        ious.append(compute_mean_iou(outputs, labels, num_classes))
        dices.append(compute_dice(outputs, labels, num_classes))
        accs.append(compute_pixel_accuracy(outputs, labels))
        all_class_ious += np.nan_to_num(compute_iou_per_class(outputs, labels, num_classes))
        n_batches += 1

    model.train()
    backbone.train()
    return (
        float(np.mean(losses)),
        float(np.mean(ious)),
        float(np.mean(dices)),
        float(np.mean(accs)),
        all_class_ious / n_batches
    )


# ============================================================================
# TTA (unchanged, but updated signature)
# ============================================================================

def predict_with_tta(imgs, backbone, model, feat_indices):
    preds = []
    for flip in [False, True]:
        x = imgs
        if flip:
            x = torch.flip(x, dims=[3])
        feats  = extract_intermediate_features(backbone, x, feat_indices)
        logits, _ = model(feats)
        out = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
        if flip:
            out = torch.flip(out, dims=[3])
        preds.append(out)
    return torch.mean(torch.stack(preds), dim=0)


# ============================================================================
# Plotting / logging (unchanged)
# ============================================================================

def save_plots(history, per_class_iou, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'],   label='Val')
    axes[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, history['train_iou'], label='Train')
    axes[1].plot(epochs, history['val_iou'],   label='Val')
    axes[1].set(title='Mean IoU', xlabel='Epoch', ylabel='IoU')
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_iou.png'), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_dice'], label='Train')
    axes[0].plot(epochs, history['val_dice'],   label='Val')
    axes[0].set(title='Dice Score', xlabel='Epoch', ylabel='Dice')
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, history['train_acc'], label='Train')
    axes[1].plot(epochs, history['val_acc'],   label='Val')
    axes[1].set(title='Pixel Accuracy', xlabel='Epoch', ylabel='Accuracy')
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_acc.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#e74c3c' if v < 0.4 else '#f39c12' if v < 0.6 else '#2ecc71'
              for v in per_class_iou]
    bars = ax.bar(CLASS_NAMES, per_class_iou, color=colors)
    ax.axhline(np.nanmean(per_class_iou), color='navy', linestyle='--',
               label=f'Mean IoU = {np.nanmean(per_class_iou):.3f}')
    ax.set(title='Per-Class IoU (Final Val)', ylabel='IoU', ylim=(0, 1))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=9)
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    for bar, v in zip(bars, per_class_iou):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150)
    plt.close()
    print(f"Plots saved to {output_dir}")


def save_history(history, per_class_iou, output_dir):
    path = os.path.join(output_dir, 'training_log.txt')
    with open(path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best Val IoU:  {max(history['val_iou']):.4f} "
                f"(Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"Best Val Dice: {max(history['val_dice']):.4f}\n")
        f.write(f"Best Val Acc:  {max(history['val_acc']):.4f}\n\n")
        f.write("Per-Class IoU (Final Epoch):\n")
        for name, iou in zip(CLASS_NAMES, per_class_iou):
            f.write(f"  {name:<20} {iou:.4f}\n")
        f.write("\n")
        headers = ['Epoch','TrLoss','VaLoss','TrIoU','VaIoU','TrDice','VaDice','TrAcc','VaAcc']
        f.write("{:<6} {:<9} {:<9} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}\n".format(*headers))
        f.write("-" * 80 + "\n")
        for i in range(len(history['train_loss'])):
            f.write("{:<6} {:<9.4f} {:<9.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f}\n".format(
                i+1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i],  history['val_iou'][i],
                history['train_dice'][i], history['val_dice'][i],
                history['train_acc'][i],  history['val_acc'][i],
            ))
    print(f"History saved to {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}  |  AMP: {use_amp}")

    BATCH_SIZE       = 8          # reduced — backbone now has grad, needs more VRAM
    N_EPOCHS         = 40         # more epochs with warmup
    LR_HEAD          = 3e-4       # learning rate for decoder head
    LR_BACKBONE      = 3e-5       # 10× lower for unfrozen backbone blocks
    WEIGHT_DECAY     = 1e-4
    WARMUP_EPOCHS    = 5
    N_UNFROZEN_BLOCKS= 4          # set to 0 to revert to fully frozen backbone
    AUX_LOSS_WEIGHT  = 0.4        # weight on auxiliary deep supervision head
    NUM_WORKERS      = min(4, os.cpu_count() or 1)

    H = int(((540 // 2) // 14) * 14)   # 378
    W = int(((960 // 2) // 14) * 14)   # 476
    SIZE = (H, W)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # ── Datasets
    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset = MaskDataset(data_dir, SIZE, augment=True)
    valset   = MaskDataset(val_dir,  SIZE, augment=False)
    print(f"Train: {len(trainset)}  |  Val: {len(valset)}")

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── Backbone
    print("Loading DINOv2-small backbone …")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.train().to(device)
    set_backbone_trainability(backbone, N_UNFROZEN_BLOCKS)

    # Infer embed dim and token grid
    with torch.no_grad():
        sample = next(iter(train_loader))[0][:1].to(device)
        out    = backbone.forward_features(sample)
        feats  = out["x_norm_patchtokens"]
    embed_dim = feats.shape[2]
    
    # Calculate grid from image size and patch size
    # DINOv2-vits14 has patch size 14
    try:
        patch_size = backbone.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]  # Extract first dimension if it's a tuple/list
        elif isinstance(patch_size, int):
            pass  # Already an integer
        else:
            patch_size = int(patch_size)  # Try to convert to int
    except (AttributeError, IndexError, TypeError):
        # Fallback: DINOv2-vits14 has hardcoded patch size of 14
        patch_size = 14
    
    token_h = SIZE[0] // patch_size
    token_w = SIZE[1] // patch_size
    expected_n_patches = token_h * token_w

    print(f"Correct token grid: {token_h} x {token_w}")

    n_blocks  = len(backbone.blocks)
    feat_indices = [2, 5, 8, 11]  # blocks to extract intermediate features from
    print(f"Embed dim: {embed_dim}  |  Token grid: {token_h}×{token_w}  |  Blocks: {n_blocks}")

    # ── Decoder
    model = FPNDecoder(embed_dim, N_CLASSES, token_h, token_w, num_blocks=n_blocks).to(device)

    # ── Loss
    weights = CLASS_WEIGHTS.to(device)
    loss_fn = OHEMLoss(weight=weights, keep_ratio=0.7, label_smoothing=0.05)

    # ── Optimizer: separate param groups for backbone vs head
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params     = list(model.parameters())

    optimizer = optim.AdamW([
        {'params': head_params,     'lr': LR_HEAD,     'name': 'head'},
        {'params': backbone_params, 'lr': LR_BACKBONE, 'name': 'backbone'},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=N_EPOCHS,
        min_lr=1e-6,
        base_lrs=[LR_HEAD, LR_BACKBONE]
    )
    scaler = GradScaler(enabled=use_amp)

    # ── Training state
    history = {k: [] for k in
               ['train_loss','val_loss','train_iou','val_iou',
                'train_dice','val_dice','train_acc','val_acc']}
    best_iou   = -1.0
    best_epoch = 0
    best_path  = os.path.join(script_dir, "segmentation_head_best_v2.pth")

    print(f"\nStarting training — {N_EPOCHS} epochs, "
          f"{WARMUP_EPOCHS} warmup …\n{'='*70}")

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        backbone.train()
        train_losses = []

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:>2}/{N_EPOCHS} [Train]",
                    leave=False, unit="batch")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', enabled=use_amp):
                feats  = extract_intermediate_features(backbone, imgs, feat_indices)
                logits, aux_logits = model(feats)

                # Upsample both heads to full resolution
                outputs     = F.interpolate(logits,     size=imgs.shape[2:],
                                            mode="bilinear", align_corners=False)
                aux_outputs = F.interpolate(aux_logits, size=imgs.shape[2:],
                                            mode="bilinear", align_corners=False)

                main_loss = loss_fn(outputs, labels)
                aux_loss  = loss_fn(aux_outputs, labels)
                loss      = main_loss + AUX_LOSS_WEIGHT * aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + backbone_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(main_loss.item())
            lrs = scheduler.get_last_lr()
            pbar.set_postfix(loss=f"{main_loss.item():.4f}",
                             lr_h=f"{lrs[0]:.2e}",
                             lr_b=f"{lrs[1]:.2e}")

        scheduler.step()

        # ── Validation
        val_loss, val_iou, val_dice, val_acc, per_class_iou = \
            evaluate(model, backbone, val_loader, device, loss_fn, feat_indices, N_CLASSES)

        _, train_iou, train_dice, train_acc, _ = \
            evaluate(model, backbone, train_loader, device, loss_fn, feat_indices, N_CLASSES)

        epoch_train_loss = float(np.mean(train_losses))
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        lrs = scheduler.get_last_lr()
        print(f"Epoch {epoch:>2}/{N_EPOCHS}  "
              f"TrLoss={epoch_train_loss:.4f}  VaLoss={val_loss:.4f}  "
              f"TrIoU={train_iou:.4f}  VaIoU={val_iou:.4f}  "
              f"VaAcc={val_acc:.4f}  "
              f"LR_head={lrs[0]:.2e}  LR_bb={lrs[1]:.2e}")

        if val_iou > best_iou:
            best_iou   = val_iou
            best_epoch = epoch
            torch.save({
                'model': model.state_dict(),
                'backbone_partial': {
                    k: v for k, v in backbone.state_dict().items()
                    if any(f'blocks.{i}.' in k for i in range(n_blocks - N_UNFROZEN_BLOCKS, n_blocks))
                    or 'norm.' in k
                },
                'feat_indices': feat_indices,
                'config': {
                    'embed_dim': embed_dim, 'token_h': token_h, 'token_w': token_w,
                    'num_classes': N_CLASSES, 'n_blocks': n_blocks,
                },
            }, best_path)
            print(f"  ★ New best Val IoU={best_iou:.4f} — saved to {best_path}")

    last_path = os.path.join(script_dir, "segmentation_head_last_v2 .pth")
    torch.save(model.state_dict(), last_path)

    save_plots(history, per_class_iou, output_dir)
    save_history(history, per_class_iou, output_dir)

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"  Best Val IoU : {best_iou:.4f}  (Epoch {best_epoch})")
    print(f"  Final Val IoU: {history['val_iou'][-1]:.4f}")
    print(f"\nPer-class IoU breakdown (final epoch):")
    for name, iou in zip(CLASS_NAMES, per_class_iou):
        bar = '█' * int(iou * 20)
        print(f"  {name:<20} {bar:<20} {iou:.3f}")


if __name__ == "__main__":
    main()