"""
Improved Segmentation Training Script
Based on original train_segmentation.py — DINOv2 backbone + enhanced segmentation head

Key improvements over baseline:
  1. Fixed value_map to include class 600 (Flowers) — was completely missing
  2. Fixed n_classes (10 → 11 including background)
  3. Nearest-neighbor mask resizing — prevents label corruption from bilinear interp
  4. Rich data augmentation — horizontal/vertical flip, color jitter, random crop
  5. AdamW optimizer + cosine annealing LR scheduler
  6. Weighted CrossEntropyLoss — handles class imbalance in desert scenes
  7. Deeper segmentation head with residual blocks and dropout
  8. Best-model checkpointing based on val IoU
  9. Mixed-precision training (AMP) — faster on GPU
 10. Improved DataLoader (num_workers, pin_memory)
 11. Per-class IoU logging for failure case analysis
 12. Label smoothing in loss for better generalization
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

plt.switch_backend('Agg')


# ============================================================================
# Utility
# ============================================================================

def save_image(img, filename):
    """Save a normalised image tensor to disk."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1)
    img  = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Class Definitions
# ============================================================================

# FIX 1: 600 (Flowers) was missing from the original map.
# FIX 2: n_classes now correctly reflects all 11 entries (0–10).
VALUE_MAP = {
    0:     0,   # Background / unlabelled
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers  ← was missing in original
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}
N_CLASSES = len(VALUE_MAP)   # 11

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

# Approximate inverse-frequency weights to counter class imbalance.
# Sky / Landscape dominate desert scenes; Logs / Flowers are rare.
# Tune these if you have class pixel counts from your dataset.
CLASS_WEIGHTS = torch.tensor([
    0.5,   # Background
    1.5,   # Trees
    2.0,   # Lush Bushes
    1.5,   # Dry Grass
    2.0,   # Dry Bushes
    3.0,   # Ground Clutter
    4.0,   # Flowers     (rare)
    4.0,   # Logs        (rare)
    3.0,   # Rocks
    0.8,   # Landscape   (very common)
    0.8,   # Sky         (very common)
], dtype=torch.float32)


def convert_mask(mask: Image.Image) -> Image.Image:
    """Map raw pixel values to sequential class IDs (0-based)."""
    arr     = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls_id in VALUE_MAP.items():
        new_arr[arr == raw] = cls_id
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset with Augmentation
# ============================================================================

class JointTransform:
    """
    Applies the same spatial transform to both image and mask.
    Color/brightness changes are applied to image only.
    """
    def __init__(self, size, augment=False):
        self.size    = size   # (H, W)
        self.augment = augment

    def __call__(self, image: Image.Image, mask: Image.Image):
        # FIX 3: always resize mask with NEAREST to avoid label blending
        image = TF.resize(image, self.size, interpolation=Image.BILINEAR)
        mask  = TF.resize(mask,  self.size, interpolation=Image.NEAREST)

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Random vertical flip (uncommon in real scenes, but helps generalise)
            if random.random() > 0.8:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # Random crop & resize back to target
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.6, 1.0), ratio=(4/3, 16/9))
            image = TF.resized_crop(image, i, j, h, w, self.size, Image.BILINEAR)
            mask  = TF.resized_crop(mask,  i, j, h, w, self.size, Image.NEAREST)

            # Colour jitter — image only
            image = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )(image)

            # Random grayscale (simulates overcast / sensor variation)
            if random.random() > 0.9:
                image = TF.to_grayscale(image, num_output_channels=3)
                image = Image.fromarray(np.array(image))

        # To tensor & normalise
        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
        # Mask: keep as long tensor of class IDs
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
# Improved Segmentation Head
# ============================================================================

class ResidualBlock(nn.Module):
    """Depthwise-separable residual block (ConvNeXt-style)."""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.dw   = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(8, channels)
        self.pw1  = nn.Conv2d(channels, channels * 4, 1)
        self.pw2  = nn.Conv2d(channels * 4, channels, 1)
        self.drop = nn.Dropout2d(dropout)
        self.act  = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw2(x)
        return x + residual


class SegmentationHead(nn.Module):
    """
    Deeper segmentation head with:
      - Linear projection from DINOv2 embedding dim
      - 3 residual blocks
      - Final classifier
    """
    def __init__(self, in_channels, out_channels, token_h, token_w):
        super().__init__()
        self.H, self.W = token_h, token_w
        hidden = 256

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GroupNorm(16, hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden, dropout=0.1),
            ResidualBlock(hidden, dropout=0.1),
            ResidualBlock(hidden, dropout=0.05),
        )
        self.classifier = nn.Conv2d(hidden, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        x = self.blocks(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou_per_class(pred_logits, target, num_classes=N_CLASSES):
    """Return per-class IoU array (NaN for absent classes)."""
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
def evaluate(model, backbone, loader, device, loss_fn, num_classes=N_CLASSES):
    model.eval()
    losses, ious, dices, accs = [], [], [], []
    all_class_ious = np.zeros((num_classes,))
    n_batches = 0

    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(device_type='cuda'):
            feats  = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(feats)
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
    return (
        float(np.mean(losses)),
        float(np.mean(ious)),
        float(np.mean(dices)),
        float(np.mean(accs)),
        all_class_ious / n_batches   # mean per-class IoU across dataset
    )


# ============================================================================
# Plotting
# ============================================================================

def save_plots(history, per_class_iou, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # ── Loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'],   label='Val')
    axes[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axes[0].legend(); axes[0].grid(True)

    # ── IoU
    axes[1].plot(epochs, history['train_iou'], label='Train')
    axes[1].plot(epochs, history['val_iou'],   label='Val')
    axes[1].set(title='Mean IoU', xlabel='Epoch', ylabel='IoU')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_iou.png'), dpi=150)
    plt.close()

    # ── Dice & Pixel Accuracy
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

    # ── Per-class IoU bar chart (great for failure case analysis)
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
        f.write(f"Best Val Dice: {max(history['val_dice']):.4f} "
                f"(Epoch {np.argmax(history['val_dice']) + 1})\n")
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
    
def predict_with_tta(imgs, backbone, classifier):
    preds = []

    for flip in [False, True]:
        x = imgs
        if flip:
            x = torch.flip(x, dims=[3])

        feats = backbone.forward_features(x)["x_norm_patchtokens"]
        logits = classifier(feats)
        out = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

        if flip:
            out = torch.flip(out, dims=[3])

        preds.append(out)

    return torch.mean(torch.stack(preds), dim=0)

# ============================================================================
# Main
# ============================================================================

def main():
    # ── Config
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp    = device.type == 'cuda'
    print(f"Device: {device}  |  AMP: {use_amp}")

    BATCH_SIZE  = 16        # increase if VRAM allows
    N_EPOCHS    = 30         # more epochs with scheduler — won't overfit
    LR          = 3e-4       # AdamW starting LR
    WEIGHT_DECAY= 1e-4
    NUM_WORKERS = min(4, os.cpu_count() or 1)

    # Image size must be a multiple of 14 (DINOv2 patch size)
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
    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True   # 🔥 add this
    )
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

    # ── Backbone (frozen)
    print("Loading DINOv2-small backbone …")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    print("Backbone loaded.")

    # Infer embedding dim
    with torch.no_grad():
        sample = next(iter(train_loader))[0][:1].to(device)
        feats  = backbone.forward_features(sample)["x_norm_patchtokens"]
    embed_dim = feats.shape[2]
    token_h   = H // 14
    token_w   = W // 14
    print(f"Embed dim: {embed_dim}  |  Token grid: {token_h}×{token_w}")

    # ── Segmentation head
    model = SegmentationHead(embed_dim, N_CLASSES, token_h, token_w).to(device)

    # ── Loss: weighted CE + label smoothing
    weights  = CLASS_WEIGHTS.to(device)
    loss_fn  = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    # ── Optimiser + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
    scaler    = GradScaler(enabled=use_amp)

    # ── Training state
    history = {k: [] for k in
               ['train_loss','val_loss','train_iou','val_iou',
                'train_dice','val_dice','train_acc','val_acc']}
    best_iou    = -1.0
    best_epoch  = 0
    best_path   = os.path.join(script_dir, "segmentation_head_best.pth")

    print(f"\nStarting training — {N_EPOCHS} epochs …\n{'='*70}")

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:>2}/{N_EPOCHS} [Train]",
                    leave=False, unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', enabled=use_amp):
                with torch.no_grad():
                    feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits  = model(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                loss    = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        # ── Validation
        val_loss, val_iou, val_dice, val_acc, per_class_iou = \
            evaluate(model, backbone, val_loader, device, loss_fn, N_CLASSES)

        # quick train metrics (subset for speed)
        _, train_iou, train_dice, train_acc, _ = \
            evaluate(model, backbone, train_loader, device, loss_fn, N_CLASSES)

        epoch_train_loss = float(np.mean(train_losses))
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:>2}/{N_EPOCHS}  "
              f"TrLoss={epoch_train_loss:.4f}  VaLoss={val_loss:.4f}  "
              f"TrIoU={train_iou:.4f}  VaIoU={val_iou:.4f}  "
              f"VaAcc={val_acc:.4f}  "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

        # ── Checkpoint best model
        if val_iou > best_iou:
            best_iou   = val_iou
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best Val IoU={best_iou:.4f} — saved to {best_path}")

    # ── Save last model too
    last_path = os.path.join(script_dir, "segmentation_head_last.pth")
    torch.save(model.state_dict(), last_path)

    # ── Plots & logs
    save_plots(history, per_class_iou, output_dir)
    save_history(history, per_class_iou, output_dir)

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"  Best Val IoU : {best_iou:.4f}  (Epoch {best_epoch})")
    print(f"  Final Val IoU: {history['val_iou'][-1]:.4f}")
    print(f"  Best model   : {best_path}")
    print(f"  Last model   : {last_path}")
    print(f"\nPer-class IoU breakdown (final epoch):")
    for name, iou in zip(CLASS_NAMES, per_class_iou):
        bar = '█' * int(iou * 20)
        print(f"  {name:<20} {bar:<20} {iou:.3f}")


if __name__ == "__main__":
    main()