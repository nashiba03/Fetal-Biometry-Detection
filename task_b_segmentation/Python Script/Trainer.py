# Trainer.py 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from Dataset import FetalSegmentationDataset
from UNet import UNet


IMAGE_DIR = r"C:\Users\BME\Desktop\Task 2 segmentation\data\images-20251229T061146Z-3-001\images"
MASK_DIR  = r"C:\Users\BME\Desktop\Task 2 segmentation\data\masks-20251229T062052Z-3-001\masks"
SAVE_DIR = r"C:\Users\BME\Desktop\Task 2 segmentation\Model_Weights"

EPOCHS = 60  
BATCH_SIZE = 8
LR = 1e-3
VAL_SPLIT = 0.2


if not os.path.exists(IMAGE_DIR):
    raise ValueError(f"IMAGE DIRECTORY NOT FOUND:\n{IMAGE_DIR}")
if not os.path.exists(MASK_DIR):
    raise ValueError(f"MASK DIRECTORY NOT FOUND:\n{MASK_DIR}")

os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


full_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
train_size = int((1 - VAL_SPLIT) * len(full_files))
val_size = len(full_files) - train_size


train_files = full_files[:train_size]
val_files = full_files[train_size:]  

class FileLimitedDataset(FetalSegmentationDataset):
    def __init__(self, image_dir, mask_dir, files, fixed_size=(384, 384), augment=False):
        super().__init__(image_dir, mask_dir, fixed_size, augment)
        self.image_files = files  

train_dataset = FileLimitedDataset(IMAGE_DIR, MASK_DIR, train_files, augment=True)
val_dataset = FileLimitedDataset(IMAGE_DIR, MASK_DIR, val_files, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)


def focal_loss(pred, target, alpha=0.25, gamma=2):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def combined_loss(pred, target):
    foc = focal_loss(pred, target)
    dice = dice_loss(pred, target)
    return foc + dice

print("\n🚀 Training started\n")
best_val_dice = 0.0

for epoch in range(EPOCHS):
    
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            val_loss += loss.item()
            val_dice += (1 - dice_loss(outputs, masks)).item()
    val_loss /= len(val_loader)
    val_dice_score = val_dice / len(val_loader)

    scheduler.step(val_dice_score)

    if val_dice_score > best_val_dice:
        best_val_dice = val_dice_score
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "hypothesis_2_best_model.pth"))

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice_score:.4f} {'<- BEST' if val_dice_score == best_val_dice else ''}")

torch.save(model.state_dict(), os.path.join(SAVE_DIR, "hypothesis_2_final_model.pth"))
print(f"\n✅ Training completed! Best Val Dice: {best_val_dice:.4f}")