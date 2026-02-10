# Tester.py 
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
import os

from Dataset import FetalSegmentationDataset
from UNet import UNet

IMAGE_DIR = r"C:\Users\BME\Desktop\Task 2 segmentation\data\images-20251229T061146Z-3-001\images"
MASK_DIR = r"C:\Users\BME\Desktop\Task 2 segmentation\data\masks-20251229T062052Z-3-001\masks"
MODEL_PATH = r"C:\Users\BME\Desktop\Task 2 segmentation\Model_Weights\hypothesis_2_best_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FetalSegmentationDataset(IMAGE_DIR, MASK_DIR, fixed_size=(384, 384), augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def fit_ellipse(mask):
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0, 0, 0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None, 0, 0, 0
    ellipse = cv2.fitEllipse(cnt)
    (_, _), (major, minor), _ = ellipse
    bpd = minor
    ofd = major
    hc = np.pi * (major + minor) / 2
    return ellipse, bpd, ofd, hc

os.makedirs("test_overlays", exist_ok=True)

print("\n📊 TEST RESULTS (First 10 Samples)\n")
total_bpd, total_ofd, total_hc = 0, 0, 0
num_valid = 0

with torch.no_grad():
    for i, (imgs, _) in enumerate(loader):
        if i >= 10:
            break
        imgs = imgs.to(device)
        pred = model(imgs)[0, 0].cpu().numpy()

        ellipse, bpd, ofd, hc = fit_ellipse(pred)
        if ellipse is not None:
            total_bpd += bpd
            total_ofd += ofd
            total_hc += hc
            num_valid += 1
            
            
            overlay = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.ellipse(overlay, ellipse, (0, 255, 0), 2)
            cv2.imwrite(f"test_overlays/overlay_{i:03d}.png", overlay)
            
            print(f"Sample {i} ({dataset.image_files[i]}): BPD: {bpd:.2f}px | OFD: {ofd:.2f}px | HC: {hc:.2f}px")
        else:
            print(f"Sample {i}: No valid ellipse")

if num_valid > 0:
    print(f"\nAvg (10 samples): BPD: {total_bpd/num_valid:.2f}px | OFD: {total_ofd/num_valid:.2f}px | HC: {total_hc/num_valid:.2f}px")

print("✅ Task B complete! Check test_overlays/ for visuals.")