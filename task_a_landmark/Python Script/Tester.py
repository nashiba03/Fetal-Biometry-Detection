#Tester.py
import os
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader
from Dataset import FetalLandmarkDataset
from UNet import UNet


CSV_PATH = r"C:\Users\BME\Desktop\Project_Fetal_Landmark_Detection\data\role_challenge_dataset_ground_truth.csv"
IMAGE_DIR = r"C:\Users\BME\Desktop\Project_Fetal_Landmark_Detection\data\images-20251229T061146Z-3-001\images"
MODEL_PATH = r"C:\Users\BME\Desktop\Project_Fetal_Landmark_Detection\model_weights\unet_epoch_30.pth"

OUTPUT_CSV = r"C:\Users\BME\Desktop\Project_Fetal_Landmark_Detection\taskA_predictions.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FetalLandmarkDataset(CSV_PATH, IMAGE_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def heatmap_to_coord(hm):
    y, x = np.unravel_index(np.argmax(hm), hm.shape)
    return int(x), int(y)

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_name",
        "x1","y1","x2","y2","x3","y3","x4","y4",
        "bpd","ofd"
    ])

    with torch.no_grad():
        for images, _, image_name in loader:
            images = images.to(device)
            pred_heatmaps = model(images)[0].cpu().numpy()

            points = [heatmap_to_coord(pred_heatmaps[i]) for i in range(4)]

            ofd = distance(points[0], points[1])
            bpd = distance(points[2], points[3])

            writer.writerow([
                image_name[0],
                *points[0], *points[1], *points[2], *points[3],
                round(bpd, 2),
                round(ofd, 2)
            ])

            print(f"{image_name[0]} → BPD: {bpd:.2f}, OFD: {ofd:.2f}")

print("\n taskA_predictions.csv saved")
