#Dataset.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FetalLandmarkDataset(Dataset):
    def __init__(self, csv_path, image_dir, sigma=5, img_size=256):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.sigma = sigma
        self.img_size = img_size

        
        required_cols = [
            "image_name",
            "ofd_1_x","ofd_1_y",
            "ofd_2_x","ofd_2_y",
            "bpd_1_x","bpd_1_y",
            "bpd_2_x","bpd_2_y",
        ]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column in CSV: {col}")

    def __len__(self):
        return len(self.df)

    def generate_heatmap(self, cx, cy):
        h = w = self.img_size
        x = np.arange(0, w, dtype=np.float32)
        y = np.arange(0, h, dtype=np.float32)
        y = y[:, None]
        heatmap = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * self.sigma ** 2))
        heatmap /= heatmap.max() + 1e-8
        return heatmap

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image_name"]

        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        orig_h, orig_w = image.shape

        
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1,256,256]

        
        landmarks = [
            (row["ofd_1_x"], row["ofd_1_y"]),
            (row["ofd_2_x"], row["ofd_2_y"]),
            (row["bpd_1_x"], row["bpd_1_y"]),
            (row["bpd_2_x"], row["bpd_2_y"]),
        ]

        
        scaled_landmarks = []
        for x, y in landmarks:
            sx = x * self.img_size / orig_w
            sy = y * self.img_size / orig_h
            scaled_landmarks.append((sx, sy))

        
        heatmaps = []
        for (x, y) in scaled_landmarks:
            hm = self.generate_heatmap(x, y)
            heatmaps.append(hm)

        heatmaps = torch.tensor(np.stack(heatmaps), dtype=torch.float32)  

        return image, heatmaps, image_name