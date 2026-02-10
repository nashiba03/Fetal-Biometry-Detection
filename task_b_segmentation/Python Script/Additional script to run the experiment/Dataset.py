# Dataset.py 
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import re

class FetalSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, fixed_size=(384, 384), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fixed_size = fixed_size
        self.augment = augment  # Train only

        if not os.path.exists(image_dir):
            raise ValueError(f"IMAGE DIRECTORY NOT FOUND: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"MASK DIRECTORY NOT FOUND: {mask_dir}")

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        print(f"🔍 Found {len(self.image_files)} image files (Augment: {augment})")
        if len(self.image_files) > 0:
            print(f"First few: {self.image_files[:5]}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        
        match = re.search(r'^(\d+)', img_name)
        if not match:
            raise ValueError(f"Cannot parse number from: {img_name}")
        base_num = match.group(1)
        mask_name = f"{int(base_num):03d}_HC_Annotation.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.float32)

        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        
        image = cv2.GaussianBlur(image, (3,3), 0)
        
        
        p2, p98 = np.percentile(image, (2, 98))
        if p98 > p2:
            image = np.clip((image - p2) / (p98 - p2), 0, 1) * 255
        image = image.astype(np.uint8)

        
        target_h, target_w = self.fixed_size
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        
        image = image.astype(np.float32) / 255.0

        
        if self.augment:
            if np.random.rand() > 0.5:  
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            angle = np.random.uniform(-10, 10)  
            center = (target_w // 2, target_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (target_w, target_h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (target_w, target_h), borderMode=cv2.BORDER_REFLECT)
        mask = (mask > 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor