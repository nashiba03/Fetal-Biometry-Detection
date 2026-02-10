#Trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Dataset import FetalLandmarkDataset
from UNet import UNet


CSV_PATH = r"C:\Users\BME\Desktop\Project_Fetal_Landmark_Detection\data\role_challenge_dataset_ground_truth.csv"
IMAGE_DIR = r"C:\Users\BME\Desktop\Project_Fetal_Landmark_Detection\data\images-20251229T061146Z-3-001\images"
SAVE_DIR = r"C:\Users\BME\Desktop\Project_Fetal_Landmark_Detection\model_weights"

EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-3
SIGMA = 5

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = FetalLandmarkDataset(CSV_PATH, IMAGE_DIR, SIGMA)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  
    )

    print("Dataset size:", len(dataset))
    print("Train loader batches:", len(loader))
    print("Epochs:", EPOCHS)
    print("Using device:", device)

    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training started", flush=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, heatmaps, _) in enumerate(loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(images)
            loss = criterion(preds, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_loss:.4f}",
            flush=True
        )

        
        epoch_model_path = os.path.join(
            SAVE_DIR, f"unet_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Saved: {epoch_model_path}", flush=True)

    print("Training completed")



if __name__ == "__main__":
    main()
