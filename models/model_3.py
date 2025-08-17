# model 3

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import zipfile
import random
import numpy as np
import cv2

# =========================
# UNZIP DATASET
# =========================
ZIP_PATH = "/content/cedar.zip"
EXTRACT_DIR = "/content/cedar_extracted"

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Dataset extracted to:", EXTRACT_DIR)
else:
    print("Using existing dataset at:", EXTRACT_DIR)

# =========================
# FIXED CONFIGURATION
# =========================
IMG_H, IMG_W = 160, 256
BATCH_SIZE = 64
EPOCHS = 10
MARGIN = 1.0
OUT_DIR = "/content/siamese_runs"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = "/content/cedar_train_siamese.csv"
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Expected CSV not found: {TRAIN_CSV}")

# =========================
# PATH AUTO-FIXER
# =========================
def fix_csv_paths(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    df["path1"] = df["path1"].str.replace("\\", "/", regex=False)
    df["path2"] = df["path2"].str.replace("\\", "/", regex=False)
    cedar_dir = os.path.join(base_dir, "cedar", "original")
    if not os.path.exists(cedar_dir):
        raise RuntimeError(f"Could not locate cedar/original under {base_dir}")
    df.to_csv(csv_path, index=False)
    return csv_path

TRAIN_CSV = fix_csv_paths(TRAIN_CSV, EXTRACT_DIR)

# =========================
# DATA AUGMENTATION FUNCTIONS
# =========================
def add_gaussian_noise(img, mean=0.0, std=0.02):
    np_img = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(mean, std, np_img.shape)
    np_img += noise
    np_img = np.clip(np_img, 0.0, 1.0)
    return Image.fromarray((np_img * 255).astype(np.uint8))

def elastic_transform(image, alpha=34, sigma=4):
    # Convert to numpy
    img = np.array(image)
    random_state = np.random.RandomState(None)
    shape = img.shape
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17,17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17,17), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(distorted)

# =========================
# DATASET CLASS
# =========================
class SiameseDataset(Dataset):
    def __init__(self, csv_file, base_dir, img_h, img_w):
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.img_h = img_h
        self.img_w = img_w
        self.transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _resolve_path(self, p):
        p = str(p).strip().replace("\\", "/")
        while p.startswith("cedar/cedar/"):
            p = p.replace("cedar/cedar/", "cedar/", 1)
        if not p.startswith("cedar/") and "cedar/" in p:
            p = "cedar/" + p.split("cedar/", 1)[-1]
        elif not p.startswith("cedar/"):
            p = "cedar/" + p
        full_path = os.path.join(self.base_dir, p)
        if not os.path.exists(full_path):
            print(f"[WARN] File not found: {full_path}")
        return full_path

    def _augment(self, img):
        # Random rotation ±5°
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, fillcolor=255)
        # Elastic distortion
        img = elastic_transform(img)
        # Gaussian noise
        img = add_gaussian_noise(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path1 = self._resolve_path(row["path1"])
        path2 = self._resolve_path(row["path2"])
        label = torch.tensor(int(row["authentic"]), dtype=torch.float32)

        img1 = Image.open(path1).convert("L")
        img2 = Image.open(path2).convert("L")

        img1 = self._augment(img1)
        img2 = self._augment(img2)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, label

# =========================
# MODEL ARCHITECTURE
# =========================
class SiameseNetwork(nn.Module):
    def __init__(self, img_h, img_w):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, img_w)
            out = self.cnn(dummy)
            flat_dim = out.view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# =========================
# CONTRASTIVE LOSS
# =========================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        return torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

# =========================
# TRAINING LOOP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SiameseDataset(TRAIN_CSV, EXTRACT_DIR, IMG_H, IMG_W)

num_workers = max(6, os.cpu_count())
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=True)

model = SiameseNetwork(IMG_H, IMG_W).to(device)
criterion = ContrastiveLoss(MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\nTraining on {device} for {EPOCHS} epochs with {num_workers} workers")

best_loss = float("inf")
best_path = os.path.join(OUT_DIR, "model_3.pth")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (img1, img2, label) in enumerate(dataloader, 1):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_path)
        print(f"Best model updated at epoch {epoch+1}, saved to {best_path}")

print(f"\nTraining complete. Best model saved at {best_path}")
