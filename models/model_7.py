# =========================
# MODEL 7 - Robust Triplet Embedding with Heavy Augmentation
# =========================
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import zipfile
import random

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
# CONFIGURATION
# =========================
IMG_H, IMG_W = 160, 256
BATCH_SIZE = 64
EPOCHS = 15
MARGIN = 1.0
OUT_DIR = "/content/triplet_runs"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = "/content/cedar_train_triplet.csv"
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Expected CSV not found: {TRAIN_CSV}")

# =========================
# PATH FIXER
# =========================
def fix_csv_paths(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    for col in ["anchor", "positive", "negative"]:
        df[col] = df[col].astype(str).str.replace("\\", "/", regex=False)
        df[col] = df[col].str.replace("cedar/cedar/", "cedar/", regex=False)
        df[col] = df[col].apply(lambda p: p if p.startswith("cedar/") else "cedar/" + p)
        df[col] = df[col].apply(lambda p: os.path.join(base_dir, p))
    df.to_csv(csv_path, index=False)
    return csv_path

TRAIN_CSV = fix_csv_paths(TRAIN_CSV, EXTRACT_DIR)

# =========================
# CUSTOM BINARIZE (B/W conversion)
# =========================
def binarize(img):
    return img.convert("L").point(lambda x: 0 if x < 128 else 255, 'L')

class BinarizeTransform:
    def __call__(self, img):
        return binarize(img)

# =========================
# DATASET CLASS
# =========================
class TripletDataset(Dataset):
    def __init__(self, csv_file, img_h, img_w, train=True):
        self.data = pd.read_csv(csv_file)

        aug_list = [
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9,1.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.RandomApply([transforms.RandomErasing(p=0.9, scale=(0.02, 0.25))], p=0.5),
        ]

        self.transform = transforms.Compose(
            ([BinarizeTransform()] if train else []) +
            [transforms.Resize((img_h, img_w))] +
            aug_list +
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        anchor_path, positive_path, negative_path = row["anchor"], row["positive"], row["negative"]

        for p in [anchor_path, positive_path, negative_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image not found: {p}")

        anchor = Image.open(anchor_path).convert("L")
        positive = Image.open(positive_path).convert("L")
        negative = Image.open(negative_path).convert("L")

        return self.transform(anchor), self.transform(positive), self.transform(negative)

# =========================
# MODEL
# =========================
class EmbeddingNet(nn.Module):
    def __init__(self, img_h, img_w):
        super(EmbeddingNet, self).__init__()
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

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.normalize(out, p=2, dim=1)

# =========================
# TRAINING
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TripletDataset(TRAIN_CSV, IMG_H, IMG_W, train=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=min(6, os.cpu_count()), pin_memory=True)

model = EmbeddingNet(IMG_H, IMG_W).to(device)
criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\nTraining on {device} for {EPOCHS} epochs")

best_loss = float("inf")
best_path = os.path.join(OUT_DIR, "model_7.pth")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader, 1):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        out_anchor = model(anchor)
        out_positive = model(positive)
        out_negative = model(negative)

        loss = criterion(out_anchor, out_positive, out_negative)

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
# accuracy 0.7