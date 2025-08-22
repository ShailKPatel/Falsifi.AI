# =========================
# MODEL 6 — Triplet EmbeddingNet with heavy augmentations
# =========================

import os
import random
import zipfile
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ------------- Repro -------------
SEED = 1337
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ------------- Paths -------------
ZIP_PATH     = "/content/cedar.zip"
EXTRACT_DIR  = "/content/cedar_extracted"
OUT_DIR      = "/content/triplet_runs"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- Unzip (if needed) -------------
if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Dataset extracted to:", EXTRACT_DIR)
else:
    print("Using existing dataset at:", EXTRACT_DIR)

# ------------- Config -------------
IMG_H, IMG_W   = 160, 256
BATCH_SIZE     = 64
EPOCHS         = 10
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
MARGIN         = 1.0   # keep same scale as your earlier models
BEST_PATH      = os.path.join(OUT_DIR, "model_6.pth")

TRAIN_CSV = "/content/cedar_train_triplet.csv"
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Expected CSV not found: {TRAIN_CSV}")

# =========================
# PATH AUTO-FIXER (applied once to CSV)
# =========================
def fix_csv_paths(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    for col in ["anchor", "positive", "negative"]:
        df[col] = df[col].astype(str).str.replace("\\", "/", regex=False)
        # remove duplicate cedar/cedar if it slipped in
        df[col] = df[col].str.replace("cedar/cedar/", "cedar/", regex=False)
        # ensure starts with cedar/
        df[col] = df[col].apply(lambda p: p if p.startswith("cedar/") else "cedar/" + p)
        # make absolute under base_dir
        df[col] = df[col].apply(lambda p: os.path.join(base_dir, p))
    df.to_csv(csv_path, index=False)
    return csv_path

TRAIN_CSV = fix_csv_paths(TRAIN_CSV, EXTRACT_DIR)

# =========================
# Augmentations (HEAVY)
# =========================
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"

train_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    # strong geometric jitter (white background fill ~ paper color)
    transforms.RandomApply([transforms.RandomRotation(30, fill=255)], p=0.9),
    transforms.RandomApply([transforms.RandomAffine(
        degrees=0, translate=(0.06, 0.06), scale=(0.9, 1.1), shear=12, fill=255
    )], p=0.9),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.6),
    # photometric jitter
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.35),
    transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.35),
    transforms.ColorJitter(brightness=0.15, contrast=0.25),
    transforms.ToTensor(),
    AddGaussianNoise(0.0, 0.02),
    transforms.Normalize((0.5,), (0.5,)),
    # occlusions / missing strokes
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0.0),
])

# A “light” transform to sanity-check samples (not used here but handy)
eval_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# DATASET
# =========================
class TripletDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _open_img(self, p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        return Image.open(p).convert("L")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        a = self._open_img(row["anchor"])
        p = self._open_img(row["positive"])
        n = self._open_img(row["negative"])
        # apply strong, *independent* random aug to each image
        a = self.transform(a)
        p = self.transform(p)
        n = self.transform(n)
        return a, p, n

# =========================
# MODEL ARCHITECTURE (same as Model 2)
# =========================
class EmbeddingNet(nn.Module):
    def __init__(self, img_h, img_w):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, img_w)
            flat_dim = self.cnn(dummy).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # (no L2 norm here to keep scale similar to earlier models)

# =========================
# TRIPLET LOSS
# =========================
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
    def forward(self, a, p, n):
        pos = F.pairwise_distance(a, p)
        neg = F.pairwise_distance(a, n)
        return F.relu(pos - neg + self.margin).mean()

# =========================
# TRAIN LOOP (with AMP, grad clip)
# =========================
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset    = TripletDataset(TRAIN_CSV, transform=train_transform)
num_workers = min(6, os.cpu_count())
loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))

model      = EmbeddingNet(IMG_H, IMG_W).to(device)
criterion  = TripletLoss(MARGIN)
optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler     = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.1)

print(f"\n[Model 6] Training on {device} for {EPOCHS} epochs, batch={BATCH_SIZE}, workers={num_workers}")
best_loss = float("inf")

for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for a, p, n in loader:
        a, p, n = a.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            ea, ep, en = model(a), model(p), model(n)
            loss = criterion(ea, ep, en)

        scaler.scale(loss).backward()
        # gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()

    scheduler.step()
    avg = running / len(loader)
    print(f"Epoch [{epoch:02d}/{EPOCHS}]  loss: {avg:.6f}  lr: {optimizer.param_groups[0]['lr']:.6f}")

    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), BEST_PATH)
        print(f"  ✅ New best — saved to {BEST_PATH}")

print(f"\nTraining complete. Best model saved at {BEST_PATH}")
print("Note: This model uses MUCH heavier augmentation (rot/affine/perspective/blur/noise/erasing).")
# accuracy # 0.6