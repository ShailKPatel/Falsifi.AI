# model 4

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import zipfile

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
IMG_H, IMG_W = 224, 224   # ResNet expects square inputs
BATCH_SIZE = 32
EPOCHS = 10
FREEZE_EPOCHS = 5
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
    if os.path.exists(cedar_dir):
        print("Found cedar/original and cedar/forgeries")
    else:
        raise RuntimeError(f"Could not locate cedar/original under {base_dir}")
    df.to_csv(csv_path, index=False)
    return csv_path

TRAIN_CSV = fix_csv_paths(TRAIN_CSV, EXTRACT_DIR)

# =========================
# DATASET CLASS
# =========================
class SiameseDataset(Dataset):
    def __init__(self, csv_file, base_dir, img_h, img_w):
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.RandomAffine(degrees=3, translate=(0.02,0.02), scale=(0.95,1.05)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) # ImageNet norms
        ])

    def _resolve_path(self, p):
        p = str(p).strip().replace("\\", "/")
        while p.startswith("cedar/cedar/"):
            p = p.replace("cedar/cedar/", "cedar/", 1)
        if not p.startswith("cedar/") and "cedar/" in p:
            p = p.split("cedar/", 1)[-1]
            p = "cedar/" + p
        elif not p.startswith("cedar/"):
            p = "cedar/" + p
        return os.path.join(self.base_dir, p)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path1 = self._resolve_path(row["path1"])
        path2 = self._resolve_path(row["path2"])
        label = torch.tensor(int(row["authentic"]), dtype=torch.float32)

        img1 = Image.open(path1).convert("RGB")  # replicate grayscale → RGB
        img2 = Image.open(path2).convert("RGB")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, label

# =========================
# MODEL ARCHITECTURE
# =========================
class ResNet18Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(pretrained=True)
        # strip off FC, keep backbone
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, 128)

    def forward_once(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # L2 norm
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# =========================
# CONTRASTIVE LOSS
# =========================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        return torch.mean((1 - label) * torch.pow(dist, 2) +
                          label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))

# =========================
# TRAINING LOOP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SiameseDataset(TRAIN_CSV, EXTRACT_DIR, IMG_H, IMG_W)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=min(6, os.cpu_count()), pin_memory=True)

model = ResNet18Siamese().to(device)
criterion = ContrastiveLoss(MARGIN)

# freeze first layers initially
for name, param in model.backbone.named_parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam([
    {"params": model.fc.parameters(), "lr": 1e-3},   # head
], lr=1e-3)

print(f"\nTraining Model 4 (ResNet18 Siamese) on {device} for {EPOCHS} epochs")

best_loss = float("inf")
best_path = os.path.join(OUT_DIR, "model_4.pth")

for epoch in range(EPOCHS):
    # unfreeze after FREEZE_EPOCHS
    print(f"Epoch {epoch+1}/{EPOCHS}")
    if epoch == FREEZE_EPOCHS:
        for param in model.backbone.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.fc.parameters(), "lr": 1e-3},
            {"params": model.backbone.parameters(), "lr": 1e-4},
        ])

    model.train()
    epoch_loss = 0.0
    for img1, img2, label in dataloader:
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

print(f"\nTraining complete. Best Model 4 saved at {best_path}")
