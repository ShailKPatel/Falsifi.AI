# model 3
import os, random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

# Faster conv kernels
torch.backends.cudnn.benchmark = True

# =========================
# CONFIG
# =========================
IMG_H, IMG_W = 160, 256
BATCH_SIZE = 64
EPOCHS = 10
MARGIN = 1.0
LR = 1e-3
AUGMENT = True
OUT_DIR = "/content/siamese_runs"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# ELASTIC FIELD PRECOMPUTE
# =========================
def make_elastic_field(H, W, alpha=34.0, sigma=4.0):
    dx = torch.randn(1, 1, H, W)
    dy = torch.randn(1, 1, H, W)
    ksize = int(4 * sigma) | 1
    coords = torch.arange(ksize) - (ksize // 2)
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    gk = g.view(1, 1, -1)

    dx = F.conv2d(dx, gk.unsqueeze(2), padding=(ksize // 2, 0))
    dx = F.conv2d(dx, gk.unsqueeze(3), padding=(0, ksize // 2))
    dy = F.conv2d(dy, gk.unsqueeze(2), padding=(ksize // 2, 0))
    dy = F.conv2d(dy, gk.unsqueeze(3), padding=(0, ksize // 2))

    dx, dy = dx * alpha, dy * alpha
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    dx_norm = dx[0, 0] * (2.0 / max(W - 1, 1))
    dy_norm = dy[0, 0] * (2.0 / max(H - 1, 1))
    grid = torch.stack((xx + dx_norm, yy + dy_norm), dim=-1).unsqueeze(0)
    return grid

def apply_elastic(img_t, grid):
    return F.grid_sample(img_t.unsqueeze(0), grid, mode="bilinear",
                         padding_mode="border", align_corners=True)[0]

# =========================
# DATASET CLASS
# =========================
class SiameseDataset(Dataset):
    def __init__(self, csv_file, base_dir, img_h, img_w, augment=False, n_fields=50):
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.img_h, self.img_w = img_h, img_w
        self.augment = augment
        self.base_transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if augment:
            self.elastic_fields = [make_elastic_field(img_h, img_w) for _ in range(n_fields)]

    def _resolve(self, rel_path):
        """Fix Cedar paths. Handles both original and forgery images."""
        rel_path = str(rel_path).replace("\\", "/")

        if "original" in rel_path.lower():
            return os.path.join(self.base_dir, "original", os.path.basename(rel_path))
        elif "forgeries" in rel_path.lower() or "forg" in rel_path.lower():
            return os.path.join(self.base_dir, "forgeries", os.path.basename(rel_path))
        else:
            # Default fallback: assume inside cedar/
            return os.path.join(self.base_dir, "cedar", os.path.basename(rel_path))

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img1 = Image.open(self._resolve(row["path1"])).convert("L")
        img2 = Image.open(self._resolve(row["path2"])).convert("L")
        label = torch.tensor(int(row["authentic"]), dtype=torch.float32)

        if self.augment:
            img1 = img1.resize((self.img_w, self.img_h))
            img2 = img2.resize((self.img_w, self.img_h))
            t1, t2 = transforms.ToTensor()(img1), transforms.ToTensor()(img2)

            angle = random.uniform(-10, 10)
            t1 = TF.rotate(t1, angle, fill=1.0)
            t2 = TF.rotate(t2, angle, fill=1.0)

            grid = random.choice(self.elastic_fields)
            t1 = apply_elastic(t1, grid)
            t2 = apply_elastic(t2, grid)

            sigma = random.uniform(0.0, 0.05)
            t1 = (t1 + torch.randn_like(t1) * sigma).clamp(0, 1)
            t2 = (t2 + torch.randn_like(t2) * sigma).clamp(0, 1)

            t1 = TF.normalize(t1, [0.5], [0.5])
            t2 = TF.normalize(t2, [0.5], [0.5])
        else:
            t1, t2 = self.base_transform(img1), self.base_transform(img2)

        return t1, t2, label

    def __len__(self): return len(self.data)

# =========================
# MODEL
# =========================
class SiameseNetwork(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            flat_dim = self.cnn(dummy).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x).view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# =========================
# CONTRASTIVE LOSS
# genuine=1, forgery=0
# =========================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin): super().__init__(); self.margin = margin
    def forward(self, o1, o2, y):
        d = F.pairwise_distance(o1, o2)
        return (y * d.pow(2) + (1 - y) * torch.clamp(self.margin - d, min=0).pow(2)).mean()

# =========================
# TRAINING LOOP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_csv = "/content/cedar_train_siamese.csv"
base_dir = "/content/cedar_extracted/cedar"

dataset = SiameseDataset(train_csv, base_dir, IMG_H, IMG_W, augment=AUGMENT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=min(6, os.cpu_count() + 2),
                    pin_memory=(device.type == "cuda"), prefetch_factor=2)

model = SiameseNetwork(IMG_H, IMG_W).to(device)
criterion, optimizer = ContrastiveLoss(MARGIN), torch.optim.Adam(model.parameters(), lr=LR)

print(f"[SigNet+Augmentation] Training on {device}")
best_loss, best_path = float("inf"), os.path.join(OUT_DIR, "model_3.pth")

scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            o1, o2 = model(x1, x2)
            loss = criterion(o1, o2, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    avg = total / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss {avg:.6f}")
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), best_path)
        print(f"Best updated at: {best_path}")
