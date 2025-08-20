import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# =========================
# MODEL ARCHITECTURE (must match training)
# =========================
class ResNet50Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, 128)

    def forward_once(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


# =========================
# LOAD TRAINED WEIGHTS
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50Siamese().to(device)

model.load_state_dict(torch.load("model_5.pth", map_location=device))
model.eval()
print("✅ Model 5 loaded successfully")
def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    return transforms(image).unsqueeze(0)  # add batch dimension

# Example images
img1_path = "test1.png"   # genuine signature
img2_path = "test2.png"   # suspected signature

img1 = load_image(img1_path).to(device)
img2 = load_image(img2_path).to(device)

with torch.no_grad():
    out1, out2 = model(img1, img2)
    distance = F.pairwise_distance(out1, out2).item()

print(f"Pairwise Distance: {distance:.4f}")

# ✅ Decision Rule (tune threshold based on validation set)
threshold = 0.8
if distance < threshold:
    print("Prediction: ✅ Same person (authentic)")
else:
    print("Prediction: ❌ Different person (forgery)")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# =========================
# MODEL ARCHITECTURE (must match training)
# =========================
class ResNet50Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, 128)

    def forward_once(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


# =========================
# LOAD TRAINED WEIGHTS
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50Siamese().to(device)

model.load_state_dict(torch.load("model_5.pth", map_location=device))
model.eval()
print("✅ Model 5 loaded successfully")
