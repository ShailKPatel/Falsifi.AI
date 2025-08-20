import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

st.markdown("""
<style>
/* Global dark background and white text */
body {
    background-color: black !important;
    color: white !important;
}

/* All text elements white (except h1 which is overridden by class) */
h2, h3, h4, h5, h6, p, li, a, label {
    color: white;
}

/* Custom H1 color override */
.custom-h1 {
    color: #374151 !important;     /* Dark gray */
    font-weight: 800 !important;   /* Bold */
    font-size: 3.5rem !important;  /* Larger font size */
}


/* Custom text selection style */
::selection {
    background: white;
    color: black;
}
::-moz-selection {
    background: white;
    color: black;
}

/* Clean block spacing */
.block-container {
    padding: 2rem 1rem;
}

/* Module boxes */
.module-box {
    background-color: #111;
    padding: 1.5rem;
    border: 1px solid #444;
    border-radius: 10px;
    margin-bottom: 1rem;
    height: 100%;
}
.module-box a {
    color: white !important;
    font-weight: bold;
    text-decoration: none;
}

.custom-h1 {
    color: #4B5563 !important;     
    font-weight: 800 !important;   
    font-size: 3.75rem !important;  
    margin-bottom: 0rem !important;
    margin-top: 0rem !important;
}

.custom-subtitle {
    font-size: 1.25rem !important;
    color: white !important;
    margin-top: 0.25rem !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL ARCHITECTURE (from model_5.py)
# =========================
class ResNet50Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # remove final fc
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
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50Siamese().to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_5.pth')
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None, None

    try:
        state_dict = torch.load(model_path, map_location=device)

        # Ensure we only load if it's really a state_dict
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
        else:
            st.error("❌ model_5.pth is not a state_dict. Please re-save using torch.save(model.state_dict(), path)")
            return None, None

        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


model, device = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return transform(image).unsqueeze(0)

# -------------------------------
# Title Section 
# -------------------------------
st.markdown("""
<h1 class="custom-h1">Signature Verification</h1>
""", unsafe_allow_html=True)

st.markdown("""
### Project Understanding

This page allows you to verify the authenticity of a signature by comparing it against an anchor image. 

1.  *Upload Anchor Image*: This is the known, genuine signature.
2.  *Upload Image to be Verified*: This is the signature you want to check.
3.  *Select a Model*: Choose the model you want to use for verification.
4.  *Submit*: Click the submit button to perform the verification.
""")

# -------------------------------
# Image Upload Section
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.header("Anchor Image")
    anchor_image = st.file_uploader("Upload an anchor image", type=['png', 'jpg', 'jpeg'], key='anchor_image')

with col2:
    st.header("Image to Verify")
    verifi_image = st.file_uploader("Upload an image to be verified", type=['png', 'jpg', 'jpeg'], key='verifi_image')

# -------------------------------
# Model Selection and Submission
# -------------------------------
st.header("Model Selection")
model_list = ["Model 5"]
selected_model = st.selectbox("Select a model", model_list)

st.write(f"You selected: {selected_model}")

if st.button("Submit"):
    if anchor_image and verifi_image and model:
        with st.spinner("Verifying..."):
            img1 = load_image(anchor_image).to(device)
            img2 = load_image(verifi_image).to(device)

            with torch.no_grad():
                out1, out2 = model(img1, img2)
                distance = F.pairwise_distance(out1, out2).item()

            st.write(f"### Result")
            st.write(f"*Pairwise Distance:* {distance:.4f}")

            # Decision Rule (tune threshold based on validation set)
            threshold = 0.8 
            if distance < threshold:
                st.success("Prediction: ✅ Same person (authentic)")
            else:
                st.error("Prediction: ❌ Different person (forgery)")
    elif not model:
        st.warning("Model is not loaded. Cannot perform verification.")
    else:
        st.warning("Please upload both images.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div style="background-color: #111827; padding: 20px 0; max-width: 100%; margin-bottom: 0;">
    <div style="text-align: center; color: gray;">
        © 2025 Shail K Patel · Crafted out of boredom.
    </div>
    <div style="text-align: center; color: gray;">
        <a href="https://github.com/ShailKPatel/Falsifi.AI/" style="color: gray; text-decoration: none;">GitHub Repo</a> · MIT License
    </div>
</div>
""", unsafe_allow_html=True)