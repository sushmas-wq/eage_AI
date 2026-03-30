import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import numpy as np
from torchvision import models, transforms
from PIL import Image
from skimage.feature import local_binary_pattern

# =============================
# CONFIG
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkptc"
CONF_THRESHOLD = 0.75

# =============================
# TRANSFORMS
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =============================
# MODELS
# =============================
class CropClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = models.mobilenet_v3_large(weights=None)
        in_f = self.net.classifier[3].in_features
        self.net.classifier[3] = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.net(x)

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = models.mobilenet_v3_large(weights=None)
        in_f = self.net.classifier[3].in_features
        self.net.classifier[3] = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.net(x)

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_crop_model():
    ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "best_crop_model.pth"),
        map_location=DEVICE
    )
    model = CropClassifier(len(ckpt["classes"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]

@st.cache_resource
def load_disease_model(crop):
    path = os.path.join(CHECKPOINT_DIR, f"best_disease_{crop}.pth")
    if not os.path.exists(path):
        return None, None

    ckpt = torch.load(path, map_location=DEVICE)
    model = DiseaseClassifier(len(ckpt["classes"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]

# =============================
# SEGMENTATION
# =============================

def segment_leaf(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Leaf = green + yellow + brown + pale
    lower_leaf = np.array([0, 20, 20])
    upper_leaf = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    # Remove obvious background using saturation
    sat = hsv[:, :, 1]
    mask[sat < 25] = 0   # removes sky, white table, paper

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep largest connected component (main leaf)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, img_bgr

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)
    return clean_mask, segmented



# =============================
# DISEASE PIXELS
# =============================
def brown_disease_mask(segmented, leaf_mask):
    hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)

    # Brown / necrotic lesions
    lower_brown = np.array([5, 50, 20])
    upper_brown = np.array([30, 255, 200])

    brown = cv2.inRange(hsv, lower_brown, upper_brown)
    brown = cv2.bitwise_and(brown, brown, mask=leaf_mask)

    return (brown > 0).astype(np.uint8)
def detect_white_fungus(segmented_bgr, leaf_mask):
    """
    Detect white / powdery fungal disease pixels
    """
    hsv = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2HSV)

    # White / gray fungal regions
    lower_white = np.array([0, 0, 180])    # low saturation, high brightness
    upper_white = np.array([180, 50, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply leaf mask
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=leaf_mask)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    return (white_mask > 0).astype(np.uint8)
def disease_color_mask(segmented, leaf_mask):
    hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)

    # Brown / rust / necrosis
    lower_brown = np.array([8, 80, 50])
    upper_brown = np.array([25, 255, 200])

    # Yellow chlorosis
    lower_yellow = np.array([18, 80, 80])
    upper_yellow = np.array([35, 255, 255])

    brown = cv2.inRange(hsv, lower_brown, upper_brown)
    yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    color_mask = cv2.bitwise_or(brown, yellow)
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=leaf_mask)

    return (color_mask > 0).astype(np.uint8)
def disease_texture_mask(segmented, leaf_mask, color_mask):
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, 16, 2, method="uniform")
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    _, texture = cv2.threshold(
        lbp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 🔑 ONLY allow texture where color already indicates disease
    texture = cv2.bitwise_and(texture, texture, mask=color_mask)
    texture = cv2.bitwise_and(texture, texture, mask=leaf_mask)

    return (texture > 0).astype(np.uint8)

def final_disease_mask(segmented, leaf_mask):
    """
    Final disease mask combining color + texture
    White fungus is handled carefully
    """

    color = disease_color_mask(segmented, leaf_mask)
    texture = disease_texture_mask(segmented, leaf_mask, color)

    # Core disease pixels (color + texture)
    core_disease = np.logical_and(color, texture)

    # White fungus pixels (color-dominant, texture optional)
    white_fungus = detect_white_fungus(segmented, leaf_mask)

    # Final mask = core disease OR white fungus
    disease = np.logical_or(core_disease, white_fungus).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease = cv2.morphologyEx(disease, cv2.MORPH_OPEN, kernel)
    disease = cv2.morphologyEx(disease, cv2.MORPH_CLOSE, kernel)

    return disease
def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
# =============================
# SEVERITY
# =============================
def compute_severity(disease_mask, leaf_mask):
    diseased_pixels = np.sum(disease_mask)
    leaf_pixels = np.sum(leaf_mask > 0)
    return (diseased_pixels / (leaf_pixels + 1e-7)) * 100

# =============================
# PREDICTION
# =============================
def predict(img_pil, model, classes):
    if isinstance(img_pil, np.ndarray):
        img_pil = Image.fromarray(img_pil)
    x = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
    idx = probs.argmax(1).item()
    return classes[idx], probs.max(1)[0].item()

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="🌱 Crop & Disease + Severity", layout="centered")
st.title("🌿 Crop, Disease & Severity Detection")

uploaded = st.file_uploader("Upload leaf image", ["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Analysis"):   
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        leaf_mask, segmented = segment_leaf(img_bgr)
        st.image(segmented,caption="Pre-Processing done",use_container_width=True)
        
        disease_mask = final_disease_mask(segmented, leaf_mask)
        severity = compute_severity(disease_mask, leaf_mask)
      
        crop_model, crop_classes = load_crop_model()
        crop, crop_conf = predict(segmented, crop_model, crop_classes)

        st.success(f"🌱 Crop: {crop} ({crop_conf:.2f})")
        disease_model, disease_classes = load_disease_model(crop)
        disease, disease_conf = predict(img, disease_model, disease_classes)
        st.error(f"🦠 Disease: {disease} ({disease_conf:.2f})")

        if "healthy" in disease.lower():
            st.success("✅ Healthy leaf")
            st.info("Disease Severity: **0.00%**")
            st.stop()

        overlay = segmented.copy()
        overlay[disease_mask == 1] = [0, 0, 255]
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        st.warning(f"🩺 Disease Severity: **{severity:.2f}%**")
        st.image(overlay, caption="Diseased pixels (red)", use_container_width=True)
        st.stop()