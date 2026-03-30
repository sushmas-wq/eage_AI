import streamlit as st
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern

# ======================================================
# APP CONFIG
# ======================================================
st.set_page_config(page_title="🌱 Crop & Disease Evaluation", layout="centered")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpt"

# ======================================================
# SEGMENTATION (CONSERVATIVE)
# ======================================================
def segment_leaves_seeded(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20, 20, 20])
    upper_green = np.array([95, 255, 255])
    seed = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)

    gc_mask = np.full(seed.shape, cv2.GC_PR_BGD, np.uint8)
    gc_mask[seed > 0] = cv2.GC_PR_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, gc_mask, None,
                bgdModel, fgdModel, 3,
                cv2.GC_INIT_WITH_MASK)

    leaf_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype("uint8")

    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=leaf_mask)
    return leaf_mask, segmented

# ======================================================
# COLOR-BASED DISEASE PIXELS
# ======================================================
def detect_disease_pixels(segmented, leaf_mask):
    hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([5, 40, 20])
    upper_brown = np.array([30, 255, 200])

    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([35, 255, 255])

    brown = cv2.inRange(hsv, lower_brown, upper_brown)
    yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    disease = cv2.bitwise_or(brown, yellow)
    disease = cv2.bitwise_and(disease, disease, mask=leaf_mask)

    return (disease > 0).astype(np.uint8)

# ======================================================
# LBP TEXTURE DISEASE PIXELS (STRIPES / RUST)
# ======================================================
def lbp_texture_mask(segmented, leaf_mask):
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, 16, 2, method="uniform")
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    _, mask = cv2.threshold(lbp_norm, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_and(mask, mask, mask=leaf_mask)
    return (mask > 0).astype(np.uint8)

# ======================================================
# FINAL DISEASE MASK (COLOR + TEXTURE)
# ======================================================
def final_disease_mask(segmented, leaf_mask):
    color_mask = detect_disease_pixels(segmented, leaf_mask)
    texture_mask = lbp_texture_mask(segmented, leaf_mask)
    return np.logical_or(color_mask, texture_mask).astype(np.uint8)
import re

def extract_label_parts(label: str) -> str:
    """
    Normalize PlantVillage / custom labels and return:
    - 'healthy' OR
    - cleaned disease name
    """

    label = str(label).strip()

    # Handle PlantVillage formats
    if "___" in label:
        disease = label.split("___", 1)[1]
    elif "__" in label:
        disease = label.split("__", 1)[1]
    elif "_" in label:
        disease = label.split("_", 1)[1]
    else:
        disease = label

    # ✅ Healthy detection (case-insensitive, early return)
    if re.search(r"healthy", disease, re.IGNORECASE):
        return "healthy"

    # Clean formatting
    disease = disease.replace("_", " ").strip()
    disease = re.sub(r"\s+", " ", disease)

    return disease.lower()

# ======================================================
# SEVERITY
# ======================================================
def compute_severity(disease_mask, leaf_mask):
    diseased = np.sum(disease_mask)
    leaf = np.sum(leaf_mask > 0)
    return (diseased / (leaf + 1e-7)) * 100

# ======================================================
# TRANSFORM
# ======================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# MODELS
# ======================================================
class Classifier(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = models.mobilenet_v3_large(weights=None)
        self.net.classifier[3] = nn.Linear(
            self.net.classifier[3].in_features, n
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = Classifier(len(ckpt["classes"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]

def predict(model, img, classes):
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = F.softmax(model(x), dim=1)
        conf, idx = torch.max(p, dim=1)
    return classes[idx.item()], conf.item()

# ======================================================
# UI
# ======================================================
st.title("🌱 Crop & Disease Evaluation")

file = st.file_uploader("Upload a leaf image", ["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("🔍 Analyze"):
        crop_model, crop_classes = load_model(
            os.path.join(CHECKPOINT_DIR, "best_crop_model.pth")
        )
        crop, crop_conf = predict(crop_model, image, crop_classes)

        st.success(f"🌾 Crop: {crop} ({crop_conf:.2f})")

        disease_path = os.path.join(
            CHECKPOINT_DIR, f"best_disease_{crop}.pth"
        )

        if not os.path.exists(disease_path):
            st.warning("No disease model available")
        else:
            disease_model, disease_classes = load_model(disease_path)
            disease, d_conf = predict(disease_model, image, disease_classes)

            st.error(f"🦠 Disease: {disease} ({d_conf:.2f})")
            disease = extract_label_parts(disease)

            if disease == "healthy":
                severity = 0.0
                st.info("🟢 Disease Severity: 0.00%")
                st.stop()   # 🚨 REQUIRED
            else:
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                leaf_mask, segmented = segment_leaves_seeded(img_bgr)
                disease_mask = final_disease_mask(segmented, leaf_mask)

                severity = compute_severity(disease_mask, leaf_mask)
                st.warning(f"🦠 Disease Severity: {severity:.2f}%")

                overlay = segmented.copy()
                overlay[disease_mask == 1] = [0, 0, 255]

                st.image(
                    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                    caption="Detected disease pixels (red)",
                    use_container_width=True
                )