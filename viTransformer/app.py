from multiprocessing.util import info
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
import re


st.set_page_config(
    page_title="AgroAI 🌱",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #f4fbf6;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headers */
h1, h2, h3 {
    color: #1b5e20;
    font-weight: 600;
}

/* Cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #43a047, #66bb6a);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: 600;
}

/* Success box */
.stSuccess {
    border-radius: 12px;
}

/* Error box */
.stError {
    border-radius: 12px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #66bb6a;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# CONFIG
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkptc"
CONF_THRESHOLD = 0.75
#  ===========================
#  INFO
# ============================
infoa = {
    "healthy": {
        "title": "Plant appears healthy 🌱",
        "cause": "No visible disease symptoms detected.",
        "advice": "Continue regular monitoring and proper irrigation."
    },
    "Apple Scab":{
        "title": "Apple Scab detected 🍏",
        "cause": "Fungal disease causing dark lesions on leaves and fruit.",
        "advice": "Remove infected material and apply fungicide."
    },
    "Black Rot":{
        "title": "Black Rot detected 🍎",
        "cause": "Fungal infection leading to black lesions on leaves and fruit.",
        "advice": "Prune affected areas and use appropriate fungicides."
    },
    "Cedar Apple Rust":{
        "title": "Cedar Apple Rust detected 🌿",
        "cause": "Fungal disease requiring both apple and cedar hosts.",
        "advice": "Remove nearby cedar trees and apply fungicide."
    },
    "Cercospora Leaf Spot":{
        "title": "Cercospora Leaf Spot detected 🌽",
        "cause": "Fungal infection causing circular spots on leaves.",
        "advice": "Improve air circulation and apply fungicide."
    },
    "Common Rust": {
        "title": "Common Rust detected 🌽",
        "cause": "Fungal disease producing rust-colored pustules on leaves.",
        "advice": "Use resistant varieties and apply fungicide if needed."
    },"Esca (Black Measles)": {
        "title": "Esca (Black Measles) detected 🍇",
        "cause": "Fungal disease affecting grapevines.",
        "advice": "Remove infected vines and apply appropriate fungicides."
    },"Haunglongbing (Citrus greening)": {
        "title": "Haunglongbing (Citrus greening) detected  🍊",
        "cause": "Bacterial disease spread by psyllid insects.",
        "advice": "Remove infected trees and control psyllid population."
    },"Tomato Yellow Leaf Curl Virus": {
        "title": "Tomato Yellow Leaf Curl Virus detected 🍅",
        "cause": "Viral disease transmitted by whiteflies.",
        "advice": "Control whitefly population and remove infected plants."
    },"Brown Rust": {
        "title": "Brown Rust detected 🌾",
        "cause": "Fungal disease producing brown pustules on wheat leaves.",
        "advice": "Use resistant varieties and apply fungicide if needed."
    },
    "Leaf Rust": {
        "title": "Leaf Rust detected 🍂",
        "cause": "Fungal disease spread by airborne spores in humid conditions.",
        "advice": "Remove infected leaves and apply fungicide if needed."
    },
    "Powdery mildew": {
        "title": "Powdery Mildew detected 🌫️",
        "cause": "Fungus growing on leaf surfaces due to poor air circulation.",
        "advice": "Improve airflow and apply sulfur-based treatment."
    },
    "Bacterial Blight": {
        "title": "Bacterial Blight detected 🦠",
        "cause": "Bacterial infection spread via rain splash and tools.",
        "advice": "Use clean tools and disease-free seeds."
    },
    "Leaf Spot": {
        "title": "Leaf Spot detected 🔴",
        "cause": "Fungal or bacterial pathogens favored by wet leaves.",
        "advice": "Avoid overhead watering and improve drainage."
    },
    "Late Blight":{
          "title": "Late Blight detected 🌧️",
        "cause": "Fungal disease thriving in cool, wet conditions.",
        "advice": """
<ul>
  <li>Remove and destroy infected plant material immediately to reduce spread</li>
  <li>Apply a bio-enzyme / microbial formulation (such as enzyme-based or beneficial microbe solutions)</li>
  <li>Ensure good air circulation and avoid overhead irrigation</li>
  <li>Apply preventively spray during high-humidity periods for best results</li>
  <li>👉 2–3 ml  of bio-enzyme per liter of water</li>
  <li>Spray every 10–14 days</li>
</ul>
"""
    },
    "Early Blight":{
        "title": "Early Blight detected 🌞",
        "cause": "Fungal disease favored by warm, dry weather.",
        "advice": "Remove infected material and apply fungicide."
    },
    "Tungro":{
        "title": "Tungo detected 🌾",
        "cause": "Caused by Rice Tungro Virus transmitted by green leafhoppers.",
        "advice": "Remove infected plants, control leafhopper population, and use resistant rice varieties."
},
}
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

def segment_leaf(img_bgr, min_area=4000):
    # 1️  Weak green seed (very permissive)
    max_size = 512
    h, w = img_bgr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    

  # GREEN
    lower_green = np.array([25, 20, 20])   # VERY loose
    upper_green = np.array([95, 255, 255])
  
 

 # YELLOW
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])

 # BROWN (low saturation, darker)
    lower_brown = np.array([5, 50, 20])
    upper_brown = np.array([20, 255, 200])

 # RED (two ranges in HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

 # DARK / BLACK (diseased spots)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])

 # PALE / WHITE

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    #dark / black (diseased spots)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])

    # seed = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    seed = (mask_green | mask_yellow | mask_brown |
        mask_red1 | mask_red2 | mask_dark | mask_white | mask_black)

    # Clean seed a bit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)

    # 2️  Initialize GrabCut mask
    h, w = seed.shape

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

# mark probable foreground
    gc_mask[seed > 0] = cv2.GC_PR_FGD

#  FORCE definite foreground (center region)
    gc_mask[h//4:3*h//4, w//4:3*w//4][seed[h//4:3*h//4, w//4:3*w//4] > 0] = cv2.GC_FGD

    gc_mask[:10, :] = cv2.GC_BGD
    gc_mask[-10:, :] = cv2.GC_BGD
    gc_mask[:, :10] = cv2.GC_BGD
    gc_mask[:, -10:] = cv2.GC_BGD
    # 3️⃣ Run GrabCut (color used internally, but shape dominates)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img_bgr,
        gc_mask,
        None,
        bgdModel,
        fgdModel,
        iterCount=5,
        mode=cv2.GC_INIT_WITH_MASK
    )

    # 4️  Extract foreground
    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)

    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(final_mask, [c], -1, 255, -1)

    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=final_mask)

    return final_mask, segmented


# ============================
# DISEASE SEGMENTATION
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
    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    color_mask = cv2.bitwise_or(brown, yellow)
    color_mask = cv2.bitwise_or(color_mask, black)
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
    return classes[idx], probs.max(1)[0]
def extract_label_parts(label):
    

    # Handle PlantVillage formats
    if "___" in label:
        disease = label.split("___", 1)[1]
    elif "__" in label:
        disease = label.split("__", 1)[1]
    elif "_" in label:
        disease = label.split("_", 1)[1]
    else:
        disease = label

    # Healthy check
    if re.search(r"healthy", disease, re.IGNORECASE):
        return "healthy"

    disease = disease.replace("_", " ").strip()
    disease = re.sub(r"\s+", " ", disease)

    return disease
   
    


# =============================
# STREAMLIT UI
# =============================

st.set_page_config(
    page_title="AgroAI 🌱",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #f4fbf6;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headers */
h1, h2, h3 {
    color: #1b5e20;
    font-weight: 600;
}

/* Cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #43a047, #66bb6a);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: 600;
}

/* Success box */
.stSuccess {
    border-radius: 12px;
}

/* Error box */
.stError {
    border-radius: 12px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #66bb6a;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <h1 style='color: green;'>🌱 AgroAI</h1>
    <p  style='color: green;'>AI-powered crop & disease detection from real-world field images.</p>
</div>
""", unsafe_allow_html=True)
#2
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png"])

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Analysis"):
      with st.spinner("Analyzing leaf... "):   
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        leaf_mask, segmented = segment_leaf(img_bgr)
        st.image(segmented,caption="Pre-Processing done",use_container_width=True)
        
        disease_mask = final_disease_mask(segmented, leaf_mask)
        severity = compute_severity(disease_mask, leaf_mask)
      
        crop_model, crop_classes = load_crop_model()
        crop, crop_conf = predict(segmented, crop_model, crop_classes)

        st.success(f"🌱 Crop: {crop} ({crop_conf.item():.2f})")
        st.progress(float(crop_conf))
        disease_model, disease_classes = load_disease_model(crop)
        disease, disease_conf = predict(img, disease_model, disease_classes)
        st.error(f"🦠 Disease: {disease} ({disease_conf.item():.2f})")
        st.progress(float(disease_conf))

        
        label = extract_label_parts(disease)
        label_key = label.title()
       

        if "healthy" in disease.lower():
            st.success("✅ Healthy leaf")
            st.info("Disease Severity: **0.00%**")
            st.stop()

        overlay = segmented.copy()
        overlay[disease_mask == 1] = [0, 0, 255]
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        st.info(f"🩺 Disease Severity: **{severity:.2f}%**")
        
        st.image(overlay, caption="Diseased pixels (red)", use_container_width=True)
        info = infoa.get(label_key, {
        "title": "Unknown condition",
        "cause": "Pattern does not match known diseases.",
        "advice": "Consult an agriculture expert."
        })
        st.markdown("## 🌿 Disease Insight")
        st.markdown(
        f"""
        <div style='color: blue;'>
        <b>{info['title']}</b><br><br>
        <b>Cause:</b> {info['cause']}<br><br>
        <b>Recommended Action:</b> {info['advice']}<br><br>
        </div>
        """,
        unsafe_allow_html=True
        ) 
#------------------------------
        st.stop()