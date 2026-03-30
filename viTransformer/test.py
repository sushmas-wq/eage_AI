import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =============================
# CONFIG
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpt"
TEST_DIR = r"D:\2\test"
BATCH_SIZE = 32

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
# FUNCTIONS
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
# ===================================
# DISEASE PIXEL DETECTION
# =================================== 
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
# LOAD CROP MODEL
# =============================
crop_ckpt = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_crop_model.pth"),
    map_location=DEVICE
)

crop_classes = crop_ckpt["classes"]
crop_model = CropClassifier(len(crop_classes)).to(DEVICE)
crop_model.load_state_dict(crop_ckpt["model_state"])
crop_model.eval()

print("\n✅ Loaded Crop Model")
print("Crop Classes:", crop_classes)

# =============================
# LOAD DISEASE MODELS (CACHE)
# =============================
disease_cache = {}

def load_disease_model(crop):
    if crop in disease_cache:
        return disease_cache[crop]

    path = os.path.join(CHECKPOINT_DIR, f"best_disease_{crop}.pth")
    if not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location=DEVICE)

    model = DiseaseClassifier(len(ckpt["classes"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    disease_cache[crop] = (model, ckpt)
    return disease_cache[crop]



# =============================
# 1️⃣ CROP LEVEL EVALUATION
# =============================
print("\n==============================")
print("📊 CROP LEVEL EVALUATION")
print("==============================")

crop_ds = ImageFolder(TEST_DIR, transform=transform)
crop_loader = DataLoader(crop_ds, batch_size=BATCH_SIZE, shuffle=False)

y_true, y_pred, confs = [], [], []

with torch.no_grad():
    for x, y in crop_loader:

        x = x.to(DEVICE)

        logits = crop_model(x)
        probs = F.softmax(logits, dim=1)

        preds = probs.argmax(1).cpu().numpy()

        y_true.extend(y.numpy())
        y_pred.extend(preds)
        confs.extend(probs.max(1)[0].cpu().numpy())

# compute accuracy once
crop_accuracy = accuracy_score(y_true, y_pred)

labels = sorted(set(y_true) | set(y_pred))

# print("\nConfusion Matrix:")
#print(confusion_matrix(y_true, y_pred, labels=labels))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[crop_classes[i] for i in labels],
        zero_division=0
    )
)

print(f"\n🌾 Overall Crop Accuracy: {crop_accuracy*100:.2f}%")
# =============================
# 2️⃣ DISEASE LEVEL EVALUATION
# =============================
print("\n==============================")
print("🦠 DISEASE LEVEL EVALUATION")
print("==============================")

TOTAL_ACC = 0.0
disease_models_used = 0
ALL_SEVERITIES = []

for crop in crop_classes:

    crop_path = os.path.join(TEST_DIR, crop)

    if not os.path.exists(crop_path):
        continue

    data = load_disease_model(crop)

    if data is None:
        print(f"\n⚠️ No disease model for {crop}")
        continue

    model, ckpt = data
    disease_classes = ckpt["classes"]

    print(f"\n📌 Crop: {crop}")
    print("Disease Classes:", disease_classes)

    ds = ImageFolder(crop_path, transform=transform)

    # enforce same class order
    ds.classes = ckpt["classes"]
    ds.class_to_idx = {c: i for i, c in enumerate(ds.classes)}

    lower_class_map = {c.lower(): c for c in ds.classes}

    new_samples = []

    for p, _ in ds.samples:

        folder = os.path.basename(os.path.dirname(p)).lower()

        if folder not in lower_class_map:
            raise ValueError(
                f"Unknown disease folder '{folder}' for crop '{crop}'."
            )

        true_class = lower_class_map[folder]
        new_samples.append((p, ds.class_to_idx[true_class]))

    ds.samples = new_samples
    ds.targets = [y for _, y in ds.samples]
    ds.imgs = ds.samples

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred, confs = [], [], []

    severity_scores = []
    img_index = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            logits = model(x)
            probs = F.softmax(logits, dim=1)

            preds = probs.argmax(1).cpu().numpy()

            y_true.extend(y.numpy())
            y_pred.extend(preds)
            confs.extend(probs.max(1)[0].cpu().numpy())

            # severity calculation
            for i in range(len(x)):

                img_path, _ = ds.samples[img_index]

                img_bgr = cv2.imread(img_path)

                leaf_mask, segmented = segment_leaf(img_bgr)

                disease_mask = final_disease_mask(segmented, leaf_mask)

                severity = compute_severity(disease_mask, leaf_mask)

                severity_scores.append(severity)
                ALL_SEVERITIES.append(severity)

                img_index += 1

    # metrics AFTER loop
    labels = list(range(len(disease_classes)))

    # print("\nConfusion Matrix:")
   # print(confusion_matrix(y_true, y_pred, labels=labels))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=disease_classes,
            zero_division=0
        )
    )

    overall = accuracy_score(y_true, y_pred)

    print(f"\n🦠 Disease Accuracy: {overall*100:.2f}%")

    avg_severity = np.mean(severity_scores)

    print(f"🌱 Average Severity for {crop}: {avg_severity:.2f}%")

    TOTAL_ACC += overall
    disease_models_used += 1
print("\n==============================")
print("📊 FINAL SYSTEM PERFORMANCE")
print("==============================")

overall_disease_acc = (TOTAL_ACC / disease_models_used) * 100

global_severity = np.mean(ALL_SEVERITIES)

print(f"\n🌾 Overall Crop Accuracy: {crop_accuracy*100:.2f}%")
print(f"🦠 Overall Disease Accuracy: {overall_disease_acc:.2f}%")
print(f"🌱 Global Average Disease Severity: {global_severity:.2f}%")

print("\n✅ Evaluation Complete.")