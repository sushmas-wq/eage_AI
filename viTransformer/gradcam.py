import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_PATH = r"D:/test/field3.JPG"         # path to image
CHECKPOINT = "super_model_81.pth"  # your trained model
print("ckecpoint.keys:",torch.load(CHECKPOINT, map_location=DEVICE).keys())
print("using:",CHECKPOINT)
IMG_SIZE = 224


# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# LOAD IMAGE
# -----------------------------




img_pil = Image.open(IMG_PATH).convert("RGB")
img_np = np.array(img_pil)

# --- HSV conversion ---
hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

# --- Green mask ---
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# --- Morphological cleanup ---
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# --- Soft mask (important!) ---
mask = cv2.GaussianBlur(mask, (15, 15), 0)
mask = mask.astype(np.float32) / 255.0
mask = np.expand_dims(mask, axis=2)

# --- Background suppression without black pixels ---
mean_color = np.array([123, 117, 104], dtype=np.float32)  # ImageNet mean * 255
masked_img = img_np * mask + mean_color * (1 - mask)
masked_img = masked_img.astype(np.uint8)

image_pil = Image.fromarray(masked_img)

# --- Final tensor ---
input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

# -----
# LOAD MODEL
# -----------------------------
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
class_names = checkpoint["class_names"]

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# -----------------------------
# GRAD-CAM CLASS
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward()

        # Global Average Pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam, class_idx

# -----------------------------
# RUN GRAD-CAM
# -----------------------------
target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

cam, pred_idx = gradcam.generate(input_tensor)
pred_name = class_names[pred_idx]

# -----------------------------
# VISUALIZE
# -----------------------------
img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

cv2.imwrite("gradcam_heatmap.jpg", heatmap)
cv2.imwrite("gradcam_overlaya2.jpg", overlay)

print(f"✅ Predicted class: {pred_name}")
print("📁 Saved: gradcam_heatmap.jpg, gradcam_overlay2.jpg")
