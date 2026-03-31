import cv2
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
IMAGE_PATH = r"D:/test/field1.jpg"

MIN_LEAF_AREA = 3000
SEED_DILATION = 7
WS_THRESHOLD = 0.40
KERNEL_SIZE = 9

# ==============================
# LOAD IMAGE
# ==============================
img = cv2.imread(IMAGE_PATH)
assert img is not None, "Image not found"

# ==============================
# 1️⃣ WEAK GREEN SEED

# ==============================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV green range
lower = np.array([25, 40, 40])
upper = np.array([95, 255, 255])

hsv_mask = cv2.inRange(hsv, lower, upper)

# Excess Green Index
B, G, R = cv2.split(img)
exg = 2*G - R - B
exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

_, exg_mask = cv2.threshold(exg, 25, 255, cv2.THRESH_BINARY)

# Combine masks
seed = cv2.bitwise_or(hsv_mask, exg_mask)

# Dilate seed
seed = cv2.dilate(
    seed,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SEED_DILATION, SEED_DILATION))
)

# ==============================
# 2️⃣ GRABCUT
# ==============================
gc_mask = np.zeros(seed.shape, np.uint8)
gc_mask[:] = cv2.GC_PR_BGD
gc_mask[seed > 0] = cv2.GC_PR_FGD

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

cv2.grabCut(
    img,
    gc_mask,
    None,
    bgdModel,
    fgdModel,
    5,
    cv2.GC_INIT_WITH_MASK
)

# ==============================
# 3️⃣ LEAF MASK
# ==============================
leaf_mask = np.where(
    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
    255, 0
).astype("uint8")

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
)

leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

leaf_mask = cv2.medianBlur(leaf_mask, 5)

# ==============================
# 4️⃣ WATERSHED SEPARATION
# ==============================
dist = cv2.distanceTransform(leaf_mask, cv2.DIST_L2, 5)
dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

_, sure_fg = cv2.threshold(dist, WS_THRESHOLD, 1.0, cv2.THRESH_BINARY)
sure_fg = (sure_fg * 255).astype("uint8")

sure_bg = cv2.dilate(leaf_mask, kernel, iterations=2)
unknown = cv2.subtract(sure_bg, sure_fg)

num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)

# ==============================
# 5️⃣ SAVE LEAF INSTANCES
# ==============================
os.makedirs("leaves", exist_ok=True)

leaf_id = 1

for label in np.unique(markers):

    if label <= 1:
        continue

    instance_mask = np.zeros_like(leaf_mask)
    instance_mask[markers == label] = 255

    if cv2.countNonZero(instance_mask) < MIN_LEAF_AREA:
        continue

    leaf = cv2.bitwise_and(img, img, mask=instance_mask)

    cv2.imwrite(f"leaves/leaf_{leaf_id}.png", leaf)
    cv2.imwrite(f"leaves/leaf_{leaf_id}_mask.png", instance_mask)

    leaf_id += 1

print(f"✅ Extracted {leaf_id-1} leaf instances")