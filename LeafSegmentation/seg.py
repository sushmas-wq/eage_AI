import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
def segment_leaves_seeded(img_bgr, min_area=4000):
    # 1️⃣ Weak green seed (very permissive)
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

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])

    # seed = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    seed = (mask_green | mask_yellow | mask_brown |
        mask_red1 | mask_red2 | mask_dark | mask_white)

    # Clean seed a bit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)

    # 2️⃣ Initialize GrabCut mask
    h, w = seed.shape

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

# mark probable foreground
    gc_mask[seed > 0] = cv2.GC_PR_FGD

# 🔥 FORCE definite foreground (center region)
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

    # 4️⃣ Extract foreground
    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    # 5️⃣ Remove small junk
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)

    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(final_mask, [c], -1, 255, -1)

    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=final_mask)

    return final_mask, segmented

base_path = r"C:\Users\sushm\OneDrive\Documents\2\2\train\\"

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.isdir(folder_path):
        continue

    i = 0  # reset per class
    print("Processing:", folder_name)

    for file in os.listdir(folder_path):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        full_path = os.path.join(folder_path, file)
        img = cv2.imread(full_path)

        if img is None:
            print("Failed:", full_path)
            continue

        mask, segmented = segment_leaves_seeded(img)

        if i >= 100:
            break

        # create folders
        mask_dir = os.path.join("D:/data/masks", folder_name)
        img_dir = os.path.join("D:/data/images", folder_name)
        seg_dir = os.path.join("D:/data/segmented", folder_name)

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        # save
        cv2.imwrite(os.path.join(mask_dir, f"{i}.png"), mask)
        cv2.imwrite(os.path.join(img_dir, f"{i}.png"), img)
        cv2.imwrite(os.path.join(seg_dir, f"{i}.png"), segmented)

        i += 1