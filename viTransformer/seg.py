import cv2
import os
import numpy as np

def segment_leaves_seeded(img_bgr, min_area=4000):

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # COLOR MASKS
    mask_green = cv2.inRange(hsv, (25,20,20), (95,255,255))
    mask_yellow = cv2.inRange(hsv, (15,40,40), (35,255,255))
    mask_brown = cv2.inRange(hsv, (5,50,20), (20,255,200))
    mask_red1 = cv2.inRange(hsv, (0,50,50), (10,255,255))
    mask_red2 = cv2.inRange(hsv, (170,50,50), (180,255,255))
    mask_dark = cv2.inRange(hsv, (0,0,0), (180,255,60))
    mask_white = cv2.inRange(hsv, (0,0,180), (180,40,255))

    seed = (mask_green | mask_yellow | mask_brown |
            mask_red1 | mask_red2 | mask_dark | mask_white)

    # clean seed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)

    # 🔥 SAFETY CHECKS
    if np.count_nonzero(seed) == 0:
        return np.zeros_like(seed), img_bgr

    if np.count_nonzero(seed) == seed.size:
        seed[0:10, 0:10] = 0  # force background

    # STRONG FG/BG
    kernel = np.ones((5,5), np.uint8)
    sure_fg = cv2.erode(seed, kernel, iterations=2)
    sure_bg = cv2.dilate(seed, kernel, iterations=2)

    gc_mask = np.full(seed.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[sure_bg == 0] = cv2.GC_BGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    # ensure both exist
    if not np.any(gc_mask == cv2.GC_FGD) or not np.any(gc_mask == cv2.GC_BGD):
        return np.zeros_like(seed), img_bgr

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    try:
        cv2.grabCut(img_bgr, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except:
        return np.zeros_like(seed), img_bgr

    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype("uint8")

    # remove small regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return final_mask, img_bgr  


# =========================
# MAIN LOOP
# =========================

base_path = r"C:\Users\sushm\OneDrive\Documents\2\2\train"

images_out = r"D:/data/images"
masks_out = r"D:/data/masks"

os.makedirs(images_out, exist_ok=True)
os.makedirs(masks_out, exist_ok=True)

i = 0

for folder_name in os.listdir(base_path):

    folder_path = os.path.join(base_path, folder_name)

    if not os.path.isdir(folder_path):
        continue

    print("Processing:", folder_name)

    for file in os.listdir(folder_path):

        if not file.lower().endswith(('.jpg','.png','.jpeg')):
            continue

        full_path = os.path.join(folder_path, file)
        img = cv2.imread(full_path)

        if img is None:
            continue

        mask, original = segment_leaves_seeded(img)

        # skip empty masks
        if np.count_nonzero(mask) == 0:
            continue

        # save
        cv2.imwrite(os.path.join(images_out, f"{i}.png"), original)
        cv2.imwrite(os.path.join(masks_out, f"{i}.png"), mask)

        print("Saved:", i)

        i += 1

        if i >= 100:
            break

    if i >= 100:
        break