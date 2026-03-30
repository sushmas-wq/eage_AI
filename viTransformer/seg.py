import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
def segment_leaves_seeded(img_bgr, min_area=4000):
    # 1️⃣ Weak green seed (very permissive)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 20, 20])   # VERY loose
    upper_green = np.array([95, 255, 255])

    seed = cv2.inRange(hsv, lower_green, upper_green)

    # Clean seed a bit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)

    # 2️⃣ Initialize GrabCut mask
    gc_mask = np.full(seed.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[seed > 0] = cv2.GC_PR_FGD

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
folder= input("enter folder path:")
img = cv2.imread(r"D:\test\field4.jpg")

mask, segmented = segment_leaves_seeded(img)

cv2.imwrite("m_field4.png", mask)
cv2.imwrite("s_field4.png", segmented)