#pip install scikit-image scikit-learn opencv-python
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, classification_report
import os
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# LBP parameters
RADIUS = 2
N_POINTS = 8 * RADIUS
METHOD = "uniform"

def extract_lbp(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))

    lbp = local_binary_pattern(img, N_POINTS, RADIUS, METHOD)

    # Uniform LBP → P + 2 bins
    n_bins = N_POINTS + 2
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return hist


def load_lbp_dataset(root_dir):
    X, y = [], []
    class_to_idx = {}

    class_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)
        class_to_idx[class_name] = idx

        images = os.listdir(class_path)

        # Progress bar per class
        for img_name in tqdm(
            images,
            desc=f"Processing {class_name}",
            leave=False
        ):
            img_path = os.path.join(class_path, img_name)
            try:
                features = extract_lbp(img_path)
                X.append(features)
                y.append(idx)
            except:
                continue

    return np.array(X), np.array(y), class_to_idx


    

X_train, y_train, class_map = load_lbp_dataset(r"C:/users/sushm/OneDrive/Documents/2/2/train")
X_test, y_test, _ = load_lbp_dataset(r"C:/users/sushm/OneDrive/Documents/2/2/test")

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    ))
])

svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc*100:.2f}%\n")

print(classification_report(y_test, y_pred))
probs = svm.predict_proba(X_test)
confidences = probs.max(axis=1)

print("Avg confidence:", confidences.mean())


