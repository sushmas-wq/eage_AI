import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

# =============================
# CONFIG
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

TRAIN_DIR = r"D:\2\train"
VAL_DIR   = r"D:\2\valid"
SAVE_DIR  = "checkpt"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# HELPERS
# =============================
def get_class_weights(dataset):
    labels = [y for _, y in dataset.samples]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(len(counts))]
    return torch.tensor(weights, dtype=torch.float32)

def make_weighted_sampler(dataset):
    labels = [y for _, y in dataset.samples]
    counts = Counter(labels)
    class_weights = {c: 1.0 / counts[c] for c in counts}
    sample_weights = [class_weights[y] for y in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


            

def evaluate_metrics(model, loader, class_names, title):
    model.eval()
    y_true, y_pred = [], []
    conf_sum, conf_count = Counter(), Counter()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            probs = F.softmax(model(x), dim=1)
            preds = probs.argmax(1)
            confs = probs.max(1).values

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            for p, c in zip(preds.cpu().numpy(), confs.cpu().numpy()):
                conf_sum[p] += c
                conf_count[p] += 1

    # ✅ ACCURACY
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = correct / len(y_true) if len(y_true) > 0 else 0.0

    print(f"\n📊 CONFUSION MATRIX ({title})")
    print(confusion_matrix(y_true, y_pred))

    print(f"\n📋 CLASSIFICATION REPORT ({title})")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print(f"\n📈 AVG CONFIDENCE ({title})")
    for i, name in enumerate(class_names):
        print(f"{name}: {conf_sum[i] / max(conf_count[i],1):.3f}")

    print(f"\n✅ ACCURACY ({title}): {acc:.4f}")

    return acc   
# =============================
# TRANSFORMS
# =============================
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3,0.3,0.3,0.1),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =============================
# MODELS
# =============================
class MobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        in_f = self.net.classifier[3].in_features
        self.net.classifier[3] = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.net(x)

# =============================
# DATASETS
# =============================
crop_train_ds = ImageFolder(TRAIN_DIR, transform=train_tf)
crop_val_ds   = ImageFolder(VAL_DIR, transform=val_tf)

crop_sampler = make_weighted_sampler(crop_train_ds)

crop_train_loader = DataLoader(
    crop_train_ds, batch_size=BATCH_SIZE, sampler=crop_sampler
)
crop_val_loader = DataLoader(
    crop_val_ds, batch_size=BATCH_SIZE, shuffle=False
)

crop_classes = crop_train_ds.classes

# =============================
# TRAIN CROP MODEL
# =============================
crop_model = MobileNetV3(len(crop_classes)).to(DEVICE)
crop_weights = get_class_weights(crop_train_ds).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=crop_weights)
optimizer = torch.optim.Adam(crop_model.parameters(), lr=LR)

best_acc = 0.0
print("\n=== TRAINING CROP CLASSIFIER ===")

for epoch in range(EPOCHS):
    crop_model.train()
    bar = tqdm(crop_train_loader, desc=f"[Crop][{epoch+1}/{EPOCHS}]")

    for x,y in bar:
        x,y = x.to(DEVICE), y.to(DEVICE)
        loss = criterion(crop_model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.set_postfix(loss=loss.item())

    acc = evaluate_metrics(crop_model, crop_val_loader, crop_classes, "CROP-VAL")
    if acc > best_acc:
        torch.save(
            {"model_state": crop_model.state_dict(), "classes": crop_classes},
            os.path.join(SAVE_DIR, "best_crop_model.pth")
        )

# =============================
# TRAIN DISEASE MODELS
# =============================
print("\n=== TRAINING DISEASE CLASSIFIERS ===")

for crop in crop_classes:
    train_path = os.path.join(TRAIN_DIR, crop)
    val_path   = os.path.join(VAL_DIR, crop)
    if not os.path.exists(train_path):
        continue

    train_ds = ImageFolder(train_path, transform=train_tf)
    val_ds   = ImageFolder(val_path, transform=val_tf)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MobileNetV3(len(train_ds.classes)).to(DEVICE)
    weights = get_class_weights(train_ds).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    
    best_dis_acc = 0.0
    print(f"\n--- {crop.upper()} DISEASE MODEL ---")

    for epoch in range(EPOCHS):
        model.train()
        bar = tqdm(train_loader, desc=f"[{crop}][{epoch+1}/{EPOCHS}]")

        for x,y in bar:
            x,y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=loss.item())

        conf=evaluate_metrics(model, val_loader, train_ds.classes, f"{crop}-DISEASE")
        if conf> best_dis_acc:

           torch.save(
            {"model_state": model.state_dict(), "classes": train_ds.classes,"class_to_idx": train_ds.class_to_idx},
            os.path.join(SAVE_DIR, f"best_disease_{crop}.pth")
          )

print("\n🎉 TRAINING + METRICS COMPLETE")