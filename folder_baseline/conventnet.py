import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import os

# ==========================
# 1. CONFIG
# ==========================

NUM_CLASSES = 49
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
num_workers = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

        
# ==========================
# 2. TRANSFORMS
# ==========================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
    transforms.RandomRotation(25),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================
# 3. DATASETS
# ==========================

train_dataset = datasets.ImageFolder(r"C:/Users/sushm/OneDrive/Documents/2/2/train", transform=train_transform)
class_names = train_dataset.classes

val_dataset   = datasets.ImageFolder(r"C:/Users/sushm/OneDrive/Documents/2/2/valid", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================
# 4. CLASS WEIGHTS
# ==========================

labels = [label for _, label in train_dataset.samples]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ==========================
# 5. MODEL
# ==========================

model = convnext_tiny(weights="IMAGENET1K_V1")
model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ==========================
# 6. LOSS + OPTIMIZER
# ==========================

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==========================
# 7. TRAIN LOOP
# ==========================

def train_one_epoch(model, loader):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


# ==========================
# 8. VALIDATION LOOP
# ==========================

def validate(model, loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total


# ==========================
# 9. MAIN TRAINING
# ==========================

best_val_acc = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
        }, "best_convnext.pth")
        print("model_state_dict:",model.state_dict().keys())
        print("✔ Saved Best Model")

print("Training Complete")
