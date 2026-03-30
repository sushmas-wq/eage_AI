from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import time
import os



class LeafBaseSuppression:
    def __init__(self, p=0.3, suppress_ratio=0.35):
        """
        p: probability of applying suppression
        suppress_ratio: fraction of image height to suppress from bottom
        """
        self.p = p
        self.suppress_ratio = suppress_ratio

    def __call__(self, img):
        # img: Tensor [C, H, W] after ToTensor()
        if random.random() > self.p:
            return img

        c, h, w = img.shape
        y_start = int((1 - self.suppress_ratio) * h)

        # Replace with image mean (NOT zeros)
        mean_val = img.mean(dim=(1, 2), keepdim=True)
        img[:, y_start:h, :] = mean_val

        return img


class DatasetManager:
    def __init__(self, data_path, batch_size=8):
        self.data_path = Path(data_path)
        self.val_data = self.data_path.parent / "valid"
        print("val_data:",self.val_data)
        self.batch_size = batch_size
        num_workers = 0
        self.transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomRotation(15),

         transforms.RandomGrayscale(p=0.1),
         transforms.RandomAutocontrast(p=0.3),
         transforms.ColorJitter(
        brightness=0.4,
        contrast=0.2,
        saturation=0.5,
        hue=0.02
         ),

         transforms.ToTensor(),

         transforms.Lambda(
        lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
      ),

         transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
   )
   ])


    

                
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def prepare(self):
        dataset = datasets.ImageFolder(
            root=self.data_path,
            transform=self.transform
        )

        self.class_names = dataset.classes
        

        self.train_dataset = dataset
        self.val_dataset   = datasets.ImageFolder(self.val_data,   transform=self.eval_transform)


        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print("VAL classes == TRAIN classes:", self.val_dataset.classes==self.train_dataset.classes)
        return self.train_loader, self.val_loader, self.class_names
    def cam_regularization(cam, leaf_mask):
       """
       cam: [B, H, W] normalized to [0,1]
       leaf_mask: [B, H, W] binary {0,1}
      """
       outside_attention = cam * (1 - leaf_mask)
       return outside_attention.mean()

class MobileNetModel:
    
    torch.utils.rename_privateuse1_backend("dml")
    def __init__(self, num_classes):
        self.device = torch.device("cpu")

        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            num_classes
        )

        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)


class Trainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3):
        self.model = model
        self.device = model.device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.model.parameters(),
            lr=lr
        )
        print("training stage ")
      


    


    def train_one_epoch(self):
        self.model.model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model.forward(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        return total_loss / len(self.train_loader), acc

    def validate(self):
        self.model.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model.forward(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        return total_loss / len(self.val_loader), acc
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier =nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return  x
class CNNModel:
    def __init__(self, num_classes):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = CNN(num_classes).to(self.device)

    def forward(self, x):
        return self.model(x)
if __name__ == "__main__":

  device = torch.device("cpu")
  print("Using device:", device)


  data_path = r"C:\Users\sushm\OneDrive\Documents\2\2\train"

    # Dataset
  data = DatasetManager(data_path)
  train_loader, val_loader, class_names = data.prepare()

    # Model
  model = MobileNetModel(num_classes=len(class_names))
  if(input("enter training:").lower() in("y","yes")):

    # Trainer
    trainer = Trainer(model, train_loader, val_loader)

    epochs = 6
    best_val_acc = 0
    prev_val_loss = None

    for epoch in range(epochs):
        start_timer = time.time()
        train_loss, train_acc = trainer.train_one_epoch()
        val_loss, val_acc = trainer.validate()
        
        print(
            f"[mobile net] Epoch {epoch+1} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% |"
            f" Val Loss: {val_loss:.4f}"
        )
        if prev_val_loss is not None and val_loss >= prev_val_loss: 
            print("Early stopping triggered")
            break
        prev_val_loss = val_loss                                
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.model.state_dict(),
                "class_names": class_names
            }, "super_model.pth")

            print("✅ Best model saved")
    stop_time = time.time() - start_timer
    print("time_taken:",stop_time)
# CNN model from scratch


    #stop_time= time.time() - start_timer
    #print("time_taken:",stop_time)


  print("\n\nTesting on a random image from test set...")

  test_dir = Path(r"C:\Users\sushm\OneDrive\Documents\2\2\test")

    # 1️⃣ Pick crop folder (e.g. corn)
  disease_dir = random.choice([
        p for p in test_dir.iterdir() if p.is_dir()
    ])
  crop = disease_dir.name.split("_")[0]
  print("Crop:", disease_dir.name)
  print("disease dir:", disease_dir)
  print("Exists:", disease_dir.exists())

    # 2️⃣ Pick disease folder (e.g. Northern Leaf Blight)
  
  print("Disease:", disease_dir.name)

    # 3️⃣ Pick ONLY image files
  image_files = [
        p for p in disease_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

  if not image_files:
        raise RuntimeError("No image files found!")

  img_path = random.choice(image_files)
  print("Using image:", img_path)

    # 4️⃣ Open image
  image = Image.open(img_path).convert("RGB")
  plt.imshow(image)
  plt.show()
  data = DatasetManager(r"C:\Users\sushm\OneDrive\Documents\2\2\train")
  image_tensor = data.eval_transform(image)
  image_tensor = image_tensor.unsqueeze(0).to(device)
  
  

  device = torch.device("cpu")
  # Load checkpoint
  checkpoint = torch.load("super_model.pth", map_location=device)

  #  Restore class names
  class_names = checkpoint["class_names"]

  # ⃣ DEFINE NUM_CLASSES (THIS WAS MISSING)
  NUM_CLASSES = len(class_names)

  #  Recreate model architecture
  model = models.mobilenet_v2(weights=None)
  model.classifier[1] = nn.Linear(
      model.classifier[1].in_features,
      NUM_CLASSES
  )
  model = model.to(device)

  #  Load trained weights
  model.load_state_dict(checkpoint["model_state"])
  model.eval()

  print("✅ Model loaded with", NUM_CLASSES, "classes")

  with torch.no_grad():
      outputs = model(image_tensor)
      probs = torch.softmax(outputs, dim=1)
      conf, pred = torch.max(probs, 1)

  print("Predicted:", class_names[pred.item()])
  print(f"Confidence: {conf.item():.4f}")
  print(r"""Training: 100%|████████████████████████| 1516/1516 [1:27:58<00:00,  3.48s/it] 
[mobile net] Epoch 1 | Train Acc: 90.10% | Train Loss: 0.3298 | Val Acc: 95.67% | Val Loss: 0.1323
✅ Best model saved
Training: 100%|████████████████████████| 1516/1516 [1:38:29<00:00,  3.90s/it] 
[mobile net] Epoch 2 | Train Acc: 95.13% | Train Loss: 0.1502 | Val Acc: 96.77% | Val Loss: 0.0954
✅ Best model saved
Training: 100%|████████████████████████| 1516/1516 [1:36:09<00:00,  3.81s/it] 
[mobile net] Epoch 3 | Train Acc: 96.08% | Train Loss: 0.1194 | Val Acc: 96.68% | Val Loss: 0.1018
Training: 100%|████████████████████████| 1516/1516 [1:24:56<00:00,  3.36s/it] 
[mobile net] Epoch 4 | Train Acc: 96.60% | Train Loss: 0.1027 | Val Acc: 97.85% | Val Loss: 0.0657
✅ Best model saved
Training images: 60619
Test images: 620
        """)


  def count_images(root_dir):
        count = 0
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                  count += 1
        return count

  train_count = count_images(r"C:\Users\sushm\OneDrive\Documents\2\2\train")
  test_count  = count_images(r"C:\Users\sushm\OneDrive\Documents\2\2\test")

  print(f"Training images: {train_count}")
  print(f"Test images: {test_count}") 
#✅ Best for documentation and verification