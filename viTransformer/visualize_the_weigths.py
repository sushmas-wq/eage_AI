import torch
import matplotlib.pyplot as plt
from torchvision import models

# -----------------------------
# LOAD CHECKPOINT
# -----------------------------
checkpoint = torch.load("best_modela.pth", map_location="cpu")
print("Checkpoint keys:", checkpoint.keys())

# -----------------------------
# RECREATE MODEL EXACTLY
# -----------------------------
NUM_CLASSES = len(checkpoint["class_names"])

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(
    model.last_channel,
    NUM_CLASSES
)

model.load_state_dict(checkpoint["model_state"])
model.eval()

print("✅ Model loaded successfully")

# ------------------------------
# MOBILENET FIRST LAYER VISUALIZATION
# ------------------------------
first_conv = model.features[0][0]   # Conv2d
weights = first_conv.weight.data    # (32, 3, 3, 3)

fig, axes = plt.subplots(1, 8, figsize=(15, 3))

for i in range(8):
    filt = weights[i].mean(dim=0)   # average RGB
    axes[i].imshow(filt, cmap="gray")
    axes[i].axis("off")

plt.suptitle("MobileNetV2 First Layer Filters")
plt.show()

# -----------------------------
# FIND DEPTHWISE CONVOLUTIONS
# -----------------------------
import torch.nn as nn

depthwise_layers = []

for layer in model.features:
    if hasattr(layer, "conv"):
        for block in layer.conv:
            if isinstance(block, nn.Sequential):
                for sub in block:
                    if isinstance(sub, nn.Conv2d):
                        if sub.groups == sub.in_channels:
                            depthwise_layers.append(sub)
            elif isinstance(block, nn.Conv2d):
                if block.groups == block.in_channels:
                    depthwise_layers.append(block)

print(f"Found {len(depthwise_layers)} depthwise conv layers")
import matplotlib.pyplot as plt

def show_depthwise(dw_layer, title, n=6):
    w = dw_layer.weight.data
    fig, axes = plt.subplots(1, n, figsize=(12, 3))
    for i in range(n):
        axes[i].imshow(w[i, 0], cmap="viridis")
        axes[i].axis("off")
    plt.suptitle(title)
    plt.show()

# Early depthwise layer
show_depthwise(depthwise_layers[0], "Early Depthwise Filters")

# Middle depthwise layer
show_depthwise(depthwise_layers[len(depthwise_layers)//2], "Middle Depthwise Filters")

# Late depthwise layer
show_depthwise(depthwise_layers[-1], "Late Depthwise Filters")
# -----------------------------
# ACTIVATIONS VISUALIZATION
# -----------------------------
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Hook a mid-level depthwise layer
target_dw = depthwise_layers[8]
target_dw.register_forward_hook(hook_fn("dw8"))
# Dummy input
input_tensor = torch.randn(1, 3, 224, 224)


_ = model(input_tensor)

act = activations["dw8"][0]  # (C, H, W)

plt.imshow(act[0], cmap="magma")
plt.title("Activation Map (Depthwise Layer)")
plt.axis("off")
plt.show()