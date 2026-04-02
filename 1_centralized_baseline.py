# =============================================================================
# FILE 1: Centralized Baseline
# =============================================================================
# Goal: Train ONE model on ALL the data (the traditional approach).
# This gives us a "best case" benchmark to compare federated learning against.
#
# Dataset: DermaMNIST - skin lesion images (28x28 pixels, 7 categories)
# Categories: actinic keratoses, basal cell carcinoma, benign keratosis,
#             dermatofibroma, melanoma, melanocytic nevi, vascular lesions
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os

# ---------- Settings ----------
DATA_PATH   = "data/dermamnist.npz"
RESULTS_DIR = "results"
EPOCHS      = 20          # How many times to loop through the training data
BATCH_SIZE  = 64         # How many images to process at once
LEARNING_RATE = 0.001    # How fast the model learns (step size)
NUM_CLASSES = 7          # DermaMNIST has 7 skin lesion categories

# Create results folder if it does not exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- Check for GPU ----------
# A GPU (graphics card) makes training much faster.
# If no GPU is available, we fall back to the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# STEP 1: Load the Data
# =============================================================================
print("\n--- Loading DermaMNIST data ---")

data = np.load(DATA_PATH)

# Images: shape (N, 28, 28, 3) — N images, 28x28 pixels, 3 color channels (RGB)
# Labels: shape (N, 1)         — one label (category) per image
train_images = data["train_images"]  # 7,007 training images
val_images   = data["val_images"]    # 1,003 validation images
test_images  = data["test_images"]   # 2,005 test images

train_labels = data["train_labels"].flatten()  # flatten (N,1) → (N,)
val_labels   = data["val_labels"].flatten()
test_labels  = data["test_labels"].flatten()

print(f"  Training images  : {train_images.shape}")
print(f"  Validation images: {val_images.shape}")
print(f"  Test images      : {test_images.shape}")
print(f"  Number of classes: {NUM_CLASSES}")


# =============================================================================
# STEP 2: Prepare PyTorch Datasets
# =============================================================================
# Neural networks expect float values between 0 and 1, and images in the
# format (channels, height, width) — so we divide by 255 and rearrange.

def to_tensor_dataset(images, labels):
    # Normalize pixel values from [0, 255] → [0.0, 1.0]
    x = torch.tensor(images / 255.0, dtype=torch.float32)
    # Change shape from (N, H, W, C) → (N, C, H, W)
    x = x.permute(0, 3, 1, 2)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)

train_dataset = to_tensor_dataset(train_images, train_labels)
val_dataset   = to_tensor_dataset(val_images,   val_labels)
test_dataset  = to_tensor_dataset(test_images,  test_labels)

# DataLoader feeds data in small batches to the model during training
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Training batches : {len(train_loader)}")


# =============================================================================
# STEP 3: Define the Neural Network Model
# =============================================================================
# We use a simple Convolutional Neural Network (CNN).
# CNN layers detect patterns (edges, textures, shapes) in images.

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()

        # --- Feature extraction layers ---
        # Conv2d: learns to detect patterns in the image
        # ReLU:   activation function — adds non-linearity
        # MaxPool2d: shrinks the image by keeping only the strongest signals
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 channels in → 32 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 → 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 filters → 64 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 → 7x7
        )

        # --- Classification layers ---
        # Flatten the 2D feature maps into a 1D vector, then classify
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 64 × 7 × 7 = 3136 values
            nn.Linear(64 * 7 * 7, 128),    # fully connected layer
            nn.ReLU(),
            nn.Linear(128, num_classes),   # output: one score per class
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# STEP 4: Train the Model
# =============================================================================

def evaluate(model, loader, device):
    """Calculate accuracy on a given dataset."""
    model.eval()  # Switch to evaluation mode (disables dropout, etc.)
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)  # Pick the class with highest score
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total  # Return accuracy as a fraction (0.0 to 1.0)


print("\n--- Training Centralized Model ---")

model     = SimpleCNN(num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()  # Standard loss for multi-class classification

history = {"train_acc": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    # ---- Training phase ----
    model.train()  # Switch to training mode
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Clear old gradients
        outputs = model(images)        # Forward pass: predict
        loss = criterion(outputs, labels)  # Compute error
        loss.backward()                # Backward pass: compute gradients
        optimizer.step()               # Update model weights

    # ---- Evaluation phase ----
    train_acc = evaluate(model, train_loader, device)
    val_acc   = evaluate(model, val_loader,   device)

    history["train_acc"].append(round(train_acc, 4))
    history["val_acc"].append(round(val_acc, 4))

    print(f"  Epoch {epoch}/{EPOCHS} | "
          f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")


# =============================================================================
# STEP 5: Final Test Evaluation
# =============================================================================
test_acc = evaluate(model, test_loader, device)
print(f"\n  Final Test Accuracy: {test_acc:.2%}")


# =============================================================================
# STEP 6: Save Results
# =============================================================================
results = {
    "method"   : "Centralized",
    "epochs"   : EPOCHS,
    "test_acc" : round(test_acc, 4),
    "history"  : history,
}

results_path = os.path.join(RESULTS_DIR, "centralized_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to: {results_path}")
print("\n=== Done! Run 2_data_partitioning.py next. ===")
