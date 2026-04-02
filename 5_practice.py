# =============================================================================
# FILE 5: Practice — Explore Federated Learning Parameters
# =============================================================================
# Goal: Modify the parameters below and observe how they affect FL performance.
#
# INSTRUCTIONS FOR STUDENTS:
#   1. Change one parameter at a time.
#   2. Run this file and note the final accuracy.
#   3. Record your observations in the table provided.
#   4. Discuss: which setting gave the best accuracy? Why?
#
# Run this file from the command line:
#   python 5_practice.py
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import os

# =============================================================================
# >>>>>>>>>>>>>>>>>>>  STUDENT PARAMETERS — CHANGE THESE!  <<<<<<<<<<<<<<<<<<<
# =============================================================================

# Q1: How many communication rounds should the server run?
#     Try: 1, 3, 5, 10
#     More rounds → better accuracy, but higher communication cost.
NUM_ROUNDS = 5                       # <-- CHANGE ME

# Q2: How many epochs should each client train locally per round?
#     Try: 1, 2, 5, 10
#     More local epochs → faster but can cause "client drift" (divergence).
LOCAL_EPOCHS = 2                     # <-- CHANGE ME

# Q3: What learning rate should clients use?
#     Try: 0.01, 0.001, 0.0001
#     Too high → unstable training. Too low → slow learning.
LEARNING_RATE = 0.001               # <-- CHANGE ME

# Q4: How non-IID should the data split be?
#     This controls how unequal the class distributions are across clients.
#     Try values between 0.1 (very non-IID) and 0.5 (balanced / IID)
#     0.5 = equal split (IID), 0.9 = Client 1 gets 90% of classes 0-3
NON_IID_DEGREE = 0.8                # <-- CHANGE ME

# =============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>  END OF STUDENT PARAMETERS  <<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================


# ---------- Fixed Settings (do not change) ----------
DATA_PATH   = "data/dermamnist.npz"
BATCH_SIZE  = 64
NUM_CLASSES = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("\n" + "="*55)
print("   YOUR EXPERIMENT SETTINGS")
print("="*55)
print(f"   NUM_ROUNDS    = {NUM_ROUNDS}")
print(f"   LOCAL_EPOCHS  = {LOCAL_EPOCHS}")
print(f"   LEARNING_RATE = {LEARNING_RATE}")
print(f"   NON_IID_DEGREE= {NON_IID_DEGREE}")
print("="*55)


# =============================================================================
# HELPERS (same as Files 1–3, copied here so this file runs standalone)
# =============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loader(images, labels, batch_size=64, shuffle=True):
    x = torch.tensor(images / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def client_update(global_model, loader, local_epochs, lr, device):
    local_model = copy.deepcopy(global_model)
    local_model.train()
    optimizer = optim.Adam(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(images), labels)
            loss.backward()
            optimizer.step()
    return local_model.state_dict()


def federated_average(global_model, client_weights, client_sizes):
    total = sum(client_sizes)
    avg   = copy.deepcopy(client_weights[0])
    for key in avg:
        avg[key] = torch.zeros_like(avg[key], dtype=torch.float32)
    for weights, n in zip(client_weights, client_sizes):
        for key in avg:
            avg[key] += weights[key].float() * (n / total)
    global_model.load_state_dict(avg)
    return global_model


# =============================================================================
# STEP 1: Load & Split Data Using NON_IID_DEGREE
# =============================================================================
print("\n--- Loading and splitting data ---")

data = np.load(DATA_PATH)
train_images = data["train_images"]
train_labels = data["train_labels"].flatten()

# Split based on NON_IID_DEGREE
# Classes 0-3: Client 1 gets NON_IID_DEGREE fraction
# Classes 4-6: Client 1 gets (1 - NON_IID_DEGREE) fraction
np.random.seed(42)
client1_idx, client2_idx = [], []

for class_id in range(NUM_CLASSES):
    idx = np.where(train_labels == class_id)[0]
    np.random.shuffle(idx)
    # Classes 0-3 are "rare" types favored by Client 1
    if class_id <= 3:
        split = int(len(idx) * NON_IID_DEGREE)
    else:
        split = int(len(idx) * (1 - NON_IID_DEGREE))
    client1_idx.extend(idx[:split])
    client2_idx.extend(idx[split:])

client1_idx = np.array(client1_idx)
client2_idx = np.array(client2_idx)

print(f"  Client 1: {len(client1_idx)} samples")
print(f"  Client 2: {len(client2_idx)} samples")

# Show per-class breakdown so students can see the non-IID effect
print(f"\n  {'Class':<5} {'Client1':>8} {'Client2':>8}  Distribution")
c1_lab = train_labels[client1_idx]
c2_lab = train_labels[client2_idx]
for c in range(NUM_CLASSES):
    n1 = int(np.sum(c1_lab == c))
    n2 = int(np.sum(c2_lab == c))
    total = n1 + n2
    pct1  = n1 / total if total > 0 else 0
    bar   = "█" * int(pct1 * 20) + "░" * (20 - int(pct1 * 20))
    print(f"  {c:<5} {n1:>8} {n2:>8}  C1|{bar}|C2")

client1_loader = make_loader(train_images[client1_idx], c1_lab)
client2_loader = make_loader(train_images[client2_idx], c2_lab)
test_loader    = make_loader(data["test_images"],
                             data["test_labels"].flatten(), shuffle=False)
client_sizes   = [len(client1_idx), len(client2_idx)]


# =============================================================================
# STEP 2: Run Federated Training
# =============================================================================
print(f"\n--- Running Federated Training ({NUM_ROUNDS} rounds) ---\n")

global_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
round_accs   = []

for round_num in range(1, NUM_ROUNDS + 1):
    weights1 = client_update(global_model, client1_loader,
                             LOCAL_EPOCHS, LEARNING_RATE, device)
    weights2 = client_update(global_model, client2_loader,
                             LOCAL_EPOCHS, LEARNING_RATE, device)
    global_model = federated_average(global_model,
                                     [weights1, weights2], client_sizes)

    acc = evaluate(global_model, test_loader, device)
    round_accs.append(acc)
    print(f"  Round {round_num:2d}/{NUM_ROUNDS} | Test Accuracy: {acc:.2%}")


# =============================================================================
# STEP 3: Print Summary
# =============================================================================
print("\n" + "="*55)
print("   EXPERIMENT SUMMARY")
print("="*55)
print(f"   NUM_ROUNDS    = {NUM_ROUNDS}")
print(f"   LOCAL_EPOCHS  = {LOCAL_EPOCHS}")
print(f"   LEARNING_RATE = {LEARNING_RATE}")
print(f"   NON_IID_DEGREE= {NON_IID_DEGREE}  (0.5=IID, 0.9=very non-IID)")
print(f"\n   Round-by-round accuracy:")
for i, acc in enumerate(round_accs, 1):
    bar = "█" * int(acc * 40)
    print(f"   Round {i:2d}: {acc:.2%}  {bar}")
print(f"\n   Final Test Accuracy: {round_accs[-1]:.2%}")

model_size_mb = 2.0
total_comm = NUM_ROUNDS * 2 * 2 * model_size_mb  # rounds × clients × 2-way
print(f"   Total Communication: {total_comm:.1f} MB")

print("""
=================================================================
   DISCUSSION QUESTIONS FOR STUDENTS:
=================================================================
   1. What happens to accuracy as you increase NUM_ROUNDS?
      Is there a point of diminishing returns?

   2. What happens when LOCAL_EPOCHS is very large (e.g., 10)?
      (Hint: look up "client drift" in FL literature)

   3. Compare NON_IID_DEGREE = 0.5 vs. 0.9.
      Why does non-IID data make FL harder?

   4. If you could only afford 20 MB of communication total,
      how would you set NUM_ROUNDS and LOCAL_EPOCHS?

   5. How does federated learning protect patient privacy?
      What is still potentially at risk? (Advanced)
=================================================================
""")
