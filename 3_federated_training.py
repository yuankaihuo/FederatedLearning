# =============================================================================
# FILE 3: Federated Learning Training
# =============================================================================
# Goal: Train a model across 2 simulated clients WITHOUT sharing their data.
#
# How Federated Learning works:
#   1. The SERVER sends the current global model to each CLIENT.
#   2. Each CLIENT trains the model on their LOCAL data only.
#   3. Each CLIENT sends the updated model WEIGHTS back to the server.
#      (Note: only model weights are shared — NOT the actual patient data!)
#   4. The SERVER AGGREGATES (averages) all the client weights.
#   5. Repeat for multiple rounds.
#
# This simulation runs both clients on the SAME machine,
# but logically they are treated as separate entities.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import json
import os

# ---------- Settings ----------
DATA_DIR     = "data"
RESULTS_DIR  = "results"
NUM_ROUNDS   = 5       # How many federated rounds (communication rounds)
LOCAL_EPOCHS = 2       # How many training epochs each client does per round
BATCH_SIZE   = 64
LEARNING_RATE = 0.001
NUM_CLASSES  = 7

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# HELPER: Same CNN architecture as in File 1
# =============================================================================
# Both clients and the server use this same model structure.

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
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# HELPER: Convert numpy arrays to a PyTorch DataLoader
# =============================================================================

def make_loader(images, labels, batch_size=64, shuffle=True):
    x = torch.tensor(images / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# HELPER: Evaluate a model on a dataset
# =============================================================================

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# =============================================================================
# STEP 1: Load Client Data and Test Data
# =============================================================================
print("\n--- Loading client datasets ---")

c1 = np.load(os.path.join(DATA_DIR, "client1.npz"))
c2 = np.load(os.path.join(DATA_DIR, "client2.npz"))
td = np.load(os.path.join(DATA_DIR, "test_data.npz"))

client1_loader = make_loader(c1["images"], c1["labels"])
client2_loader = make_loader(c2["images"], c2["labels"])
test_loader    = make_loader(td["images"], td["labels"], shuffle=False)

print(f"  Client 1 samples: {len(c1['labels'])}")
print(f"  Client 2 samples: {len(c2['labels'])}")
print(f"  Test samples    : {len(td['labels'])}")


# =============================================================================
# STEP 2: Define the Federated Learning Functions
# =============================================================================

def client_update(global_model, loader, local_epochs, lr, device):
    """
    CLIENT-SIDE: Train the global model on local data for a few epochs.
    Returns the updated model weights.

    In a real FL system, this function runs on the client's own machine.
    The client never sends its raw data — only the model weights.
    """
    # Start from a copy of the global model (don't modify the original)
    local_model = copy.deepcopy(global_model)
    local_model.train()

    optimizer = optim.Adam(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(local_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Return only the updated weights (not the data!)
    return local_model.state_dict()


def federated_average(global_model, client_weights, client_sizes):
    """
    SERVER-SIDE: Average the weights from all clients (FedAvg algorithm).
    Clients with more data have proportionally more influence.

    FedAvg (McMahan et al., 2017) is the classic FL aggregation method.
    """
    total_samples = sum(client_sizes)

    # Start with zeros for every parameter
    avg_weights = copy.deepcopy(client_weights[0])
    for key in avg_weights:
        avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)

    # Weighted average: each client contributes proportionally to its data size
    for weights, n_samples in zip(client_weights, client_sizes):
        weight_fraction = n_samples / total_samples
        for key in avg_weights:
            avg_weights[key] += weights[key].float() * weight_fraction

    # Load the averaged weights into the global model
    global_model.load_state_dict(avg_weights)
    return global_model


# =============================================================================
# STEP 3: Run Federated Training
# =============================================================================
print("\n--- Starting Federated Training ---")
print(f"    Rounds       : {NUM_ROUNDS}")
print(f"    Local epochs : {LOCAL_EPOCHS} per round per client")
print(f"    Clients      : 2 (Client 1 and Client 2)\n")

# Initialize the GLOBAL model on the server (random weights to start)
global_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

client_sizes = [len(c1["labels"]), len(c2["labels"])]

history = {"round": [], "test_acc": []}

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"  ===== Round {round_num}/{NUM_ROUNDS} =====")

    # --- Step A: Server sends global model to all clients ---
    # (In our simulation, clients receive global_model directly)

    # --- Step B: Each client trains locally ---
    print(f"    [Client 1] Training on local data ({client_sizes[0]} samples)...")
    weights1 = client_update(global_model, client1_loader,
                             LOCAL_EPOCHS, LEARNING_RATE, device)

    print(f"    [Client 2] Training on local data ({client_sizes[1]} samples)...")
    weights2 = client_update(global_model, client2_loader,
                             LOCAL_EPOCHS, LEARNING_RATE, device)

    # --- Step C: Server aggregates (averages) the weights ---
    print(f"    [Server]   Aggregating weights using FedAvg...")
    global_model = federated_average(global_model,
                                     [weights1, weights2],
                                     client_sizes)

    # --- Step D: Evaluate the new global model ---
    test_acc = evaluate(global_model, test_loader, device)
    print(f"    [Server]   Global model test accuracy: {test_acc:.2%}\n")

    history["round"].append(round_num)
    history["test_acc"].append(round(test_acc, 4))


# =============================================================================
# STEP 4: Final Summary & Save Results
# =============================================================================
final_acc = history["test_acc"][-1]
print(f"  Final Federated Test Accuracy: {final_acc:.2%}")

results = {
    "method"     : "Federated",
    "num_rounds" : NUM_ROUNDS,
    "local_epochs": LOCAL_EPOCHS,
    "test_acc"   : final_acc,
    "history"    : history,
}

results_path = os.path.join(RESULTS_DIR, "federated_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to: {results_path}")
print("\n=== Done! Run 4_evaluation_visualization.py next. ===")
