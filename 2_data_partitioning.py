# =============================================================================
# FILE 2: Data Partitioning
# =============================================================================
# Goal: Split the training data between 2 simulated "community health centers"
#       (clients) in a realistic, non-IID way.
#
# What is Non-IID?
#   IID     = Independent and Identically Distributed
#             → each client has the same balanced mix of all classes
#   Non-IID = each client has a DIFFERENT mix of classes
#             → more realistic! A rural clinic sees different cases than an
#               urban dermatology center.
#
# Our split:
#   Client 1 (Community Health Center A): sees mostly rare lesion types
#   Client 2 (Community Health Center B): sees mostly common lesion types
# =============================================================================

import numpy as np
import os

# ---------- Settings ----------
DATA_PATH   = "data/dermamnist.npz"
DATA_DIR    = "data"
# What fraction of each class goes to Client 1 (the rest goes to Client 2)
# Classes 0-3 (rarer types)  → Client 1 gets 80%
# Classes 4-6 (common types) → Client 1 gets 20%
CLIENT1_FRACTION = {0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8,
                    4: 0.2, 5: 0.2, 6: 0.2}

CLASS_NAMES = [
    "Actinic keratoses",       # Class 0
    "Basal cell carcinoma",    # Class 1
    "Benign keratosis",        # Class 2
    "Dermatofibroma",          # Class 3
    "Melanoma",                # Class 4
    "Melanocytic nevi",        # Class 5
    "Vascular lesions",        # Class 6
]

os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# STEP 1: Load Original Training Data
# =============================================================================
print("--- Loading DermaMNIST training data ---")

data = np.load(DATA_PATH)
train_images = data["train_images"]  # (7007, 28, 28, 3)
train_labels = data["train_labels"].flatten()  # (7007,)

print(f"  Total training samples: {len(train_labels)}")
print(f"  Number of classes: {len(CLASS_NAMES)}\n")

# Show overall class distribution
print("  Class distribution in the full dataset:")
for c in range(len(CLASS_NAMES)):
    count = np.sum(train_labels == c)
    bar   = "█" * (count // 30)
    print(f"    Class {c} ({CLASS_NAMES[c]:<25}): {count:4d}  {bar}")


# =============================================================================
# STEP 2: Split Data into 2 Clients (Non-IID)
# =============================================================================
print("\n--- Splitting data into 2 clients (Non-IID) ---")

client1_idx = []  # List of sample indices for Client 1
client2_idx = []  # List of sample indices for Client 2

np.random.seed(42)  # Seed for reproducibility (same split every run)

for class_id in range(len(CLASS_NAMES)):
    # Find all indices belonging to this class
    class_indices = np.where(train_labels == class_id)[0]
    np.random.shuffle(class_indices)

    # Decide the split point
    n_client1 = int(len(class_indices) * CLIENT1_FRACTION[class_id])

    client1_idx.extend(class_indices[:n_client1])
    client2_idx.extend(class_indices[n_client1:])

# Convert to arrays
client1_idx = np.array(client1_idx)
client2_idx = np.array(client2_idx)


# =============================================================================
# STEP 3: Show the Distribution in Each Client
# =============================================================================
print(f"\n  Client 1 total samples: {len(client1_idx)}")
print(f"  Client 2 total samples: {len(client2_idx)}")

print("\n  Per-class breakdown:")
print(f"  {'Class':<30} {'Client 1':>10} {'Client 2':>10}")
print(f"  {'-'*52}")

c1_labels = train_labels[client1_idx]
c2_labels = train_labels[client2_idx]

for c in range(len(CLASS_NAMES)):
    n1 = np.sum(c1_labels == c)
    n2 = np.sum(c2_labels == c)
    print(f"  {CLASS_NAMES[c]:<30} {n1:>10} {n2:>10}")

print(f"\n  KEY INSIGHT: Client 1 has more rare lesion types (0-3),")
print(f"               Client 2 has more common lesion types (4-6).")
print(f"               This simulates different hospital populations!")


# =============================================================================
# STEP 4: Save Client Datasets
# =============================================================================
print("\n--- Saving client datasets ---")

# Save Client 1 data
client1_path = os.path.join(DATA_DIR, "client1.npz")
np.savez(client1_path,
         images=train_images[client1_idx],
         labels=train_labels[client1_idx])
print(f"  Client 1 data saved to: {client1_path}")

# Save Client 2 data
client2_path = os.path.join(DATA_DIR, "client2.npz")
np.savez(client2_path,
         images=train_images[client2_idx],
         labels=train_labels[client2_idx])
print(f"  Client 2 data saved to: {client2_path}")

# Also save the test/val data (shared — used for global evaluation)
np.savez(os.path.join(DATA_DIR, "test_data.npz"),
         images=data["test_images"],
         labels=data["test_labels"].flatten())
print(f"  Test data saved to:     {DATA_DIR}/test_data.npz")

print("\n=== Done! Run 3_federated_training.py next. ===")
