# Federated Learning Lab
**TSU NSF AI Workshop 2026 — Session 4, Day 2 Afternoon**
Instructor: Dr. Yuankai Huo

---

## What is Federated Learning?

Imagine three hospitals want to build an AI model to detect skin cancer.
The problem: **patient data is private** and cannot be shared between hospitals.

**Traditional (Centralized) AI:**
All hospitals send their patient images to one central server → train one model.
This works well, but it **violates patient privacy**.

**Federated Learning:**
Each hospital trains the model **locally on their own data**.
Only the model weights (not the images) are sent to a central server.
The server **averages** the weights and sends the improved model back.
Patient data **never leaves the hospital**.

```
Hospital A  ──┐
              ├──► Server (averages weights) ──► Better Global Model
Hospital B  ──┘
```

This lab simulates this process using **DermaMNIST** — a dataset of 7,007
skin lesion images across 7 categories — split between 2 simulated hospitals.

---

## Dataset: DermaMNIST

| Property | Value |
|----------|-------|
| Source | MedMNIST (dermatology) |
| Image size | 28 × 28 pixels, color (RGB) |
| Training images | 7,007 |
| Validation images | 1,003 |
| Test images | 2,005 |
| Classes | 7 skin lesion types |

**The 7 classes:**
- 0 — Actinic keratoses
- 1 — Basal cell carcinoma
- 2 — Benign keratosis-like lesions
- 3 — Dermatofibroma
- 4 — Melanoma
- 5 — Melanocytic nevi *(most common — 67% of training data)*
- 6 — Vascular lesions

---

## Setup

Make sure you have Python and the following packages installed:

```bash
pip install torch numpy matplotlib
```

All 5 scripts should be run **in order** from the project folder:

```bash
cd FederatedLearning
python 1_centralized_baseline.py
python 2_data_partitioning.py
python 3_federated_training.py
python 4_evaluation_visualization.py
python 5_practice.py
```

---

## Lab Steps

### Step 1 — Centralized Baseline (`1_centralized_baseline.py`)

**Goal:** Train a single model on *all* the data the traditional way.

This is our **benchmark**. In a real scenario, this would require collecting
all hospital data in one place — which is a privacy concern. Here, we do it
just to see how well a model *can* perform with full access to all data.

What happens:
- Loads all 7,007 training images
- Trains a small CNN for 20 epochs
- Evaluates accuracy on the held-out test set
- Saves results to `results/centralized_results.json`

> **Key concept:** A Convolutional Neural Network (CNN) learns to detect
> patterns (edges, textures, shapes) in images by stacking layers of filters.

---

### Step 2 — Data Partitioning (`2_data_partitioning.py`)

**Goal:** Split the training data between 2 simulated community health centers.

In federated learning, each "client" (hospital, clinic, device) has its own
local data. Importantly, the data is **non-IID** — each client sees a
*different mix* of cases, just like real hospitals in different communities.

What happens:
- **Client 1 (Health Center A):** receives 80% of the rare lesion types (classes 0–3)
- **Client 2 (Health Center B):** receives 80% of the common lesion types (classes 4–6)
- Saves the split data to `data/client1.npz`, `data/client2.npz`, `data/test_data.npz`
- Prints a visual breakdown of which client has which classes

> **Key concept:** Non-IID (non-independent, non-identically distributed) data
> means different clients have different class distributions. This is the core
> challenge in federated learning — and what makes it hard!

---

### Step 3 — Federated Training (`3_federated_training.py`)

**Goal:** Train a model across 2 clients using the **FedAvg** algorithm —
without ever sharing raw data between clients.

Each **communication round** follows 4 steps:
1. **Broadcast:** Server sends the current global model to both clients
2. **Local training:** Each client trains the model on their own data only
3. **Upload:** Each client sends updated model weights back to the server
4. **Aggregate:** Server computes a weighted average of all client weights (FedAvg)

```
Round 1:  Global Model → Client 1 trains → weights₁
                       → Client 2 trains → weights₂
          Server: new_model = average(weights₁, weights₂)

Round 2:  new_model → Client 1 trains → ...
          ...
```

What happens:
- Runs 5 communication rounds
- Each client trains for 2 local epochs per round
- Evaluates the global model on the shared test set after each round
- Saves results to `results/federated_results.json`

> **Key concept:** FedAvg (Federated Averaging) was introduced by Google in
> 2017. It averages model weights proportionally to each client's data size.
> Only weights travel over the network — never the patient images.

---

### Step 4 — Evaluation & Visualization (`4_evaluation_visualization.py`)

**Goal:** Compare centralized vs. federated results with charts and numbers.

Loads the saved results from Steps 1 and 3 and produces:

| Chart | What it shows |
|-------|--------------|
| `accuracy_comparison.png` | Centralized accuracy per epoch vs. federated accuracy per round |
| `final_accuracy_bar.png` | Side-by-side bar chart of final test accuracy |
| `convergence_vs_cost.png` | How accuracy improves as communication cost increases |

Also prints a **communication cost analysis** — how many megabytes of model
weights are sent across the network in total — and compares that to the size
of the raw data that was *protected*.

All plots are saved to `results/plots/`.

> **Key concept:** Communication cost is a real constraint in FL. Sending
> model weights every round is expensive. More rounds = better accuracy,
> but higher bandwidth usage.

---

### Step 5 — Practice (`5_practice.py`)

**Goal:** Experiment with federated learning parameters and observe the effects.

This is a **standalone** script — it reloads the raw data and runs FL from
scratch using whatever parameters you set. Find the four lines marked
`# <-- CHANGE ME` at the top of the file:

```python
NUM_ROUNDS    = 5      # Try: 1, 3, 5, 10
LOCAL_EPOCHS  = 2      # Try: 1, 2, 5, 10
LEARNING_RATE = 0.001  # Try: 0.01, 0.001, 0.0001
NON_IID_DEGREE = 0.8   # Try: 0.5 (IID) to 0.9 (very non-IID)
```

**Discussion questions:**
1. What happens to accuracy as you increase `NUM_ROUNDS`? Is there a point of diminishing returns?
2. What happens when `LOCAL_EPOCHS` is very large (e.g., 10)? *(Hint: "client drift")*
3. Compare `NON_IID_DEGREE = 0.5` vs `0.9`. Why does non-IID data make FL harder?
4. If you could only afford 20 MB of communication total, how would you choose `NUM_ROUNDS` and `LOCAL_EPOCHS`?
5. How does federated learning protect patient privacy? What information might still be at risk?

---

## Project Structure

```
FederatedLearning/
├── data/
│   ├── dermamnist.npz        ← original dataset
│   ├── client1.npz           ← created by Step 2
│   ├── client2.npz           ← created by Step 2
│   └── test_data.npz         ← created by Step 2
├── results/
│   ├── centralized_results.json   ← created by Step 1
│   ├── federated_results.json     ← created by Step 3
│   └── plots/                     ← created by Step 4
│       ├── accuracy_comparison.png
│       ├── final_accuracy_bar.png
│       └── convergence_vs_cost.png
├── 1_centralized_baseline.py
├── 2_data_partitioning.py
├── 3_federated_training.py
├── 4_evaluation_visualization.py
├── 5_practice.py
└── README.md
```

---

## Further Reading

- McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data* (the original FedAvg paper)
- MedMNIST — https://medmnist.com (source of the DermaMNIST dataset)
- Flower Framework — https://flower.dev (a popular open-source FL library)
