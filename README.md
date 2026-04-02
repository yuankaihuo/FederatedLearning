# Federated Learning Lab
**TSU NSF AI Workshop 2026 — Session 4, Day 2 Afternoon (1:30 PM – 4:30 PM)**
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

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy matplotlib
```

For Step 6 (privacy extensions), also install:
```bash
pip install tenseal
```

### 2. Run the scripts in order

```bash
python 1_centralized_baseline.py
python 2_data_partitioning.py
python 3_federated_training.py
python 4_evaluation_visualization.py
python 5_practice.py
python 6_privacy_extension.py
```

> Each script prints step-by-step output explaining what is happening.
> Read the printed output as you go — it is part of the lesson.

### 3. Prefer Jupyter / Google Colab?

Open the matching notebook in the `CoLab/` folder. Every script has an
equivalent notebook with the same code and richer markdown explanations.

---

## Lab Steps

---

### Step 1 — Centralized Baseline
**File:** `1_centralized_baseline.py`

**Goal:** Train a single model on *all* the data the traditional way.

This is our **benchmark**. In a real hospital scenario this would require
collecting all patient images on one server — a major privacy risk.
Here we do it just to measure how well a model *can* perform with full data access.

**What the script does:**
- Loads all 7,007 DermaMNIST training images
- Builds a simple CNN (2 conv layers + 2 fully connected layers)
- Trains for 20 epochs, printing train and validation accuracy each epoch
- Evaluates on the held-out test set
- Saves results to `results/centralized_results.json`

**What to look for while it runs:**
- Accuracy starts low (random ~14%) and rises each epoch
- Training accuracy is usually higher than validation — this is normal (the model has seen training data)
- The final test accuracy is our **gold standard** for comparison

> **Key concept — CNN:**
> A Convolutional Neural Network scans images with small filters to detect
> edges, textures, and shapes. Each layer learns increasingly complex patterns.
> The final layer outputs one score per class — the highest score is the prediction.

**Output files:**
```
results/centralized_results.json
```

---

### Step 2 — Data Partitioning
**File:** `2_data_partitioning.py`

**Goal:** Split the training data across 2 simulated community health centers (non-IID).

In federated learning, data lives on separate machines and is never pooled.
This step simulates that by dividing the dataset into two realistic "hospital" slices.

**What is Non-IID?**

| Term | Meaning |
|------|---------|
| IID | Every client has the same balanced mix of all classes |
| Non-IID | Each client has a *different* disease distribution — realistic! |

**How the split works:**
- Client 1 (Health Center A) gets 80% of the *rare* lesion types (classes 0–3)
- Client 2 (Health Center B) gets 80% of the *common* lesion types (classes 4–6)
- This simulates a specialist center (Client 1) vs. a general clinic (Client 2)

**What to look for while it runs:**
- The printed table and bar chart show how lopsided the distributions are
- Notice that Client 1 has very few samples of class 5 (melanocytic nevi)
  while Client 2 has thousands — they are seeing completely different patients

> **Key concept — Non-IID challenge:**
> When clients have different class distributions, federated learning is harder.
> Each client's model pulls in a different direction, making the global average
> less accurate. This is the central open problem in FL research.

**Output files:**
```
data/client1.npz
data/client2.npz
data/test_data.npz
```

---

### Step 3 — Federated Training
**File:** `3_federated_training.py`

**Goal:** Train a global model across 2 clients using the FedAvg algorithm —
without any client sharing its raw data.

**How each communication round works:**

```
┌─────────────────────────────────────────────────────┐
│  1. SERVER  → sends current global model to clients │
│  2. CLIENT 1 → trains on its own data locally       │
│     CLIENT 2 → trains on its own data locally       │
│  3. CLIENTS  → send updated weights back to server  │
│     (weights only — no patient images!)             │
│  4. SERVER  → averages all weights (FedAvg)         │
│  5. Repeat for next round                           │
└─────────────────────────────────────────────────────┘
```

**Default settings (change at top of file):**
- `NUM_ROUNDS = 5` — how many server↔client communication cycles
- `LOCAL_EPOCHS = 2` — how many epochs each client trains per round

**What to look for while it runs:**
- Global model accuracy after each round — it should increase round by round
- The server never prints any patient data — only weight values and accuracy
- Compare the final accuracy to Step 1's centralized result

> **Key concept — FedAvg:**
> Federated Averaging (McMahan et al., 2017) is the foundational FL algorithm.
> The server computes a weighted average of client weights, where each client's
> contribution is proportional to its number of training samples.

**Output files:**
```
results/federated_results.json
```

---

### Step 4 — Evaluation & Visualization
**File:** `4_evaluation_visualization.py`

**Goal:** Compare centralized vs. federated results side by side with charts.

**Requires:** Run Steps 1 and 3 first so the result JSON files exist.

**What the script produces:**

| Chart | What it shows |
|-------|--------------|
| `accuracy_comparison.png` | Centralized (per epoch) vs. federated (per round) accuracy curves |
| `final_accuracy_bar.png` | Side-by-side bar chart of final test accuracy |
| `convergence_vs_cost.png` | How accuracy improves as communication cost accumulates |

**What to look for:**
- The accuracy *gap* between centralized and federated — how large is it?
- Does federated accuracy improve smoothly round by round, or does it jump?
- The communication cost chart shows a classic FL tradeoff:
  more rounds = better accuracy, but higher network cost

**Communication cost explained:**
- Each round: server → clients (download) + clients → server (upload)
- 2 clients × 2 directions × ~2 MB model = ~8 MB per round
- Patient images (~167 MB) are never transmitted — that is the privacy gain

**Output files:**
```
results/plots/accuracy_comparison.png
results/plots/final_accuracy_bar.png
results/plots/convergence_vs_cost.png
```

---

### Step 5 — Practice
**File:** `5_practice.py`

**Goal:** Experiment with FL parameters and observe how they change outcomes.

This script is **standalone** — it reloads raw data and runs FL from scratch
using whatever parameters you set. No need to re-run earlier steps.

**Find these four lines at the top of the file and change them:**

```python
NUM_ROUNDS     = 5      # Try: 1, 3, 5, 10
LOCAL_EPOCHS   = 2      # Try: 1, 2, 5, 10
LEARNING_RATE  = 0.001  # Try: 0.01, 0.001, 0.0001
NON_IID_DEGREE = 0.8    # Try: 0.5 (balanced) → 0.9 (very skewed)
```

**Suggested experiments:**

| Experiment | Change | Expected observation |
|-----------|--------|---------------------|
| More rounds | `NUM_ROUNDS = 10` | Higher accuracy, but diminishing returns |
| Client drift | `LOCAL_EPOCHS = 10` | Accuracy may drop — clients diverge too far |
| IID data | `NON_IID_DEGREE = 0.5` | Accuracy gap vs. centralized nearly disappears |
| Very non-IID | `NON_IID_DEGREE = 0.9` | Larger accuracy gap — harder for FL |
| Fast learning | `LEARNING_RATE = 0.01` | Training is faster but may be unstable |

**Discussion questions printed at the end of the script:**
1. What happens to accuracy as you increase `NUM_ROUNDS`?
2. What happens when `LOCAL_EPOCHS` is very large? (client drift)
3. Compare `NON_IID_DEGREE = 0.5` vs `0.9`. Why does skew hurt?
4. With only 20 MB total budget, what is your best configuration?
5. What privacy risks remain even in federated learning?

---

### Step 6 — Privacy Extensions
**File:** `6_privacy_extension.py`

**Goal:** Go beyond basic FL privacy with two advanced cryptographic techniques.

**Prerequisite:** Run Step 2 first (needs client data files).
For real HE (not simulation), install TenSEAL first:
```bash
pip install tenseal
```

---

#### Part A — Homomorphic Encryption (HE)

Standard FL protects raw data, but the server still sees each client's
plaintext weights. A sophisticated attacker could try to reverse-engineer
training images from those weights (*gradient inversion attack*).

Homomorphic Encryption (HE) solves this: the server aggregates weights
**without ever decrypting them**.

```
Client 1: weights  →  Encrypt  →  Enc(w1)  ──┐
                                               ├─► Server computes:
Client 2: weights  →  Encrypt  →  Enc(w2)  ──┘   Enc(avg) = Enc(w1)×f1 + Enc(w2)×f2
                                                        ↓
                                               Client decrypts → avg  (same result!)
```

**What to look for:**
- The script prints how long encryption, aggregation, and decryption take
- Compare HE timing to standard FL — this is the real cost of stronger privacy
- The verification step confirms HE gives the same answer as plaintext averaging
- If TenSEAL is not installed, the script runs a conceptual simulation instead

> **Key concept — CKKS scheme:**
> CKKS is a homomorphic encryption scheme designed for floating-point arithmetic.
> It supports approximate computation — tiny numerical errors (~1e-5) that are
> completely negligible for neural network weight averaging.

---

#### Part B — Secure Aggregation (Additive Masking)

A faster alternative to HE. Each client adds a random mask to their weights
before sending. The masks are designed to cancel when the server sums them.

```
Client 1 sends:  w1 + mask          ← looks like noise
Client 2 sends:  w2 - mask          ← looks like noise
Server computes: (w1+mask) + (w2-mask) = w1 + w2  ← masks cancel!
```

The server learns only the **sum** — it cannot recover w1 or w2 individually.

**What to look for:**
- The printed output shows what the server actually sees for one weight value
- The "true" weight vs. the "sent" value look completely different
- Accuracy of secure aggregation vs. standard FL should be nearly identical
- This technique is already used in production at Google (Gboard keyboard)

---

#### Part C — Comparison Summary

The script prints a table comparing all approaches:

| Technique | Server sees | Accuracy loss | Speed |
|-----------|-------------|--------------|-------|
| Standard FL | Plaintext weights | None | Fast |
| Secure Aggregation | Only the SUM | None | Fast |
| Homomorphic Encryption | Encrypted ciphertext | None | Slow (10–100×) |

**Output files:**
```
results/privacy_results.json
results/plots/privacy_comparison.png
```

---

## Project Structure

```
FederatedLearning/
├── data/
│   ├── dermamnist.npz             ← original dataset (you provide this)
│   ├── client1.npz                ← created by Step 2
│   ├── client2.npz                ← created by Step 2
│   └── test_data.npz              ← created by Step 2
│
├── results/
│   ├── centralized_results.json   ← created by Step 1
│   ├── federated_results.json     ← created by Step 3
│   ├── privacy_results.json       ← created by Step 6
│   └── plots/
│       ├── accuracy_comparison.png
│       ├── final_accuracy_bar.png
│       ├── convergence_vs_cost.png
│       └── privacy_comparison.png
│
├── 1_centralized_baseline.py      ← train on all data (benchmark)
├── 2_data_partitioning.py         ← split data across 2 clients (non-IID)
├── 3_federated_training.py        ← FedAvg across 2 clients
├── 4_evaluation_visualization.py  ← compare & plot results
├── 5_practice.py                  ← experiment with parameters
├── 6_privacy_extension.py         ← homomorphic encryption + secure aggregation
│
└── CoLab/                         ← Jupyter notebooks for Google Colab
    ├── 1_centralized_baseline.ipynb
    ├── 2_data_partitioning.ipynb
    ├── 3_federated_training.ipynb
    ├── 4_evaluation_visualization.ipynb
    ├── 5_practice.ipynb
    └── 6_privacy_extension.ipynb
```

---

## Conceptual Roadmap

```
Step 1             Step 2              Step 3              Step 4
Centralized   →   Split Data    →   Federated FL    →   Compare
Baseline          (Non-IID)         (FedAvg)            & Visualize
"What's the       "Who has           "Train without      "Did FL work?"
best we           what data?"        sharing data"
can do?"
                                          ↓
                                       Step 5
                                       Practice
                                    "Change params,
                                     what happens?"
                                          ↓
                                       Step 6
                                    Privacy Extensions
                                   "Make it even more
                                    private with HE
                                    and Secure Agg."
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: torch` | Run `pip install torch numpy matplotlib` |
| `FileNotFoundError: client1.npz` | Run Step 2 before Step 3 or 6 |
| `FileNotFoundError: centralized_results.json` | Run Step 1 before Step 4 |
| `ModuleNotFoundError: tenseal` | Run `pip install tenseal` (Step 6 only) — script works without it |
| Step 6 TenSEAL install fails | Script auto-falls back to simulation — no action needed |
| Training is slow | Enable GPU: check that CUDA is installed, or use the Colab notebooks (free GPU) |

---

## Further Reading

- McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data* (original FedAvg paper)
- Bonawitz et al. (2017) — *Practical Secure Aggregation for Privacy-Preserving Machine Learning*
- MedMNIST — https://medmnist.com (source of the DermaMNIST dataset)
- TenSEAL — https://github.com/OpenMined/TenSEAL (homomorphic encryption library)
- Flower Framework — https://flower.dev (production-grade FL framework)
