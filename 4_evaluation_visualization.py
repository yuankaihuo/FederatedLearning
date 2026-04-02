# =============================================================================
# FILE 4: Evaluation & Visualization
# =============================================================================
# Goal: Compare centralized vs. federated training results side by side.
#
# We will look at:
#   1. Final test accuracy        — how well does each approach classify?
#   2. Convergence behavior       — how does accuracy improve over time?
#   3. Communication cost         — how much data is sent over the network?
#   4. Privacy benefit            — conceptual, not a number
# =============================================================================

import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model size estimate (used for communication cost calculation)
# SimpleCNN has ~500K parameters, each a 32-bit float = 4 bytes → ~2 MB
MODEL_SIZE_MB = 2.0


# =============================================================================
# STEP 1: Load Saved Results
# =============================================================================
print("--- Loading results ---")

centralized_path = os.path.join(RESULTS_DIR, "centralized_results.json")
federated_path   = os.path.join(RESULTS_DIR, "federated_results.json")

# Check that both result files exist
if not os.path.exists(centralized_path):
    print("  ERROR: centralized_results.json not found.")
    print("  Please run 1_centralized_baseline.py first!")
    exit(1)

if not os.path.exists(federated_path):
    print("  ERROR: federated_results.json not found.")
    print("  Please run 3_federated_training.py first!")
    exit(1)

with open(centralized_path) as f:
    central = json.load(f)

with open(federated_path) as f:
    federated = json.load(f)

print(f"  Centralized epochs : {central['epochs']}")
print(f"  Federated rounds   : {federated['num_rounds']}")
print(f"  Local epochs/round : {federated['local_epochs']}")


# =============================================================================
# STEP 2: Accuracy Comparison (Text)
# =============================================================================
print("\n" + "="*55)
print("   RESULTS COMPARISON")
print("="*55)
print(f"   {'Method':<25} {'Test Accuracy':>15}")
print(f"   {'-'*42}")
print(f"   {'Centralized (baseline)':<25} {central['test_acc']:>14.2%}")
print(f"   {'Federated (2 clients)':<25} {federated['test_acc']:>14.2%}")

diff = federated["test_acc"] - central["test_acc"]
sign = "+" if diff >= 0 else ""
print(f"\n   Accuracy difference  : {sign}{diff:.2%}")
if abs(diff) < 0.03:
    print("   → Federated achieves similar accuracy to centralized!")
elif diff > 0:
    print("   → Federated outperformed centralized!")
else:
    print("   → Federated is slightly lower — typical with non-IID data.")


# =============================================================================
# STEP 3: Communication Cost Analysis
# =============================================================================
print("\n" + "="*55)
print("   COMMUNICATION COST")
print("="*55)

n_rounds  = federated["num_rounds"]
n_clients = 2

# Each round: server sends model to all clients, clients send back
# = 2 transmissions per client per round
upload_total   = n_rounds * n_clients * MODEL_SIZE_MB  # clients → server
download_total = n_rounds * n_clients * MODEL_SIZE_MB  # server → clients
total_comm     = upload_total + download_total

print(f"   Rounds           : {n_rounds}")
print(f"   Clients          : {n_clients}")
print(f"   Model size       : {MODEL_SIZE_MB} MB")
print(f"   Upload (clients) : {upload_total:.1f} MB total")
print(f"   Download (server): {download_total:.1f} MB total")
print(f"   Total comm. cost : {total_comm:.1f} MB")
print(f"\n   KEY POINT: Only model weights are transmitted.")
print(f"   Raw patient data NEVER leaves the hospital!")
print(f"   Total data protected: ~{7007 * 28 * 28 * 3 / 1e6:.1f} MB of patient images")


# =============================================================================
# STEP 4: Visualizations
# =============================================================================
print("\n--- Creating plots ---")

# ---- Plot A: Accuracy over epochs/rounds ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Centralized vs. Federated Learning — DermaMNIST", fontsize=14)

# Centralized: plot val accuracy per epoch
ax = axes[0]
epochs = list(range(1, central["epochs"] + 1))
ax.plot(epochs, central["history"]["train_acc"],
        marker="o", label="Train Accuracy", color="steelblue")
ax.plot(epochs, central["history"]["val_acc"],
        marker="s", label="Val Accuracy",   color="darkorange")
ax.set_title("Centralized Training")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
# Annotate final val accuracy
final_val = central["history"]["val_acc"][-1]
ax.annotate(f"{final_val:.2%}", xy=(epochs[-1], final_val),
            xytext=(-30, 10), textcoords="offset points", fontsize=9)

# Federated: plot test accuracy per round
ax = axes[1]
rounds = federated["history"]["round"]
acc    = federated["history"]["test_acc"]
ax.plot(rounds, acc, marker="o", label="Global Test Accuracy", color="seagreen")
ax.set_title("Federated Training (2 Clients, FedAvg)")
ax.set_xlabel("Communication Round")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
# Annotate final accuracy
ax.annotate(f"{acc[-1]:.2%}", xy=(rounds[-1], acc[-1]),
            xytext=(-30, 10), textcoords="offset points", fontsize=9)

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "accuracy_comparison.png")
plt.savefig(plot_path, dpi=120)
print(f"  Saved: {plot_path}")
plt.show()


# ---- Plot B: Final accuracy bar chart ----
fig, ax = plt.subplots(figsize=(6, 5))
methods = ["Centralized", "Federated\n(2 clients)"]
accs    = [central["test_acc"], federated["test_acc"]]
colors  = ["steelblue", "seagreen"]
bars    = ax.bar(methods, accs, color=colors, width=0.45, edgecolor="white")

# Add value labels on top of bars
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.2%}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylim(0, 1.0)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Final Test Accuracy: Centralized vs. Federated", fontsize=13)
ax.axhline(central["test_acc"], color="steelblue",
           linestyle="--", alpha=0.4, label="Centralized baseline")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
bar_path = os.path.join(PLOTS_DIR, "final_accuracy_bar.png")
plt.savefig(bar_path, dpi=120)
print(f"  Saved: {bar_path}")
plt.show()


# ---- Plot C: Communication cost vs. accuracy tradeoff ----
fig, ax = plt.subplots(figsize=(7, 5))

# Each round's cumulative communication cost
cumulative_comm = [r * n_clients * 2 * MODEL_SIZE_MB
                   for r in federated["history"]["round"]]
ax.plot(cumulative_comm, federated["history"]["test_acc"],
        marker="o", color="seagreen", label="Federated")
ax.axhline(central["test_acc"], color="steelblue",
           linestyle="--", label=f"Centralized ({central['test_acc']:.2%})")

# Label each round
for comm, acc, r in zip(cumulative_comm, federated["history"]["test_acc"], rounds):
    ax.annotate(f"R{r}", xy=(comm, acc),
                xytext=(4, 4), textcoords="offset points", fontsize=8)

ax.set_xlabel("Cumulative Communication Cost (MB)", fontsize=11)
ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_title("Convergence vs. Communication Cost", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
conv_path = os.path.join(PLOTS_DIR, "convergence_vs_cost.png")
plt.savefig(conv_path, dpi=120)
print(f"  Saved: {conv_path}")
plt.show()


# =============================================================================
# STEP 5: Key Takeaways
# =============================================================================
print("\n" + "="*55)
print("   KEY TAKEAWAYS")
print("="*55)
print("""
   1. PRIVACY: In federated learning, raw patient data
      never leaves the hospital. Only model weights are shared.

   2. ACCURACY: Federated learning can achieve accuracy
      close to centralized training — even with non-IID data.

   3. COMMUNICATION: FL requires sending model weights
      each round. More rounds = higher cost but better accuracy.

   4. NON-IID CHALLENGE: When clients have very different
      data distributions, federated training is harder.
      This is an active research area!

   5. SCALABILITY: This demo used 2 clients. Real FL systems
      can coordinate hundreds or thousands of hospitals/devices.
""")

print("=== Done! Plots saved in results/plots/ ===")
print("    Run 5_practice.py to experiment with FL parameters.")
