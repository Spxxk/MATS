import os, json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIRS = {
    "C1: Prior vs false context": "outputs/main_30",
    "C2: Context vs context": "outputs/c2_15",
    "C3: Prior vs true context + misleading framing": "outputs/c3_15",
}

lags = {}

for label, base in BASE_DIRS.items():
    vals = []
    for item in sorted(os.listdir(base)):
        meta_path = os.path.join(base, item, "meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["lag"] is not None:
            vals.append(meta["lag"])
    lags[label] = np.array(vals)

# Plot
plt.figure(figsize=(8, 5))

bins = np.arange(-5, 60, 4)

for label, vals in lags.items():
    plt.hist(
        vals,
        bins=bins,
        alpha=0.5,
        label=f"{label} (n={len(vals)})"
    )

plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Lag = ack_time − commit_time (tokens)")
plt.ylabel("Count")
plt.title("Commitment–acknowledgment lag across conflict types")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figure3_condition_comparison.png", dpi=200)
plt.close()

print("Saved outputs/figure3_condition_comparison.png")
