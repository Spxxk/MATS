import os, json
import numpy as np
import matplotlib.pyplot as plt

# file name specific
BASE_DIRS = {
    "C1: Prior vs false context (n=30)": "outputs/main_30",
    "B1: No document (n=15)": "outputs/b1_15",
    "B3: Question before document (n=15)": "outputs/b3_15",
}

def collect_lags(base):
    vals = []
    for item in sorted(os.listdir(base)):
        meta_path = os.path.join(base, item, "meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("lag") is not None:
            vals.append(meta["lag"])
    return np.array(vals, dtype=np.float32)

lags = {label: collect_lags(path) for label, path in BASE_DIRS.items()}

plt.figure(figsize=(8, 5))
bins = np.arange(-5, 60, 4)

for label, vals in lags.items():
    plt.hist(vals, bins=bins, alpha=0.5, label=f"{label} (usable={len(vals)})")

plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Lag = ack_time − commit_time (tokens)")
plt.ylabel("Count")
plt.title("Baseline checks: is lag specific to conflict prompts?")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figure4_baselines.png", dpi=200)
plt.close()

print("Saved outputs/figure4_baselines.png")
