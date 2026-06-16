"""Plot distribution of number of rules per dataset tier.

Usage:
    python plot_rules_distribution.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from puzzlescript_jax.globals import GAMES_METADATA_PATH, PLOTS_DIR
from puzzlescript_jax.utils import get_list_of_games_for_testing

DATASETS = ["priority", "gallery", "pedro", "increpare"]
RULES_THRESHOLD = 44


def main():
    with open(GAMES_METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Strip .txt from metadata keys for matching
    meta_by_name = {}
    for k, v in metadata.items():
        name = k[:-4] if k.endswith(".txt") else k
        meta_by_name[name] = v

    # Collect n_rules per dataset
    dataset_rules = {}
    for ds in DATASETS:
        games = get_list_of_games_for_testing(dataset=ds)
        rules = []
        for g in games:
            m = meta_by_name.get(g)
            if m and "n_rules" in m:
                rules.append(m["n_rules"])
        dataset_rules[ds] = np.array(rules)

    # Print stats
    print(f"{'Dataset':<12} {'Count':>6} {'Median':>8} {'<=44 rules':>12}")
    print("-" * 42)
    for ds in DATASETS:
        r = dataset_rules[ds]
        median = np.median(r)
        pct = 100.0 * np.mean(r <= RULES_THRESHOLD)
        print(f"{ds:<12} {len(r):>6} {median:>8.1f} {pct:>11.1f}%")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = {"priority": "C0", "gallery": "C1", "pedro": "C2", "increpare": "C3"}

    for ax, ds in zip(axes.flat, DATASETS):
        r = dataset_rules[ds]
        median = np.median(r)
        pct_below = 100.0 * np.mean(r <= RULES_THRESHOLD)

        ax.hist(r, bins=30, color=colors[ds], edgecolor="white", alpha=0.85)
        ax.axvline(median, color="black", linestyle="--", linewidth=1.5, label=f"median={median:.0f}")
        ax.axvline(RULES_THRESHOLD, color="red", linestyle=":", linewidth=1.5,
                   label=f"<={RULES_THRESHOLD}: {pct_below:.0f}%")
        ax.set_title(f"{ds} (n={len(r)})")
        ax.set_xlabel("Number of rules")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Distribution of Number of Rules per Dataset", fontsize=14)
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, "rules_distribution.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
