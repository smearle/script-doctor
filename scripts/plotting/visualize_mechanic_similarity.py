"""Compute and visualize mechanic similarity between PuzzleScript games.

Usage:
    python visualize_mechanic_similarity.py [--dataset pedro] [--max-games 200]
"""
import argparse
import os
import pickle
import sys
import traceback
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

EMBEDDINGS_PLOT_DIR = os.path.join('plots', 'embeddings')

from puzzlescript_jax.gen_tree import GenPSTree
from puzzlescript_jax.globals import TREES_DIR
from puzzlescript_jax.utils import get_list_of_games_for_testing

from puzzlejax.mechanic_graph import build_mechanic_graph, canonical_label_vector


def load_game_tree(game):
    pkl_path = os.path.join(TREES_DIR, game + ".pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        min_tree = pickle.load(f)
    return GenPSTree().transform(min_tree)


def wl_similarity(v1: Counter, v2: Counter) -> float:
    intersection = sum((v1 & v2).values())
    union = sum((v1 | v2).values())
    return intersection / union if union > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pedro")
    ap.add_argument("--max-games", type=int, default=300,
                    help="Cap number of games for readability")
    ap.add_argument("--wl-iterations", type=int, default=3)
    ap.add_argument("--n-clusters", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(EMBEDDINGS_PLOT_DIR, exist_ok=True)

    games = get_list_of_games_for_testing(dataset=args.dataset, include_random=True)
    print(f"Loading {len(games)} games from '{args.dataset}'...")

    # Build mechanic graphs and WL label vectors
    game_names = []
    label_vectors = []
    for game in games:
        try:
            tree = load_game_tree(game)
            if tree is None:
                continue
            G = build_mechanic_graph(tree)
            lv = canonical_label_vector(G, iterations=args.wl_iterations)
            game_names.append(game)
            label_vectors.append(lv)
        except Exception:
            continue

    print(f"Successfully processed {len(game_names)} games")

    # Subsample if too many for the heatmap
    if len(game_names) > args.max_games:
        # Keep priority games + random sample
        from puzzlescript_jax.globals import PRIORITY_GAMES
        priority_set = set(PRIORITY_GAMES)
        priority_idxs = [i for i, g in enumerate(game_names) if g in priority_set]
        other_idxs = [i for i, g in enumerate(game_names) if g not in priority_set]
        np.random.seed(42)
        n_sample = args.max_games - len(priority_idxs)
        sampled_other = list(np.random.choice(other_idxs, size=min(n_sample, len(other_idxs)), replace=False))
        keep_idxs = sorted(priority_idxs + sampled_other)
        game_names = [game_names[i] for i in keep_idxs]
        label_vectors = [label_vectors[i] for i in keep_idxs]
        print(f"Subsampled to {len(game_names)} games for visualization")

    N = len(game_names)

    # Compute pairwise similarity matrix
    print(f"Computing {N}x{N} similarity matrix...")
    sim_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, N):
            s = wl_similarity(label_vectors[i], label_vectors[j])
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    # Convert to distance matrix for clustering
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    # Ensure non-negative (floating point)
    dist_matrix = np.maximum(dist_matrix, 0)

    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method='ward')
    clusters = fcluster(Z, t=args.n_clusters, criterion='maxclust')

    # Reorder by cluster
    order = np.argsort(clusters)
    sim_ordered = sim_matrix[np.ix_(order, order)]
    names_ordered = [game_names[i] for i in order]
    clusters_ordered = clusters[order]

    # --- Plot 1: Clustered heatmap ---
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(1, 2, width_ratios=[1, 12], wspace=0.02)

    # Dendrogram
    ax_dendro = fig.add_subplot(gs[0])
    dendro = dendrogram(Z, orientation='left', no_labels=True, ax=ax_dendro,
                        color_threshold=Z[-(args.n_clusters - 1), 2])
    ax_dendro.set_xticks([])
    ax_dendro.invert_yaxis()

    # Heatmap
    ax_heat = fig.add_subplot(gs[1])
    # Reorder to match dendrogram
    dendro_order = dendro['leaves']
    sim_dendro = sim_matrix[np.ix_(dendro_order, dendro_order)]
    names_dendro = [game_names[i] for i in dendro_order]

    im = ax_heat.imshow(sim_dendro, cmap='magma', aspect='auto', vmin=0, vmax=1)
    ax_heat.set_title(f'Mechanic Similarity (WL graph kernel, {N} games)', fontsize=14)

    # Label every Nth game for readability
    label_step = max(1, N // 40)
    tick_positions = list(range(0, N, label_step))
    ax_heat.set_xticks(tick_positions)
    ax_heat.set_xticklabels([names_dendro[i] for i in tick_positions],
                             rotation=90, fontsize=5)
    ax_heat.set_yticks(tick_positions)
    ax_heat.set_yticklabels([names_dendro[i] for i in tick_positions], fontsize=5)

    plt.colorbar(im, ax=ax_heat, label='Mechanic Similarity', shrink=0.6)
    plt.tight_layout()
    heatmap_path = os.path.join(EMBEDDINGS_PLOT_DIR, 'mechanic_similarity_heatmap.png')
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    print(f'Saved {heatmap_path}')
    plt.close()

    # --- Plot 2: UMAP of WL label vectors ---
    # Convert WL label vectors to a sparse feature matrix
    all_labels = set()
    for lv in label_vectors:
        all_labels.update(lv.keys())
    all_labels = sorted(all_labels)
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    X = np.zeros((N, len(all_labels)), dtype=np.float32)
    for i, lv in enumerate(label_vectors):
        for label, count in lv.items():
            X[i, label_to_idx[label]] = count

    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X)

    # Color by cluster
    fig, ax = plt.subplots(figsize=(16, 12))
    cmap = plt.cm.get_cmap('tab20', args.n_clusters)
    for cl in range(1, args.n_clusters + 1):
        mask = clusters == cl
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(cl - 1)], s=20, alpha=0.7, label=f'Cluster {cl}')

    # Label some games
    from puzzlescript_jax.globals import PRIORITY_GAMES
    for g in PRIORITY_GAMES:
        if g in game_names:
            i = game_names.index(g)
            ax.annotate(g, (embedding[i, 0], embedding[i, 1]), fontsize=5, alpha=0.9,
                        arrowprops=dict(arrowstyle='->', alpha=0.4, lw=0.5),
                        textcoords='offset points', xytext=(5, 5))

    ax.set_title(f'PuzzleScript Games - UMAP of WL Graph Kernel Features ({N} games)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(fontsize=7, loc='upper left', ncol=2, framealpha=0.7)
    plt.tight_layout()
    umap_path = os.path.join(EMBEDDINGS_PLOT_DIR, 'mechanic_similarity_umap.png')
    plt.savefig(umap_path, dpi=200)
    print(f'Saved {umap_path}')
    plt.close()

    # --- Plot 3: Cluster summary ---
    fig, ax = plt.subplots(figsize=(14, 8))
    cluster_games = {}
    for cl in range(1, args.n_clusters + 1):
        members = [game_names[i] for i in range(N) if clusters[i] == cl]
        cluster_games[cl] = members

    # For each cluster, find the most representative game (highest avg similarity to cluster)
    cluster_reps = {}
    for cl, members in cluster_games.items():
        if not members:
            continue
        member_idxs = [game_names.index(m) for m in members]
        sub_sim = sim_matrix[np.ix_(member_idxs, member_idxs)]
        avg_sim = sub_sim.mean(axis=1)
        best = member_idxs[np.argmax(avg_sim)]
        cluster_reps[cl] = game_names[best]

    # Bar chart of cluster sizes with representative game labels
    cls = sorted(cluster_games.keys())
    sizes = [len(cluster_games[cl]) for cl in cls]
    bars = ax.bar(range(len(cls)), sizes, color=[cmap(cl - 1) for cl in cls])
    ax.set_xticks(range(len(cls)))
    xlabels = []
    for cl in cls:
        rep = cluster_reps.get(cl, "?")
        n_members = len(cluster_games[cl])
        # Show up to 3 example games
        examples = cluster_games[cl][:3]
        xlabels.append(f"C{cl}\n({n_members})\n" + "\n".join(examples[:3]))
    ax.set_xticklabels(xlabels, fontsize=6, ha='center')
    ax.set_ylabel('Number of games')
    ax.set_title(f'Mechanic Clusters ({args.n_clusters} clusters, {N} games)')
    plt.tight_layout()
    clusters_path = os.path.join(EMBEDDINGS_PLOT_DIR, 'mechanic_clusters.png')
    plt.savefig(clusters_path, dpi=200, bbox_inches='tight')
    print(f'Saved {clusters_path}')
    plt.close()

    # --- Print nearest neighbors for priority games ---
    print("\n=== Nearest mechanic neighbors (priority games) ===")
    for g in PRIORITY_GAMES:
        if g not in game_names:
            continue
        i = game_names.index(g)
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        top5 = np.argsort(sims)[-5:][::-1]
        neighbors = [(game_names[j], sims[j]) for j in top5]
        print(f"\n  {g} (cluster {clusters[i]}):")
        for name, s in neighbors:
            print(f"    {s:.3f}  {name}")

    # Save data
    np.savez('mechanic_similarity.npz',
             names=game_names, sim_matrix=sim_matrix,
             clusters=clusters, umap=embedding)
    print('\nSaved mechanic_similarity.npz')


if __name__ == "__main__":
    main()
