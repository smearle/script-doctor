"""Visualize search metrics overlaid on mechanic-graph embeddings.

Loads precomputed mechanic similarity data (mechanic_similarity.npz) and
search results ({algo}_results.json) to produce:
  1. UMAP scatter colored by solve rate (one subplot per algo)
  2. UMAP scatter colored by search effort (iterations to solve)
  3. Cluster-level performance bar charts (mean solve rate per cluster per algo)
  4. Mechanic similarity vs search-outcome similarity scatter
  5. Correlation: mechanic cluster membership vs search difficulty

Usage:
    python plot_search_vs_embeddings.py [--depth 1000000] [--out-dir data/plots/search_embeddings]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

ALGO_NAMES = ['astar', 'bfs', 'gbfs', 'mcts']
ALGO_LABELS = {'bfs': 'BFS', 'astar': 'A*', 'gbfs': 'GBFS', 'mcts': 'MCTS'}


def _load_search_results(algo: str, depth: int) -> dict[str, dict]:
    """Load search results for a given algo, returning {game: metrics} at best depth."""
    path = os.path.join('data', f'{algo}_results.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        raw = json.load(f)

    # Results may be nested by depth or flat
    sample = next(iter(raw.values()), None)
    if isinstance(sample, dict) and 'pct_solved' in sample:
        return raw  # flat

    # Nested by depth — pick requested depth, fall back to largest available
    if str(depth) in raw:
        return raw[str(depth)]
    if depth in raw:
        return raw[depth]
    # Fallback: largest depth key
    int_keys = sorted(int(k) for k in raw.keys())
    return raw[str(int_keys[-1])] if int_keys else {}


def main():
    ap = argparse.ArgumentParser(description='Search metrics × mechanic embeddings')
    ap.add_argument('--depth', type=int, default=1_000_000,
                    help='Search depth to use from results')
    ap.add_argument('--out-dir', default=os.path.join('plots', 'embeddings'))
    ap.add_argument('--sim-npz', default='mechanic_similarity.npz')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load mechanic embedding data ---
    npz = np.load(args.sim_npz, allow_pickle=True)
    emb_names = list(npz['names'])
    sim_matrix = npz['sim_matrix']
    clusters = npz['clusters']
    umap_coords = npz['umap']
    n_clusters = int(clusters.max())

    name_to_idx = {n: i for i, n in enumerate(emb_names)}

    # --- Load search results for all algos ---
    results_by_algo: dict[str, dict[str, dict]] = {}
    for algo in ALGO_NAMES:
        r = _load_search_results(algo, args.depth)
        if r:
            results_by_algo[algo] = r

    if not results_by_algo:
        print('No search results found. Exiting.')
        return

    # Find games present in both embeddings and at least one algo's results
    all_search_games = set()
    for r in results_by_algo.values():
        all_search_games.update(r.keys())
    common_games = [g for g in emb_names if g in all_search_games]
    print(f'{len(emb_names)} games in embeddings, {len(all_search_games)} in search results, '
          f'{len(common_games)} overlap')

    if len(common_games) < 5:
        print('Too few overlapping games. Exiting.')
        return

    common_idxs = np.array([name_to_idx[g] for g in common_games])
    common_umap = umap_coords[common_idxs]
    common_clusters = clusters[common_idxs]

    algos_present = sorted(results_by_algo.keys())

    # =====================================================================
    # Plot 1: UMAP colored by solve rate (one subplot per algo)
    # =====================================================================
    n_algos = len(algos_present)
    n_cols = min(n_algos, 2)
    n_rows = (n_algos + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)

    for ax_i, algo in enumerate(algos_present):
        ax = axes[ax_i // n_cols][ax_i % n_cols]
        r = results_by_algo[algo]

        pct_solved = np.array([r.get(g, {}).get('pct_solved', np.nan) for g in common_games])
        has_data = ~np.isnan(pct_solved)

        # Background: games without data
        ax.scatter(common_umap[~has_data, 0], common_umap[~has_data, 1],
                   c='lightgray', s=15, alpha=0.4, edgecolors='none', label='no data')

        sc = ax.scatter(common_umap[has_data, 0], common_umap[has_data, 1],
                        c=pct_solved[has_data], cmap='RdYlGn', vmin=0, vmax=1,
                        s=25, alpha=0.8, edgecolors='k', linewidths=0.3)
        ax.set_title(ALGO_LABELS.get(algo, algo.upper()), fontsize=13)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(sc, ax=ax, label='% Solved', shrink=0.7)

    for ax in axes.flat[n_algos:]:
        ax.axis('off')

    fig.suptitle('Solve Rate on Mechanic-Graph UMAP', fontsize=15, y=1.01)
    fig.tight_layout()
    path = os.path.join(args.out_dir, 'umap_solve_rate.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # =====================================================================
    # Plot 2: UMAP colored by search effort (log iterations to solve)
    # =====================================================================
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)

    for ax_i, algo in enumerate(algos_present):
        ax = axes[ax_i // n_cols][ax_i % n_cols]
        r = results_by_algo[algo]

        iters = np.array([r.get(g, {}).get('mean_solved_iters', np.nan) for g in common_games])
        has_data = ~np.isnan(iters) & (iters > 0)

        ax.scatter(common_umap[~has_data, 0], common_umap[~has_data, 1],
                   c='lightgray', s=15, alpha=0.4, edgecolors='none', label='unsolved/no data')

        if has_data.any():
            sc = ax.scatter(common_umap[has_data, 0], common_umap[has_data, 1],
                            c=np.log10(iters[has_data]), cmap='plasma',
                            s=25, alpha=0.8, edgecolors='k', linewidths=0.3)
            plt.colorbar(sc, ax=ax, label='log₁₀(iterations)', shrink=0.7)

        ax.set_title(ALGO_LABELS.get(algo, algo.upper()), fontsize=13)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

    for ax in axes.flat[n_algos:]:
        ax.axis('off')

    fig.suptitle('Search Effort on Mechanic-Graph UMAP', fontsize=15, y=1.01)
    fig.tight_layout()
    path = os.path.join(args.out_dir, 'umap_search_effort.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # =====================================================================
    # Plot 3: Mean solve rate per mechanic cluster per algo
    # =====================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    cluster_ids = sorted(set(common_clusters))
    x = np.arange(len(cluster_ids))
    bar_width = 0.8 / n_algos

    for algo_i, algo in enumerate(algos_present):
        r = results_by_algo[algo]
        means = []
        stds = []
        counts = []
        for cl in cluster_ids:
            mask = common_clusters == cl
            cl_games = [common_games[i] for i in range(len(common_games)) if mask[i]]
            vals = [r.get(g, {}).get('pct_solved', np.nan) for g in cl_games]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)
            counts.append(len(vals))

        offset = (algo_i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                      label=ALGO_LABELS.get(algo, algo), alpha=0.8,
                      capsize=2, error_kw={'lw': 0.8})

        # Annotate with game count
        for xi, cnt in zip(x + offset, counts):
            if cnt > 0:
                ax.text(xi, -0.06, str(cnt), ha='center', va='top', fontsize=6, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels([f'C{cl}' for cl in cluster_ids], fontsize=10)
    ax.set_xlabel('Mechanic Cluster')
    ax.set_ylabel('Mean % Levels Solved')
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('Search Performance by Mechanic Cluster')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out_dir, 'cluster_solve_rate.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # =====================================================================
    # Plot 4: Mechanic similarity vs search-outcome similarity
    # =====================================================================
    # For each pair of games, compare mechanic similarity to similarity in
    # search outcomes (absolute difference in pct_solved, inverted).
    # Use best algo (A* at max depth) for a clean signal.
    best_algo = 'astar' if 'astar' in results_by_algo else algos_present[0]
    r = results_by_algo[best_algo]

    # Build vectors for common games that have solve data
    solve_vals = np.array([r.get(g, {}).get('pct_solved', np.nan) for g in common_games])
    valid = ~np.isnan(solve_vals)
    valid_idxs_local = np.where(valid)[0]
    n_valid = len(valid_idxs_local)

    if n_valid >= 10:
        # Sample pairs (full pairwise may be huge)
        n_pairs = min(n_valid * (n_valid - 1) // 2, 50_000)
        rng = np.random.RandomState(42)

        mech_sims = []
        outcome_diffs = []
        for _ in range(n_pairs):
            i, j = rng.choice(n_valid, size=2, replace=False)
            gi, gj = valid_idxs_local[i], valid_idxs_local[j]
            emb_i, emb_j = common_idxs[gi], common_idxs[gj]
            mech_sims.append(sim_matrix[emb_i, emb_j])
            outcome_diffs.append(abs(solve_vals[gi] - solve_vals[gj]))

        mech_sims = np.array(mech_sims)
        outcome_diffs = np.array(outcome_diffs)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 4a: Scatter
        ax = axes[0]
        ax.scatter(mech_sims, outcome_diffs, alpha=0.05, s=5, c='steelblue', edgecolors='none')
        ax.set_xlabel('Mechanic Similarity (WL kernel)')
        ax.set_ylabel('|Δ Solve Rate|')
        ax.set_title(f'{ALGO_LABELS[best_algo]}: Mechanic Similarity vs Solve-Rate Difference')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

        # Binned trend line
        bin_edges = np.linspace(0, 1, 21)
        bin_centers = []
        bin_means = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (mech_sims >= lo) & (mech_sims < hi)
            if mask.sum() >= 5:
                bin_centers.append((lo + hi) / 2)
                bin_means.append(outcome_diffs[mask].mean())
        if bin_centers:
            ax.plot(bin_centers, bin_means, 'r-o', markersize=4, linewidth=2, label='binned mean')
            ax.legend()

        rho, pval = scipy_stats.spearmanr(mech_sims, outcome_diffs)
        ax.text(0.02, 0.98, f'Spearman ρ = {rho:.3f}\np = {pval:.2e}\nn = {n_pairs:,}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # 4b: Hex-bin density view
        ax = axes[1]
        hb = ax.hexbin(mech_sims, outcome_diffs, gridsize=30, cmap='YlOrRd', mincnt=1)
        ax.set_xlabel('Mechanic Similarity (WL kernel)')
        ax.set_ylabel('|Δ Solve Rate|')
        ax.set_title('Density')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.colorbar(hb, ax=ax, label='count')

        fig.tight_layout()
        path = os.path.join(args.out_dir, 'mechsim_vs_solve_diff.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')

    # =====================================================================
    # Plot 5: Per-cluster performance radar / summary table
    # =====================================================================
    metrics = ['pct_solved', 'mean_solved_iters', 'mean_score_progress']
    metric_labels = ['% Solved', 'Iterations (log₁₀)', 'Score Progress']

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    cmap = plt.cm.get_cmap('tab20', n_clusters)

    for mi, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[mi]
        for algo_i, algo in enumerate(algos_present):
            r = results_by_algo[algo]
            cluster_means = []
            for cl in cluster_ids:
                mask = common_clusters == cl
                cl_games = [common_games[i] for i in range(len(common_games)) if mask[i]]
                vals = [r.get(g, {}).get(metric, np.nan) for g in cl_games]
                vals = [v for v in vals if not np.isnan(v) and v > 0]
                if vals:
                    m = np.mean(vals)
                    if metric == 'mean_solved_iters':
                        m = np.log10(m) if m > 0 else 0
                    cluster_means.append(m)
                else:
                    cluster_means.append(np.nan)

            ax.plot(cluster_ids, cluster_means, 'o-', markersize=4, alpha=0.8,
                    label=ALGO_LABELS.get(algo, algo))

        ax.set_xticks(cluster_ids)
        ax.set_xticklabels([f'C{cl}' for cl in cluster_ids], fontsize=8)
        ax.set_xlabel('Mechanic Cluster')
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle('Search Metrics by Mechanic Cluster', fontsize=14)
    fig.tight_layout()
    path = os.path.join(args.out_dir, 'cluster_metrics_detail.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # =====================================================================
    # Print summary
    # =====================================================================
    print(f'\nAll plots saved to {args.out_dir}/')
    print(f'  umap_solve_rate.png       - UMAP colored by % solved per algo')
    print(f'  umap_search_effort.png    - UMAP colored by search iterations')
    print(f'  cluster_solve_rate.png    - Bar chart: cluster × algo solve rates')
    print(f'  mechsim_vs_solve_diff.png - Mechanic similarity vs solve-rate gap')
    print(f'  cluster_metrics_detail.png - Line plots: metrics across clusters')


if __name__ == '__main__':
    main()
