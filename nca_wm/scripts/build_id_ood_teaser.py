"""Build the ID/OOD teaser figure for the paper.

Renders a few frames each from a small set of in-distribution training games
and a small set of held-out (OOD) games, composes them into a single
matplotlib figure with a dashed horizontal border separating the two halves,
and arrows indicating the role of each frame in the NCA WM training/eval
loop:

  - Training (top half):  s_t, s_{t+1} ... feed in *and* drive the loss.
  - Held-out (bottom):    s_t feeds in; the model rolls forward
                          autoregressively against the engine's s_{t+1}.

Output:  nca_wm/paper/figures/id_ood_teaser/teaser.{pdf,png}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_jax.utils import init_ps_lark_parser  # noqa: E402
from puzzlescript_cpp import CppPuzzleScriptBackend  # noqa: E402

ID_GAMES = [
    ("nekopuzzle", 0, "nekopuzzle"),
    ("sokoban_basic", 0, "sokoban_basic"),
    ("Modality", 0, "Modality"),
]
OOD_GAMES = [
    ("in_the_way_by_giovanni_mota", 0, "in_the_way (Mota)"),
    ("link_by_jeffjeff123456", 0, "link (jeffjeff123456)"),
    ("Hamiltwo", 0, "Hamiltwo"),
]
N_FRAMES = 4
PAPER_DIR = _REPO_ROOT / "nca_wm" / "paper"
OUT_DIR = PAPER_DIR / "figures" / "id_ood_teaser"
OUT_DIR.mkdir(parents=True, exist_ok=True)


_PARSER = None
def _parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = init_ps_lark_parser()
    return _PARSER


def _bfs_cache_path(name: str, level: int) -> Path | None:
    """Return BFS solution cache for game/level if any candidate exists."""
    candidates = [
        _REPO_ROOT / "rollout_data" / name / f"level_{level}"
            / "search_bfs_100000_60000.npz",
        _REPO_ROOT / "rollout_data" / name / f"level_{level}"
            / "search_bfs_100000_-1.npz",
        _REPO_ROOT / "nca_wm" / "data_cache" / "heldout_search"
            / f"{name}_L{level}_bfs_100000_60000.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_actions(name: str, level: int, fallback_seed: int = 0,
                  n_steps: int = N_FRAMES - 1) -> list[int]:
    cache = _bfs_cache_path(name, level)
    if cache is not None:
        d = np.load(cache, allow_pickle=True)
        if "actions" in d.files and len(d["actions"]) > 0:
            return [int(a) for a in d["actions"]]
    # Random fallback: 5 actions (up/left/down/right/use).
    rng = np.random.default_rng(fallback_seed)
    return [int(rng.integers(5)) for _ in range(n_steps * 3)]


def _evenly_spaced_indices(n_total: int, n_pick: int) -> list[int]:
    if n_total <= n_pick:
        return list(range(n_total))
    return [int(round(i * (n_total - 1) / (n_pick - 1))) for i in range(n_pick)]


def _render_frames(name: str, level: int, n_frames: int) -> list[np.ndarray]:
    backend = CppPuzzleScriptBackend()
    backend.compile_game(_parser(), name)
    backend.cpp_engine.load_level(level)
    eng = backend.cpp_engine

    actions = _load_actions(name, level, n_steps=n_frames - 1)
    frames = [backend.render_frame()]
    for a in actions:
        eng.process_input(int(a))
        again = 0
        while eng.againing and again < 8:
            eng.process_input(-1)
            again += 1
        frames.append(backend.render_frame())
    # Subsample to n_frames including the initial state.
    idx = _evenly_spaced_indices(len(frames), n_frames)
    return [frames[i] for i in idx]


def _render_section(games):
    out = []
    for name, level, _label in games:
        frames = _render_frames(name, level, N_FRAMES)
        out.append(frames)
    return out


def _add_frame(ax, img: np.ndarray, *, edge_color: str, edge_lw: float = 0.8):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edge_color)
        s.set_linewidth(edge_lw)


def build_figure():
    print("Rendering ID frames...")
    id_rollouts = _render_section(ID_GAMES)
    print("Rendering OOD frames...")
    ood_rollouts = _render_section(OOD_GAMES)

    n_id = len(ID_GAMES)
    n_ood = len(OOD_GAMES)
    n_rows = n_id + n_ood

    # Layout: per-frame columns + a label column on the left.
    label_w_in = 1.4
    frame_w_in = 1.05
    frame_h_in = 0.78
    band_h_in = 0.28  # vertical room for banner above each section
    fig_w = label_w_in + N_FRAMES * frame_w_in + 0.3
    fig_h = (n_rows * frame_h_in) + 2 * band_h_in + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Use GridSpec with header bands above each section, plus rows for games.
    from matplotlib.gridspec import GridSpec
    height_ratios = (
        [band_h_in]
        + [frame_h_in] * n_id
        + [band_h_in]
        + [frame_h_in] * n_ood
    )
    width_ratios = [label_w_in] + [frame_w_in] * N_FRAMES
    gs = GridSpec(
        nrows=len(height_ratios), ncols=len(width_ratios),
        height_ratios=height_ratios, width_ratios=width_ratios,
        left=0.02, right=0.98, top=0.97, bottom=0.03,
        hspace=0.18, wspace=0.08,
    )

    id_color = "#2A6FB7"   # blue
    ood_color = "#C0463E"  # red

    # ID banner row
    ax_id_band = fig.add_subplot(gs[0, :])
    ax_id_band.axis("off")
    ax_id_band.text(
        0.5, 0.5,
        "In-distribution training games  "
        r"($s_t \rightarrow$ NCA-WM $\rightarrow \hat s_{t+1}$;  engine $s_{t+1}$ supervises the loss)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=id_color, transform=ax_id_band.transAxes,
    )

    # ID rows
    for i, (name, level, label) in enumerate(ID_GAMES):
        ax_lbl = fig.add_subplot(gs[1 + i, 0])
        ax_lbl.axis("off")
        ax_lbl.text(
            1.0, 0.5, label, ha="right", va="center",
            fontsize=8, color="black",
            transform=ax_lbl.transAxes,
        )
        for j in range(N_FRAMES):
            ax = fig.add_subplot(gs[1 + i, 1 + j])
            _add_frame(ax, id_rollouts[i][j], edge_color=id_color)
            if i == 0:
                ax.set_title(
                    f"$s_{{{j}}}$",
                    fontsize=9, color=id_color, pad=2,
                )

    # OOD banner row
    ax_ood_band = fig.add_subplot(gs[1 + n_id, :])
    ax_ood_band.axis("off")
    ax_ood_band.text(
        0.5, 0.5,
        "Held-out (OOD) games  "
        r"(only $s_0$ fed in;  $\hat s_{t+1}$ rolls forward autoregressively, no supervision)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=ood_color, transform=ax_ood_band.transAxes,
    )

    # OOD rows
    for i, (name, level, label) in enumerate(OOD_GAMES):
        ax_lbl = fig.add_subplot(gs[2 + n_id + i, 0])
        ax_lbl.axis("off")
        ax_lbl.text(
            1.0, 0.5, label, ha="right", va="center",
            fontsize=8, color="black",
            transform=ax_lbl.transAxes,
        )
        for j in range(N_FRAMES):
            ax = fig.add_subplot(gs[2 + n_id + i, 1 + j])
            _add_frame(ax, ood_rollouts[i][j], edge_color=ood_color)
            if i == 0:
                title = f"$s_{{{j}}}$" if j == 0 else r"$\hat s_{" + str(j) + "}$"
                ax.set_title(title, fontsize=9, color=ood_color, pad=2)

    # Dashed horizontal divider centered between the two banded sections.
    # Compute y in figure-fraction at the boundary between the last ID row
    # and the OOD banner.
    # We can grab the bbox from the last ID row and first OOD banner row.
    fig.canvas.draw()
    last_id_ax = fig.axes[1 + N_FRAMES * 1 + 1 + 1]  # rough; use bbox from ax_ood_band
    bbox_id_band = ax_id_band.get_position()
    bbox_ood_band = ax_ood_band.get_position()
    y_div = (bbox_id_band.y0 + bbox_ood_band.y1) / 2.0  # midpoint... wait
    # ax_id_band is the TOP banner; ax_ood_band is the MIDDLE banner that
    # separates the two halves. Place the dashed line just above ax_ood_band.
    y_div = bbox_ood_band.y1 + 0.005

    line = Line2D(
        [0.04, 0.96], [y_div, y_div],
        transform=fig.transFigure,
        color="black", linestyle=(0, (5, 4)), linewidth=1.0,
    )
    fig.add_artist(line)

    pdf_path = OUT_DIR / "teaser.pdf"
    png_path = OUT_DIR / "teaser.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {pdf_path}")
    print(f"  wrote {png_path}")


if __name__ == "__main__":
    plt.rcParams.update({
        "text.usetex": False,  # rendered text uses mathtext only
        "font.family": "serif",
    })
    build_figure()
