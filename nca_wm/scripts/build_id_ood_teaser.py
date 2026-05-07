"""Build the ID/OOD teaser figure for the paper.

Top half (in-distribution training): for each of N_ID games, the engine's
$s_0$ enters a shared NCA WM block, the model outputs $\hat s_1$, and the
engine's true $s_1$ supervises the loss.

(Vertical "training updates" arrow descends across a dashed horizontal
border that separates training from evaluation.)

Bottom half (out-of-distribution evaluation): for each of N_OOD held-out
games, only $s_0$ enters the *trained* NCA WM, which then predicts a
short autoregressive rollout $\hat s_1, \hat s_2, \hat s_3$. Those frames
come from `dump_ood_rollout_frames.py` so the slight prediction errors
are real, not staged.

Output: nca_wm/paper/figures/id_ood_teaser/teaser.{pdf,png}
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
import imageio.v2 as imageio

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_jax.utils import init_ps_lark_parser  # noqa: E402
from puzzlescript_cpp import CppPuzzleScriptBackend  # noqa: E402

# (game name, level, display label, BFS-action sub-list).
ID_GAMES = [
    ("nekopuzzle", 0, "nekopuzzle"),
    ("sokoban_basic", 0, "sokoban_basic"),
    ("Modality", 0, "Modality"),
]
OOD_GAMES = [
    ("in_the_way_by_giovanni_mota", 0, "in_the_way"),
    ("my_nana_ate_my_kid_brother_by_ruth_williams", 0, "my_nana"),
    ("Hamiltwo", 0, "Hamiltwo"),
]
N_OOD_FRAMES = 4   # s_0, ŝ_1, ŝ_2, ŝ_3
PAPER_DIR = _REPO_ROOT / "nca_wm" / "paper"
OUT_DIR = PAPER_DIR / "figures" / "id_ood_teaser"
ROLLOUT_DIR = OUT_DIR / "rollouts"
NCA_MINI_PNG = PAPER_DIR / "figures" / "architecture" / "nca_wm_mini.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)


_PARSER = None
def _parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = init_ps_lark_parser()
    return _PARSER


def _bfs_cache_path(name: str, level: int) -> Path | None:
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
                  n_steps: int = 4) -> list[int]:
    cache = _bfs_cache_path(name, level)
    if cache is not None:
        d = np.load(cache, allow_pickle=True)
        if "actions" in d.files and len(d["actions"]) > 0:
            return [int(a) for a in d["actions"]]
    rng = np.random.default_rng(fallback_seed)
    return [int(rng.integers(5)) for _ in range(n_steps * 3)]


def _render_engine_pair(name: str, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (s_0, s_1) frames from the engine, where s_1 is one BFS step."""
    backend = CppPuzzleScriptBackend()
    backend.compile_game(_parser(), name)
    backend.cpp_engine.load_level(level)
    f0 = backend.render_frame()
    actions = _load_actions(name, level, n_steps=1)
    a = actions[0]
    backend.cpp_engine.process_input(int(a))
    again = 0
    while backend.cpp_engine.againing and again < 8:
        backend.cpp_engine.process_input(-1)
        again += 1
    f1 = backend.render_frame()
    return f0, f1


def _load_ood_rollout(name: str, n_frames: int) -> tuple[list[np.ndarray], list[dict]]:
    """Read precomputed predicted frames + per-step diff metadata."""
    d = ROLLOUT_DIR / name
    if not d.exists():
        raise FileNotFoundError(
            f"Missing OOD rollout dir {d}. Run "
            f"`nca_wm.scripts.dump_ood_rollout_frames` first."
        )
    import json as _json
    frames = []
    for t in range(n_frames):
        p = d / f"pred_t{t}.png"
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(imageio.imread(p))
    meta = _json.loads((d / "diffs.json").read_text())
    return frames, meta["diffs"][:n_frames]


def _show_image(ax, img, edge_color, edge_lw=0.8):
    """Project the PuzzleScript frame onto the front face of a 3D box,
    matching the embed/readout look in nca_wm_mini.tex (same isometric
    offset, light-fill side/top faces).
    """
    H, W = img.shape[:2]
    # Isometric depth offset for the slanted top/right faces.
    dx = max(2, int(round(0.18 * W)))
    dy = max(2, int(round(0.18 * H)))

    # y-up coordinates. Front face occupies x=[0,W], y=[0,H]; the top
    # face is above (y > H) and the right face is to the right (x > W),
    # both receding in the (+dx, +dy) direction.
    ax.set_xlim(0, W + dx)
    ax.set_ylim(0, H + dy)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Front-face image. extent=(left, right, bottom, top); origin='upper'
    # places image[0,0] at the top of the extent, which is what we want.
    ax.imshow(img, interpolation="nearest",
              extent=(0, W, 0, H), origin="upper", zorder=2)

    # Top face (parallelogram receding to upper-right).
    top = mpatches.Polygon(
        [(0, H), (W, H), (W + dx, H + dy), (dx, H + dy)],
        closed=True, facecolor="#F2F2F2", edgecolor=edge_color,
        linewidth=edge_lw, joinstyle="miter", zorder=1,
    )
    # Right face (parallelogram receding to upper-right from the right edge).
    right = mpatches.Polygon(
        [(W, 0), (W + dx, dy), (W + dx, H + dy), (W, H)],
        closed=True, facecolor="#E5E5E5", edgecolor=edge_color,
        linewidth=edge_lw, joinstyle="miter", zorder=1,
    )
    ax.add_patch(top)
    ax.add_patch(right)
    # Crisp front-face outline drawn last.
    front = mpatches.Rectangle(
        (0, 0), W, H, fill=False, edgecolor=edge_color,
        linewidth=edge_lw, zorder=3,
    )
    ax.add_patch(front)


def _arrow(ax, x0, y0, x1, y1, *, color="black", lw=1.0, mutation=12,
           style="-|>"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        mutation_scale=mutation),
    )


def build_figure():
    n_id = len(ID_GAMES)
    n_ood = len(OOD_GAMES)

    print("Rendering ID engine s_0/s_1 frames...")
    id_frames = []  # list[(s0, s1)]
    for name, level, _ in ID_GAMES:
        f0, f1 = _render_engine_pair(name, level)
        id_frames.append((f0, f1))

    print("Loading precomputed OOD predicted rollout frames...")
    ood_frames = []  # list[list[frame]] (s0, ŝ1, ŝ2, ŝ3)
    ood_diffs = []   # list[list[{wrong_cells,total_cells,t}]]
    for name, level, _ in OOD_GAMES:
        f, d = _load_ood_rollout(name, N_OOD_FRAMES)
        ood_frames.append(f)
        ood_diffs.append(d)

    # ----- Layout (figure-fraction coords) -----
    # Two section heights; each contains 3 game rows; central NCA WM column
    # spans the section vertically.
    fig_w = 8.5
    fig_h = 6.4
    fig = plt.figure(figsize=(fig_w, fig_h))

    id_color = "#2A6FB7"   # blue (forward, ID inputs)
    ood_color = "#C0463E"  # red (forward + AR feedback, OOD)
    nca_color = "#E5B800"  # gold (legacy; mini diagram now used in its place)
    loss_color = "#D2691E" # dark orange (loss feedback to NCA output)
    nca_mini_img = imageio.imread(NCA_MINI_PNG)

    # Section vertical extents (figure fraction).
    title_h = 0.04
    section_pad = 0.02
    band_h = 0.03
    id_top = 0.97
    id_bot = 0.58
    ood_top = 0.46
    ood_bot = 0.04
    div_y = 0.52  # dashed border y
    label_x = 0.05  # left edge of name column
    s0_x = 0.18    # center x of s_0 column
    nca_x = 0.40   # center x of NCA WM block
    out_x_id = 0.62  # center x of single ŝ_1 column (ID)
    loss_x = 0.83

    frame_w = 0.085
    frame_h_id = 0.10
    frame_h_ood = 0.10

    # ---- Banners ----
    fig.text(
        0.5, id_top + 0.005,
        "Training on in-distribution PuzzleScript games",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=id_color,
    )
    fig.text(
        0.5, ood_top + 0.005,
        "Evaluation on held-out (OOD) games",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=ood_color,
    )
    # Small italic note about training transition, between the dashed line
    # and the OOD banner so it doesn't strike through anything.
    fig.text(
        0.99, div_y - 0.005,
        r"trained $\theta \rightarrow \theta^{*}$",
        ha="right", va="top", fontsize=8, style="italic",
        color="#444444",
    )

    # ---- Dashed border ----
    line = Line2D(
        [0.02, 0.98], [div_y, div_y],
        transform=fig.transFigure,
        color="black", linestyle=(0, (5, 4)), linewidth=1.0,
    )
    fig.add_artist(line)

    # ---- ID section ----
    id_rows_top = id_top - title_h
    id_rows_bot = id_bot + section_pad
    id_row_h = (id_rows_top - id_rows_bot) / n_id
    id_centers = [
        id_rows_top - (i + 0.5) * id_row_h for i in range(n_id)
    ]

    # NCA WM mini diagram for ID: a TikZ-rendered boxes-and-edges sketch
    # of the dynamics block (embed -> NCA body -> readout, with a slot
    # column conditioning above and an iteration loop below). Sized to
    # span the section vertically.
    nca_w = 0.18
    nca_h = (n_id) * id_row_h * 0.95
    nca_y_center = (id_rows_top + id_rows_bot) / 2
    ax_nca_id = fig.add_axes(
        [nca_x - nca_w/2, nca_y_center - nca_h/2, nca_w, nca_h],
    )
    ax_nca_id.imshow(nca_mini_img, interpolation="bilinear",
                     aspect="auto")
    ax_nca_id.set_xticks([]); ax_nca_id.set_yticks([])
    for s in ax_nca_id.spines.values():
        s.set_visible(False)
    fig.text(nca_x, nca_y_center - nca_h/2 - 0.005,
             r"NCA-WM ($\theta$)",
             ha="center", va="top", fontsize=9, fontweight="bold",
             color="black")

    # Per-row ID content.
    for i, ((name, level, label), (f0, f1)) in enumerate(zip(ID_GAMES, id_frames)):
        cy = id_centers[i]
        # Name label
        fig.text(label_x, cy, label, ha="left", va="center",
                 fontsize=8, color="black")
        # s_0 frame
        ax_s0 = fig.add_axes([s0_x - frame_w/2, cy - frame_h_id/2,
                              frame_w, frame_h_id])
        _show_image(ax_s0, f0, edge_color=id_color)
        if i == 0:
            ax_s0.set_title("$s_0$", fontsize=9, color=id_color, pad=2)
        # ŝ_1 frame (here we display the engine's s_1 since this is the
        # supervision target — model and engine agree at training optimum).
        ax_s1 = fig.add_axes([out_x_id - frame_w/2, cy - frame_h_id/2,
                              frame_w, frame_h_id])
        _show_image(ax_s1, f1, edge_color=id_color)
        if i == 0:
            ax_s1.set_title("$\\hat s_1\\;\\;(\\!\\approx\\! s_1)$",
                            fontsize=9, color=id_color, pad=2)

        # Forward arrow: s_0 -> NCA box (blue, rightward)
        fig.add_artist(Line2D(
            [s0_x + frame_w/2 + 0.005, nca_x - nca_w/2 - 0.003],
            [cy, cy],
            transform=fig.transFigure,
            color=id_color, lw=1.0,
        ))
        fig.add_artist(plt.matplotlib.lines.Line2D(
            [nca_x - nca_w/2 - 0.003], [cy],
            transform=fig.transFigure, marker=">", color=id_color,
            markersize=5, lw=0,
        ))
        # Loss arrow: ŝ_1 -> NCA output (dark orange, leftward).
        fig.add_artist(Line2D(
            [nca_x + nca_w/2 + 0.003, out_x_id - frame_w/2 - 0.005],
            [cy, cy],
            transform=fig.transFigure,
            color=loss_color, lw=1.2,
        ))
        fig.add_artist(plt.matplotlib.lines.Line2D(
            [nca_x + nca_w/2 + 0.003], [cy],
            transform=fig.transFigure, marker="<", color=loss_color,
            markersize=6, lw=0,
        ))

    # Single "loss" label centered above the dark-orange ŝ_1->NCA arrows.
    cy_mid = (id_centers[0] + id_centers[-1]) / 2
    fig.text(
        (nca_x + nca_w/2 + out_x_id - frame_w/2) / 2,
        id_centers[0] + frame_h_id/2 + 0.012,
        r"loss  $\|\hat s_1 - s_1\|$  $\rightarrow$  back-prop $\nabla_\theta$",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        color=loss_color,
    )

    # ---- OOD section ----
    ood_rows_top = ood_top - title_h * 0.5
    ood_rows_bot = ood_bot + section_pad
    ood_row_h = (ood_rows_top - ood_rows_bot) / n_ood
    ood_centers = [
        ood_rows_top - (i + 0.5) * ood_row_h for i in range(n_ood)
    ]

    nca_h_ood = n_ood * ood_row_h * 0.95
    nca_y_ood = (ood_rows_top + ood_rows_bot) / 2

    ax_nca_ood = fig.add_axes(
        [nca_x - nca_w/2, nca_y_ood - nca_h_ood/2, nca_w, nca_h_ood],
    )
    ax_nca_ood.imshow(nca_mini_img, interpolation="bilinear",
                      aspect="auto")
    ax_nca_ood.set_xticks([]); ax_nca_ood.set_yticks([])
    for s in ax_nca_ood.spines.values():
        s.set_visible(False)
    fig.text(nca_x, nca_y_ood - nca_h_ood/2 - 0.005,
             r"NCA-WM ($\theta^{*}$, trained)",
             ha="center", va="top", fontsize=9, fontweight="bold",
             color="black")

    # OOD predicted-frame x-positions (after NCA box)
    # We have N_OOD_FRAMES=4 frames; place s_0 left of NCA, then ŝ_1..ŝ_3 right.
    pred_xs = np.linspace(nca_x + nca_w/2 + 0.05, 0.93,
                          N_OOD_FRAMES - 1)

    for i, ((name, level, label), frames, diffs) in enumerate(
            zip(OOD_GAMES, ood_frames, ood_diffs)):
        cy = ood_centers[i]
        fig.text(label_x, cy, label, ha="left", va="center",
                 fontsize=8, color="black")

        # s_0 frame on left
        ax_s0 = fig.add_axes([s0_x - frame_w/2, cy - frame_h_ood/2,
                              frame_w, frame_h_ood])
        _show_image(ax_s0, frames[0], edge_color=ood_color)
        if i == 0:
            ax_s0.set_title("$s_0$", fontsize=9, color=ood_color, pad=2)

        # arrow into NCA
        fig.add_artist(Line2D(
            [s0_x + frame_w/2 + 0.005, nca_x - nca_w/2 - 0.003],
            [cy, cy],
            transform=fig.transFigure, color=ood_color, lw=1.0,
        ))
        fig.add_artist(plt.matplotlib.lines.Line2D(
            [nca_x - nca_w/2 - 0.003], [cy],
            transform=fig.transFigure, marker=">", color=ood_color,
            markersize=5, lw=0,
        ))

        # arrow out of NCA to first predicted frame
        fig.add_artist(Line2D(
            [nca_x + nca_w/2 + 0.003, pred_xs[0] - frame_w/2 - 0.003],
            [cy, cy],
            transform=fig.transFigure, color=ood_color, lw=1.0,
        ))
        fig.add_artist(plt.matplotlib.lines.Line2D(
            [pred_xs[0] - frame_w/2 - 0.003], [cy],
            transform=fig.transFigure, marker=">", color=ood_color,
            markersize=5, lw=0,
        ))

        # predicted frames + chained arrows
        for j, x in enumerate(pred_xs):
            ax = fig.add_axes([x - frame_w/2, cy - frame_h_ood/2,
                               frame_w, frame_h_ood])
            _show_image(ax, frames[j + 1], edge_color=ood_color)
            if i == 0:
                ax.set_title(f"$\\hat s_{{{j+1}}}$", fontsize=9,
                             color=ood_color, pad=2)
            # Tiny "k/N cells off engine" caption underneath the frame.
            d = diffs[j + 1]
            fig.text(
                x, cy - frame_h_ood/2 - 0.008,
                f"{d['wrong_cells']}/{d['total_cells']} cells off",
                ha="center", va="top", fontsize=6.5,
                color="#993333",
            )
            if j > 0:
                # arrow from previous predicted frame to this one
                xprev = pred_xs[j - 1]
                fig.add_artist(Line2D(
                    [xprev + frame_w/2 + 0.003, x - frame_w/2 - 0.003],
                    [cy, cy],
                    transform=fig.transFigure, color=ood_color,
                    lw=1.0, linestyle=(0, (2, 2)),
                ))
                fig.add_artist(plt.matplotlib.lines.Line2D(
                    [x - frame_w/2 - 0.003], [cy],
                    transform=fig.transFigure, marker=">",
                    color=ood_color, markersize=5, lw=0,
                ))

    # (Inline subtitles removed; the figure caption carries that info.)

    pdf_path = OUT_DIR / "teaser.pdf"
    png_path = OUT_DIR / "teaser.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {pdf_path}")
    print(f"  wrote {png_path}")


if __name__ == "__main__":
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
    })
    build_figure()
