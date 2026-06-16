"""Combined architecture diagram with evolution progress inset.

Places the evolution rollout figure above the architecture flowchart,
with zoom-style connector lines from the Evolution node to the initial
levels column and from the Search node to the bottom solution trace row.

Usage:
    python plot_architecture_evo.py --run_dir data/evolved_levels_cpp/Atlas_Shrank/<run>
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Reuse data-loading from plot_evolution
from plot_evolution import (
    _get_champion_improvement_gens,
    _select_evenly_spaced,
    _composite_ghost_trail,
    replay_champion_frames,
)
import json


# ============================================================================
# Architecture diagram drawing (adapted from plot_architecture.py)
# ============================================================================

# Colors
C_CPP = '#4A90D9'
C_JAX = '#D94A4A'
C_JS = '#E8A838'
C_RL = '#3DAA5C'
C_SEARCH = '#8B5CF6'
C_EVO = '#D4A017'
C_LLM = '#2CAAA8'
C_EXIT = '#D95BA3'
C_DATA = '#E07020'
C_BG = '#EEEEEE'
C_ALGO = '#E0E0E0'

# Shared node dimensions
NODE_W = 1.2
NODE_H = 0.55
NODE_GAP = 0.3
OUTER_PAD = 0.18
ROW_GAP = 0.4
FONT_LABEL = 12
FONT_SUB = FONT_LABEL
FONT_ROW = 15
RIGHT_MARGIN = 10.3
CONTENT_W = 12.5
OUTER_H = NODE_H + 2 * OUTER_PAD


def rounded_box(ax, x, y, w, h, color, label, sublabel=None,
                linewidth=1.5, fontsize=FONT_LABEL, alpha=0.15, text_color=None):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=(*plt.cm.colors.to_rgb(color), alpha),
                         edgecolor=color, linewidth=linewidth)
    ax.add_patch(box)
    tc = text_color or 'black'
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.12, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=tc, family='sans-serif')
        ax.text(x + w/2, y + h/2 - 0.14, sublabel,
                ha='center', va='center', fontsize=FONT_SUB,
                color='#555555', family='sans-serif')
    else:
        ax.text(x + w/2, y + h/2, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=tc, family='sans-serif')


def curved_arrow(ax, x0, y0, x1, y1, color='#888', lw=1.2, rad=0.15):
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle='->,head_width=3,head_length=2.5',
                        connectionstyle=f'arc3,rad={rad}',
                        color=color, lw=lw, mutation_scale=1)
    ax.add_patch(a)


def straight_arrow(ax, x0, y0, x1, y1, color='#888', lw=1.2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->,head_width=0.12,head_length=0.1',
                                color=color, lw=lw))


def draw_architecture(ax):
    """Draw the architecture diagram into *ax* and return node positions.

    Returns dict with keys 'evo_top_center', 'search_top_center' giving
    (x, y) in data coordinates for the top-centre of those nodes.
    """
    ax.set_xlim(-0.2, 14.5)
    ax.set_ylim(3.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # ===== ALGORITHM ROW =====
    total_5 = 5 * NODE_W + 4 * NODE_GAP
    margin_5 = (CONTENT_W - total_5) / 2
    algo_xs = [margin_5 + i * (NODE_W + NODE_GAP) for i in range(5)]
    content_cx = CONTENT_W / 2

    algo_y = 7.8
    algo_outer_x = algo_xs[0] - OUTER_PAD
    algo_outer_w = total_5 + 2 * OUTER_PAD

    algo_bg = FancyBboxPatch((algo_outer_x, algo_y - OUTER_PAD),
                              algo_outer_w, OUTER_H,
                              boxstyle="round,pad=0.12",
                              facecolor='none', edgecolor='none', linewidth=0)
    ax.add_patch(algo_bg)

    consumers = [
        (algo_xs[0], algo_y, C_RL,     'RL', 'PPO'),
        (algo_xs[1], algo_y, C_RL,     'Search',      'A* \u00b7 MCTS'),
        (algo_xs[2], algo_y, C_RL,     'ExIt',        'BC \u00b7 Q*'),
        (algo_xs[3], algo_y, C_RL,     'LLMs',        'vLLM, APIs'),
        (algo_xs[4], algo_y, C_SEARCH, 'Evolution',   '(1+\u03bb) ES'),
    ]
    for x, y, c, label, sub in consumers:
        rounded_box(ax, x, y, NODE_W, NODE_H, c, label, sub)

    # JIT dashed boxes
    jit_items = [(0, 'purejaxrl'), (1, 'JAXtar'), (2, 'JAXtar'), (4, None)]
    for i, label in jit_items:
        x0 = algo_xs[i] - 0.08
        jit_box = FancyBboxPatch((x0, algo_y - 0.08), NODE_W + 0.16, NODE_H + 0.16,
                                  boxstyle="round,pad=0.05",
                                  facecolor='none',
                                  edgecolor=C_JAX, linewidth=1.3,
                                  linestyle=(0, (5, 3)))
        ax.add_patch(jit_box)
        if label is not None:
            cx = algo_xs[i] + NODE_W / 2
            ax.text(cx, algo_y + NODE_H + 0.3, label, ha='center', va='center',
                    fontsize=FONT_LABEL, color=C_JAX, family='monospace', style='italic',
                    fontweight='bold')

    algo_bot = algo_y - OUTER_PAD

    ax.text(RIGHT_MARGIN, algo_y + NODE_H / 2, 'Players and\nGenerators',
            ha='left', va='center', fontsize=FONT_ROW, style='italic',
            fontweight='bold', color='black', family='sans-serif')

    # ===== BACKENDS =====
    be_y = algo_bot - ROW_GAP - NODE_H
    total_3 = 3 * NODE_W + 2 * NODE_GAP
    margin_3 = (CONTENT_W - total_3) / 2
    be_xs = [margin_3 + i * (NODE_W + NODE_GAP) for i in range(3)]

    bg = FancyBboxPatch((be_xs[0] - OUTER_PAD, be_y - OUTER_PAD),
                         total_3 + 2 * OUTER_PAD, OUTER_H,
                         boxstyle="round,pad=0.12",
                         facecolor='none', edgecolor='none', linewidth=0)
    ax.add_patch(bg)

    backends = [
        (be_xs[0], be_y, C_JAX, 'JAX',  None),
        (be_xs[1], be_y, C_CPP, 'C++',  None),
        (be_xs[2], be_y, C_JS,  'JS',   None),
    ]
    for x, y, c, label, sub in backends:
        rounded_box(ax, x, y, NODE_W, NODE_H, c, label, sub)

    jax_cx = be_xs[0] + NODE_W / 2
    cpp_cx = be_xs[1] + NODE_W / 2
    js_cx = be_xs[2] + NODE_W / 2
    be_top = be_y + NODE_H
    be_bot = be_y

    # PuzzleScript mascot
    try:
        mascot_img = mpimg.imread('PuzzleScript/src/images/mascot_128.png')
        mascot_box = OffsetImage(mascot_img, zoom=0.18)
        mascot_ab = AnnotationBbox(mascot_box, (be_xs[2] + NODE_W - 0.10, be_y + NODE_H / 2),
                                   frameon=False, box_alignment=(1.0, 0.5))
        ax.add_artist(mascot_ab)
    except FileNotFoundError:
        pass

    ax.text(RIGHT_MARGIN, be_y + NODE_H / 2, 'PuzzleScript\nBackends',
            ha='left', va='center', fontsize=FONT_ROW, style='italic',
            fontweight='bold', color='black', family='sans-serif')

    straight_arrow(ax, content_cx, algo_bot, content_cx, be_top + OUTER_PAD, 'black', lw=2.5)

    # ===== VALIDATION =====
    val_y = be_bot - OUTER_PAD - ROW_GAP - NODE_H
    total_2 = 2 * NODE_W + NODE_GAP
    margin_2 = (CONTENT_W - total_2) / 2
    val_xs = [margin_2 + i * (NODE_W + NODE_GAP) for i in range(2)]

    val_bg = FancyBboxPatch((val_xs[0] - OUTER_PAD, val_y - OUTER_PAD),
                             total_2 + 2 * OUTER_PAD, OUTER_H,
                             boxstyle="round,pad=0.12",
                             facecolor='none', edgecolor='none', linewidth=0)
    ax.add_patch(val_bg)

    vjax_cx = val_xs[0] + NODE_W / 2
    vcpp_cx = val_xs[1] + NODE_W / 2

    rounded_box(ax, val_xs[0], val_y, NODE_W, NODE_H, C_JAX,
                'JAX\nValidation', None, fontsize=FONT_LABEL, alpha=0.1)
    rounded_box(ax, val_xs[1], val_y, NODE_W, NODE_H, C_CPP,
                'C++\nValidation', None, fontsize=FONT_LABEL, alpha=0.1)

    val_top = val_y + NODE_H
    val_bot = val_y

    curved_arrow(ax, vjax_cx, val_top, jax_cx + 0.2, be_bot, C_JAX, lw=1.5, rad=-0.15)
    curved_arrow(ax, vcpp_cx, val_top, cpp_cx + 0.2, be_bot, C_CPP, lw=1.5, rad=-0.15)
    curved_arrow(ax, js_cx - 0.2, be_bot, vjax_cx + 0.6, val_top, C_JS, lw=1.5, rad=-0.25)
    ax.text((js_cx + vcpp_cx) / 2 + 0.38, (be_bot + val_top) / 2 - 0.1, 'Solutions',
            fontsize=FONT_LABEL, color=C_JS, family='sans-serif', style='italic', fontweight='bold')
    curved_arrow(ax, js_cx, be_bot, vcpp_cx + 0.6, val_top, C_JS, lw=1.5, rad=-0.15)

    ax.text(RIGHT_MARGIN, val_y + NODE_H / 2, 'Validation\nHarness',
            ha='left', va='center', fontsize=FONT_ROW, style='italic',
            fontweight='bold', color='black', family='sans-serif')

    # ===== DATASETS =====
    ds_base_x = algo_outer_x
    ds_base_y = val_bot - OUTER_PAD - ROW_GAP - NODE_H
    ds_gists_w = algo_outer_w
    ds_gists_h = NODE_H

    ds_info = [
        ('gists',    1.0,  0.15),
        ("Archive",  0.72, 0.10),
        ('Gallery',  0.48, 0.10),
        ('Priority', 0.28, 0.05),
    ]

    ds_top = ds_base_y + ds_gists_h
    for name, frac, alpha in ds_info:
        w = ds_gists_w * frac
        h = ds_gists_h * frac
        box = FancyBboxPatch((ds_base_x, ds_base_y), w, h,
                             boxstyle="round,pad=0.08",
                             facecolor=(*plt.cm.colors.to_rgb(C_DATA), alpha),
                             edgecolor=C_DATA, linewidth=1.2)
        ax.add_patch(box)
        label_x = ds_base_x + w - 0.15 if name != 'gists' else ds_base_x + w - 0.6
        ax.text(label_x, ds_base_y + h / 2, name,
                ha='right', va='center', fontsize=11,
                fontweight='bold', color='black', family='sans-serif')

    try:
        github_img = mpimg.imread('paper/assets/github_mark_transparent.png')
        github_box = OffsetImage(github_img, zoom=0.055)
        gists_label_x = ds_base_x + ds_gists_w - 0.8
        gists_label_y = ds_base_y + ds_gists_h / 2
        github_ab = AnnotationBbox(github_box, (gists_label_x + 0.35, gists_label_y),
                                   frameon=False, box_alignment=(0.0, 0.5))
        ax.add_artist(github_ab)
    except FileNotFoundError:
        pass

    ds_right = ds_base_x + ds_gists_w
    curved_arrow(ax, ds_right, ds_top, be_xs[2] + NODE_W, be_bot, C_DATA, lw=1.5, rad=0.15)
    ax.text(ds_right - 0.75, (ds_top + be_bot) / 2 + 0.2, 'Games',
            fontsize=FONT_LABEL, color=C_DATA, family='sans-serif', style='italic', fontweight='bold')

    ax.text(RIGHT_MARGIN, ds_base_y + ds_gists_h / 2, 'Datasets',
            ha='left', va='center', fontsize=FONT_ROW, style='italic',
            fontweight='bold', color='black', family='sans-serif')

    # ===== LEGEND (top-left, above purejaxrl) =====
    ly = algo_y + NODE_H + 0.7
    lx = algo_xs[0] - 0.08
    ax.plot([lx, lx + 0.6], [ly, ly], color=C_JAX, lw=1.3, ls=(0, (5, 3)))
    ax.text(lx + 0.75, ly, 'JIT outer loop (JAX)', fontsize=FONT_LABEL, va='center',
            color='#666', family='sans-serif', fontweight='bold')

    # Return anchor positions for zoom lines
    evo_top = (algo_xs[4] + NODE_W / 2, algo_y + NODE_H)
    search_top = (algo_xs[1] + NODE_W / 2, algo_y + NODE_H)
    return {
        'evo_top_center': evo_top,
        'search_top_center': search_top,
    }


# ============================================================================
# Load evolution data (reused from plot_evolution)
# ============================================================================

def load_evolution_rows(run_dir, backend, n_rows, n_cols, scale):
    """Load and prepare evolution row data. Returns (rows_data, game, level_i)."""
    result_files = [f for f in os.listdir(run_dir) if f.endswith("_result.json")]
    if not result_files:
        raise FileNotFoundError(f"No result JSON found in {run_dir}")
    with open(os.path.join(run_dir, result_files[0])) as f:
        result_data = json.load(f)

    game = result_data["game"]
    level_i = result_data["level_i"]
    history = result_data["history"]

    improvement_gens = _get_champion_improvement_gens(history)
    best_at_gen = {}
    best_fitness = -float("inf")
    for entry in history:
        if entry["fitness"] > best_fitness:
            best_fitness = entry["fitness"]
            best_at_gen[entry["gen"]] = entry

    selected_gens, _ = _select_evenly_spaced(improvement_gens, n_rows)
    print(f"Selected generations: {selected_gens}")

    rows_data = []
    for gen in selected_gens:
        entry = best_at_gen[gen]
        actions = entry.get("actions", [])
        fitness = entry.get("fitness", 0)
        cost = entry.get("cost", "?")
        if not actions:
            continue

        frames = replay_champion_frames(
            game, level_i, run_dir, gen, actions, backend, scale=scale)
        if not frames:
            continue

        init_frame = frames[0]
        trace_frames_all = frames[1:]
        if trace_frames_all:
            # Evenly space n_cols+1 timesteps across [0, len(frames)-1],
            # then drop the first (t=0, shown as init column).
            total_steps = len(frames) - 1  # last timestep index
            all_timesteps = [int(round(i * total_steps / n_cols))
                             for i in range(n_cols + 1)]
            # Drop t=0 (already the init column)
            trace_timesteps = all_timesteps[1:]
            # trace_timesteps are 1-indexed into frames; convert to
            # 0-indexed into trace_frames_all
            trace_indices = [t - 1 for t in trace_timesteps]
            trace_frames = [
                _composite_ghost_trail(trace_frames_all, idx, n_ghosts=3)
                for idx in trace_indices
            ]
        else:
            trace_frames, trace_timesteps = [], []

        rows_data.append((gen, fitness, cost, init_frame,
                          trace_frames, trace_timesteps))
        print(f"  Gen {gen}: fitness={fitness}, cost={cost}, "
              f"{len(frames)} total -> 1 init + {len(trace_frames)} trace, "
              f"timesteps=[0] + {trace_timesteps}")

    return rows_data, game, level_i


# ============================================================================
# Combined figure
# ============================================================================

def plot_combined(run_dir, backend="cpp", n_rows=2, n_cols=3,
                  scale=10, out_path=None, dpi=200):
    rows_data, game, level_i = load_evolution_rows(
        run_dir, backend, n_rows, n_cols, scale)
    if not rows_data:
        print("No evolution data to plot.")
        return

    actual_rows = len(rows_data)
    trace_cols = max(len(rd[4]) for rd in rows_data)
    total_evo_cols = 1 + trace_cols

    # Figure sizing — evo grid must not exceed architecture width
    fig_w = 14.0
    arch_h = 5.5  # architecture diagram portion
    gap_h = 0.4   # gap between evo and arch for zoom lines

    # Fit evo cells into ~70% of the available width (with margins)
    evo_content_w = fig_w * (0.95 - 0.07) * 0.67
    sample = rows_data[0][3]
    fh, fw = sample.shape[:2]
    aspect = fw / fh
    cell_w_in = evo_content_w / total_evo_cols
    cell_h_in = cell_w_in / aspect

    evo_h = cell_h_in * actual_rows + 0.6
    fig_h = evo_h + gap_h + arch_h

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    # Heights as fractions of figure
    evo_frac = evo_h / fig_h
    gap_frac = gap_h / fig_h
    arch_frac = arch_h / fig_h

    # --- Top: evolution grid ---
    # Use a sub-gridspec for evolution frames within the top portion
    # Align evo grid left edge with architecture content left edge
    evo_left = 0.20
    evo_right = evo_left + evo_content_w / fig_w
    gs_top = fig.add_gridspec(
        actual_rows, total_evo_cols,
        left=evo_left, right=evo_right,
        top=1.0 - 0.02, bottom=1.0 - evo_frac + 0.02,
        wspace=0.05, hspace=0.18,
        width_ratios=[1] * total_evo_cols,
    )

    # Track axes positions for zoom lines
    evo_axes = {}  # (row, col) -> axes

    for row_i, (gen, fitness, cost, init_frame,
                trace_frames, trace_timesteps) in enumerate(rows_data):

        # Init column
        ax_init = fig.add_subplot(gs_top[row_i, 0], zorder=5)
        ax_init.set_facecolor('white')
        frame = init_frame
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        ax_init.imshow(frame, interpolation="nearest", aspect="equal")
        ax_init.set_xticks([])
        ax_init.set_yticks([])
        for spine in ax_init.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(1.5)
        evo_axes[(row_i, 0)] = ax_init

        # Trace columns
        for col_i in range(trace_cols):
            ax = fig.add_subplot(gs_top[row_i, 1 + col_i], zorder=5)
            ax.set_facecolor('white')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_i < len(trace_frames):
                frame = trace_frames[col_i]
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                ax.imshow(frame, interpolation="nearest", aspect="equal")
            else:
                ax.axis("off")
            evo_axes[(row_i, 1 + col_i)] = ax


    # --- Bottom: architecture diagram ---
    ax_arch = fig.add_axes([0.0, 0.0, 1.0, arch_frac])
    positions = draw_architecture(ax_arch)

    # --- Zoom connectors ---
    # We need rendered positions, so force a draw first.
    fig.canvas.draw()

    evo_top_data = positions['evo_top_center']
    search_top_data = positions['search_top_center']

    zoom_color_evo = C_SEARCH   # match Evolution node color
    zoom_color_search = C_RL    # match Search node color
    zoom_pad = 0.006  # padding around highlight boxes (figure fraction)

    # Helper: get bounding box in figure-fraction coords for a set of axes
    def _axes_group_bbox_fig(axes_list):
        """Return (x0, y0, x1, y1) in figure-fraction coords."""
        x0s, y0s, x1s, y1s = [], [], [], []
        for a in axes_list:
            bb = a.get_position()  # Bbox in figure fraction
            x0s.append(bb.x0)
            y0s.append(bb.y0)
            x1s.append(bb.x1)
            y1s.append(bb.y1)
        return min(x0s), min(y0s), max(x1s), max(y1s)

    # Background overlay axes (behind frame axes at zorder=5)
    ax_bg = fig.add_axes([0, 0, 1, 1], facecolor='none', zorder=1)
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    ax_bg.axis('off')

    # ---- Evolution → init levels column: filled translucent background ----
    init_axes = [evo_axes[(r, 0)] for r in range(actual_rows)]
    ix0, iy0, ix1, iy1 = _axes_group_bbox_fig(init_axes)
    init_inset_r = 0.011  # pull right edge inward
    init_rect = FancyBboxPatch(
        (ix0 - zoom_pad +0.005, iy0 - zoom_pad),
        (ix1 - ix0) + 2 * zoom_pad - init_inset_r - 0.005,
        (iy1 - iy0) + 2 * zoom_pad,
        boxstyle="round,pad=0.008",
        facecolor=(*plt.cm.colors.to_rgb(zoom_color_evo), 0.12),
        edgecolor=zoom_color_evo, linewidth=1.2,
        transform=ax_bg.transAxes,
    )
    ax_bg.add_patch(init_rect)

    # Single line: Evolution node top → bottom-center of init highlight
    init_anchor = ((ix0 + ix1) / 2, iy0 - zoom_pad)
    conn_evo = ConnectionPatch(
        xyA=evo_top_data, coordsA=ax_arch.transData,
        xyB=init_anchor, coordsB=fig.transFigure,
        color=zoom_color_evo, lw=1.5, ls=(0, (5, 3)),
        alpha=0.7, zorder=5,
    )
    fig.add_artist(conn_evo)

    # ---- Search → bottom trace row: filled translucent background ----
    last_row = actual_rows - 1
    trace_axes = [evo_axes[(last_row, c)] for c in range(1, total_evo_cols)]
    tx0, ty0, tx1, ty1 = _axes_group_bbox_fig(trace_axes)
    trace_rect = FancyBboxPatch(
        (tx0 - zoom_pad + 0.012, ty0 - zoom_pad),
        (tx1 - tx0) + 2 * zoom_pad - 0.02,
        (ty1 - ty0) + 2 * zoom_pad - 0.005,
        boxstyle="round,pad=0.008",
        facecolor=(*plt.cm.colors.to_rgb(zoom_color_search), 0.12),
        edgecolor=zoom_color_search, linewidth=1.2,
        transform=ax_bg.transAxes,
    )
    ax_bg.add_patch(trace_rect)

    # Single line: Search node top → bottom-center of trace highlight
    trace_anchor = ((tx0 + tx1) / 2, ty0 - zoom_pad)
    conn_search = ConnectionPatch(
        xyA=search_top_data, coordsA=ax_arch.transData,
        xyB=trace_anchor, coordsB=fig.transFigure,
        color=zoom_color_search, lw=1.5, ls=(0, (5, 3)),
        alpha=0.7, zorder=5,
    )
    fig.add_artist(conn_search)

    # ---- Directional arrows ----
    arrow_color = '#555555'

    # Vertical arrow: left of init column, downward (0.7 length, centered)
    arrow_x = ix0 - zoom_pad - 0.010
    v_full_top = iy1 + zoom_pad
    v_full_bot = iy0 - zoom_pad
    v_mid = (v_full_top + v_full_bot) / 2
    v_half = (v_full_top - v_full_bot) * 0.7 / 2
    ax_bg.annotate('', xy=(arrow_x, v_mid - v_half),
                   xytext=(arrow_x, v_mid + v_half),
                   arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.15',
                                   color=arrow_color, lw=1.5),
                   transform=ax_bg.transAxes)

    # Horizontal arrow: below last trace row, rightward (0.7 length, centered)
    arrow_y = ty0 - zoom_pad - 0.015
    h_full_left = tx0 - zoom_pad
    h_full_right = tx1 + zoom_pad
    h_mid = (h_full_left + h_full_right) / 2
    h_half = (h_full_right - h_full_left) * 0.7 / 2
    ax_bg.annotate('', xy=(h_mid + h_half, arrow_y),
                   xytext=(h_mid - h_half, arrow_y),
                   arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.15',
                                   color=arrow_color, lw=1.5),
                   transform=ax_bg.transAxes)

    # Save
    if out_path is None:
        out_path = "paper/codebase_architecture_evo.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight',
                pad_inches=0.05, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined architecture + evolution progress figure.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--backend", default=None, choices=["cpp", "nodejs", "jax"])
    parser.add_argument("--n_rows", type=int, default=2)
    parser.add_argument("--n_cols", type=int, default=3)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    backend = args.backend
    if backend is None:
        if "evolved_levels_cpp" in args.run_dir:
            backend = "cpp"
        elif "evolved_levels_nodejs" in args.run_dir:
            backend = "nodejs"
        else:
            backend = "jax"

    plot_combined(
        run_dir=args.run_dir,
        backend=backend,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        scale=args.scale,
        out_path=args.out,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
