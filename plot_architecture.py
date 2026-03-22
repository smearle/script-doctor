"""Generate a codebase architecture diagram for Script Doctor."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(-0.2, 14.5)
ax.set_ylim(3.5, 9.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

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
ROW_GAP = 0.4  # gap between outer boxes
FONT_LABEL = 12
FONT_SUB = FONT_LABEL
FONT_ROW = 15
RIGHT_MARGIN = 9.7
CONTENT_W = 12.5


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


# Outer box height is the same for all rows
OUTER_H = NODE_H + 2 * OUTER_PAD

# Legend will be drawn at the bottom after all rows are laid out

# ===== ALGORITHM ROW =====
total_4 = 4 * NODE_W + 3 * NODE_GAP
margin_4 = (CONTENT_W - total_4) / 2
algo_xs = [margin_4 + i * (NODE_W + NODE_GAP) for i in range(4)]
content_cx = CONTENT_W / 2

algo_y = 7.8
algo_outer_x = algo_xs[0] - OUTER_PAD
algo_outer_w = total_4 + 2 * OUTER_PAD

algo_bg = FancyBboxPatch((algo_outer_x, algo_y - OUTER_PAD),
                          algo_outer_w, OUTER_H,
                          boxstyle="round,pad=0.12",
                          facecolor='none', edgecolor='none', linewidth=0)
ax.add_patch(algo_bg)

# New order: RL, Search, ExIt, LLM
consumers = [
    (algo_xs[0], algo_y, C_RL,     'RL', 'PPO'),
    (algo_xs[1], algo_y, C_RL,     'Search',                  'A* · MCTS'),
    (algo_xs[2], algo_y, C_RL,     'ExIt',        'BC · Q*'),
    (algo_xs[3], algo_y, C_RL,     'LLMs',              'vLLM, APIs'),
]
for x, y, c, label, sub in consumers:
    rounded_box(ax, x, y, NODE_W, NODE_H, c, label, sub)

# JIT dashed boxes: RL(0), Search(1), ExIt(2)
jit_items = [
    (0, 'purejaxrl'),
    (1, 'JAXtar'),
    (2, 'JAXtar'),
]
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

ax.text(RIGHT_MARGIN, algo_y + NODE_H / 2, 'Player\nAlgorithms',
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

be_outer_right = be_xs[2] + NODE_W + OUTER_PAD

backends = [
    (be_xs[0], be_y, C_JAX, 'JAX',    None),
    (be_xs[1], be_y, C_CPP, 'C++',    None),
    (be_xs[2], be_y, C_JS,  'JS', None),
]
for x, y, c, label, sub in backends:
    rounded_box(ax, x, y, NODE_W, NODE_H, c, label, sub)

jax_cx = be_xs[0] + NODE_W / 2
cpp_cx = be_xs[1] + NODE_W / 2
js_cx  = be_xs[2] + NODE_W / 2
be_top = be_y + NODE_H
be_bot = be_y

# PuzzleScript mascot inside the right side of NodeJS node
mascot_img = mpimg.imread('PuzzleScript/src/images/mascot_128.png')
mascot_box = OffsetImage(mascot_img, zoom=0.18)
mascot_ab = AnnotationBbox(mascot_box, (be_xs[2] + NODE_W - 0.10, be_y + NODE_H / 2),
                           frameon=False, box_alignment=(1.0, 0.5))
ax.add_artist(mascot_ab)

ax.text(RIGHT_MARGIN, be_y + NODE_H / 2, 'PuzzleScript\nBackends',
        ha='left', va='center', fontsize=FONT_ROW, style='italic',
        fontweight='bold', color='black', family='sans-serif')

# Arrow: algo → backends
straight_arrow(ax, content_cx, algo_bot, content_cx, be_top + OUTER_PAD, 'black', lw=2.5)

# ===== VALIDATION ROW =====
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

# Validation → backends (curved left) — start at val node top, end at backend node bottom
curved_arrow(ax, vjax_cx, val_top, jax_cx + 0.2, be_bot, C_JAX, lw=1.5, rad=-0.15)
curved_arrow(ax, vcpp_cx, val_top, cpp_cx + 0.2, be_bot, C_CPP, lw=1.5, rad=-0.15)

# NodeJS → validation (curved) — start at NodeJS bottom, end at val node top
curved_arrow(ax, js_cx - 0.2, be_bot, vjax_cx + 0.6, val_top, C_JS, lw=1.5, rad=-0.25)
ax.text((js_cx + vcpp_cx) / 2 + 0.38, (be_bot + val_top) / 2 - 0.1, 'Solutions',
        fontsize=FONT_LABEL, color=C_JS, family='sans-serif', style='italic', fontweight='bold')
curved_arrow(ax, js_cx, be_bot, vcpp_cx + 0.6, val_top, C_JS, lw=1.5, rad=-0.15)

ax.text(RIGHT_MARGIN, val_y + NODE_H / 2, 'Validation\nHarness',
        ha='left', va='center', fontsize=FONT_ROW, style='italic',
        fontweight='bold', color='black', family='sans-serif')

# ===== DATASETS — nested boxes, gists outer = same OUTER_H as other rows =====
ds_base_x = algo_outer_x
ds_base_y = val_bot - OUTER_PAD - ROW_GAP - NODE_H
ds_gists_w = algo_outer_w
ds_gists_h = NODE_H  # same height as inner nodes

# Nested proportions (width scales, height = full)
ds_info = [
    ('Gists',                  1.0,  0.15),
    ("Archive",        0.72, 0.10),
    ('Gallery',  0.48, 0.10),
    ('Select',             0.28, 0.05),
]

ds_top = ds_base_y + ds_gists_h
for name, frac, alpha in ds_info:
    w = ds_gists_w * frac
    h = ds_gists_h * frac  # all same height, nested by width
    box = FancyBboxPatch((ds_base_x, ds_base_y), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=(*plt.cm.colors.to_rgb(C_DATA), alpha),
                         edgecolor=C_DATA, linewidth=1.2)
    ax.add_patch(box)
    # Label right-aligned, vertically centered in its node
    label_x = ds_base_x + w - 0.15 if name != 'Gists' else ds_base_x + w - 0.6
    ax.text(label_x, ds_base_y + h / 2, name,
            ha='right', va='center', fontsize=11,
            fontweight='bold', color='black', family='sans-serif')

# GitHub logo to the right of "gists" label — snug inside the dataset node
github_img = mpimg.imread('github_mark_transparent.png')
github_box = OffsetImage(github_img, zoom=0.055)
# gists label is at upper-right of the outermost box
gists_label_x = ds_base_x + ds_gists_w - 0.8
gists_label_y = ds_base_y + ds_gists_h / 2  # vertically centered in gists node
github_ab = AnnotationBbox(github_box, (gists_label_x + 0.35, gists_label_y),
                           frameon=False, box_alignment=(0.0, 0.5))
ax.add_artist(github_ab)

ds_right = ds_base_x + ds_gists_w

# Single arrow: dataset upper-right up to backends box
# Arrow from gists to NodeJS backend
curved_arrow(ax, ds_right, ds_top, be_xs[2] + NODE_W, be_bot, C_DATA, lw=1.5, rad=0.35)
ax.text(be_xs[2] + NODE_W + 0.1, (ds_top + be_bot) / 2 - 0.1, 'Games',
        fontsize=FONT_LABEL, color=C_DATA, family='sans-serif', style='italic', fontweight='bold')

ax.text(RIGHT_MARGIN, ds_base_y + ds_gists_h / 2, 'Datasets',
        ha='left', va='center', fontsize=FONT_ROW, style='italic',
        fontweight='bold', color='black', family='sans-serif')

# ===== LEGEND (bottom, centered) =====
ly = ds_base_y - 0.45
line_len = 0.6
legend_text = 'JIT outer loop (JAX)'
lcx = content_cx
ax.plot([lcx - 1.8, lcx - 1.8 + line_len], [ly, ly], color=C_JAX, lw=1.3, ls=(0, (5, 3)))
ax.text(lcx - 1.8 + line_len, ly, legend_text, fontsize=FONT_LABEL, va='center',
        ha='left', color='#666', family='sans-serif', fontweight='bold')

plt.tight_layout()
plt.savefig('codebase_architecture_simple.png', dpi=300, bbox_inches='tight',
            pad_inches=0.05, facecolor='white', edgecolor='none')
plt.savefig('codebase_architecture_simple.pdf', bbox_inches='tight',
            pad_inches=0.05, facecolor='white', edgecolor='none')
plt.savefig('codebase_architecture_simple.svg', bbox_inches='tight',
            pad_inches=0.05, facecolor='white', edgecolor='none')
print("Saved codebase_architecture_simple.png, .pdf, and .svg")
