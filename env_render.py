import numpy as np
import os
import re
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import jax.numpy as jnp
import jax
from jax import random
from jax import jit, vmap


# TODO: double check these colors
color_hex_map = {
    "black": "#000000",
    "white": "#FFFFFF",
    "lightgray": "#D3D3D3",
    "lightgrey": "#D3D3D3",
    "gray": "#808080",
    "grey": "#808080",
    "darkgray": "#A9A9A9",
    "darkgrey": "#A9A9A9",
    "red": "#FF0000",
    "darkred": "#8B0000",
    "lightred": "#FF6666",
    "brown": "#A52A2A",
    "darkbrown": "#5C4033",
    "lightbrown": "#C4A484",
    "orange": "#FFA500",
    "yellow": "#FFFF00",
    "green": "#008000",
    "darkgreen": "#006400",
    "lightgreen": "#90EE90",
    "blue": "#0000FF",
    "lightblue": "#ADD8E6",
    "darkblue": "#00008B",
    "purple": "#800080",
    "pink": "#FFC0CB",
    "transparent": "#00000000"  # Transparent in RGBA format
}

def render_solid_color(color):
    color = color.lower()
    alpha = 255
    if color == 'transparent':
        c = color_hex_map[color]
        alpha = 0
    elif color in color_hex_map:
        c = color_hex_map[color]
    else:
        c = '#000000'
    c = hex_to_rgba(c, alpha)
    im = np.zeros((5, 5, 4), dtype=np.uint8)
    im[:, :, :] = np.array(c)
    return im

def render_sprite(colors, sprite):
    sprite = np.vectorize(replace_bg_tiles)(sprite)
    colors = np.array(['transparent'] + colors)
    colors_vec = np.zeros((len(colors), 4), dtype=np.uint8)
    for i, c in enumerate(colors):
        c = c.lower()
        alpha = 255
        if c in color_hex_map:
            if c == 'transparent':
                alpha = 0
            c = color_hex_map[c]
        c = hex_to_rgba(c, alpha)
        colors_vec[i] = np.array(c)
    im = colors_vec[sprite]

    # Anywhere the sprite is 0, set to transparent
    im[sprite == 0] = [0, 0, 0, 0]

    return im


def replace_bg_tiles(x):
    if x == '.':
        return 0
    else:
        return int(x) + 1

def hex_to_rgba(hex_code, alpha):
    """Converts a hex color code to RGBA values."""
    hex_code = hex_code.lstrip('#')
    if len(hex_code) == 3:
        hex_code = ''.join(c*2 for c in hex_code)
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return (*rgb, alpha)
