import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from utils.image_utils import get_grid, load_images, compute_image_gradients, save_image

# fork of gsplat from https://github.com/NYU-ICL/image-gs
from gsplat import (
    project_gaussians_2d_scale_rot,
    rasterize_gaussians_no_tiles,
)

dtype = torch.float32
device = "cuda" # has to be on cuda

gamma = 1.0
init_scale = 40.0
num_gaussians = 2000
init_random_ratio = 0.2 
block_h, block_w = 16, 16  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
images, input_channels, image_fnames  = load_images("./AALL/anime-1_2k.png")
img_h, img_w = images.shape[1:]
images = torch.from_numpy(images)

def compute_gmap(images):
    print("Generating Gradient Map")
    gy, gx = compute_image_gradients(np.power(images.clone().numpy(), 1.0/gamma))
    g_norm = np.hypot(gy, gx).astype(np.float32)
    g_norm = g_norm / g_norm.max()
    save_image(g_norm, f"./AALL/test_gradient_map.png")
    g_norm = np.power(g_norm.reshape(-1), 2.0)
    return g_norm / g_norm.sum()

xy = nn.Parameter(torch.rand(num_gaussians, 2, dtype=dtype, device=device), requires_grad=True)
scale = nn.Parameter(torch.ones(num_gaussians, 2, dtype=dtype, device=device), requires_grad=True)
rot = nn.Parameter(torch.zeros(num_gaussians, 1, dtype=dtype, device=device), requires_grad=True)
feat_dim = sum(input_channels)
feat = nn.Parameter(torch.rand(num_gaussians, feat_dim, dtype=dtype, device=device), requires_grad=True)
vis_feat = nn.Parameter(torch.rand_like(feat), requires_grad=False)  # Only used for Gaussian ID visualization

tile_bounds = ((img_w + block_w - 1) // block_w, (img_h + block_h - 1) // block_h, 1)

def forward(xy, scale, rot, upsample_ratio=None, benchmark=False):
    tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
    xy, radii, conics, num_tiles_hit = tmp
    tmp = xy, conics, feat, img_h, img_w
    out_image = rasterize_gaussians_no_tiles(*tmp)
    out_image = out_image.view(-1, img_h, img_w, feat_dim).permute(0, 3, 1, 2).contiguous()
    return out_image.squeeze(dim=0)

def separate_image_channels(images, input_channels):
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_list = []
    curr_channel = 0
    for num_channels in input_channels:
        image_list.append(images[curr_channel:curr_channel+num_channels])
        curr_channel += num_channels
    return image_list

def separate_and_save_images(images, channels, path):
    images_sep = separate_image_channels(images=images, input_channels=channels)
    for idx, image in enumerate(images_sep, 1):
        suffix = "" if len(images_sep) == 1 else f"_{idx:d}"
        save_image(image, f"{path}{suffix}.png", gamma=gamma)

def sample_pos(pixel_xy, prob):
    num_random = round(init_random_ratio*num_gaussians)
    selected_random = np.random.choice(img_h * img_w, num_random, replace=False, p=None)
    selected_other = np.random.choice(img_h * img_w, num_gaussians-num_random, replace=False, p=prob)
    return torch.cat([pixel_xy.detach().clone()[selected_random], pixel_xy.detach().clone()[selected_other]], dim=0)

def get_target_features(positions):
    with torch.no_grad():
        # gt_images [1, C, H, W]; positions [1, 1, P, 2]; top-left [-1, -1]; bottom-right [1, 1]
        target_features = F.grid_sample(images.unsqueeze(0).to(device=device), positions[None, None, ...] * 2.0 - 1.0, align_corners=False)
        target_features = target_features[0, :, 0, :].permute(1, 0)  # [P, C]
    return target_features

def init_pos_scale_feat(images):
    pixel_xy = get_grid(h=img_h, w=img_w).to(dtype=dtype, device=device).reshape(-1, 2)
    with torch.no_grad():
        # Position
        image_gradients = compute_gmap(images)
        xy.copy_(sample_pos(pixel_xy, prob=image_gradients))
        # Scale
        scale.fill_(init_scale)
        # Feature
        feat.copy_(get_target_features(positions=xy).detach().clone())

init_pos_scale_feat(images)
separate_and_save_images(images=forward(xy, scale, rot), channels=input_channels, path=f"./AALL/test_gaussian_map")