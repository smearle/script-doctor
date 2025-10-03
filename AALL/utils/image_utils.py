import os

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from numpy.linalg import norm
from PIL import Image
from scipy.ndimage import sobel

ALLOWED_IMAGE_FILE_FORMATS = [".jpeg", ".jpg", ".png"]
ALLOWED_IMAGE_TYPES = {"RGB": 3, "RGBA": 3, "L": 1}

PLOT_DPI = 72.0
GAUSSIAN_ZOOM = 5
GAUSSIAN_COLOR = "#80ed99"


def get_psnr(image1, image2, max_value=1.0):
    mse = torch.mean((image1-image2)**2)
    if mse.item() <= 1e-7:
        return float('inf')
    psnr = 20*torch.log10(max_value/torch.sqrt(mse))
    return psnr


def get_grid(h, w, x_lim=np.asarray([0, 1]), y_lim=np.asarray([0, 1])):
    x = torch.linspace(x_lim[0], x_lim[1], steps=w + 1)[:-1] + 0.5 / w
    y = torch.linspace(y_lim[0], y_lim[1], steps=h + 1)[:-1] + 0.5 / h
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid


def compute_image_gradients(image):
    gy, gx = [], []
    for image_channel in image:
        gy.append(sobel(image_channel, 0))
        gx.append(sobel(image_channel, 1))
    gy = norm(np.stack(gy, axis=0), ord=2, axis=0).astype(np.float32)
    gx = norm(np.stack(gx, axis=0), ord=2, axis=0).astype(np.float32)
    return gy, gx


def load_images(load_path, downsample_ratio=None, gamma=None):
    """
    Load target images or textures from a directory or a single file.
    """
    image_list = []
    image_path_list = []
    image_fname_list = []
    num_channels_list = []
    if os.path.isfile(load_path) and os.path.splitext(load_path)[1].lower() in ALLOWED_IMAGE_FILE_FORMATS:
        image_path_list.append(load_path)
    elif os.path.isdir(load_path):
        for file in sorted(os.listdir(load_path), key=str.lower):
            if os.path.splitext(file)[1].lower() in ALLOWED_IMAGE_FILE_FORMATS:
                image_path_list.append(os.path.join(load_path, file))
    if len(image_path_list) == 0:
        raise FileNotFoundError(f"No supported image file found at '{load_path}'")
    for image_path in image_path_list:
        image_fname_list.append(os.path.splitext(os.path.basename(image_path))[0])
        image = Image.open(image_path)
        # Warning: Only support images of type L, RGB, or RGBA in JPEG or PNG format
        if not image.mode in ALLOWED_IMAGE_TYPES:
            raise TypeError(f"Only support images of type {list(ALLOWED_IMAGE_TYPES.keys())} in JPEG or PNG format")
        num_channels = ALLOWED_IMAGE_TYPES[image.mode]
        num_channels_list.append(num_channels)
        if downsample_ratio is not None:
            image = image.resize((round(image.width/downsample_ratio), round(image.height/downsample_ratio)), resample=Image.Resampling.BILINEAR)
        # Warning: Assume 8 bit color depth
        image = np.asarray(image, dtype=np.float32) / 255.0
        if gamma is not None:
            image = np.power(image, gamma)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        image = image.transpose(2, 0, 1)
        image = image[:num_channels]
        image_list.append(image)
    return np.concatenate(image_list, axis=0), num_channels_list, image_fname_list


def to_output_format(image, gamma):
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Wrong image format: shape = {image.shape}")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().clone().numpy()
    if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
        image = image.transpose(1, 2, 0)
        if image.shape[2] not in [1, 3]:
            raise ValueError(f"Wrong image format: shape = {image.shape}")
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)
    image = np.clip(image, 0.0, 1.0)
    if gamma is not None:
        image = np.power(image, 1.0/gamma)
    image = (255.0 * image).astype(np.uint8)
    return image


def save_image(image, save_path, gamma=None, zoom=None):
    image = to_output_format(image, gamma)
    image = Image.fromarray(image)
    if zoom is not None and zoom > 0.0:
        width, height = image.size
        image = image.resize((round(width*zoom), round(height*zoom)), resample=Image.Resampling.BOX)
    image.save(save_path)


def separate_image_channels(images, input_channels):
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_list = []
    curr_channel = 0
    for num_channels in input_channels:
        image_list.append(images[curr_channel:curr_channel+num_channels])
        curr_channel += num_channels
    return image_list


def visualize_gaussians(filepath, xy, scale, rot, feat, img_h, img_w, input_channels, alpha=0.8, gamma=None):
    """
    Visualize Gaussians as colored elliptical disks.
    """
    if feat.shape[1] != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {feat.shape[1]:d} vs {sum(input_channels):d}")
    xy = xy.detach().cpu().clone().numpy()
    y, x = xy[:, 1] * img_h, xy[:, 0] * img_w
    scale = GAUSSIAN_ZOOM * scale.detach().cpu().clone().numpy()
    rot = rot.detach().cpu().clone().numpy()
    if gamma is not None:
        feat = torch.pow(feat, 1.0/gamma)
    feat = np.clip(feat.detach().cpu().clone().numpy(), 0.0, 1.0)

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        curr_feat = feat[:, curr_channel:curr_channel+num_channels]
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=img_w/PLOT_DPI, h=img_h/PLOT_DPI, forward=False)
        ax = plt.gca()
        for gid in range(len(xy)):
            ellipse = Ellipse(xy=(x[gid], y[gid]), width=scale[gid, 0], height=scale[gid, 1],
                              angle=rot[gid, 0]*180/np.pi, alpha=alpha, ec=None, fc=curr_feat[gid], lw=None)
            ax.add_patch(ellipse)
        plt.xlim(0, img_w)
        plt.ylim(img_h, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.png", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels


def visualize_added_gaussians(filepath, images, old_xy, new_xy, input_channels, size=500, every_n=5, alpha=0.8, gamma=None):
    """
    Visualize the positions of added Gaussians during error-guided progressive optimization.
    """
    if len(images) != sum(input_channels):
        raise ValueError(f"Incompatible number of channels: {len(images):d} vs {sum(input_channels):d}")
    image_height, image_width = images.shape[1:]
    old_xy = old_xy.detach().cpu().clone().numpy()[::every_n]
    new_xy = new_xy.detach().cpu().clone().numpy()[::every_n]
    old_x, old_y = old_xy[:, 0] * image_width, old_xy[:, 1] * image_height
    new_x, new_y = new_xy[:, 0] * image_width, new_xy[:, 1] * image_height

    curr_channel = 0
    for image_id, num_channels in enumerate(input_channels, 1):
        image = images[curr_channel:curr_channel+num_channels]
        image = to_output_format(image, gamma)
        fig = plt.figure()
        fig.set_dpi(PLOT_DPI)
        fig.set_size_inches(w=image_width/PLOT_DPI, h=image_height/PLOT_DPI, forward=False)
        plt.imshow(Image.fromarray(image), cmap='gray', vmin=0, vmax=255)
        plt.scatter(old_x, old_y, s=size, c="#ef476f", marker='o', alpha=alpha)  # red
        plt.scatter(new_x, new_y, s=size, c="#06d6a0", marker='o', alpha=alpha)  # green
        plt.xlim(0, image_width)
        plt.ylim(image_height, 0)
        plt.axis('off')
        plt.tight_layout()
        suffix = "" if len(input_channels) == 1 else f"_{image_id:d}"
        plt.savefig(f"{filepath}{suffix}.png", bbox_inches='tight', pad_inches=0, dpi=PLOT_DPI)
        plt.close()
        curr_channel += num_channels
