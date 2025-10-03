"""Python bindings for custom Cuda functions"""

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects


def rasterize_gaussians_sum(
    xys: Float[Tensor, "*batch 2"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    img_height: int,
    img_width: int,
    BLOCK_H: int=16,
    BLOCK_W: int=16,
    topk_norm: bool=False
) -> Tensor:
    if colors.dtype == torch.uint8:
        # make sure colors are float [0,1]
        colors = colors.float() / 255

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    return _RasterizeGaussiansSum.apply(
        xys.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        img_height,
        img_width,
        BLOCK_H, 
        BLOCK_W,
        topk_norm
    )


class _RasterizeGaussiansSum(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        img_height: int,
        img_width: int,
        BLOCK_H: int=16,
        BLOCK_W: int=16,
        topk_norm: bool=False
    ) -> Tensor:
        num_points = xys.size(0)
        BLOCK_X, BLOCK_Y = BLOCK_W, BLOCK_H
        tile_bounds = (
            (img_width + BLOCK_X - 1) // BLOCK_X,
            (img_height + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        block = (BLOCK_X, BLOCK_Y, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = (
                torch.zeros(img_height, img_width, colors.shape[-1], device=xys.device)
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
            tile_bins =           torch.zeros(0, 2, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                radii,
                cum_tiles_hit,
                tile_bounds,
            )
        
        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.BLOCK_H = BLOCK_H
        ctx.BLOCK_W = BLOCK_W
        ctx.num_intersects = num_intersects
        ctx.topk_norm = topk_norm

        if not topk_norm:
        # if colors.shape[-1] == 3:
        #     rasterize_fn = _C.rasterize_forward
        # else:
            rasterize_fn = _C.nd_rasterize_forward

            out_img,  = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
            )

            ctx.save_for_backward(
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors
            )
        else:
            out_img, pixel_topk = _C.nd_rasterize_forward_topk_norm(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
            )
            ctx.save_for_backward(
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                pixel_topk.contiguous()
            )

        return out_img

    @staticmethod
    def backward(ctx, v_out_img):
        img_height = ctx.img_height
        img_width = ctx.img_width
        BLOCK_H = ctx.BLOCK_H
        BLOCK_W = ctx.BLOCK_W
        num_intersects = ctx.num_intersects
        topk_norm = ctx.topk_norm

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            v_colors = torch.zeros_like(colors)
        
        elif not topk_norm:
            (
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors
            ) = ctx.saved_tensors
            # if colors.shape[-1] == 3:
            #     rasterize_fn = _C.rasterize_backward
            # else:
            rasterize_fn = _C.nd_rasterize_backward
            v_xy, v_conic, v_colors = rasterize_fn(
                img_height,
                img_width,
                BLOCK_H,
                BLOCK_W,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                v_out_img
            )
        else:
            (
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                pixel_topk
            ) = ctx.saved_tensors
            v_xy, v_conic, v_colors = _C.nd_rasterize_backward_topk_norm(
                img_height,
                img_width,
                BLOCK_H,
                BLOCK_W,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                v_out_img,
                pixel_topk
            )

        return (
            v_xy,  # xys
            None,  # radii
            v_conic,  # conics
            None,  # num_tiles_hit
            v_colors,  # colors
            None,  # img_height
            None,  # img_width
            None,  # block_w
            None,  # block_h
            None,  # topk_norm
        )
