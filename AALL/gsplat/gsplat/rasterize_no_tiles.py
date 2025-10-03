"""Python bindings for custom Cuda functions"""

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


def rasterize_gaussians_no_tiles(
    xys: Float[Tensor, "*batch 2"],
    conics: Float[Tensor, "*batch 3"],
    colors: Float[Tensor, "*batch channels"],
    img_height: int,
    img_width: int
) -> Tensor:
    if colors.dtype == torch.uint8:
        # make sure colors are float [0,1]
        colors = colors.float() / 255

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    return _RasterizeGaussiansNoTiles.apply(
        xys.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        img_height,
        img_width
    )


class _RasterizeGaussiansNoTiles(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        conics: Float[Tensor, "*batch 3"],
        colors: Float[Tensor, "*batch channels"],
        img_height: int,
        img_width: int
    ) -> Tensor:
        num_points = xys.size(0)
        img_size = (img_width, img_height, 1)

        ctx.img_width = img_width
        ctx.img_height = img_height

        out_img, pixel_topk = _C.nd_rasterize_forward_no_tiles(
            img_size,
            num_points,
            xys,
            conics,
            colors
        )
        ctx.save_for_backward(
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
        
        (
            xys,
            conics,
            colors,
            pixel_topk
        ) = ctx.saved_tensors
        v_xy, v_conic, v_colors = _C.nd_rasterize_backward_no_tiles(
            img_height,
            img_width,
            xys,
            conics,
            colors,
            v_out_img,
            pixel_topk
        )

        return (
            v_xy,  # xys
            v_conic,  # conics
            v_colors,  # colors
            None,  # img_height
            None,  # img_width
        )
    
def rasterize_gaussians_simple(
    xys:    Float[Tensor, "*batch 2"],
    scale:  Float[Tensor, "*batch 2"],
    rot:    Float[Tensor, "*batch"],
    feat: Float[Tensor, "*batch channels"],
    img_height: int,
    img_width: int
) -> Tensor:
    
    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")
    if feat.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    return _RasterizeGaussiansSimple.apply(
        xys.contiguous(),
        scale.contiguous(),
        rot.contiguous(),
        feat.contiguous(),
        img_height,
        img_width
    )


class _RasterizeGaussiansSimple(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys:    Float[Tensor, "*batch 2"],
        scale:  Float[Tensor, "*batch 2"],
        rot:    Float[Tensor, "*batch"],
        feat:   Float[Tensor, "*batch channels"],
        img_height: int,
        img_width: int
    ) -> Tensor:
        num_points = xys.size(0)
        img_size = (img_width, img_height, 1)

        ctx.img_width = img_width
        ctx.img_height = img_height

        out_img, pixel_topk = _C.nd_rasterize_forward_simple(
            img_size,
            num_points,
            xys,
            scale,
            rot,
            feat
        )
        ctx.save_for_backward(
            xys,
            scale,
            rot,
            feat,
            pixel_topk.contiguous()
        )

        return out_img

    @staticmethod
    def backward(ctx, v_out_img):
        img_height = ctx.img_height
        img_width = ctx.img_width
        
        (
            xys,
            scale,
            rot,
            feat,
            pixel_topk
        ) = ctx.saved_tensors

        v_xy, v_scale, v_rot, v_feat = _C.nd_rasterize_backward_simple(
            img_height,
            img_width,
            xys,
            scale,
            rot,
            feat,
            v_out_img,
            pixel_topk
        )

        return (
            v_xy,    # xys
            v_scale, # scale
            v_rot,   # rot
            v_feat,  # feat
            None,  # img_height
            None,  # img_width
        )