from typing import Any
import torch
from .project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from .rasterize_sum import rasterize_gaussians_sum
from .rasterize_no_tiles import rasterize_gaussians_no_tiles, rasterize_gaussians_simple
from .utils import (
    map_gaussian_to_intersects,
    bin_and_sort_gaussians,
    compute_cumulative_intersects,
    compute_cov2d_bounds,
    get_tile_bin_edges,
)
import warnings


__all__ = [
    "project_gaussians_2d_scale_rot",
    "rasterize_gaussians_sum",
    "rasterize_gaussians_no_tiles",
    # utils
    "bin_and_sort_gaussians",
    "compute_cumulative_intersects",
    "compute_cov2d_bounds",
    "get_tile_bin_edges",
    "map_gaussian_to_intersects",
    "ProjectGaussians2dScaleRot",
    "RasterizeGaussiansSum",
    "BinAndSortGaussians",
    "ComputeCumulativeIntersects",
    "ComputeCov2dBounds",
    "GetTileBinEdges",
    "MapGaussiansToIntersects",
]

# Define these for backwards compatibility

class MapGaussiansToIntersects(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "MapGaussiansToIntersects is deprecated, use map_gaussian_to_intersects instead",
            DeprecationWarning,
        )
        return map_gaussian_to_intersects(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class ComputeCumulativeIntersects(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ComputeCumulativeIntersects is deprecated, use compute_cumulative_intersects instead",
            DeprecationWarning,
        )
        return compute_cumulative_intersects(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class ComputeCov2dBounds(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ComputeCov2dBounds is deprecated, use compute_cov2d_bounds instead",
            DeprecationWarning,
        )
        return compute_cov2d_bounds(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class GetTileBinEdges(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "GetTileBinEdges is deprecated, use get_tile_bin_edges instead",
            DeprecationWarning,
        )
        return get_tile_bin_edges(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class BinAndSortGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "BinAndSortGaussians is deprecated, use bin_and_sort_gaussians instead",
            DeprecationWarning,
        )
        return bin_and_sort_gaussians(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError



class ProjectGaussians2dScaleRot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ProjectGaussians2dScaleRot is deprecated, use project_gaussians_2d_scale_rot instead",
            DeprecationWarning,
        )
        return project_gaussians_2d_scale_rot(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError



class RasterizeGaussiansSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "RasterizeGaussiansSum is deprecated, use rasterize_gaussians instead",
            DeprecationWarning,
        )
        return rasterize_gaussians_sum(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


