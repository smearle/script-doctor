from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


nd_rasterize_forward = _make_lazy_cuda_func("nd_rasterize_forward")
nd_rasterize_backward = _make_lazy_cuda_func("nd_rasterize_backward")

nd_rasterize_forward_no_tiles = _make_lazy_cuda_func("nd_rasterize_forward_no_tiles")
nd_rasterize_backward_no_tiles = _make_lazy_cuda_func("nd_rasterize_backward_no_tiles")

nd_rasterize_forward_simple = _make_lazy_cuda_func("nd_rasterize_forward_simple")
nd_rasterize_backward_simple = _make_lazy_cuda_func("nd_rasterize_backward_simple")

nd_rasterize_forward_topk_norm = _make_lazy_cuda_func("nd_rasterize_forward_topk_norm")
nd_rasterize_backward_topk_norm = _make_lazy_cuda_func("nd_rasterize_backward_topk_norm")

rasterize_forward = _make_lazy_cuda_func("rasterize_forward")
rasterize_backward = _make_lazy_cuda_func("rasterize_backward")

compute_cov2d_bounds = _make_lazy_cuda_func("compute_cov2d_bounds")
project_gaussians_2d_scale_rot_forward = _make_lazy_cuda_func("project_gaussians_2d_scale_rot_forward")
project_gaussians_2d_scale_rot_backward = _make_lazy_cuda_func("project_gaussians_2d_scale_rot_backward")
map_gaussian_to_intersects = _make_lazy_cuda_func("map_gaussian_to_intersects")
get_tile_bin_edges = _make_lazy_cuda_func("get_tile_bin_edges")

