#include "cuda_runtime.h"
#include "forward.cuh"
#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_tensor(const int num_pts, torch::Tensor &A);


std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
);

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects,
    int num_tiles,
    const torch::Tensor &isect_ids_sorted
);


std::tuple<
    torch::Tensor
> nd_rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors
);

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor  // dL_dcolors
        >
    nd_rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned BLOCK_H,
        const unsigned BLOCK_W,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &v_output  // dL_dout_color
    );

std::tuple<
    torch::Tensor, // out_img 
    torch::Tensor  // pixel_topk 
> nd_rasterize_forward_topk_norm_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors
);

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor  // dL_dcolors
        >
    nd_rasterize_backward_topk_norm_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned BLOCK_H,
        const unsigned BLOCK_W,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &v_output,  // dL_dout_color
        const torch::Tensor &pixel_topk
    );

std::tuple<
    torch::Tensor, // out_img 
    torch::Tensor  // pixel_topk 
> nd_rasterize_forward_no_tiles_tensor(
    const std::tuple<int, int, int> img_size,
    const unsigned num_points,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors
);

std::tuple<
    torch::Tensor, // dL_dxy
    torch::Tensor, // dL_dconic
    torch::Tensor  // dL_dcolors
    >
nd_rasterize_backward_no_tiles_tensor(
    const unsigned img_height,
    const unsigned img_width,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &v_output,  // dL_dout_color
    const torch::Tensor &pixel_topk
);

std::tuple<
    torch::Tensor, // out_img 
    torch::Tensor  // pixel_topk 
> nd_rasterize_forward_simple_tensor(
    const std::tuple<int, int, int> img_size,
    const unsigned num_points,
    const torch::Tensor &xys,
    const torch::Tensor &scale,
    const torch::Tensor &rot,
    const torch::Tensor &feat
);

std::tuple<
    torch::Tensor, // dL_dxy
    torch::Tensor, // dL_dscale
    torch::Tensor, // dL_drot
    torch::Tensor  // dL_dfeat
    >
nd_rasterize_backward_simple_tensor(
    const unsigned img_height,
    const unsigned img_width,
    const torch::Tensor &xys,
    const torch::Tensor &scale,
    const torch::Tensor &rot,
    const torch::Tensor &feat,
    const torch::Tensor &v_output,  // dL_dout_color
    const torch::Tensor &pixel_topk
);


std::tuple<
    torch::Tensor
> rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors
);

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor  // dL_dcolors
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned BLOCK_H,
        const unsigned BLOCK_W,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &v_output  // dL_dout_color
    );


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_scale_rot_forward_tensor(
    const int num_points,
    torch::Tensor &means2d,
    torch::Tensor &scales2d,
    torch::Tensor &rotation,
    const unsigned img_height,
    const unsigned img_width,
    const std::tuple<int, int, int> tile_bounds
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_2d_scale_rot_backward_tensor(
    const int num_points,
    torch::Tensor &means2d,
    torch::Tensor &scales2d,
    torch::Tensor &rotation,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_conic
);