#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>


__global__ void nd_rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    float* __restrict__ out_img
);

__global__ void nd_rasterize_forward_topk_norm(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    float* __restrict__ out_img,
    int* __restrict__ pixel_topk
);

__global__ void nd_rasterize_forward_no_tiles(
    const dim3 img_size,
    const unsigned channels,
    const unsigned num_points,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    float* __restrict__ out_img,
    int* __restrict__ pixel_topk
);


// device helper to approximate projected 2d cov from 3d mean and cov
__device__ float3 project_cov3d_ewa(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy
);

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
);

__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
);

__global__ void get_tile_bin_edges(
    const int num_intersects, 
    const int64_t* __restrict__ isect_ids_sorted,
    int2* __restrict__ tile_bins
);


__global__ void rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    float3* __restrict__ out_img
);