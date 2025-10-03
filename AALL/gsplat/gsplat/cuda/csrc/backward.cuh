#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>


// compute jacobians of output image wrt binned and sorted gaussians
__global__ void nd_rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb
);

__global__ void nd_rasterize_backward_topk_norm_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    int* __restrict__ pixel_topk
);

__global__ void nd_rasterize_backward_no_tiles_kernel(
    const dim3 img_size,
    const unsigned channels,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    int* __restrict__ pixel_topk
);


__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float3* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb
);


__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *v_cov3d,
    float3 &v_scale,
    float4 &v_quat
);
