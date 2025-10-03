#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void project_gaussians_2d_scale_rot_forward_kernel(
    const int num_points,
    const float2* __restrict__ means2d,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const dim3 img_size,
    const dim3 tile_bounds,
    float2* __restrict__ xys,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
);
