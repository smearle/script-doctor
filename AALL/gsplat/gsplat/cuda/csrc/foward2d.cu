#include "forward2d.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
namespace cg = cooperative_groups;

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
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // Retrieve the 2D Gaussian parameters
    float2 center = {means2d[idx].x * img_size.x, means2d[idx].y * img_size.y};

    // cov   = R * S * S * T^(-1)
    // conic = R * S^(-1) * S^(-1) * T^(-1)
    glm::mat2 R = rotmat2d(rotation[idx]);
    glm::mat2 S = scale_to_mat2d(scales2d[idx]);
    glm::mat2 M = R * S;
    glm::mat2 tmp = M * glm::transpose(M);

    // upper triangular
    float3 cov2d = make_float3(tmp[0][0], tmp[0][1], tmp[1][1]);
    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok){
        // printf("compute_cov2d_bounds failed\n");
        return; // zero determinant
    }
    conics[idx] = conic;
    
    xys[idx] = center;
    radii[idx] = (int)radius;
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;

}