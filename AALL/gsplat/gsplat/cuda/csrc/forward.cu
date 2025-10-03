#include "forward.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>

namespace cg = cooperative_groups;

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    // get the tile bbox for gaussian
    uint2 tile_min, tile_max;
    float2 center = xys[idx];
    get_tile_bbox(center, radii[idx], tile_bounds, tile_min, tile_max);

    // update the intersection info for all tiles this gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];
    int64_t depth_id = 0;
    for (int i = tile_min.y; i < tile_max.y; ++i) {
        for (int j = tile_min.x; j < tile_max.x; ++j) {
            // isect_id is tile ID and depth as int32
            int64_t tile_id = i * tile_bounds.x + j; // tile within image
            isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            ++cur_idx; // handles gaussians that hit more than one tile
        }
    }
}

// kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
__global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t* __restrict__ isect_ids_sorted, int2* __restrict__ tile_bins
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0)
            tile_bins[cur_tile_idx].x = 0;
        if (idx == num_intersects - 1)
            tile_bins[cur_tile_idx].y = num_intersects;
    }
    if (idx == 0)
        return;
    int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x = idx;
        return;
    }
}

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
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
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    
    // **** max 12 channels for speed ***
    float pix_out[MAX_CHANNELS] = {0.f};

    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            xy_batch[tr] = {xy.x, xy.y};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                        conic.y * delta.x * delta.y;
            
            if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
                continue;
            }
            
            const float alpha = __expf(-sigma);
            int32_t g = id_batch[t];
            const float vis = alpha;
            
            for (int c = 0; c < channels; ++c) {
                pix_out[c] += colors[g * channels + c] * vis;
            }
        }
    }

    if (inside) {
        for (int c = 0; c < channels; ++c) {
            out_img[pix_id * channels + c] = pix_out[c];
        }
    }
}

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
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    
    // **** max 12 channels for speed ***
    float pix_out[MAX_CHANNELS] = {0.f};
    
    // top k Gaussian ids
    int32_t topk[TOP_K];
    float topk_vals[TOP_K] = {0.f};
    for (int k = 0; k < TOP_K; ++k)
        topk[k] = -1;

    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            xy_batch[tr] = {xy.x, xy.y};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                        conic.y * delta.x * delta.y;
            
            if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
                continue;
            }
            
            const float alpha = __expf(-sigma);
            int32_t g = id_batch[t];

            // find the minimum value in topk
            int32_t min_topk_id = -1;
            float min_topk_val = 1e30f;
            for (int32_t k = 0; k < TOP_K; ++k) {
                if (topk[k] < 0) {
                    min_topk_id = k;
                    min_topk_val = -1.0f;
                    break;
                } else if (topk_vals[k] < min_topk_val) {
                    min_topk_id = k;
                    min_topk_val = topk_vals[k];
                }
            }
            if (alpha > min_topk_val) {
                topk[min_topk_id] = g;
                topk_vals[min_topk_id] = alpha;
            }
        }
    }

    for (int c = 0; c < channels; ++c) {
        float sum_val = 0.f;
        for (int k = 0; k < TOP_K; ++k) {
            if (topk[k] < 0) continue;
            sum_val += topk_vals[k];
        }
        for (int k = 0; k < TOP_K; ++k) {
            int32_t g = topk[k];
            if (g < 0) continue;
            // normalize by sum of topk values
            float vis = topk_vals[k] / (sum_val + EPS);
            pix_out[c] += colors[g * channels + c] * vis;
        }
    }

    if (inside) {
        for (int c = 0; c < channels; ++c) {
            out_img[pix_id * channels + c] = pix_out[c];
        }
        for (int k = 0; k < TOP_K; ++k) {
            pixel_topk[pix_id * TOP_K + k] = topk[k];
        }
    }
}

__global__ void nd_rasterize_forward_no_tiles(
    const dim3 img_size,
    const unsigned channels,
    const unsigned num_points,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    float* __restrict__ out_img,
    int* __restrict__ pixel_topk
) {
    auto block = cg::this_thread_block();
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // **** max 12 channels for speed ***
    float pix_out[MAX_CHANNELS] = {0.f};
    
    // top k Gaussian ids
    int32_t topk[TOP_K];
    float topk_vals[TOP_K] = {0.f};
    for (int k = 0; k < TOP_K; ++k)
        topk[k] = -1;

    for (int g = 0; g < num_points; ++g) {
        const float3 conic = conics[g];
        const float2 xy = xys[g];
        const float2 delta = {xy.x - px, xy.y - py};
        const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
        
        if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
            continue;
        }
        
        const float alpha = __expf(-sigma);
        int32_t g_id = g;

        // find the minimum value in topk
        int32_t min_topk_id = -1;
        float min_topk_val = 1e30f;
        for (int32_t k = 0; k < TOP_K; ++k) {
            if (topk[k] < 0) {
                min_topk_id = k;
                min_topk_val = -1.0f;
                break;
            } else if (topk_vals[k] < min_topk_val) {
                min_topk_id = k;
                min_topk_val = topk_vals[k];
            }
        }
        if (alpha > min_topk_val) {
            topk[min_topk_id] = g_id;
            topk_vals[min_topk_id] = alpha;
        }
    }

    for (int c = 0; c < channels; ++c) {
        float sum_val = 0.f;
        for (int k = 0; k < TOP_K; ++k) {
            if (topk[k] < 0) continue;
            sum_val += topk_vals[k];
        }
        for (int k = 0; k < TOP_K; ++k) {
            int32_t g = topk[k];
            if (g < 0) continue;
            // normalize by sum of topk values
            float vis = topk_vals[k] / (sum_val + EPS_no_tiles);
            pix_out[c] += colors[g * channels + c] * vis;
        }
    }

    if (inside) {
        for (int c = 0; c < channels; ++c) {
            out_img[pix_id * channels + c] = pix_out[c];
        }
        for (int k = 0; k < TOP_K; ++k) {
            pixel_topk[pix_id * TOP_K + k] = topk[k];
        }
    }
}

__global__ void rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    float3* __restrict__ out_img
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];

    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            xy_batch[tr] = {xy.x, xy.y};
            conic_batch[tr] = conics[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float2 xy = xy_batch[t];
            const float2 delta = {xy.x - px, xy.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                        conic.y * delta.x * delta.y;
            
            // const float alpha = min(1.f, __expf(-sigma));
            if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
            //     printf("wrong value sigma %f delta %f %f conic %f %f %f\n", 
            //         sigma, delta.x, delta.y, conic.x, conic.y, conic.z);
                continue;
            }
            
            const float alpha = __expf(-sigma);
            int32_t g = id_batch[t];
            const float vis = alpha;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        out_img[pix_id] = pix_out;
    }
}

// device helper to approximate projected 2d cov from 3d mean and cov
__device__ float3 project_cov3d_ewa(
    const float3& __restrict__ mean3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy
) {
    // clip the
    // we expect row major matrices as input, glm uses column major
    // upper 3x3 submatrix
    glm::mat3 W = glm::mat3(
        viewmat[0],
        viewmat[4],
        viewmat[8],
        viewmat[1],
        viewmat[5],
        viewmat[9],
        viewmat[2],
        viewmat[6],
        viewmat[10]
    );
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;

    // clip so that the covariance
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    t.x = t.z * std::min(lim_x, std::max(-lim_x, t.x / t.z));
    t.y = t.z * std::min(lim_y, std::max(-lim_y, t.y / t.z));

    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    glm::mat3 J = glm::mat3(
        fx * rz,
        0.f,
        0.f,
        0.f,
        fy * rz,
        0.f,
        -fx * t.x * rz2,
        -fy * t.y * rz2,
        0.f
    );
    glm::mat3 T = J * W;

    glm::mat3 V = glm::mat3(
        cov3d[0],
        cov3d[1],
        cov3d[2],
        cov3d[1],
        cov3d[3],
        cov3d[4],
        cov3d[2],
        cov3d[4],
        cov3d[5]
    );

    glm::mat3 cov = T * V * glm::transpose(T);

    // add a little blur along axes and save upper triangular elements
    return make_float3(float(cov[0][0]) + 0.3f, float(cov[0][1]), float(cov[1][1]) + 0.3f);
}

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
) {
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.x, quat.y, quat.z, quat.w);
    glm::mat3 R = quat_to_rotmat(quat);
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    glm::mat3 M = R * S;
    glm::mat3 tmp = M * glm::transpose(M);
    // printf("tmp %.2f %.2f %.2f\n", tmp[0][0], tmp[1][1], tmp[2][2]);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}
