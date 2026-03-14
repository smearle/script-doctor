#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

/// Find player position in column-major objects array using player bitmask.
/// Returns {x, y}. If not found, returns {-1, -1}.
inline std::pair<int, int> findPlayerInObjects(
    const int32_t* objects, int width, int height, int n_objs,
    const std::vector<int32_t>& player_mask_words, bool player_mask_aggregate)
{
    const int stride_obj = (n_objs + 31) / 32;
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            const int flat_idx = (x * height + y) * stride_obj;
            bool matches;
            if (player_mask_aggregate) {
                matches = true;
                for (int w = 0; w < static_cast<int>(player_mask_words.size()); ++w) {
                    if ((player_mask_words[w] & objects[flat_idx + w]) != player_mask_words[w]) {
                        matches = false;
                        break;
                    }
                }
            } else {
                matches = false;
                for (int w = 0; w < static_cast<int>(player_mask_words.size()); ++w) {
                    if (player_mask_words[w] & objects[flat_idx + w]) {
                        matches = true;
                        break;
                    }
                }
            }
            if (matches) return {x, y};
        }
    }
    return {-1, -1};
}

/// Compute viewport bounds from player position and screen config.
/// Returns {min_x, min_y, max_x, max_y} (exclusive end).
/// If player not found (px < 0), returns full level bounds.
inline std::array<int, 4> computeScreenBounds(
    int px, int py, int level_w, int level_h,
    const std::optional<std::pair<int, int>>& flickscreen,
    const std::optional<std::pair<int, int>>& zoomscreen)
{
    if (!flickscreen && !zoomscreen) {
        return {0, 0, level_w, level_h};
    }
    if (px < 0) {
        return {0, 0, level_w, level_h};
    }
    if (flickscreen) {
        int sw = flickscreen->first, sh = flickscreen->second;
        int mini = (px / sw) * sw;
        int minj = (py / sh) * sh;
        return {mini, minj, std::min(mini + sw, level_w), std::min(minj + sh, level_h)};
    }
    // zoomscreen
    int sw = zoomscreen->first, sh = zoomscreen->second;
    int mini = std::max(std::min(px - sw / 2, level_w - sw), 0);
    int minj = std::max(std::min(py - sh / 2, level_h - sh), 0);
    return {mini, minj, std::min(mini + sw, level_w), std::min(minj + sh, level_h)};
}
