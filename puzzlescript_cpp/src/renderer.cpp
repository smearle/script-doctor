#include "renderer.h"
#include "json.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

using json = nlohmann::json;

// ============================================================
// Hex colour helpers
// ============================================================
uint8_t Renderer::hexCharVal(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
    return 0;
}

bool Renderer::hexToRGB(const std::string& hex, uint8_t out[3]) {
    // Accept "#RGB" or "#RRGGBB"
    std::string h = hex;
    if (h == "transparent" || h == "Transparent" || h == "TRANSPARENT") {
        return false;
    }
    if (!h.empty() && h[0] == '#') h = h.substr(1);
    if (h.size() == 3) {
        out[0] = hexCharVal(h[0]) * 17;
        out[1] = hexCharVal(h[1]) * 17;
        out[2] = hexCharVal(h[2]) * 17;
        return true;
    } else if (h.size() >= 6) {
        out[0] = hexCharVal(h[0]) * 16 + hexCharVal(h[1]);
        out[1] = hexCharVal(h[2]) * 16 + hexCharVal(h[3]);
        out[2] = hexCharVal(h[4]) * 16 + hexCharVal(h[5]);
        return true;
    } else {
        return false;
    }
}

// ============================================================
// loadSpriteData
// ============================================================
bool Renderer::loadSpriteData(const std::string& json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (...) {
        return false;
    }

    // bgcolor
    if (j.contains("bgcolor") && j["bgcolor"].is_string()) {
        if (!hexToRGB(j["bgcolor"].get<std::string>(), bgcolor_)) {
            bgcolor_[0] = bgcolor_[1] = bgcolor_[2] = 0;
        }
    }

    if (!j.contains("sprites") || !j["sprites"].is_array()) {
        return false;
    }

    const auto& sprites_arr = j["sprites"];
    sprites_.clear();
    sprites_.reserve(sprites_arr.size());

    for (const auto& spr : sprites_arr) {
        SpriteInfo info;

        // Parse colors
        std::vector<uint8_t> color_rgb; // flat R,G,B per palette entry
        std::vector<uint8_t> color_valid;
        if (spr.contains("colors") && spr["colors"].is_array()) {
            for (const auto& c : spr["colors"]) {
                uint8_t rgb[3];
                bool valid = hexToRGB(c.get<std::string>(), rgb);
                color_rgb.push_back(rgb[0]);
                color_rgb.push_back(rgb[1]);
                color_rgb.push_back(rgb[2]);
                color_valid.push_back(valid ? 1 : 0);
            }
        }

        // Parse spritematrix
        if (spr.contains("spritematrix") && spr["spritematrix"].is_array()) {
            const auto& mat = spr["spritematrix"];
            int rows = static_cast<int>(mat.size());
            int cols = 0;
            if (rows > 0 && mat[0].is_array()) {
                cols = static_cast<int>(mat[0].size());
            }
            info.cell_h = rows;
            info.cell_w = cols;
            info.pixels.resize(rows * cols * 3, 0);
            info.mask.resize(rows * cols, false);

            for (int r = 0; r < rows; ++r) {
                if (!mat[r].is_array()) continue;
                for (int c = 0; c < cols && c < static_cast<int>(mat[r].size()); ++c) {
                    int val = mat[r][c].get<int>();
                    int idx = r * cols + c;
                    if (val >= 0
                        && val < static_cast<int>(color_valid.size())
                        && color_valid[val]
                        && (val * 3 + 2) < static_cast<int>(color_rgb.size())) {
                        info.mask[idx] = true;
                        info.pixels[idx * 3 + 0] = color_rgb[val * 3 + 0];
                        info.pixels[idx * 3 + 1] = color_rgb[val * 3 + 1];
                        info.pixels[idx * 3 + 2] = color_rgb[val * 3 + 2];
                    }
                    // val == -1 → transparent, mask stays false
                }
            }
        }

        sprites_.push_back(std::move(info));
    }

    // Determine common cell size from first sprite
    if (!sprites_.empty()) {
        cell_h_ = sprites_[0].cell_h;
        cell_w_ = sprites_[0].cell_w;
    }

    return true;
}

bool Renderer::loadRenderConfig(const std::string& json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (...) {
        return false;
    }

    player_mask_words_.clear();
    player_mask_aggregate_ = false;
    flickscreen_.reset();
    zoomscreen_.reset();
    has_old_bounds_ = false;

    if (j.contains("playerMask") && j["playerMask"].is_array() && j["playerMask"].size() == 2) {
        player_mask_aggregate_ = j["playerMask"][0].get<bool>();
        if (j["playerMask"][1].is_array()) {
            for (const auto& word : j["playerMask"][1]) {
                player_mask_words_.push_back(word.get<int32_t>());
            }
        }
    }

    if (j.contains("metadata") && j["metadata"].is_object()) {
        const auto& metadata = j["metadata"];
        if (metadata.contains("flickscreen") && metadata["flickscreen"].is_array() && metadata["flickscreen"].size() == 2) {
            flickscreen_ = std::make_pair(metadata["flickscreen"][0].get<int>(), metadata["flickscreen"][1].get<int>());
        }
        if (metadata.contains("zoomscreen") && metadata["zoomscreen"].is_array() && metadata["zoomscreen"].size() == 2) {
            zoomscreen_ = std::make_pair(metadata["zoomscreen"][0].get<int>(), metadata["zoomscreen"][1].get<int>());
        }
    }

    return true;
}

void Renderer::resetViewport(int level_width, int level_height) {
    if (flickscreen_.has_value()) {
        old_bounds_ = {
            0,
            0,
            std::min(flickscreen_->first, level_width),
            std::min(flickscreen_->second, level_height),
        };
        has_old_bounds_ = true;
        return;
    }
    if (zoomscreen_.has_value()) {
        old_bounds_ = {
            0,
            0,
            std::min(zoomscreen_->first, level_width),
            std::min(zoomscreen_->second, level_height),
        };
        has_old_bounds_ = true;
        return;
    }
    old_bounds_ = {0, 0, level_width, level_height};
    has_old_bounds_ = false;
}

std::array<int, 4> Renderer::getVisibleBounds(
    const int32_t* objects, int width, int height, int n_objs) const
{
    if (!flickscreen_.has_value() && !zoomscreen_.has_value()) {
        return {0, 0, width, height};
    }

    const int stride_obj = (n_objs + 31) / 32;
    int player_position = -1;
    for (int x = 0; x < width && player_position < 0; ++x) {
        for (int y = 0; y < height; ++y) {
            const int flat_idx = (x * height + y) * stride_obj;
            bool matches = player_mask_aggregate_;
            if (player_mask_aggregate_) {
                for (int word = 0; word < static_cast<int>(player_mask_words_.size()); ++word) {
                    if ((player_mask_words_[word] & objects[flat_idx + word]) != player_mask_words_[word]) {
                        matches = false;
                        break;
                    }
                }
            } else {
                matches = false;
                for (int word = 0; word < static_cast<int>(player_mask_words_.size()); ++word) {
                    if (player_mask_words_[word] & objects[flat_idx + word]) {
                        matches = true;
                        break;
                    }
                }
            }
            if (matches) {
                player_position = y + x * height;
                break;
            }
        }
    }

    if (player_position >= 0) {
        const int px = player_position / height;
        const int py = player_position % height;
        if (flickscreen_.has_value()) {
            const int screen_width = flickscreen_->first;
            const int screen_height = flickscreen_->second;
            const int mini = (px / screen_width) * screen_width;
            const int minj = (py / screen_height) * screen_height;
            old_bounds_ = {
                mini,
                minj,
                std::min(mini + screen_width, width),
                std::min(minj + screen_height, height),
            };
        } else {
            const int screen_width = zoomscreen_->first;
            const int screen_height = zoomscreen_->second;
            const int mini = std::max(std::min(px - screen_width / 2, width - screen_width), 0);
            const int minj = std::max(std::min(py - screen_height / 2, height - screen_height), 0);
            old_bounds_ = {
                mini,
                minj,
                std::min(mini + screen_width, width),
                std::min(minj + screen_height, height),
            };
        }
        has_old_bounds_ = true;
        return old_bounds_;
    }

    if (has_old_bounds_) {
        return old_bounds_;
    }
    return {0, 0, width, height};
}

// ============================================================
// renderFromObjects  (raw column-major objects array from Engine)
// ============================================================
std::vector<uint8_t> Renderer::renderFromObjects(
    const int32_t* objects, int width, int height, int n_objs) const
{
    const auto bounds = getVisibleBounds(objects, width, height, n_objs);
    const int mini = bounds[0];
    const int minj = bounds[1];
    const int maxi = bounds[2];
    const int maxj = bounds[3];
    const int view_w = std::max(maxi - mini, 0);
    const int view_h = std::max(maxj - minj, 0);
    int frame_h = view_h * cell_h_;
    int frame_w = view_w * cell_w_;
    std::vector<uint8_t> frame(frame_h * frame_w * 3);

    // Fill with bgcolor
    for (int i = 0; i < frame_h * frame_w; ++i) {
        frame[i * 3 + 0] = bgcolor_[0];
        frame[i * 3 + 1] = bgcolor_[1];
        frame[i * 3 + 2] = bgcolor_[2];
    }

    int stride_obj = (n_objs + 31) / 32;

    for (int x = mini; x < maxi; ++x) {
        for (int y = minj; y < maxj; ++y) {
            int flat_idx = (x * height + y) * stride_obj;

            // Gather which objects are present at this cell
            for (int obj_id = 0; obj_id < n_objs && obj_id < static_cast<int>(sprites_.size()); ++obj_id) {
                int word = obj_id / 32;
                int bit  = obj_id % 32;
                if (word >= stride_obj) continue;
                if (!(objects[flat_idx + word] & (1 << bit))) continue;

                const SpriteInfo& spr = sprites_[obj_id];
                int px = (x - mini) * cell_w_;
                int py = (y - minj) * cell_h_;

                for (int sr = 0; sr < spr.cell_h && (py + sr) < frame_h; ++sr) {
                    for (int sc = 0; sc < spr.cell_w && (px + sc) < frame_w; ++sc) {
                        int si = sr * spr.cell_w + sc;
                        if (!spr.mask[si]) continue;
                        int fi = ((py + sr) * frame_w + (px + sc)) * 3;
                        frame[fi + 0] = spr.pixels[si * 3 + 0];
                        frame[fi + 1] = spr.pixels[si * 3 + 1];
                        frame[fi + 2] = spr.pixels[si * 3 + 2];
                    }
                }
            }
        }
    }

    return frame;
}

// ============================================================
// renderFromObs  (multihot uint8 observation, shape n_objs×H×W)
// ============================================================
std::vector<uint8_t> Renderer::renderFromObs(
    const uint8_t* obs, int n_objs, int height, int width) const
{
    int frame_h = height * cell_h_;
    int frame_w = width  * cell_w_;
    std::vector<uint8_t> frame(frame_h * frame_w * 3);

    // Fill with bgcolor
    for (int i = 0; i < frame_h * frame_w; ++i) {
        frame[i * 3 + 0] = bgcolor_[0];
        frame[i * 3 + 1] = bgcolor_[1];
        frame[i * 3 + 2] = bgcolor_[2];
    }

    for (int obj_id = 0; obj_id < n_objs && obj_id < static_cast<int>(sprites_.size()); ++obj_id) {
        const SpriteInfo& spr = sprites_[obj_id];
        int obs_plane_offset = obj_id * height * width;  // obs[obj_id][y][x]

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (!obs[obs_plane_offset + y * width + x]) continue;

                int px = x * cell_w_;
                int py = y * cell_h_;

                for (int sr = 0; sr < spr.cell_h && (py + sr) < frame_h; ++sr) {
                    for (int sc = 0; sc < spr.cell_w && (px + sc) < frame_w; ++sc) {
                        int si = sr * spr.cell_w + sc;
                        if (!spr.mask[si]) continue;
                        int fi = ((py + sr) * frame_w + (px + sc)) * 3;
                        frame[fi + 0] = spr.pixels[si * 3 + 0];
                        frame[fi + 1] = spr.pixels[si * 3 + 1];
                        frame[fi + 2] = spr.pixels[si * 3 + 2];
                    }
                }
            }
        }
    }

    return frame;
}

std::pair<int, int> Renderer::getRenderGridSize(
    const int32_t* objects, int width, int height, int n_objs) const
{
    const auto bounds = getVisibleBounds(objects, width, height, n_objs);
    return {
        std::max(bounds[2] - bounds[0], 0),
        std::max(bounds[3] - bounds[1], 0),
    };
}
