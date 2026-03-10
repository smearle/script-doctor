#include "renderer.h"
#include "json.hpp"

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

// ============================================================
// renderFromObjects  (raw column-major objects array from Engine)
// ============================================================
std::vector<uint8_t> Renderer::renderFromObjects(
    const int32_t* objects, int width, int height, int n_objs) const
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

    int stride_obj = (n_objs + 31) / 32;

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            int flat_idx = (x * height + y) * stride_obj;

            // Gather which objects are present at this cell
            for (int obj_id = 0; obj_id < n_objs && obj_id < static_cast<int>(sprites_.size()); ++obj_id) {
                int word = obj_id / 32;
                int bit  = obj_id % 32;
                if (word >= stride_obj) continue;
                if (!(objects[flat_idx + word] & (1 << bit))) continue;

                const SpriteInfo& spr = sprites_[obj_id];
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
