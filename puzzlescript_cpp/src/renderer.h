#pragma once
#include <cstdint>
#include <string>
#include <vector>

/**
 * SpriteInfo: per-object sprite definition.
 *   pixels: RGB pixels in row-major order, shape (cell_h, cell_w, 3).
 *   mask:   alpha mask, true where sprite has color (not transparent).
 */
struct SpriteInfo {
    int cell_h = 0;
    int cell_w = 0;
    std::vector<uint8_t> pixels;  // (cell_h * cell_w * 3)
    std::vector<bool>    mask;    // (cell_h * cell_w)
};

/**
 * Renderer: loads sprite data from JSON produced by
 * ``serializeSpriteDataJSON()`` and renders RGB frames from the engine's
 * raw objects array or from multihot observations.
 */
class Renderer {
public:
    Renderer() = default;

    /// Load sprite data from JSON string (from serializeSpriteDataJSON).
    bool loadSpriteData(const std::string& json_str);

    /// Is sprite data loaded?
    bool ready() const { return !sprites_.empty(); }

    /// Cell size (pixels per tile side).
    int cellWidth()  const { return cell_w_; }
    int cellHeight() const { return cell_h_; }

    /// Number of sprites loaded (should match objectCount).
    int numSprites() const { return static_cast<int>(sprites_.size()); }

    // -----------------------------------------------------------------
    // Render from raw objects array (column-major, as stored in Engine).
    // Returns RGB uint8 buffer of shape (height * cell_h, width * cell_w, 3).
    // -----------------------------------------------------------------
    std::vector<uint8_t> renderFromObjects(
        const int32_t* objects, int width, int height, int n_objs) const;

    // -----------------------------------------------------------------
    // Render from multihot observation (n_objs, H, W) row-major uint8.
    // Returns RGB uint8 buffer of shape (H * cell_h, W * cell_w, 3).
    // -----------------------------------------------------------------
    std::vector<uint8_t> renderFromObs(
        const uint8_t* obs, int n_objs, int height, int width) const;

private:
    std::vector<SpriteInfo> sprites_;
    uint8_t bgcolor_[3] = {0, 0, 0};
    int cell_w_ = 5;
    int cell_h_ = 5;

    static uint8_t hexCharVal(char c);
    static bool hexToRGB(const std::string& hex, uint8_t out[3]);
};
