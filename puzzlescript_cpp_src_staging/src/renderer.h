#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <array>
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

    /// Load render config from compiled game JSON (player mask, screen metadata).
    bool loadRenderConfig(const std::string& json_str);

    /// Reset cached viewport state for a newly loaded level.
    void resetViewport(int level_width, int level_height);

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

    std::pair<int, int> getRenderGridSize(
        const int32_t* objects, int width, int height, int n_objs) const;

private:
    std::vector<SpriteInfo> sprites_;
    uint8_t bgcolor_[3] = {0, 0, 0};
    int cell_w_ = 5;
    int cell_h_ = 5;
    std::vector<int32_t> player_mask_words_;
    bool player_mask_aggregate_ = false;
    std::optional<std::pair<int, int>> flickscreen_;
    std::optional<std::pair<int, int>> zoomscreen_;
    mutable std::array<int, 4> old_bounds_ = {0, 0, 0, 0};
    mutable bool has_old_bounds_ = false;

    static uint8_t hexCharVal(char c);
    static bool hexToRGB(const std::string& hex, uint8_t out[3]);
    std::array<int, 4> getVisibleBounds(
        const int32_t* objects, int width, int height, int n_objs) const;
};
