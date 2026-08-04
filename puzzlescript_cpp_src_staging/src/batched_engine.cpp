#include "batched_engine.h"
#include "viewport.h"
#include "json.hpp"
#include <stdexcept>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

using json = nlohmann::json;

// ============================================================
// Construction
// ============================================================
BatchedEngine::BatchedEngine(int batch_size)
    : batch_size_(batch_size),
      engines_(batch_size),
      level_indices_(batch_size, 0),
      rewards_(batch_size, 0.0f),
      scores_(batch_size, 0.0f),
      score_deltas_(batch_size, 0.0f),
      dones_(batch_size, false),
      wins_(batch_size, false),
      prev_winning_(batch_size, false),
      prev_scores_(batch_size, 0.0f)
{}

// ============================================================
// Setup
// ============================================================
bool BatchedEngine::loadFromJSON(const std::string& json_str) {
    json_str_ = json_str;
    for (int i = 0; i < batch_size_; ++i) {
        if (!engines_[i].loadFromJSON(json_str)) {
            return false;
        }
    }
    // Cache object count and stride from first engine
    n_objs_ = engines_[0].getObjectCount();
    return true;
}

void BatchedEngine::setLevels(const std::vector<int>& level_indices) {
    if (static_cast<int>(level_indices.size()) != batch_size_) {
        throw std::runtime_error("level_indices size must equal batch_size");
    }
    level_indices_ = level_indices;
}

void BatchedEngine::setObsShape(int height, int width) {
    if (height < 0 || width < 0) {
        throw std::runtime_error("obs shape must be non-negative");
    }
    height_ = height;
    width_ = width;
}

void BatchedEngine::setDedupMap(const std::vector<int>& raw_to_canonical, int n_canonical) {
    if (static_cast<int>(raw_to_canonical.size()) != n_objs_) {
        throw std::runtime_error("dedup map size must equal raw object count");
    }
    dedup_map_ = raw_to_canonical;
    n_canonical_ = n_canonical;
    use_dedup_ = true;
}

void BatchedEngine::loadViewportConfig(const std::string& json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (...) {
        throw std::runtime_error("Failed to parse JSON for viewport config");
    }

    player_mask_words_.clear();
    player_mask_aggregate_ = false;
    flickscreen_.reset();
    zoomscreen_.reset();

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
        if (metadata.contains("flickscreen") && metadata["flickscreen"].is_array()
            && metadata["flickscreen"].size() == 2) {
            flickscreen_ = std::make_pair(
                metadata["flickscreen"][0].get<int>(),
                metadata["flickscreen"][1].get<int>());
        }
        if (metadata.contains("zoomscreen") && metadata["zoomscreen"].is_array()
            && metadata["zoomscreen"].size() == 2) {
            zoomscreen_ = std::make_pair(
                metadata["zoomscreen"][0].get<int>(),
                metadata["zoomscreen"][1].get<int>());
        }
    }
}

void BatchedEngine::setNumThreads(int num_threads) {
    num_threads_ = num_threads;
}

int BatchedEngine::numThreads() const {
#ifdef _OPENMP
    if (num_threads_ > 0) {
        return num_threads_;
    }
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// ============================================================
// Reset
// ============================================================
void BatchedEngine::resetEnv(int env_idx) {
    engines_[env_idx].loadLevel(level_indices_[env_idx]);
    prev_scores_[env_idx] = static_cast<float>(engines_[env_idx].getScore());
    scores_[env_idx] = prev_scores_[env_idx];
    score_deltas_[env_idx] = 0.0f;
    prev_winning_[env_idx] = false;
}

void BatchedEngine::resetAll() {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (batch_size_ > 1) num_threads(numThreads())
#endif
    for (int i = 0; i < batch_size_; ++i) {
        resetEnv(i);
    }
    refreshObsGeometry();
    fillObsAll();
}

void BatchedEngine::reset(const std::vector<int>& env_indices) {
    if (env_indices.empty()) {
        resetAll();
        return;
    }
    const int num_indices = static_cast<int>(env_indices.size());
    for (int i = 0; i < num_indices; ++i) {
        const int idx = env_indices[i];
        if (idx < 0 || idx >= batch_size_) {
            throw std::runtime_error("env index out of range");
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_indices > 1) num_threads(numThreads())
#endif
    for (int i = 0; i < num_indices; ++i) {
        resetEnv(env_indices[i]);
    }

    // Resetting selected environments can change the required padding size
    // when level assignments vary across the batch.
    if (refreshObsGeometry()) {
        fillObsAll();
        return;
    }

    #ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_indices > 1) num_threads(numThreads())
    #endif
    for (int i = 0; i < num_indices; ++i) {
        fillObs(env_indices[i]);
    }
}

// ============================================================
// Step
// ============================================================
static constexpr int MAX_AGAIN = 50;

void BatchedEngine::step(const std::vector<int>& actions) {
    if (static_cast<int>(actions.size()) != batch_size_) {
        throw std::runtime_error("actions size must equal batch_size");
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (batch_size_ > 1) num_threads(numThreads())
#endif
    for (int i = 0; i < batch_size_; ++i) {
        Engine& eng = engines_[i];
        eng.processInput(actions[i]);

        // Handle 'again' ticks
        int ag = 0;
        while (eng.isAgaining() && ag < MAX_AGAIN) {
            eng.processInput(-1);
            ++ag;
        }

        bool won = eng.isWinning();
        const float score = static_cast<float>(eng.getScore());
        const float score_delta = prev_scores_[i] - score;
        wins_[i]  = won;
        dones_[i] = won;
        scores_[i] = score;
        score_deltas_[i] = score_delta;

        rewards_[i] = score_delta + (won ? 1.0f : 0.0f) - 0.01f;

        prev_scores_[i] = score;
        prev_winning_[i] = won;

        // Preserve the current fast RL path by keeping auto-reset enabled by default.
        if (auto_reset_ && dones_[i]) {
            resetEnv(i);
        }

        // Fill this env's observation slice
        fillObs(i);
    }
}

// ============================================================
// Observation helpers
// ============================================================
void BatchedEngine::fillObs(int env_idx) {
    const Engine& eng = engines_[env_idx];
    const auto& objects = eng.getObjects();
    int w = eng.getWidth();
    int h = eng.getHeight();

    // Output obs shape: (out_n_objs, out_height, out_width)
    int env_stride = out_n_objs_ * out_height_ * out_width_;
    uint8_t* dst = obs_.data() + env_idx * env_stride;
    std::memset(dst, 0, env_stride);

    int local_stride_obj = static_cast<int>(objects.size()) / (w * h);

    // Compute viewport offset
    int ox = 0, oy = 0;
    int crop_w = std::min(w, out_width_);
    int crop_h = std::min(h, out_height_);

    if (flickscreen_.has_value() || zoomscreen_.has_value()) {
        auto [px, py] = findPlayerInObjects(
            objects.data(), w, h, n_objs_,
            player_mask_words_, player_mask_aggregate_);
        auto bounds = computeScreenBounds(px, py, w, h, flickscreen_, zoomscreen_);
        ox = bounds[0];
        oy = bounds[1];
        crop_w = std::min(bounds[2] - bounds[0], out_width_);
        crop_h = std::min(bounds[3] - bounds[1], out_height_);
    }

    // Objects array is column-major: index = (x * h + y) * stride_obj
    // obs layout: obs[obj][y][x] at offset obj * (out_h * out_w) + y * out_w + x
    const int plane_stride = out_height_ * out_width_;
    for (int dx = 0; dx < crop_w; ++dx) {
        int src_x = dx + ox;
        if (src_x >= w) break;
        for (int dy = 0; dy < crop_h; ++dy) {
            int src_y = dy + oy;
            if (src_y >= h) break;
            int flat_idx = (src_x * h + src_y) * local_stride_obj;
            for (int raw_obj = 0; raw_obj < n_objs_; ++raw_obj) {
                int word = raw_obj / 32;
                int bit  = raw_obj % 32;
                if (word < local_stride_obj &&
                    (objects[flat_idx + word] & (1 << bit))) {
                    int out_obj = use_dedup_ ? dedup_map_[raw_obj] : raw_obj;
                    dst[out_obj * plane_stride + dy * out_width_ + dx] = 1;
                }
            }
        }
    }
}

void BatchedEngine::fillObsAll() {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (batch_size_ > 1) num_threads(numThreads())
#endif
    for (int i = 0; i < batch_size_; ++i) {
        fillObs(i);
    }
}

bool BatchedEngine::refreshObsGeometry() {
    int max_width = width_;
    int max_height = height_;
    for (const auto& eng : engines_) {
        max_width = std::max(max_width, eng.getWidth());
        max_height = std::max(max_height, eng.getHeight());
    }
    width_ = max_width;
    height_ = max_height;
    stride_obj_ = (n_objs_ + 31) / 32;

    // Compute output dimensions accounting for dedup and viewport
    int new_out_n_objs = use_dedup_ ? n_canonical_ : n_objs_;
    int new_out_h, new_out_w;
    if (flickscreen_.has_value()) {
        new_out_w = std::min(flickscreen_->first, max_width);
        new_out_h = std::min(flickscreen_->second, max_height);
    } else if (zoomscreen_.has_value()) {
        new_out_w = std::min(zoomscreen_->first, max_width);
        new_out_h = std::min(zoomscreen_->second, max_height);
    } else {
        new_out_w = max_width;
        new_out_h = max_height;
    }

    const size_t new_size = static_cast<size_t>(batch_size_) * new_out_n_objs * new_out_h * new_out_w;
    const bool shape_changed = (new_out_w != out_width_) || (new_out_h != out_height_)
                             || (new_out_n_objs != out_n_objs_) || (obs_.size() != new_size);
    out_n_objs_ = new_out_n_objs;
    out_width_ = new_out_w;
    out_height_ = new_out_h;
    if (shape_changed) {
        obs_.assign(new_size, 0);
    }
    return shape_changed;
}

// ============================================================
// Queries
// ============================================================
std::vector<int> BatchedEngine::getObsShape() const {
    return {batch_size_, out_n_objs_, out_height_, out_width_};
}

int BatchedEngine::numLevels() const {
    if (engines_.empty()) return 0;
    return engines_[0].getNumLevels();
}

const std::vector<int32_t>& BatchedEngine::getObjects(int env_idx) const {
    if (env_idx < 0 || env_idx >= batch_size_) {
        throw std::runtime_error("env index out of range");
    }
    return engines_[env_idx].getObjects();
}

int BatchedEngine::getWidth(int env_idx) const {
    return engines_[env_idx].getWidth();
}

int BatchedEngine::getHeight(int env_idx) const {
    return engines_[env_idx].getHeight();
}
