#include "batched_engine.h"
#include <stdexcept>
#include <cstring>

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
    for (int i = 0; i < batch_size_; ++i) {
        resetEnv(i);
    }
    // Cache geometry from first engine (after a level is loaded)
    width_  = engines_[0].getWidth();
    height_ = engines_[0].getHeight();
    stride_obj_ = (n_objs_ + 31) / 32;

    // Allocate obs buffer
    obs_.resize(batch_size_ * n_objs_ * height_ * width_);
    fillObsAll();
}

void BatchedEngine::reset(const std::vector<int>& env_indices) {
    if (env_indices.empty()) {
        resetAll();
        return;
    }
    for (int idx : env_indices) {
        if (idx < 0 || idx >= batch_size_) {
            throw std::runtime_error("env index out of range");
        }
        resetEnv(idx);
        fillObs(idx);
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

        // Auto-reset on done
        if (dones_[i]) {
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

    // Offset into the flat obs buffer for this env
    // obs shape: (batch, n_objs, height, width) — row-major
    int env_stride = n_objs_ * height_ * width_;
    uint8_t* dst = obs_.data() + env_idx * env_stride;

    // Zero out first (handles the case where w/h differ from cached)
    std::memset(dst, 0, env_stride);

    int local_stride_obj = static_cast<int>(objects.size()) / (w * h);

    // Objects array is column-major: index = (x * h + y) * stride_obj
    // obs layout: obs[obj][y][x] at offset obj * (height_ * width_) + y * width_ + x
    for (int x = 0; x < w && x < width_; ++x) {
        for (int y = 0; y < h && y < height_; ++y) {
            int flat_idx = (x * h + y) * local_stride_obj;
            for (int obj = 0; obj < n_objs_; ++obj) {
                int word = obj / 32;
                int bit  = obj % 32;
                if (word < local_stride_obj &&
                    (objects[flat_idx + word] & (1 << bit))) {
                    dst[obj * (height_ * width_) + y * width_ + x] = 1;
                }
            }
        }
    }
}

void BatchedEngine::fillObsAll() {
    for (int i = 0; i < batch_size_; ++i) {
        fillObs(i);
    }
}

// ============================================================
// Queries
// ============================================================
std::vector<int> BatchedEngine::getObsShape() const {
    return {batch_size_, n_objs_, height_, width_};
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
