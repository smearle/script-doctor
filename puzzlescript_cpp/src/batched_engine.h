#pragma once
#include "engine.h"
#include <vector>
#include <cstdint>
#include <string>

/**
 * BatchedEngine: manages N independent Engine instances for vectorized
 * gym-style reset/step.  All instances share the same compiled game JSON.
 *
 * Observation convention:
 *   multihot uint8 array of shape (batch, n_objs, height, width)
 *   where obs[b][obj][y][x] = 1 iff object `obj` is present at (x, y).
 *
 * Action convention:  0=up, 1=left, 2=down, 3=right, 4=action
 */
class BatchedEngine {
public:
    explicit BatchedEngine(int batch_size);
    ~BatchedEngine() = default;

    // ---- Setup -------------------------------------------------------

    /// Load the compiled game JSON into all engines.
    bool loadFromJSON(const std::string& json_str);

    /// Assign level indices per environment. Length must equal batch_size.
    void setLevels(const std::vector<int>& level_indices);

    // ---- Gym interface -----------------------------------------------

    /// Reset specific environments (by index) to their assigned levels.
    /// If env_indices is empty, resets ALL environments.
    void reset(const std::vector<int>& env_indices);

    /// Reset ALL environments.
    void resetAll();

    /// Step all environments with the given actions.
    /// actions.size() must equal batch_size.
    /// Fills the internal result buffers (obs, rewards, dones, wins).
    /// Environments that are done are auto-reset.
    void step(const std::vector<int>& actions);

    /// Control whether terminal environments are automatically reset on step.
    void setAutoReset(bool auto_reset) { auto_reset_ = auto_reset; }
    bool autoReset() const { return auto_reset_; }

    // ---- Observation access ------------------------------------------

    /// Return the current observations as flat uint8 buffer.
    /// Shape: (batch, n_objs, height, width), row-major.
    const std::vector<uint8_t>& getObs() const { return obs_; }

    /// Return shape tuple (batch, n_objs, height, width).
    std::vector<int> getObsShape() const;

    // ---- Result buffers (valid after step) ---------------------------

    const std::vector<float>& getRewards() const  { return rewards_; }
    const std::vector<float>& getScores() const   { return scores_; }
    const std::vector<float>& getScoreDeltas() const { return score_deltas_; }
    const std::vector<bool>&  getDones()   const  { return dones_; }
    const std::vector<bool>&  getWins()    const  { return wins_; }

    // ---- Queries -----------------------------------------------------

    int batchSize()   const { return batch_size_; }
    int numObjects()  const { return n_objs_; }
    int levelWidth()  const { return width_; }
    int levelHeight() const { return height_; }
    int numLevels()   const;

    /// Get the objects array for a single environment (for rendering etc.)
    const std::vector<int32_t>& getObjects(int env_idx) const;

    /// Get width/height for a specific environment (can differ if levels
    /// have different sizes, but typically the same).
    int getWidth(int env_idx) const;
    int getHeight(int env_idx) const;

private:
    int batch_size_;
    std::string json_str_;
    std::vector<Engine> engines_;
    std::vector<int> level_indices_;

    // Cached geometry (from first environment after first reset)
    int n_objs_  = 0;
    int width_   = 0;
    int height_  = 0;
    int stride_obj_ = 0;

    // Result buffers
    std::vector<uint8_t> obs_;
    std::vector<float>   rewards_;
    std::vector<float>   scores_;
    std::vector<float>   score_deltas_;
    std::vector<bool>    dones_;
    std::vector<bool>    wins_;

    // Per-env state for reward computation
    std::vector<bool> prev_winning_;
    std::vector<float> prev_scores_;
    bool auto_reset_ = true;

    void fillObs(int env_idx);
    void fillObsAll();
    void resetEnv(int env_idx);
};
