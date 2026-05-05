#pragma once

#include "engine.h"

#include <vector>

struct RandomRolloutResult {
    int iterations = 0;
    double time = 0.0;
    bool timeout = false;
};

struct SolverResult {
    bool won = false;
    std::vector<int> actions;
    int iterations = 0;
    double time = 0.0;
    double score = 0.0;
    LevelBackup state;
    bool timeout = false;
    std::vector<std::string> idDict;
};

struct MCTSOptions {
    int maxSimLength = 100;
    bool useScore = true;
    bool exploreDeadends = false;
    double deadendBonus = -25.0;
    double winBonus = 100.0;
    bool mostVisited = true;
    double explorationConstant = 1.4142135623730951;
    int maxIterations = 100000;
};

struct TransitionData {
    // Flat list of all unique (state, action, next_state) transitions visited.
    // States are stored as raw LevelBackup.dat vectors (bitpacked int32).
    std::vector<std::vector<int32_t>> states;
    std::vector<int> actions;
    std::vector<std::vector<int32_t>> nextStates;
    // 1 if the corresponding next_state satisfies the win conditions.
    std::vector<uint8_t> wons;
    // Per-transition list of rule globalIndex values that fired during the
    // corresponding processInput. Empty unless trackRulesFired=true was
    // passed to the collector. Each element is the sorted-ascending set of
    // fired rules for that transition.
    std::vector<std::vector<int32_t>> rulesFired;
    int width = 0;
    int height = 0;
    int iterations = 0;
    double time = 0.0;
    bool timeout = false;
    int n_rules = 0;  // total rule count (for decoding rulesFired indices)
    std::vector<std::string> idDict;
};

TransitionData collectTransitionsBFS(Engine& engine, int maxIters = 100000, int timeoutMs = -1,
                                     bool trackRulesFired = false);
TransitionData collectTransitionsAStar(Engine& engine, int maxIters = 100000, int timeoutMs = -1);

RandomRolloutResult randomRolloutRaw(Engine& engine, int maxIters = 100000, int timeoutMs = -1);
SolverResult solveRandom(Engine& engine, int maxLength = 100, int maxIters = 100000, int timeoutMs = 60000);
SolverResult solveBFS(Engine& engine, int maxIters = 100000, int timeoutMs = -1);
SolverResult solveAStar(Engine& engine, int maxIters = 100000, int timeoutMs = -1);
SolverResult solveGBFS(Engine& engine, int maxIters = 100000, int timeoutMs = -1);
SolverResult solveMCTS(Engine& engine, const MCTSOptions& options = MCTSOptions());
