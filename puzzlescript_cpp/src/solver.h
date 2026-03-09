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

RandomRolloutResult randomRolloutRaw(Engine& engine, int maxIters = 100000, int timeoutMs = -1);
SolverResult solveRandom(Engine& engine, int maxLength = 100, int maxIters = 100000, int timeoutMs = 60000);
SolverResult solveBFS(Engine& engine, int maxIters = 100000, int timeoutMs = -1);
SolverResult solveAStar(Engine& engine, int maxIters = 100000, int timeoutMs = -1);
SolverResult solveGBFS(Engine& engine, int maxIters = 100000, int timeoutMs = -1);
SolverResult solveMCTS(Engine& engine, const MCTSOptions& options = MCTSOptions());
