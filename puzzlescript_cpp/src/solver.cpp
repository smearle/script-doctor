#include "solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <unordered_map>

namespace {

using Clock = std::chrono::steady_clock;
using StateVec = std::vector<int32_t>;

struct StateVecHash {
    std::size_t operator()(const StateVec& state) const noexcept {
        std::size_t seed = state.size();
        for (int32_t value : state) {
            seed ^= static_cast<std::size_t>(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct ParentInfo {
    StateVec parent;
    int action = -1;
};

struct PriorityNode {
    double priority = 0.0;
    LevelBackup state;
    int numSteps = 0;
    double heuristic = 0.0;
};

struct AStarComparator {
    bool operator()(const PriorityNode& a, const PriorityNode& b) const {
        if (a.priority != b.priority) {
            return a.priority > b.priority;
        }
        return a.numSteps < b.numSteps;
    }
};

struct GBFSComparator {
    bool operator()(const PriorityNode& a, const PriorityNode& b) const {
        if (a.heuristic != b.heuristic) {
            return a.heuristic > b.heuristic;
        }
        return a.numSteps > b.numSteps;
    }
};

struct MCTSNode {
    int parent = -1;
    int action = -1;
    std::vector<int> children;
    int visits = 0;
    double score = 0.0;
};

double elapsedSeconds(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool timedOut(const Clock::time_point& start, int timeoutMs) {
    if (timeoutMs <= 0) {
        return false;
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count() > timeoutMs;
}

int actionCount(const Engine& engine) {
    return engine.hasMetadata("noaction") ? 4 : 5;
}

std::vector<int> actionsForEngine(const Engine& engine) {
    std::vector<int> actions(actionCount(engine));
    for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
        actions[i] = i;
    }
    return actions;
}

bool processInputSearch(Engine& engine, int action) {
    bool changed = engine.processInput(action);
    while (engine.isAgaining()) {
        changed = engine.processInput(-1) || changed;
    }
    return changed;
}

SolverResult makeSolverResult(
    Engine& engine,
    bool won,
    std::vector<int> actions,
    int iterations,
    const Clock::time_point& start,
    double score,
    const LevelBackup& state,
    bool timeout
) {
    SolverResult result;
    result.won = won;
    result.actions = std::move(actions);
    result.iterations = iterations;
    result.time = elapsedSeconds(start);
    result.score = score;
    result.state = state;
    result.timeout = timeout;
    result.idDict = engine.getIdDict();
    return result;
}

std::vector<int> reconstructSolution(
    const StateVec& finalState,
    const std::unordered_map<StateVec, ParentInfo, StateVecHash>& parents
) {
    std::vector<int> actions;
    StateVec current = finalState;
    while (true) {
        auto it = parents.find(current);
        if (it == parents.end() || it->second.action == -1) {
            break;
        }
        actions.push_back(it->second.action);
        current = it->second.parent;
    }
    std::reverse(actions.begin(), actions.end());
    return actions;
}

int selectMCTSChild(const std::vector<MCTSNode>& nodes, int nodeIndex, double explorationConstant) {
    const MCTSNode& node = nodes[nodeIndex];
    double bestScore = -std::numeric_limits<double>::infinity();
    int bestChild = -1;
    for (int childIndex : node.children) {
        if (childIndex < 0) {
            return -1;
        }
        const MCTSNode& child = nodes[childIndex];
        if (child.visits == 0) {
            return childIndex;
        }
        const double exploit = child.score / static_cast<double>(child.visits);
        const double explore = explorationConstant * std::sqrt(
            std::log(std::max(1, node.visits)) / static_cast<double>(child.visits)
        );
        const double ucb = exploit + explore;
        if (ucb > bestScore) {
            bestScore = ucb;
            bestChild = childIndex;
        }
    }
    return bestChild;
}

bool mctsFullyExpanded(const MCTSNode& node) {
    return std::all_of(node.children.begin(), node.children.end(), [](int child) { return child >= 0; });
}

int expandMCTSNode(std::vector<MCTSNode>& nodes, int nodeIndex) {
    const int actionSpace = static_cast<int>(nodes[nodeIndex].children.size());
    for (int action = 0; action < actionSpace; ++action) {
        if (nodes[nodeIndex].children[action] >= 0) {
            continue;
        }
        MCTSNode child;
        child.parent = nodeIndex;
        child.action = action;
        child.children.assign(actionSpace, -1);
        nodes.push_back(std::move(child));
        nodes[nodeIndex].children[action] = static_cast<int>(nodes.size()) - 1;
        return nodes[nodeIndex].children[action];
    }
    return -1;
}

void backupMCTSValue(std::vector<MCTSNode>& nodes, int nodeIndex, double value) {
    int current = nodeIndex;
    while (current >= 0) {
        nodes[current].score += value;
        nodes[current].visits += 1;
        current = nodes[current].parent;
    }
}

std::vector<int> extractMCTSActions(const std::vector<MCTSNode>& nodes, const MCTSOptions& options) {
    std::vector<int> actions;
    int current = 0;
    while (mctsFullyExpanded(nodes[current])) {
        int next = -1;
        if (options.mostVisited) {
            int bestVisits = -1;
            for (int childIndex : nodes[current].children) {
                if (childIndex < 0) {
                    continue;
                }
                if (nodes[childIndex].visits > bestVisits) {
                    bestVisits = nodes[childIndex].visits;
                    next = childIndex;
                }
            }
        } else {
            double bestValue = -std::numeric_limits<double>::infinity();
            for (int childIndex : nodes[current].children) {
                if (childIndex < 0 || nodes[childIndex].visits == 0) {
                    continue;
                }
                const double value = nodes[childIndex].score / static_cast<double>(nodes[childIndex].visits);
                if (value > bestValue) {
                    bestValue = value;
                    next = childIndex;
                }
            }
        }
        if (next < 0) {
            break;
        }
        actions.push_back(nodes[next].action);
        current = next;
    }
    return actions;
}

double simulateMCTS(
    Engine& engine,
    std::mt19937& rng,
    int maxSimLength,
    bool useScore,
    double winBonus
) {
    if (maxSimLength <= 0) {
        return useScore ? engine.getScoreNormalized() : 0.0;
    }

    std::uniform_int_distribution<int> actionDist(0, actionCount(engine) - 1);
    int changes = 0;
    for (int i = 0; i < maxSimLength; ++i) {
        const bool changed = processInputSearch(engine, actionDist(rng));
        if (changed) {
            changes += 1;
        }
        if (engine.isWinning()) {
            return winBonus;
        }
    }
    if (useScore) {
        return engine.getScoreNormalized();
    }
    return static_cast<double>(changes) / static_cast<double>(maxSimLength);
}

}  // namespace

RandomRolloutResult randomRolloutRaw(Engine& engine, int maxIters, int timeoutMs) {
    RandomRolloutResult result;
    const auto start = Clock::now();
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> actionDist(0, actionCount(engine) - 1);

    for (int i = 0; i < maxIters; ++i) {
        if (i % 1000 == 0 && timedOut(start, timeoutMs)) {
            result.iterations = i;
            result.time = elapsedSeconds(start);
            result.timeout = true;
            return result;
        }

        processInputSearch(engine, actionDist(rng));
        if (engine.isWinning()) {
            engine.restart();
        }
        result.iterations = i + 1;
    }

    result.time = elapsedSeconds(start);
    return result;
}

SolverResult solveRandom(Engine& engine, int maxLength, int maxIters, int timeoutMs) {
    const auto start = Clock::now();
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> actionDist(0, actionCount(engine) - 1);

    std::vector<int> solution;
    double bestScore = engine.getScore();
    LevelBackup bestState = engine.backupLevel();

    for (int i = 0; i < maxIters; ++i) {
        if (timedOut(start, timeoutMs)) {
            return makeSolverResult(engine, false, {}, i, start, bestScore, bestState, true);
        }

        if (maxLength > 0 && i % maxLength == 0) {
            engine.restart();
            solution.clear();
        }

        const int action = actionDist(rng);
        solution.push_back(action);
        const bool changed = processInputSearch(engine, action);
        if (!changed) {
            continue;
        }

        const double score = engine.getScore();
        if (score <= bestScore) {
            bestScore = score;
            bestState = engine.backupLevel();
        }
        if (engine.isWinning()) {
            return makeSolverResult(engine, true, solution, i, start, score, engine.backupLevel(), false);
        }
    }

    return makeSolverResult(engine, false, {}, maxIters, start, bestScore, bestState, false);
}

SolverResult solveBFS(Engine& engine, int maxIters, int timeoutMs) {
    const auto start = Clock::now();
    const LevelBackup initialState = engine.backupLevel();
    std::queue<LevelBackup> frontier;
    std::unordered_map<StateVec, ParentInfo, StateVecHash> parents;
    frontier.push(initialState);
    parents.emplace(initialState.dat, ParentInfo{initialState.dat, -1});

    std::vector<int> bestActions;
    LevelBackup bestState = initialState;
    double bestScore = engine.getScore();
    int iterations = 0;

    while (!frontier.empty() && iterations < maxIters) {
        if (iterations % 1000 == 0 && timedOut(start, timeoutMs)) {
            return makeSolverResult(engine, false, bestActions, iterations, start, bestScore, bestState, true);
        }

        const LevelBackup parentState = frontier.front();
        frontier.pop();

        for (int action : actionsForEngine(engine)) {
            engine.restoreLevel(parentState);
            const bool changed = processInputSearch(engine, action);
            if (!changed) {
                continue;
            }

            LevelBackup nextState = engine.backupLevel();
            if (parents.find(nextState.dat) != parents.end()) {
                continue;
            }

            parents.emplace(nextState.dat, ParentInfo{parentState.dat, action});
            const std::vector<int> currentActions = reconstructSolution(nextState.dat, parents);
            const double score = engine.getScore();

            if (engine.isWinning()) {
                return makeSolverResult(engine, true, currentActions, iterations, start, score, nextState, false);
            }

            if (score < bestScore || (score == bestScore && currentActions.size() > bestActions.size())) {
                bestScore = score;
                bestState = nextState;
                bestActions = currentActions;
            }

            frontier.push(nextState);
        }

        iterations += 1;
    }

    return makeSolverResult(engine, false, bestActions, iterations, start, bestScore, bestState, false);
}

SolverResult solveAStar(Engine& engine, int maxIters, int timeoutMs) {
    const auto start = Clock::now();
    const LevelBackup initialState = engine.backupLevel();
    std::priority_queue<PriorityNode, std::vector<PriorityNode>, AStarComparator> frontier;
    std::unordered_map<StateVec, ParentInfo, StateVecHash> parents;
    parents.emplace(initialState.dat, ParentInfo{initialState.dat, -1});
    frontier.push(PriorityNode{0.0, initialState, 0, 0.0});

    LevelBackup bestState = initialState;
    double bestScore = engine.getScore();
    int totalIters = 0;
    std::mt19937 rng(std::random_device{}());

    while (!frontier.empty() && totalIters < maxIters) {
        if (totalIters % 1000 == 0 && timedOut(start, timeoutMs)) {
            return makeSolverResult(
                engine,
                false,
                reconstructSolution(bestState.dat, parents),
                totalIters,
                start,
                bestScore,
                bestState,
                true
            );
        }

        PriorityNode current = frontier.top();
        frontier.pop();
        std::vector<int> actions = actionsForEngine(engine);
        std::shuffle(actions.begin(), actions.end(), rng);

        for (int action : actions) {
            engine.restoreLevel(current.state);
            const bool changed = processInputSearch(engine, action);
            if (!changed) {
                continue;
            }

            LevelBackup nextState = engine.backupLevel();
            if (parents.find(nextState.dat) != parents.end()) {
                continue;
            }

            const double score = engine.getScore();
            if (score < bestScore) {
                bestScore = score;
                bestState = nextState;
            }

            parents.emplace(nextState.dat, ParentInfo{current.state.dat, action});
            if (engine.isWinning()) {
                return makeSolverResult(
                    engine,
                    true,
                    reconstructSolution(nextState.dat, parents),
                    totalIters,
                    start,
                    score,
                    nextState,
                    false
                );
            }

            frontier.push(PriorityNode{score + current.numSteps + 1.0, nextState, current.numSteps + 1, score});
        }

        totalIters += 1;
    }

    return makeSolverResult(
        engine,
        false,
        reconstructSolution(bestState.dat, parents),
        totalIters,
        start,
        bestScore,
        bestState,
        false
    );
}

SolverResult solveGBFS(Engine& engine, int maxIters, int timeoutMs) {
    const auto start = Clock::now();
    const LevelBackup initialState = engine.backupLevel();
    std::priority_queue<PriorityNode, std::vector<PriorityNode>, GBFSComparator> frontier;
    std::unordered_map<StateVec, ParentInfo, StateVecHash> parents;
    parents.emplace(initialState.dat, ParentInfo{initialState.dat, -1});
    frontier.push(PriorityNode{0.0, initialState, 0, 0.0});

    LevelBackup bestState = initialState;
    double bestScore = engine.getScore();
    int totalIters = 0;
    std::mt19937 rng(std::random_device{}());

    while (!frontier.empty() && totalIters < maxIters) {
        if (totalIters % 1000 == 0 && timedOut(start, timeoutMs)) {
            return makeSolverResult(
                engine,
                false,
                reconstructSolution(bestState.dat, parents),
                totalIters,
                start,
                bestScore,
                bestState,
                true
            );
        }

        PriorityNode current = frontier.top();
        frontier.pop();
        std::vector<int> actions = actionsForEngine(engine);
        std::shuffle(actions.begin(), actions.end(), rng);

        for (int action : actions) {
            engine.restoreLevel(current.state);
            const bool changed = processInputSearch(engine, action);
            if (!changed) {
                continue;
            }

            LevelBackup nextState = engine.backupLevel();
            if (parents.find(nextState.dat) != parents.end()) {
                continue;
            }

            const double score = engine.getScore();
            if (score < bestScore) {
                bestScore = score;
                bestState = nextState;
            }

            parents.emplace(nextState.dat, ParentInfo{current.state.dat, action});
            if (engine.isWinning()) {
                return makeSolverResult(
                    engine,
                    true,
                    reconstructSolution(nextState.dat, parents),
                    totalIters,
                    start,
                    score,
                    nextState,
                    false
                );
            }

            frontier.push(PriorityNode{score, nextState, current.numSteps + 1, score});
        }

        totalIters += 1;
    }

    return makeSolverResult(
        engine,
        false,
        reconstructSolution(bestState.dat, parents),
        totalIters,
        start,
        bestScore,
        bestState,
        false
    );
}

SolverResult solveMCTS(Engine& engine, const MCTSOptions& options) {
    const auto start = Clock::now();
    const LevelBackup initialState = engine.backupLevel();
    LevelBackup bestState = initialState;
    double bestScore = engine.getScore();

    std::vector<MCTSNode> nodes;
    MCTSNode root;
    root.children.assign(actionCount(engine), -1);
    nodes.push_back(std::move(root));

    std::mt19937 rng(std::random_device{}());

    for (int i = 0; options.maxIterations <= 0 || i < options.maxIterations; ++i) {
        int current = 0;
        engine.restoreLevel(initialState);
        bool changed = true;

        while (mctsFullyExpanded(nodes[current])) {
            const int next = selectMCTSChild(nodes, current, options.explorationConstant);
            if (next < 0) {
                break;
            }
            current = next;
            changed = processInputSearch(engine, nodes[current].action);

            const double score = engine.getScore();
            if (score < bestScore) {
                bestScore = score;
                bestState = engine.backupLevel();
            }

            if (engine.isWinning()) {
                std::vector<int> actions;
                for (int node = current; node > 0; node = nodes[node].parent) {
                    actions.push_back(nodes[node].action);
                }
                std::reverse(actions.begin(), actions.end());
                return makeSolverResult(engine, true, actions, i, start, score, engine.backupLevel(), false);
            }

            if (!options.exploreDeadends && !changed) {
                break;
            }
        }

        if (!options.exploreDeadends && !changed) {
            nodes[current].score += options.deadendBonus;
            backupMCTSValue(nodes, current, 0.0);
            continue;
        }

        const int child = expandMCTSNode(nodes, current);
        if (child < 0) {
            continue;
        }
        current = child;
        changed = processInputSearch(engine, nodes[current].action);

        if (engine.isWinning()) {
            std::vector<int> actions;
            for (int node = current; node > 0; node = nodes[node].parent) {
                actions.push_back(nodes[node].action);
            }
            std::reverse(actions.begin(), actions.end());
            return makeSolverResult(engine, true, actions, i, start, engine.getScore(), engine.backupLevel(), false);
        }

        if (!options.exploreDeadends && !changed) {
            nodes[current].score += options.deadendBonus;
            backupMCTSValue(nodes, current, 0.0);
            continue;
        }

        const double value = simulateMCTS(
            engine,
            rng,
            options.maxSimLength,
            options.useScore,
            options.winBonus
        );
        backupMCTSValue(nodes, current, value);
    }

    return makeSolverResult(
        engine,
        false,
        extractMCTSActions(nodes, options),
        options.maxIterations,
        start,
        bestScore,
        bestState,
        false
    );
}
