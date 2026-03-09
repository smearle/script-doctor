#include "engine.h"

#include <algorithm>

int Engine::manhattanDistance(int tileIndex1, int tileIndex2) const {
    const int x1 = tileIndex1 / level_.height;
    const int y1 = tileIndex1 % level_.height;
    const int x2 = tileIndex2 / level_.height;
    const int y2 = tileIndex2 % level_.height;
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}

double Engine::getScore() const {
    double score = 0.0;
    const int maxDistance = level_.width + level_.height;

    for (const auto& wc : winconditions_) {
        if (wc.num == -1) {
            for (int i = 0; i < level_.n_tiles; ++i) {
                const bool f1 = cellMatchesWinMask(wc, wc.mask1, wc.aggr1, false, i);
                const bool f2 = cellMatchesWinMask(wc, wc.mask2, wc.aggr2, wc.mask2_is_all, i);
                if (f1 && f2) {
                    score += 1.0;
                }
            }
            continue;
        }

        if (wc.num == 0) {
            int globalMinDistance = maxDistance;
            for (int i = 0; i < level_.n_tiles; ++i) {
                if (!cellMatchesWinMask(wc, wc.mask1, wc.aggr1, false, i)) {
                    continue;
                }
                for (int j = 0; j < level_.n_tiles; ++j) {
                    if (!cellMatchesWinMask(wc, wc.mask2, wc.aggr2, wc.mask2_is_all, j)) {
                        continue;
                    }
                    globalMinDistance = std::min(globalMinDistance, manhattanDistance(i, j));
                }
            }
            score += globalMinDistance;
            continue;
        }

        for (int i = 0; i < level_.n_tiles; ++i) {
            if (!cellMatchesWinMask(wc, wc.mask1, wc.aggr1, false, i)) {
                continue;
            }
            int minDistance = maxDistance;
            for (int j = 0; j < level_.n_tiles; ++j) {
                if (!cellMatchesWinMask(wc, wc.mask2, wc.aggr2, wc.mask2_is_all, j)) {
                    continue;
                }
                minDistance = std::min(minDistance, manhattanDistance(i, j));
            }
            score += minDistance;
        }
    }

    return score;
}

double Engine::getScoreNormalized() const {
    double score = 0.0;
    double normalValue = 0.0;
    const int maxDistance = level_.width + level_.height;

    for (const auto& wc : winconditions_) {
        if (wc.num == -1) {
            for (int i = 0; i < level_.n_tiles; ++i) {
                const bool f1 = cellMatchesWinMask(wc, wc.mask1, wc.aggr1, false, i);
                const bool f2 = cellMatchesWinMask(wc, wc.mask2, wc.aggr2, wc.mask2_is_all, i);
                if (f1 && f2) {
                    score += 1.0;
                    normalValue += maxDistance;
                }
            }
            continue;
        }

        if (wc.num == 0) {
            int globalMinDistance = maxDistance;
            for (int i = 0; i < level_.n_tiles; ++i) {
                if (!cellMatchesWinMask(wc, wc.mask1, wc.aggr1, false, i)) {
                    continue;
                }
                for (int j = 0; j < level_.n_tiles; ++j) {
                    if (!cellMatchesWinMask(wc, wc.mask2, wc.aggr2, wc.mask2_is_all, j)) {
                        continue;
                    }
                    globalMinDistance = std::min(globalMinDistance, manhattanDistance(i, j));
                }
            }
            score += globalMinDistance;
            normalValue += maxDistance;
            continue;
        }

        for (int i = 0; i < level_.n_tiles; ++i) {
            if (!cellMatchesWinMask(wc, wc.mask1, wc.aggr1, false, i)) {
                continue;
            }
            int minDistance = maxDistance;
            for (int j = 0; j < level_.n_tiles; ++j) {
                if (!cellMatchesWinMask(wc, wc.mask2, wc.aggr2, wc.mask2_is_all, j)) {
                    continue;
                }
                minDistance = std::min(minDistance, manhattanDistance(i, j));
            }
            score += minDistance;
            normalValue += maxDistance;
        }
    }

    if (normalValue <= 0.0) {
        return 0.0;
    }
    return 1.0 - (score / normalValue);
}
