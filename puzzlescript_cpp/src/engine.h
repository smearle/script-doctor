#pragma once
#include "bitvec.h"
#include "level.h"
#include "rng.h"

#include <cstdint>
#include <cmath>
#include <map>
#include <string>
#include <vector>

// ---- CellReplacement ----
struct CellReplacement {
    BitVec objectsClear;
    BitVec objectsSet;
    BitVec movementsClear;
    BitVec movementsSet;
    BitVec movementsLayerMask;
    BitVec randomEntityMask;
    BitVec randomDirMask;
};

// ---- CellPattern ----
struct CellPattern {
    BitVec objectsPresent;
    BitVec objectsMissing;
    std::vector<BitVec> anyObjectsPresent;
    BitVec movementsPresent;
    BitVec movementsMissing;
    CellReplacement* replacement = nullptr;  // nullptr if no replacement
    bool isEllipsis = false;

    bool matches(int index, const int32_t* objects, const int32_t* movements,
                 int STRIDE_OBJ, int STRIDE_MOV) const;
};

// ---- WinCondition ----
struct WinCondition {
    int num;             // -1=NO, 0=SOME, 1=ALL
    BitVec mask1;
    BitVec mask2;
    bool mask2_is_all;   // true if mask2 was "\nall\n" in JS
    int lineNumber;
    bool aggr1;
    bool aggr2;
};

// ---- Rule ----
struct Rule {
    int direction;
    // patterns[i] = cell row; each cell row is a vector of CellPattern*
    // (ellipsis entries have isEllipsis=true)
    std::vector<std::vector<CellPattern*>> patterns;
    bool hasReplacements;
    int lineNumber;
    std::vector<int> ellipsisCount;  // per pattern row
    int groupNumber;
    bool rigid;
    std::vector<std::vector<std::string>> commands;
    bool isRandom;
    std::vector<BitVec> cellRowMasks;
    std::vector<BitVec> cellRowMasks_Movements;
    BitVec ruleMask;
};

// ---- LevelDef ----
struct LevelDef {
    bool isMessage = false;
    int index = 0;
    int lineNumber = 0;
    int width = 0;
    int height = 0;
    int layerCount = 0;
    std::vector<int32_t> objects;
};

// ---- ObjectInfo ----
struct ObjectInfo {
    int layer = 0;
};

// ---- Engine ----
class Engine {
public:
    Engine();
    ~Engine();

    // Non-copyable (owns raw pointers), but movable
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&& other) noexcept;
    Engine& operator=(Engine&& other) noexcept;

    // Load from serialized JSON (as produced by JS serializeCompiledState)
    bool loadFromJSON(const std::string& json_str);

    // Load a specific level
    void loadLevel(int levelIndex);

    // Process a single input. dir: 0=up, 1=left, 2=down, 3=right, 4=action, -1=tick
    // Returns true if anything changed.
    bool processInput(int dir);

    // Check win conditions. Returns true if won.
    bool checkWin();
    double getScore() const;
    double getScoreNormalized() const;
    bool hasMetadata(const std::string& key) const;

    // Get the current level state as a flat int32 array (objects)
    const std::vector<int32_t>& getObjects() const;
    int getWidth() const;
    int getHeight() const;
    int getObjectCount() const;

    // Is the game in "winning" state?
    bool isWinning() const { return winning_; }
    bool isAgaining() const { return againing_; }

    // Get idDict for interpreting object bits
    const std::vector<std::string>& getIdDict() const { return idDict_; }

    // Get number of levels
    int getNumLevels() const { return static_cast<int>(levels_.size()); }

    // Reset to current level's initial state
    void restart();

    // Backup/restore for undo
    LevelBackup backupLevel() const;
    void restoreLevel(const LevelBackup& bak);

private:
    // ---- State from compilation ----
    int objectCount_ = 0;
    int layerCount_ = 0;
    int STRIDE_OBJ_ = 1;
    int STRIDE_MOV_ = 1;
    bool rigid_ = false;

    std::vector<std::string> idDict_;
    BitVec playerMask_;
    bool playerMaskAggregate_ = false;
    std::vector<BitVec> layerMasks_;
    std::vector<std::vector<Rule*>> rules_;       // rule groups
    std::vector<std::vector<Rule*>> lateRules_;
    std::map<int, int> loopPoint_;
    std::map<int, int> lateLoopPoint_;
    std::vector<WinCondition> winconditions_;
    std::vector<LevelDef> levels_;

    int backgroundid_ = 0;
    int backgroundlayer_ = 0;
    std::vector<int> rigidGroupIndex_to_GroupIndex_;
    std::map<int, int> groupNumber_to_RigidGroupIndex_;

    std::map<std::string, std::string> metadata_;

    // Object info: layer per object ID
    std::vector<ObjectInfo> objectInfos_;

    // ---- Runtime state ----
    Level level_;
    bool winning_ = false;
    bool againing_ = false;
    RNG rng_;
    int curLevel_ = 0;

    // Scratch BitVecs (to avoid repeated allocation in hot paths)
    BitVec _o1, _o2, _o2_5, _o3, _o4, _o5, _o6, _o7, _o8;
    BitVec _m1, _m2, _m3;

    BitVec sfxCreateMask_;
    BitVec sfxDestroyMask_;

    // Ownership of all allocated CellPatterns and CellReplacements
    std::vector<CellPattern*> allCellPatterns_;
    std::vector<CellReplacement*> allCellReplacements_;
    std::vector<Rule*> allRules_;

    // ---- Internal engine functions ----
    void startMovement(int dirMask);
    std::vector<int> getPlayerPositions() const;
    void moveEntitiesAtIndex(int positionIndex, const BitVec& entityMask, int dirMask);
    bool repositionEntitiesOnLayer(int positionIndex, int layer, int dirMask);
    bool repositionEntitiesAtCell(int positionIndex);
    bool resolveMovements(std::vector<bool>& bannedGroup);
    void applyRules(const std::vector<std::vector<Rule*>>& rules,
                    const std::map<int, int>& loopPoint,
                    std::vector<bool>* bannedGroup);
    bool applyRuleGroup(std::vector<Rule*>& ruleGroup);
    bool applyRandomRuleGroup(std::vector<Rule*>& ruleGroup);
    bool ruleTryApply(Rule& rule);
    // Each row returns a list of matches, where each match is {startIdx} or {startIdx, k} or {startIdx, k1, k2}
    std::vector<std::vector<std::vector<int>>> ruleFindMatches(Rule& rule);
    bool ruleApplyAt(Rule& rule, const std::vector<std::vector<int>>& tuple, bool check, int delta);
    void ruleQueueCommands(Rule& rule);
    bool cellPatternReplace(CellPattern& cp, Rule& rule, int currentIndex);

    // Match functions for cell rows
    bool matchCellRowNoEllipsis(const std::vector<CellPattern*>& cellRow, int startIdx, int delta,
                                std::vector<std::vector<int>>& result);
    void matchCellRowEllipsis1(const std::vector<CellPattern*>& cellRow, int startIdx,
                               int kmax, int kmin, int delta,
                               std::vector<std::vector<int>>& result);
    void matchCellRowEllipsis2(const std::vector<CellPattern*>& cellRow, int startIdx,
                               int kmax, int kmin,
                               int k1max, int k1min,
                               int k2max, int k2min,
                               int delta,
                               std::vector<std::vector<int>>& result);

    // Cartesian product of match lists
    std::vector<std::vector<std::vector<int>>> generateTuples(const std::vector<std::vector<std::vector<int>>>& lists);

    void initScratchVecs();
    void clearEngine();
    bool cellMatchesWinMask(const WinCondition& wc, const BitVec& mask, bool aggregate,
                            bool mask_is_all, int tileIndex) const;
    int manhattanDistance(int tileIndex1, int tileIndex2) const;
};
