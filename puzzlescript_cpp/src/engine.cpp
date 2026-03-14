#include "engine.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// We use nlohmann/json for parsing. This is bundled as a single header.
#include "json.hpp"
using json = nlohmann::json;

// ============================================================
// CellPattern::matches
// ============================================================
bool CellPattern::matches(int index, const int32_t* objects, const int32_t* movements,
                          int STRIDE_OBJ, int STRIDE_MOV) const {
    int objBase = index * STRIDE_OBJ;
    int movBase = index * STRIDE_MOV;

    for (int i = 0; i < STRIDE_OBJ; ++i) {
        int co = objects[objBase + i];
        int op = objectsPresent.data[i];
        int om = objectsMissing.data[i];
        if (op && ((co & op) != op)) return false;
        if (om && (co & om)) return false;
    }

    for (int i = 0; i < STRIDE_MOV; ++i) {
        int cm = movements[movBase + i];
        int mp = movementsPresent.data[i];
        int mm = movementsMissing.data[i];
        if (mp && ((cm & mp) != mp)) return false;
        if (mm && (cm & mm)) return false;
    }

    // anyObjectsPresent: each entry is an OR-mask — at least one bit must match
    for (const auto& aop : anyObjectsPresent) {
        bool found = false;
        for (int i = 0; i < STRIDE_OBJ; ++i) {
            if (objects[objBase + i] & aop.data[i]) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }

    return true;
}

// ============================================================
// Engine construction / destruction
// ============================================================
Engine::Engine() {}

Engine::~Engine() {
    clearEngine();
}

Engine::Engine(Engine&& other) noexcept
    : objectCount_(other.objectCount_),
      layerCount_(other.layerCount_),
      STRIDE_OBJ_(other.STRIDE_OBJ_),
      STRIDE_MOV_(other.STRIDE_MOV_),
      rigid_(other.rigid_),
      idDict_(std::move(other.idDict_)),
      playerMask_(std::move(other.playerMask_)),
      playerMaskAggregate_(other.playerMaskAggregate_),
      layerMasks_(std::move(other.layerMasks_)),
      rules_(std::move(other.rules_)),
      lateRules_(std::move(other.lateRules_)),
      loopPoint_(std::move(other.loopPoint_)),
      lateLoopPoint_(std::move(other.lateLoopPoint_)),
      winconditions_(std::move(other.winconditions_)),
      levels_(std::move(other.levels_)),
      backgroundid_(other.backgroundid_),
      backgroundlayer_(other.backgroundlayer_),
      rigidGroupIndex_to_GroupIndex_(std::move(other.rigidGroupIndex_to_GroupIndex_)),
      groupNumber_to_RigidGroupIndex_(std::move(other.groupNumber_to_RigidGroupIndex_)),
      metadata_(std::move(other.metadata_)),
      objectInfos_(std::move(other.objectInfos_)),
      level_(std::move(other.level_)),
      winning_(other.winning_),
      againing_(other.againing_),
      rng_(other.rng_),
      curLevel_(other.curLevel_),
      _o1(std::move(other._o1)), _o2(std::move(other._o2)),
      _o2_5(std::move(other._o2_5)), _o3(std::move(other._o3)),
      _o4(std::move(other._o4)), _o5(std::move(other._o5)),
      _o6(std::move(other._o6)), _o7(std::move(other._o7)),
      _o8(std::move(other._o8)),
      _m1(std::move(other._m1)), _m2(std::move(other._m2)),
      _m3(std::move(other._m3)),
      sfxCreateMask_(std::move(other.sfxCreateMask_)),
      sfxDestroyMask_(std::move(other.sfxDestroyMask_)),
      allCellPatterns_(std::move(other.allCellPatterns_)),
      allCellReplacements_(std::move(other.allCellReplacements_)),
      allRules_(std::move(other.allRules_))
{
    // Clear other's ownership vectors so its destructor doesn't free
    other.allCellPatterns_.clear();
    other.allCellReplacements_.clear();
    other.allRules_.clear();
}

Engine& Engine::operator=(Engine&& other) noexcept {
    if (this != &other) {
        clearEngine();

        objectCount_ = other.objectCount_;
        layerCount_ = other.layerCount_;
        STRIDE_OBJ_ = other.STRIDE_OBJ_;
        STRIDE_MOV_ = other.STRIDE_MOV_;
        rigid_ = other.rigid_;
        idDict_ = std::move(other.idDict_);
        playerMask_ = std::move(other.playerMask_);
        playerMaskAggregate_ = other.playerMaskAggregate_;
        layerMasks_ = std::move(other.layerMasks_);
        rules_ = std::move(other.rules_);
        lateRules_ = std::move(other.lateRules_);
        loopPoint_ = std::move(other.loopPoint_);
        lateLoopPoint_ = std::move(other.lateLoopPoint_);
        winconditions_ = std::move(other.winconditions_);
        levels_ = std::move(other.levels_);
        backgroundid_ = other.backgroundid_;
        backgroundlayer_ = other.backgroundlayer_;
        rigidGroupIndex_to_GroupIndex_ = std::move(other.rigidGroupIndex_to_GroupIndex_);
        groupNumber_to_RigidGroupIndex_ = std::move(other.groupNumber_to_RigidGroupIndex_);
        metadata_ = std::move(other.metadata_);
        objectInfos_ = std::move(other.objectInfos_);
        level_ = std::move(other.level_);
        winning_ = other.winning_;
        againing_ = other.againing_;
        rng_ = other.rng_;
        curLevel_ = other.curLevel_;
        _o1 = std::move(other._o1); _o2 = std::move(other._o2);
        _o2_5 = std::move(other._o2_5); _o3 = std::move(other._o3);
        _o4 = std::move(other._o4); _o5 = std::move(other._o5);
        _o6 = std::move(other._o6); _o7 = std::move(other._o7);
        _o8 = std::move(other._o8);
        _m1 = std::move(other._m1); _m2 = std::move(other._m2);
        _m3 = std::move(other._m3);
        sfxCreateMask_ = std::move(other.sfxCreateMask_);
        sfxDestroyMask_ = std::move(other.sfxDestroyMask_);
        allCellPatterns_ = std::move(other.allCellPatterns_);
        allCellReplacements_ = std::move(other.allCellReplacements_);
        allRules_ = std::move(other.allRules_);

        other.allCellPatterns_.clear();
        other.allCellReplacements_.clear();
        other.allRules_.clear();
    }
    return *this;
}

void Engine::clearEngine() {
    for (auto* cp : allCellPatterns_) delete cp;
    allCellPatterns_.clear();
    for (auto* cr : allCellReplacements_) delete cr;
    allCellReplacements_.clear();
    for (auto* r : allRules_) delete r;
    allRules_.clear();
    rules_.clear();
    lateRules_.clear();
    levels_.clear();
    winconditions_.clear();
}

void Engine::initScratchVecs() {
    _o1 = BitVec(STRIDE_OBJ_);
    _o2 = BitVec(STRIDE_OBJ_);
    _o2_5 = BitVec(STRIDE_OBJ_);
    _o3 = BitVec(STRIDE_OBJ_);
    _o4 = BitVec(STRIDE_OBJ_);
    _o5 = BitVec(STRIDE_OBJ_);
    _o6 = BitVec(STRIDE_OBJ_);
    _o7 = BitVec(STRIDE_OBJ_);
    _o8 = BitVec(STRIDE_OBJ_);
    _m1 = BitVec(STRIDE_MOV_);
    _m2 = BitVec(STRIDE_MOV_);
    _m3 = BitVec(STRIDE_MOV_);
    sfxCreateMask_ = BitVec(STRIDE_OBJ_);
    sfxDestroyMask_ = BitVec(STRIDE_OBJ_);
}

// ============================================================
// JSON loading
// ============================================================
static BitVec bvFromJSON(const json& arr, int expectedSize) {
    BitVec bv(expectedSize);
    for (int i = 0; i < expectedSize && i < static_cast<int>(arr.size()); ++i)
        bv.data[i] = arr[i].get<int32_t>();
    return bv;
}

bool Engine::loadFromJSON(const std::string& json_str) {
    clearEngine();

    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return false;
    }

    objectCount_ = j["objectCount"].get<int>();
    layerCount_ = j["layerCount"].get<int>();
    STRIDE_OBJ_ = j["STRIDE_OBJ"].get<int>();
    STRIDE_MOV_ = j["STRIDE_MOV"].get<int>();
    rigid_ = j["rigid"].get<bool>();
    backgroundid_ = j.value("backgroundid", 0);
    backgroundlayer_ = j.value("backgroundlayer", 0);

    // idDict
    idDict_.clear();
    for (const auto& v : j["idDict"])
        idDict_.push_back(v.get<std::string>());

    // Build objectInfos from collisionLayers
    objectInfos_.resize(objectCount_);
    if (j.contains("collisionLayers")) {
        // Build name->id lookup from idDict
        std::unordered_map<std::string, int> nameToId;
        for (int i = 0; i < static_cast<int>(idDict_.size()); ++i)
            nameToId[idDict_[i]] = i;

        const auto& cl = j["collisionLayers"];
        for (int layerIdx = 0; layerIdx < static_cast<int>(cl.size()); ++layerIdx) {
            for (const auto& objEntry : cl[layerIdx]) {
                int oid = -1;
                if (objEntry.is_number()) {
                    oid = objEntry.get<int>();
                } else if (objEntry.is_string()) {
                    auto it = nameToId.find(objEntry.get<std::string>());
                    if (it != nameToId.end()) oid = it->second;
                }
                if (oid >= 0 && oid < objectCount_)
                    objectInfos_[oid].layer = layerIdx;
            }
        }
    }

    // playerMask
    playerMaskAggregate_ = j["playerMask"][0].get<bool>();
    playerMask_ = bvFromJSON(j["playerMask"][1], STRIDE_OBJ_);

    // layerMasks
    layerMasks_.clear();
    for (const auto& m : j["layerMasks"])
        layerMasks_.push_back(bvFromJSON(m, STRIDE_OBJ_));

    // Helper lambda to parse CellPattern from JSON
    auto parseCellPattern = [&](const json& cpj) -> CellPattern* {
        auto* cp = new CellPattern();
        allCellPatterns_.push_back(cp);

        cp->objectsPresent = bvFromJSON(cpj["objectsPresent"], STRIDE_OBJ_);
        cp->objectsMissing = bvFromJSON(cpj["objectsMissing"], STRIDE_OBJ_);
        cp->movementsPresent = bvFromJSON(cpj["movementsPresent"], STRIDE_MOV_);
        cp->movementsMissing = bvFromJSON(cpj["movementsMissing"], STRIDE_MOV_);

        for (const auto& aop : cpj["anyObjectsPresent"])
            cp->anyObjectsPresent.push_back(bvFromJSON(aop, STRIDE_OBJ_));

        if (!cpj["replacement"].is_null()) {
            auto* cr = new CellReplacement();
            allCellReplacements_.push_back(cr);
            const auto& rj = cpj["replacement"];
            cr->objectsClear = bvFromJSON(rj["objectsClear"], STRIDE_OBJ_);
            cr->objectsSet = bvFromJSON(rj["objectsSet"], STRIDE_OBJ_);
            cr->movementsClear = bvFromJSON(rj["movementsClear"], STRIDE_MOV_);
            cr->movementsSet = bvFromJSON(rj["movementsSet"], STRIDE_MOV_);
            cr->movementsLayerMask = bvFromJSON(rj["movementsLayerMask"], STRIDE_MOV_);
            cr->randomEntityMask = bvFromJSON(rj["randomEntityMask"], STRIDE_OBJ_);
            cr->randomDirMask = bvFromJSON(rj["randomDirMask"], STRIDE_MOV_);
            cp->replacement = cr;
        }

        return cp;
    };

    // Helper lambda to parse a rule group
    auto parseRuleGroups = [&](const json& groupsJ) -> std::vector<std::vector<Rule*>> {
        std::vector<std::vector<Rule*>> groups;
        for (const auto& groupJ : groupsJ) {
            std::vector<Rule*> group;
            for (const auto& ruleJ : groupJ) {
                auto* rule = new Rule();
                allRules_.push_back(rule);

                rule->direction = ruleJ["direction"].get<int>();
                rule->hasReplacements = ruleJ["hasReplacements"].get<bool>();
                rule->lineNumber = ruleJ["lineNumber"].get<int>();
                rule->groupNumber = ruleJ["groupNumber"].get<int>();
                rule->rigid = ruleJ["rigid"].get<bool>();
                rule->isRandom = ruleJ["isRandom"].get<bool>();

                for (const auto& ec : ruleJ["ellipsisCount"])
                    rule->ellipsisCount.push_back(ec.get<int>());

                for (const auto& cmd : ruleJ["commands"]) {
                    std::vector<std::string> cmdVec;
                    for (const auto& c : cmd)
                        cmdVec.push_back(c.get<std::string>());
                    rule->commands.push_back(cmdVec);
                }

                // Parse patterns: array of cell rows
                for (const auto& rowJ : ruleJ["patterns"]) {
                    std::vector<CellPattern*> row;
                    for (const auto& cellJ : rowJ) {
                        if (cellJ.is_string() && cellJ.get<std::string>() == "ellipsis") {
                            auto* ep = new CellPattern();
                            allCellPatterns_.push_back(ep);
                            ep->isEllipsis = true;
                            row.push_back(ep);
                        } else {
                            row.push_back(parseCellPattern(cellJ));
                        }
                    }
                    rule->patterns.push_back(row);
                }

                for (const auto& m : ruleJ["cellRowMasks"])
                    rule->cellRowMasks.push_back(bvFromJSON(m, STRIDE_OBJ_));
                for (const auto& m : ruleJ["cellRowMasks_Movements"])
                    rule->cellRowMasks_Movements.push_back(bvFromJSON(m, STRIDE_MOV_));

                // Compute ruleMask = OR of all cellRowMasks
                rule->ruleMask = BitVec(STRIDE_OBJ_);
                for (const auto& m : rule->cellRowMasks)
                    rule->ruleMask.ior(m);

                group.push_back(rule);
            }
            groups.push_back(group);
        }
        return groups;
    };

    rules_ = parseRuleGroups(j["rules"]);
    lateRules_ = parseRuleGroups(j["lateRules"]);

    // loopPoints
    loopPoint_.clear();
    if (j.contains("loopPoint")) {
        for (auto& [key, val] : j["loopPoint"].items())
            loopPoint_[std::stoi(key)] = val.get<int>();
    }
    lateLoopPoint_.clear();
    if (j.contains("lateLoopPoint")) {
        for (auto& [key, val] : j["lateLoopPoint"].items())
            lateLoopPoint_[std::stoi(key)] = val.get<int>();
    }

    // Win conditions
    winconditions_.clear();
    for (const auto& wcj : j["winconditions"]) {
        WinCondition wc;
        wc.num = wcj["num"].get<int>();
        wc.mask1 = bvFromJSON(wcj["mask1"], STRIDE_OBJ_);
        wc.mask2_is_all = wcj["mask2"].is_null();
        if (!wc.mask2_is_all)
            wc.mask2 = bvFromJSON(wcj["mask2"], STRIDE_OBJ_);
        wc.lineNumber = wcj["lineNumber"].get<int>();
        wc.aggr1 = wcj["aggr1"].get<bool>();
        wc.aggr2 = wcj["aggr2"].get<bool>();
        winconditions_.push_back(wc);
    }

    // Levels
    levels_.clear();
    for (const auto& lvj : j["levels"]) {
        LevelDef ld;
        if (lvj["type"].get<std::string>() == "message") {
            ld.isMessage = true;
            ld.index = lvj["index"].get<int>();
        } else {
            ld.isMessage = false;
            ld.index = lvj["index"].get<int>();
            ld.lineNumber = lvj["lineNumber"].get<int>();
            ld.width = lvj["width"].get<int>();
            ld.height = lvj["height"].get<int>();
            ld.layerCount = lvj["layerCount"].get<int>();
            for (const auto& v : lvj["objects"])
                ld.objects.push_back(v.get<int32_t>());
        }
        levels_.push_back(ld);
    }

    // Rigid group mappings
    rigidGroupIndex_to_GroupIndex_.clear();
    if (j.contains("rigidGroupIndex_to_GroupIndex")) {
        for (const auto& v : j["rigidGroupIndex_to_GroupIndex"])
            rigidGroupIndex_to_GroupIndex_.push_back(v.get<int>());
    }
    groupNumber_to_RigidGroupIndex_.clear();
    if (j.contains("groupNumber_to_RigidGroupIndex")) {
        for (auto& [key, val] : j["groupNumber_to_RigidGroupIndex"].items())
            groupNumber_to_RigidGroupIndex_[std::stoi(key)] = val.get<int>();
    }

    // Metadata
    metadata_.clear();
    if (j.contains("metadata")) {
        for (auto& [key, val] : j["metadata"].items()) {
            if (val.is_string()) metadata_[key] = val.get<std::string>();
            else if (val.is_number()) metadata_[key] = std::to_string(val.get<double>());
            else if (val.is_boolean()) metadata_[key] = val.get<bool>() ? "true" : "false";
        }
    }

    initScratchVecs();
    return true;
}

// ============================================================
// Level loading
// ============================================================
void Engine::loadLevel(int levelIndex) {
    // Find the actual level (skipping messages)
    int actualIdx = -1;
    int countPlayable = 0;
    for (int i = 0; i < static_cast<int>(levels_.size()); ++i) {
        if (!levels_[i].isMessage) {
            if (countPlayable == levelIndex) {
                actualIdx = i;
                break;
            }
            countPlayable++;
        }
    }
    if (actualIdx < 0) {
        throw std::runtime_error("Level index out of range: " + std::to_string(levelIndex));
    }

    const auto& ld = levels_[actualIdx];
    level_ = Level(ld.lineNumber, ld.width, ld.height, ld.layerCount, STRIDE_OBJ_, STRIDE_MOV_);
    level_.objects = ld.objects;
    level_.initMasks(rigid_);
    level_.calculateRowColMasks();

    winning_ = false;
    againing_ = false;
    curLevel_ = levelIndex;

    // Match JS behavior: run rules once on level start if metadata flag is set.
    // JS calls processInput(-1, dontDoWin=true) here, suppressing win detection.
    if (metadata_.count("run_rules_on_level_start")) {
        processInput(-1);
        winning_ = false;
        againing_ = false;
    }
}

// ============================================================
// Backup / Restore
// ============================================================
LevelBackup Engine::backupLevel() const {
    LevelBackup bak;
    bak.dat = level_.objects;
    bak.width = level_.width;
    bak.height = level_.height;
    return bak;
}

void Engine::restoreLevel(const LevelBackup& bak) {
    level_.objects = bak.dat;
    level_.calculateRowColMasks();
    winning_ = false;
    againing_ = false;
}

void Engine::restart() {
    loadLevel(curLevel_);
}

// ============================================================
// Player positions
// ============================================================
std::vector<int> Engine::getPlayerPositions() const {
    std::vector<int> positions;
    for (int i = 0; i < level_.n_tiles; ++i) {
        BitVec cell = level_.getCell(i);
        if (playerMaskAggregate_) {
            if (playerMask_.bitsSetInArray(cell.data.data()))
                positions.push_back(i);
        } else {
            if (playerMask_.anyBitsInCommon(cell))
                positions.push_back(i);
        }
    }
    return positions;
}

// ============================================================
// Movement application
// ============================================================
void Engine::moveEntitiesAtIndex(int positionIndex, const BitVec& entityMask, int dirMask) {
    BitVec cellObj(STRIDE_OBJ_);
    level_.getCellInto(positionIndex, cellObj);

    // Find which objects from entityMask are at this position
    BitVec matchedEntities = cellObj.clone();
    matchedEntities.iand(entityMask);
    if (matchedEntities.iszero()) return;

    // For each layer that has a matched entity, set the movement direction
    for (int layer = 0; layer < layerCount_; ++layer) {
        BitVec layerObj = matchedEntities.clone();
        layerObj.iand(layerMasks_[layer]);
        if (!layerObj.iszero()) {
            BitVec mov = level_.getMovements(positionIndex);
            mov.ishiftor(dirMask, 5 * layer);
            level_.setMovements(positionIndex, mov);
        }
    }
}

void Engine::startMovement(int dirMask) {
    std::vector<int> playerPositions = getPlayerPositions();
    for (int pos : playerPositions) {
        moveEntitiesAtIndex(pos, playerMask_, dirMask);
    }
}

bool Engine::repositionEntitiesOnLayer(int positionIndex, int layer, int dirMask) {
    int dx = 0, dy = 0;
    switch (dirMask) {
        case 1:  dy = -1; break;  // up
        case 2:  dy = 1;  break;  // down
        case 4:  dx = -1; break;  // left
        case 8:  dx = 1;  break;  // right
        default: return false;     // action (16) or composite — no spatial movement
    }

    int tx = positionIndex / level_.height;
    int ty = positionIndex % level_.height;
    int maxx = level_.width - 1;
    int maxy = level_.height - 1;

    if ((tx == 0 && dx < 0) || (tx == maxx && dx > 0) ||
        (ty == 0 && dy < 0) || (ty == maxy && dy > 0)) {
        return false;
    }

    int targetIndex = positionIndex + dy + dx * level_.height;

    const BitVec& layerMask = layerMasks_[layer];

    level_.getCellInto(targetIndex, _o7);
    level_.getCellInto(positionIndex, _o8);

    // If target cell already has something on this layer (and it's not action),
    // movement is blocked
    if (layerMask.anyBitsInCommon(_o7) && dirMask != 16) {
        return false;
    }

    // Move entities from source to target
    BitVec movingEntities = _o8.clone();
    _o8.iclear(layerMask);
    movingEntities.iand(layerMask);
    _o7.ior(movingEntities);

    level_.setCell(positionIndex, _o8);
    level_.setCell(targetIndex, _o7);

    int colIndex = targetIndex / level_.height;
    int rowIndex = targetIndex % level_.height;
    level_.colCellContents[colIndex].ior(movingEntities);
    level_.rowCellContents[rowIndex].ior(movingEntities);

    return true;
}

bool Engine::repositionEntitiesAtCell(int positionIndex) {
    BitVec movementMask = level_.getMovements(positionIndex);
    if (movementMask.iszero()) return false;

    bool moved = false;
    for (int layer = 0; layer < layerCount_; ++layer) {
        int layerMovement = movementMask.getshiftor(0x1f, 5 * layer);
        if (layerMovement != 0) {
            bool thisMoved = repositionEntitiesOnLayer(positionIndex, layer, layerMovement);
            if (thisMoved) {
                movementMask.ishiftclear(layerMovement, 5 * layer);
                moved = true;
            }
        }
    }

    // Write back remaining movements
    int base = positionIndex * STRIDE_MOV_;
    for (int i = 0; i < STRIDE_MOV_; ++i)
        level_.movements[base + i] = movementMask.data[i];

    // Update row/col/map masks
    int colIndex = positionIndex / level_.height;
    int rowIndex = positionIndex % level_.height;
    level_.colCellContents_Movements[colIndex].ior(movementMask);
    level_.rowCellContents_Movements[rowIndex].ior(movementMask);
    level_.mapCellContents_Movements.ior(movementMask);

    return moved;
}

bool Engine::resolveMovements(std::vector<bool>& bannedGroup) {
    // Keep repositioning until nothing moves
    bool moved = true;
    while (moved) {
        moved = false;
        for (int i = 0; i < level_.n_tiles; ++i) {
            moved = repositionEntitiesAtCell(i) || moved;
        }
    }

    bool doUndo = false;

    // Check for unresolved rigid movements
    for (int i = 0; i < level_.n_tiles; ++i) {
        BitVec movementMask = level_.getMovements(i);
        if (!movementMask.iszero()) {
            if (rigid_ && i < static_cast<int>(level_.rigidMovementAppliedMask.size())) {
                BitVec rigidMovApplied = level_.rigidMovementAppliedMask[i];
                if (!rigidMovApplied.iszero()) {
                    BitVec testMov = movementMask.clone();
                    testMov.iand(rigidMovApplied);
                    if (!testMov.iszero()) {
                        // Find which layer was restricted
                        for (int layer = 0; layer < layerCount_; ++layer) {
                            int layerSection = testMov.getshiftor(0x1f, 5 * layer);
                            if (layerSection != 0) {
                                BitVec rigidGIM = level_.rigidGroupIndexMask[i];
                                int rigidGroupIndex = rigidGIM.getshiftor(0x1f, 5 * layer);
                                rigidGroupIndex--;  // stored incremented by 1
                                if (rigidGroupIndex >= 0 &&
                                    rigidGroupIndex < static_cast<int>(rigidGroupIndex_to_GroupIndex_.size())) {
                                    int groupIndex = rigidGroupIndex_to_GroupIndex_[rigidGroupIndex];
                                    if (groupIndex < static_cast<int>(bannedGroup.size()) && !bannedGroup[groupIndex]) {
                                        bannedGroup[groupIndex] = true;
                                        doUndo = true;
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Clear movements
        level_.clearMovementsRaw(i);

        // Clear rigid masks
        if (rigid_ && i < static_cast<int>(level_.rigidGroupIndexMask.size())) {
            level_.rigidGroupIndexMask[i].setZero();
            level_.rigidMovementAppliedMask[i].setZero();
        }
    }

    return doUndo;
}

// ============================================================
// Rule matching and application
// ============================================================

// Match a cell row without ellipsis against the level
// Returns true and sets result = [startIdx] if a match is found at startIdx
bool Engine::matchCellRowNoEllipsis(const std::vector<CellPattern*>& cellRow, int startIdx, int delta,
                                     std::vector<std::vector<int>>& result) {
    if (!cellRow[0]->matches(startIdx, level_.objects.data(), level_.movements.data(),
                              STRIDE_OBJ_, STRIDE_MOV_))
        return false;

    for (int k = 1; k < static_cast<int>(cellRow.size()); ++k) {
        if (!cellRow[k]->matches(startIdx + k * delta, level_.objects.data(), level_.movements.data(),
                                  STRIDE_OBJ_, STRIDE_MOV_))
            return false;
    }
    result.push_back({startIdx});
    return true;
}

void Engine::matchCellRowEllipsis1(const std::vector<CellPattern*>& cellRow, int startIdx,
                                    int kmax, int kmin, int delta,
                                    std::vector<std::vector<int>>& result) {
    // Find ellipsis position
    int ellipsisIdx = -1;
    for (int i = 0; i < static_cast<int>(cellRow.size()); ++i) {
        if (cellRow[i]->isEllipsis) { ellipsisIdx = i; break; }
    }

    // Match cells before the ellipsis
    int idx = startIdx;
    for (int i = 0; i < ellipsisIdx; ++i) {
        if (!cellRow[i]->matches(idx, level_.objects.data(), level_.movements.data(),
                                  STRIDE_OBJ_, STRIDE_MOV_))
            return;
        idx += delta;
    }

    // Try varying gap sizes
    for (int k = kmin; k < kmax; ++k) {
        bool match = true;
        for (int i = ellipsisIdx + 1; i < static_cast<int>(cellRow.size()); ++i) {
            int offset = i - 1;  // adjusted for the ellipsis
            if (!cellRow[i]->matches(startIdx + delta * (k + offset),
                                      level_.objects.data(), level_.movements.data(),
                                      STRIDE_OBJ_, STRIDE_MOV_)) {
                match = false;
                break;
            }
        }
        if (match) {
            result.push_back({startIdx, k});
        }
    }
}

void Engine::matchCellRowEllipsis2(const std::vector<CellPattern*>& cellRow, int startIdx,
                                    int kmax, int kmin,
                                    int k1max, int k1min,
                                    int k2max, int k2min,
                                    int delta,
                                    std::vector<std::vector<int>>& result) {
    // Find two ellipsis positions
    int e1 = -1, e2 = -1;
    for (int i = 0; i < static_cast<int>(cellRow.size()); ++i) {
        if (cellRow[i]->isEllipsis) {
            if (e1 < 0) e1 = i;
            else { e2 = i; break; }
        }
    }

    // Match cells before first ellipsis
    for (int i = 0; i < e1; ++i) {
        if (!cellRow[i]->matches(startIdx + i * delta, level_.objects.data(), level_.movements.data(),
                                  STRIDE_OBJ_, STRIDE_MOV_))
            return;
    }

    // Try varying first gap
    for (int k1 = k1min; k1 < k1max; ++k1) {
        // Check cells between first and second ellipsis
        bool matchMiddle = true;
        for (int i = e1 + 1; i < e2; ++i) {
            int offset = i - 1;
            if (!cellRow[i]->matches(startIdx + delta * (k1 + offset),
                                      level_.objects.data(), level_.movements.data(),
                                      STRIDE_OBJ_, STRIDE_MOV_)) {
                matchMiddle = false;
                break;
            }
        }
        if (!matchMiddle) continue;

        // Try varying second gap
        for (int k2 = k2min; k1 + k2 < kmax && k2 < k2max; ++k2) {
            bool matchEnd = true;
            for (int i = e2 + 1; i < static_cast<int>(cellRow.size()); ++i) {
                int offset = i - 2;  // adjusted for two ellipses
                if (!cellRow[i]->matches(startIdx + delta * (k1 + k2 + offset),
                                          level_.objects.data(), level_.movements.data(),
                                          STRIDE_OBJ_, STRIDE_MOV_)) {
                    matchEnd = false;
                    break;
                }
            }
            if (matchEnd) {
                result.push_back({startIdx, k1, k2});
            }
        }
    }
}

// Find all matches for a rule
std::vector<std::vector<std::vector<int>>> Engine::ruleFindMatches(Rule& rule) {
    if (!rule.ruleMask.bitsSetInArray(level_.mapCellContents.data.data()))
        return {};

    int d = level_.delta_index(rule.direction);
    std::vector<std::vector<std::vector<int>>> allRowMatches;

    for (int rowIdx = 0; rowIdx < static_cast<int>(rule.patterns.size()); ++rowIdx) {
        auto& cellRow = rule.patterns[rowIdx];
        const BitVec& cellRowMask = rule.cellRowMasks[rowIdx];
        const BitVec& cellRowMask_Mov = rule.cellRowMasks_Movements[rowIdx];

        // Quick check: does the board contain what this row needs?
        if (!cellRowMask.bitsSetInArray(level_.mapCellContents.data.data()) ||
            !cellRowMask_Mov.bitsSetInArray(level_.mapCellContents_Movements.data.data())) {
            return {};
        }

        // Count non-ellipsis cells for bounds
        int nonEllipsisCells = 0;
        int wildcardCount = rule.ellipsisCount[rowIdx];
        for (auto* cp : cellRow)
            if (!cp->isEllipsis) nonEllipsisCells++;

        int len = nonEllipsisCells;
        int xmin = 0, xmax = level_.width;
        int ymin = 0, ymax = level_.height;
        int direction = rule.direction;

        switch (direction) {
            case 1: ymin += (len - 1); break;    // up
            case 2: ymax -= (len - 1); break;    // down
            case 4: xmin += (len - 1); break;    // left
            case 8: xmax -= (len - 1); break;    // right
        }

        bool horizontal = direction > 2;
        // All matches for this row: each is {startIdx} or {startIdx, k} or {startIdx, k1, k2}
        std::vector<std::vector<int>> rowMatches;

        if (wildcardCount == 0) {
            if (horizontal) {
                for (int y = ymin; y < ymax; ++y) {
                    if (!cellRowMask.bitsSetInArray(level_.rowCellContents[y].data.data()) ||
                        !cellRowMask_Mov.bitsSetInArray(level_.rowCellContents_Movements[y].data.data()))
                        continue;
                    for (int x = xmin; x < xmax; ++x) {
                        int idx = x * level_.height + y;
                        matchCellRowNoEllipsis(cellRow, idx, d, rowMatches);
                    }
                }
            } else {
                for (int x = xmin; x < xmax; ++x) {
                    if (!cellRowMask.bitsSetInArray(level_.colCellContents[x].data.data()) ||
                        !cellRowMask_Mov.bitsSetInArray(level_.colCellContents_Movements[x].data.data()))
                        continue;
                    for (int y = ymin; y < ymax; ++y) {
                        int idx = x * level_.height + y;
                        matchCellRowNoEllipsis(cellRow, idx, d, rowMatches);
                    }
                }
            }
        } else {
            // With ellipsis — rowMatches will contain {startIdx, k} or {startIdx, k1, k2}
            if (horizontal) {
                for (int y = ymin; y < ymax; ++y) {
                    if (!cellRowMask.bitsSetInArray(level_.rowCellContents[y].data.data()) ||
                        !cellRowMask_Mov.bitsSetInArray(level_.rowCellContents_Movements[y].data.data()))
                        continue;
                    for (int x = xmin; x < xmax; ++x) {
                        int idx = x * level_.height + y;
                        int kmax_val;
                        if (direction == 4)  // left
                            kmax_val = x - len + 2;
                        else  // right
                            kmax_val = level_.width - (x + len) + 1;

                        if (wildcardCount == 1) {
                            matchCellRowEllipsis1(cellRow, idx, kmax_val, 0, d, rowMatches);
                        } else {
                            matchCellRowEllipsis2(cellRow, idx, kmax_val, 0,
                                                   kmax_val, 0, kmax_val, 0, d, rowMatches);
                        }
                    }
                }
            } else {
                for (int x = xmin; x < xmax; ++x) {
                    if (!cellRowMask.bitsSetInArray(level_.colCellContents[x].data.data()) ||
                        !cellRowMask_Mov.bitsSetInArray(level_.colCellContents_Movements[x].data.data()))
                        continue;
                    for (int y = ymin; y < ymax; ++y) {
                        int idx = x * level_.height + y;
                        int kmax_val;
                        if (direction == 2)  // down
                            kmax_val = level_.height - (y + len) + 1;
                        else  // up
                            kmax_val = y - len + 2;

                        if (wildcardCount == 1) {
                            matchCellRowEllipsis1(cellRow, idx, kmax_val, 0, d, rowMatches);
                        } else {
                            matchCellRowEllipsis2(cellRow, idx, kmax_val, 0,
                                                   kmax_val, 0, kmax_val, 0, d, rowMatches);
                        }
                    }
                }
            }
        }

        if (rowMatches.empty()) return {};
        allRowMatches.push_back(std::move(rowMatches));
    }

    return allRowMatches;
}

// Generate all cartesian product tuples from per-row match lists
// Each row's matches are vectors: {startIdx} or {startIdx, k} or {startIdx, k1, k2}
// Tuples are vectors of such vectors (one per row)
std::vector<std::vector<std::vector<int>>> Engine::generateTuples(const std::vector<std::vector<std::vector<int>>>& lists) {
    std::vector<std::vector<std::vector<int>>> tuples = {{}};

    for (const auto& row : lists) {
        std::vector<std::vector<std::vector<int>>> newTuples;
        for (const auto& match : row) {
            for (const auto& tuple : tuples) {
                auto newtuple = tuple;
                newtuple.push_back(match);
                newTuples.push_back(newtuple);
            }
        }
        tuples = std::move(newTuples);
    }
    return tuples;
}

bool Engine::cellPatternReplace(CellPattern& cp, Rule& rule, int currentIndex) {
    if (!cp.replacement) return false;

    auto& rep = *cp.replacement;

    BitVec objectsSet = rep.objectsSet.clone();
    BitVec objectsClear = rep.objectsClear.clone();
    BitVec movementsSet = rep.movementsSet.clone();

    // Combine movementsClear with movementsLayerMask
    BitVec movementsClear(STRIDE_MOV_);
    for (int i = 0; i < STRIDE_MOV_; ++i)
        movementsClear.data[i] = rep.movementsClear.data[i] | rep.movementsLayerMask.data[i];

    // Handle random entity
    if (!rep.randomEntityMask.iszero()) {
        std::vector<int> choices;
        for (int i = 0; i < 32 * STRIDE_OBJ_; ++i) {
            if (rep.randomEntityMask.get(i))
                choices.push_back(i);
        }
        if (!choices.empty()) {
            int rand = choices[rng_.random_int(static_cast<int>(choices.size()))];
            int objId = rand;
            if (objId < static_cast<int>(objectInfos_.size())) {
                int objLayer = objectInfos_[objId].layer;
                objectsSet.ibitset(rand);
                objectsClear.ior(layerMasks_[objLayer]);
                movementsClear.ishiftor(0x1f, 5 * objLayer);
            }
        }
    }

    // Handle random direction
    if (!rep.randomDirMask.iszero()) {
        for (int layerIndex = 0; layerIndex < layerCount_; ++layerIndex) {
            if (rep.randomDirMask.get(5 * layerIndex)) {
                int randomDir = rng_.random_int(4);
                movementsSet.ibitset(randomDir + 5 * layerIndex);
            }
        }
    }

    // Get current cell state
    BitVec oldCellMask(STRIDE_OBJ_);
    level_.getCellInto(currentIndex, oldCellMask);

    BitVec curCellMask(STRIDE_OBJ_);
    for (int i = 0; i < STRIDE_OBJ_; ++i)
        curCellMask.data[i] = (oldCellMask.data[i] & ~objectsClear.data[i]) | objectsSet.data[i];

    BitVec oldMovementMask = level_.getMovements(currentIndex);
    BitVec curMovementMask(STRIDE_MOV_);
    for (int i = 0; i < STRIDE_MOV_; ++i)
        curMovementMask.data[i] = (oldMovementMask.data[i] & ~movementsClear.data[i]) | movementsSet.data[i];

    // Handle rigid body
    bool rigidChange = false;
    if (rule.rigid && rigid_) {
        auto it = groupNumber_to_RigidGroupIndex_.find(rule.groupNumber);
        if (it != groupNumber_to_RigidGroupIndex_.end()) {
            int rigidGroupIndex = it->second + 1;  // incremented for bitfield storage

            BitVec rigidMask(STRIDE_MOV_);
            for (int layer = 0; layer < layerCount_; ++layer)
                rigidMask.ishiftor(rigidGroupIndex, layer * 5);
            rigidMask.iand(rep.movementsLayerMask);

            // Ensure rigidGroupIndexMask and rigidMovementAppliedMask exist for this cell
            if (currentIndex < static_cast<int>(level_.rigidGroupIndexMask.size())) {
                BitVec& curRigidGIM = level_.rigidGroupIndexMask[currentIndex];
                BitVec& curRigidMAM = level_.rigidMovementAppliedMask[currentIndex];

                if (!rigidMask.bitsSetInArray(curRigidGIM.data.data()) &&
                    !rep.movementsLayerMask.bitsSetInArray(curRigidMAM.data.data())) {
                    curRigidGIM.ior(rigidMask);
                    curRigidMAM.ior(rep.movementsLayerMask);
                    rigidChange = true;
                }
            }
        }
    }

    // Check if anything actually changed
    if (oldCellMask.equals(curCellMask) && oldMovementMask.equals(curMovementMask) && !rigidChange) {
        return false;
    }

    // Track created and destroyed objects for sfx
    for (int i = 0; i < STRIDE_OBJ_; ++i) {
        int created = curCellMask.data[i] & ~oldCellMask.data[i];
        sfxCreateMask_.data[i] |= created;
        int destroyed = oldCellMask.data[i] & ~curCellMask.data[i];
        sfxDestroyMask_.data[i] |= destroyed;
    }

    // Apply the changes
    level_.setCell(currentIndex, curCellMask);
    level_.setMovements(currentIndex, curMovementMask);

    int colIndex = currentIndex / level_.height;
    int rowIndex = currentIndex % level_.height;
    level_.colCellContents[colIndex].ior(curCellMask);
    level_.rowCellContents[rowIndex].ior(curCellMask);
    level_.mapCellContents.ior(curCellMask);

    return true;
}

bool Engine::ruleApplyAt(Rule& rule, const std::vector<std::vector<int>>& tuple, bool check, int delta) {
    // Double check matches if check is true (for multi-tuple rules)
    if (check) {
        for (int rowIdx = 0; rowIdx < static_cast<int>(rule.patterns.size()); ++rowIdx) {
            auto& cellRow = rule.patterns[rowIdx];
            int startIdx = tuple[rowIdx][0];
            if (rule.ellipsisCount[rowIdx] == 0) {
                for (int k = 0; k < static_cast<int>(cellRow.size()); ++k) {
                    if (!cellRow[k]->matches(startIdx + k * delta,
                                              level_.objects.data(), level_.movements.data(),
                                              STRIDE_OBJ_, STRIDE_MOV_))
                        return false;
                }
            } else if (rule.ellipsisCount[rowIdx] == 1) {
                // Re-verify with the gap value
                int gapK = tuple[rowIdx][1];
                int cellIdx = 0;
                int pos = startIdx;
                for (int i = 0; i < static_cast<int>(cellRow.size()); ++i) {
                    if (cellRow[i]->isEllipsis) {
                        pos += delta * gapK;
                        continue;
                    }
                    if (!cellRow[i]->matches(pos, level_.objects.data(), level_.movements.data(),
                                              STRIDE_OBJ_, STRIDE_MOV_))
                        return false;
                    pos += delta;
                    cellIdx++;
                }
            }
        }
    }

    bool result = false;

    // Apply the rule
    for (int rowIdx = 0; rowIdx < static_cast<int>(rule.patterns.size()); ++rowIdx) {
        auto& cellRow = rule.patterns[rowIdx];
        int currentIndex = tuple[rowIdx][0];
        int ellipseIdx = 0;

        for (int cellIdx = 0; cellIdx < static_cast<int>(cellRow.size()); ++cellIdx) {
            if (cellRow[cellIdx]->isEllipsis) {
                // Advance by gap size * delta
                int gapK = tuple[rowIdx][1 + ellipseIdx];
                currentIndex += delta * gapK;
                ellipseIdx++;
                continue;
            }
            result = cellPatternReplace(*cellRow[cellIdx], rule, currentIndex) || result;
            currentIndex += delta;
        }
    }

    return result;
}

void Engine::ruleQueueCommands(Rule& rule) {
    if (rule.commands.empty()) return;

    bool preexistingCancel = false;
    bool preexistingRestart = false;
    for (const auto& cmd : level_.commandQueue) {
        if (cmd == "cancel") preexistingCancel = true;
        if (cmd == "restart") preexistingRestart = true;
    }

    bool curruleCancel = false;
    bool curruleRestart = false;
    for (const auto& cmd : rule.commands) {
        if (!cmd.empty()) {
            if (cmd[0] == "cancel") curruleCancel = true;
            if (cmd[0] == "restart") curruleRestart = true;
        }
    }

    if (preexistingCancel) return;
    if (preexistingRestart && !curruleCancel) return;

    if (curruleCancel || curruleRestart) {
        level_.commandQueue.clear();
        level_.commandQueueSourceRules.clear();
    }

    for (const auto& cmd : rule.commands) {
        if (cmd.empty()) continue;
        // Check if already in queue
        bool already = false;
        for (const auto& existing : level_.commandQueue) {
            if (existing == cmd[0]) { already = true; break; }
        }
        if (already) continue;
        level_.commandQueue.push_back(cmd[0]);
    }
}

bool Engine::ruleTryApply(Rule& rule) {
    int delta = level_.delta_index(rule.direction);

    auto matches = ruleFindMatches(rule);
    if (matches.empty()) return false;

    bool result = false;
    if (rule.hasReplacements) {
        auto tuples = generateTuples(matches);
        for (int ti = 0; ti < static_cast<int>(tuples.size()); ++ti) {
            bool shouldCheck = (ti > 0);
            bool success = ruleApplyAt(rule, tuples[ti], shouldCheck, delta);
            result = success || result;
        }
    }

    if (!matches.empty()) {
        ruleQueueCommands(rule);
    }

    return result;
}

bool Engine::applyRandomRuleGroup(std::vector<Rule*>& ruleGroup) {
    struct Match {
        int ruleIndex;
        std::vector<std::vector<int>> tuple;
    };
    std::vector<Match> allMatches;

    for (int ri = 0; ri < static_cast<int>(ruleGroup.size()); ++ri) {
        auto ruleMatches = ruleFindMatches(*ruleGroup[ri]);
        if (!ruleMatches.empty()) {
            auto tuples = generateTuples(ruleMatches);
            for (auto& tuple : tuples)
                allMatches.push_back({ri, std::move(tuple)});
        }
    }

    if (allMatches.empty()) return false;

    int chosen = rng_.random_int(static_cast<int>(allMatches.size()));
    auto& match = allMatches[chosen];
    Rule& rule = *ruleGroup[match.ruleIndex];
    int delta = level_.delta_index(rule.direction);
    bool modified = ruleApplyAt(rule, match.tuple, false, delta);

    ruleQueueCommands(rule);
    return modified;
}

bool Engine::applyRuleGroup(std::vector<Rule*>& ruleGroup) {
    if (ruleGroup[0]->isRandom) {
        return applyRandomRuleGroup(ruleGroup);
    }

    const int MAX_LOOP_COUNT = 200;
    int groupLength = static_cast<int>(ruleGroup.size());
    bool hasChanges = false;
    bool madeChangeThisLoop = true;
    int loopcount = 0;

    while (madeChangeThisLoop && loopcount++ < MAX_LOOP_COUNT) {
        madeChangeThisLoop = false;
        int consecutiveFailures = 0;

        for (int ri = 0; ri < groupLength; ++ri) {
            if (ruleTryApply(*ruleGroup[ri])) {
                madeChangeThisLoop = true;
                consecutiveFailures = 0;
            } else {
                consecutiveFailures++;
                if (consecutiveFailures == groupLength) break;
            }
        }

        if (madeChangeThisLoop) hasChanges = true;
    }

    return hasChanges;
}

void Engine::applyRules(const std::vector<std::vector<Rule*>>& rules,
                         const std::map<int, int>& loopPoint,
                         std::vector<bool>* bannedGroup) {
    bool loopPropagated = false;
    int loopCount = 0;
    int ruleGroupIndex = 0;
    int rulesCount = static_cast<int>(rules.size());

    while (ruleGroupIndex < rulesCount) {
        // Apply rules if not banned
        if (!bannedGroup || !(*bannedGroup)[ruleGroupIndex]) {
            auto& group = const_cast<std::vector<Rule*>&>(rules[ruleGroupIndex]);
            loopPropagated = applyRuleGroup(group) || loopPropagated;
        }

        // Handle mid-sequence loop point
        auto it = loopPoint.find(ruleGroupIndex);
        if (loopPropagated && it != loopPoint.end()) {
            ruleGroupIndex = it->second;
            loopPropagated = false;
            loopCount++;
            if (loopCount > 200) break;
            continue;
        }

        ruleGroupIndex++;

        // Handle end-sequence loop point
        if (ruleGroupIndex == rulesCount && loopPropagated) {
            auto endIt = loopPoint.find(ruleGroupIndex);
            if (endIt != loopPoint.end()) {
                ruleGroupIndex = endIt->second;
                loopPropagated = false;
                loopCount++;
                if (loopCount > 200) break;
            }
        }
    }
}

// ============================================================
// Win condition checking
// ============================================================
bool Engine::checkWin() {
    // Check for explicit 'win' command in queue
    for (const auto& cmd : level_.commandQueue) {
        if (cmd == "win") {
            winning_ = true;
            return true;
        }
    }

    // Check all win conditions
    bool won = true;
    for (const auto& wc : winconditions_) {
        bool conditionMet = true;

        for (int i = 0; i < level_.n_tiles; ++i) {
            BitVec cell = level_.getCell(i);

            // Check filter1
            bool f1;
            if (wc.aggr1)
                f1 = wc.mask1.bitsSetInArray(cell.data.data());
            else
                f1 = wc.mask1.anyBitsInCommon(cell);

            // Check filter2
            bool f2;
            if (wc.mask2_is_all) {
                f2 = true;
            } else if (wc.aggr2) {
                f2 = wc.mask2.bitsSetInArray(cell.data.data());
            } else {
                f2 = wc.mask2.anyBitsInCommon(cell);
            }

            switch (wc.num) {
                case -1:  // NO: fails if ANY tile has both f1 AND f2
                    if (f1 && f2) { conditionMet = false; goto nextCondition; }
                    break;
                case 0:   // SOME: fails if NO tile has both f1 AND f2
                    if (f1 && f2) goto nextCondition;  // found one, condition met
                    break;
                case 1:   // ALL: fails if ANY tile has f1 but NOT f2
                    if (f1 && !f2) { conditionMet = false; goto nextCondition; }
                    break;
            }
        }

        // For SOME conditions, if we never found a match, condition is not met
        if (wc.num == 0) conditionMet = false;

        nextCondition:
        if (!conditionMet) { won = false; break; }
    }

    if (!winconditions_.empty()) {
        winning_ = won;
    }
    return winning_;
}

bool Engine::cellMatchesWinMask(const WinCondition& wc, const BitVec& mask, bool aggregate,
                                bool mask_is_all, int tileIndex) const {
    if (mask_is_all) {
        return true;
    }
    BitVec cell(level_.STRIDE_OBJ);
    level_.getCellInto(tileIndex, cell);
    if (aggregate) {
        return mask.bitsSetInArray(cell.data.data());
    }
    return mask.anyBitsInCommon(cell);
}

// ============================================================
// processInput
// ============================================================
bool Engine::processInput(int dir) {
    againing_ = false;

    LevelBackup bak = backupLevel();

    // Capture player positions BEFORE movement for require_player_movement check
    std::vector<int> playerPositions;
    if (dir >= 0) {
        playerPositions = getPlayerPositions();
        int dirMask;
        switch (dir) {
            case 0: dirMask = 0b00001; break;  // up
            case 1: dirMask = 0b00100; break;  // left
            case 2: dirMask = 0b00010; break;  // down
            case 3: dirMask = 0b01000; break;  // right
            case 4: dirMask = 0b10000; break;  // action
            default: dirMask = 0; break;
        }
        startMovement(dirMask);
    }

    // Initialize turn state
    std::vector<bool> bannedGroup(rules_.size(), false);
    level_.commandQueue.clear();
    level_.commandQueueSourceRules.clear();

    // Save start state for rigid rollback
    std::vector<int32_t> startObjects = level_.objects;
    std::vector<int32_t> startMovements = level_.movements;
    std::vector<BitVec> startRigidGIM;
    std::vector<BitVec> startRigidMAM;
    if (rigid_) {
        startRigidGIM = level_.rigidGroupIndexMask;
        startRigidMAM = level_.rigidMovementAppliedMask;
    }

    sfxCreateMask_.setZero();
    sfxDestroyMask_.setZero();
    level_.calculateRowColMasks();

    // Main loop (with rigid body rollback)
    int iteration = 0;
    bool rigidloop = false;
    do {
        rigidloop = false;
        iteration++;

        applyRules(rules_, loopPoint_, &bannedGroup);
        bool shouldUndo = resolveMovements(bannedGroup);

        if (shouldUndo) {
            rigidloop = true;
            // Rollback to start state
            level_.objects = startObjects;
            level_.movements = startMovements;
            if (rigid_) {
                level_.rigidGroupIndexMask = startRigidGIM;
                level_.rigidMovementAppliedMask = startRigidMAM;
            }
            level_.commandQueue.clear();
            level_.commandQueueSourceRules.clear();
            sfxCreateMask_.setZero();
            sfxDestroyMask_.setZero();
            level_.calculateRowColMasks();
        } else {
            // Apply late rules
            if (!lateRules_.empty()) {
                applyRules(lateRules_, lateLoopPoint_, nullptr);
            }
        }
    } while (iteration < 50 && rigidloop);

    // Check for require_player_movement
    // Uses pre-movement playerPositions captured before startMovement
    // JS checks if player is no longer at ANY of the original positions
    if (dir >= 0 && metadata_.count("require_player_movement") && !playerPositions.empty()) {
        bool someMoved = false;
        for (int pos : playerPositions) {
            int base = pos * STRIDE_OBJ_;
            // Check if player mask bits are ALL clear at this old position
            // (meaning the player has left this cell)
            bool allClear = true;
            for (int i = 0; i < STRIDE_OBJ_; ++i) {
                if (playerMask_.data[i] & level_.objects[base + i]) {
                    allClear = false;
                    break;
                }
            }
            if (allClear) {
                someMoved = true;
                break;
            }
        }
        if (!someMoved) {
            restoreLevel(bak);
            return false;
        }
    }

    // Process command queue
    // CANCEL
    for (const auto& cmd : level_.commandQueue) {
        if (cmd == "cancel") {
            restoreLevel(bak);
            return false;
        }
    }

    // RESTART
    for (const auto& cmd : level_.commandQueue) {
        if (cmd == "restart") {
            restart();
            return true;
        }
    }

    // Check if anything changed
    bool modified = (level_.objects != bak.dat);

    // AGAIN - only if state actually changed
    if (modified) {
        for (const auto& cmd : level_.commandQueue) {
            if (cmd == "again") {
                againing_ = true;
                break;
            }
        }
    }

    // Skip win check when a "message" command is present (mirrors JS textMode behavior)
    bool hasMessage = false;
    for (const auto& cmd : level_.commandQueue) {
        if (cmd == "message") {
            hasMessage = true;
            break;
        }
    }
    if (!hasMessage) {
        checkWin();
    }

    if (winning_) againing_ = false;

    return modified;
}

// ============================================================
// Accessors
// ============================================================
const std::vector<int32_t>& Engine::getObjects() const {
    return level_.objects;
}

int Engine::getWidth() const { return level_.width; }
int Engine::getHeight() const { return level_.height; }
int Engine::getObjectCount() const { return objectCount_; }
bool Engine::hasMetadata(const std::string& key) const { return metadata_.find(key) != metadata_.end(); }
