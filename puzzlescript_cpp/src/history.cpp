#include "engine.h"
#if PUZZLESCRIPT_CPP_HISTORY
#include "json.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
using json = nlohmann::json;

bool Engine::undo(bool force, bool ignoreDuplicates) {
    if (!force && hasMetadata("noundo")) return false;
    preserveUndoHistory();
    if (ignoreDuplicates) {
        while (!undoHistory_.empty() && undoHistory_.back().dat == level_.objects)
            undoHistory_.pop_back();
    }
    if (undoHistory_.empty()) return false;
    const bool wasWinning = winning_;
    restoreLevel(undoHistory_.back());
    winning_ = wasWinning;  // native restoreLevel does not clear the win flag
    undoHistory_.pop_back();
    return true;
}

void Engine::restartGame(bool force) {
    ExecutionScope execution(*this);
    if (restarting_ || (!force && hasMetadata("norestart"))) return;
    if (restartTarget_.dat.empty())
        throw std::logic_error("Load a level before restarting");
    if (againing_) undo(force, true);
    if (!force) undoHistory_.push_back(backupLevel());
    const bool wasWinning = winning_;
    restarting_ = true;
    try {
        restoreLevel(restartTarget_);
        winning_ = wasWinning;
        if (hasMetadata("run_rules_on_level_start")) processInput(-1, true);
        level_.commandQueue.clear();
        level_.commandQueueSourceRules.clear();
    } catch (...) { restarting_ = false; throw; }
    restarting_ = false;
}

namespace {
json boardJSON(const LevelBackup& b) {
    return {{"width", b.width}, {"height", b.height}, {"dat", b.dat}};
}
json vectorsJSON(const std::vector<BitVec>& vectors) {
    auto a = json::array();
    for (const auto& v : vectors) a.push_back(v.data);
    return a;
}
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
int64_t integer(const json& value, int64_t minimum, int64_t maximum) {
    require(value.is_number_integer(), "Snapshot value must be an integer");
    if (value.is_number_unsigned())
        require(value.get<uint64_t>() <= static_cast<uint64_t>(maximum),
                "Snapshot integer out of range");
    const auto n = value.get<int64_t>();
    require(n >= minimum && n <= maximum, "Snapshot integer out of range");
    return n;
}
std::vector<int32_t> words(const json& a, size_t size) {
    require(a.is_array() && a.size() == size, "Invalid snapshot array size");
    std::vector<int32_t> result;
    result.reserve(size);
    for (const auto& x : a) {
        const auto n = integer(x, INT32_MIN, INT32_MAX);
        result.push_back(static_cast<int32_t>(n));
    }
    return result;
}
BitVec bits(const json& a, size_t size) {
    BitVec result(static_cast<int>(size));
    result.data = words(a, size);
    return result;
}
std::vector<BitVec> vectors(const json& a, size_t count, size_t stride) {
    require(a.is_array() && a.size() == count, "Invalid snapshot vector count");
    std::vector<BitVec> result;
    result.reserve(count);
    for (const auto& v : a) result.push_back(bits(v, stride));
    return result;
}
LevelBackup board(const json& j, int w, int h, int stride) {
    require(j.at("width") == w && j.at("height") == h,
            "Snapshot board dimensions do not match the compiled level");
    LevelBackup result;
    result.width = w; result.height = h;
    result.dat = words(j.at("dat"), size_t(w) * h * stride);
    return result;
}
} // namespace

std::string Engine::serializeState() const {
    if (compiledGame_.empty() || restartTarget_.dat.empty())
        throw std::logic_error("Load a compiled game and level before serializing");
    auto history = json::array();
    for (const auto& b : undoHistory_) history.push_back(boardJSON(b));
    std::vector<int> permutation(rng_.rc4.s, rng_.rc4.s + 256);
    return json{
        {"format", "puzzlescript-cpp-runtime"}, {"version", 2},
        {"execution_budget", executionBudget_},
        // Exact identity, not a collision-prone hash. This also preserves the
        // compiler output needed to construct a fresh engine in another process.
        {"compiled_game", compiledGame_},
        {"level_index", curLevel_}, {"board", boardJSON(backupLevel())},
        {"movements", level_.movements},
        {"rigid_group_masks", vectorsJSON(level_.rigidGroupIndexMask)},
        {"rigid_movement_masks", vectorsJSON(level_.rigidMovementAppliedMask)},
        {"commands", level_.commandQueue},
        {"command_sources", level_.commandQueueSourceRules},
        {"winning", winning_}, {"againing", againing_},
        {"restart_target", boardJSON(restartTarget_)}, {"undo", history},
        {"rng", {{"s", permutation}, {"i", rng_.rc4.i}, {"j", rng_.rc4.j},
                 {"has_normal", rng_.has_normal}, {"cached_normal", rng_.cached_normal}}},
        {"telemetry", {{"enabled", trackRulesFired_},
                       {"current", firedThisCall_.data},
                       {"accumulated", firedAccumulated_.data}}},
        {"sfx_create", sfxCreateMask_.data}, {"sfx_destroy", sfxDestroyMask_.data}
    }.dump();
}

void Engine::restoreState(const std::string& state) {
    // Parse and validate everything in temporaries. A rejected snapshot must
    // leave the live engine (including its RNG and history) unchanged.
    const auto j = json::parse(state);
    require(j.at("format") == "puzzlescript-cpp-runtime" && (j.at("version") == 1 || j.at("version") == 2),
            "Unsupported PuzzleScript runtime snapshot");
    require(!compiledGame_.empty() && j.at("compiled_game") == compiledGame_,
            "Load the same compiled game before restoring this snapshot");
    const uint64_t budget = j.at("version") == 1 ? DEFAULT_EXECUTION_BUDGET
        : static_cast<uint64_t>(integer(j.at("execution_budget"), 1, INT64_MAX));
    const int index = static_cast<int>(integer(j.at("level_index"), 0, INT32_MAX));
    const LevelDef* definition = nullptr;
    int playable = 0;
    for (const auto& ld : levels_) {
        if (!ld.isMessage && playable++ == index) { definition = &ld; break; }
    }
    require(definition != nullptr, "Snapshot level index out of range");
    const auto& ld = *definition;
    const auto b = board(j.at("board"), ld.width, ld.height, STRIDE_OBJ_);
    Level next(ld.lineNumber, ld.width, ld.height, ld.layerCount, STRIDE_OBJ_, STRIDE_MOV_);
    next.initMasks(rigid_);
    next.objects = b.dat;
    next.movements = words(j.at("movements"), size_t(next.n_tiles) * STRIDE_MOV_);
    next.rigidGroupIndexMask = vectors(j.at("rigid_group_masks"),
                                      rigid_ ? next.n_tiles : 0, STRIDE_MOV_);
    next.rigidMovementAppliedMask = vectors(j.at("rigid_movement_masks"),
                                           rigid_ ? next.n_tiles : 0, STRIDE_MOV_);
    next.commandQueue = j.at("commands").get<std::vector<std::string>>();
    next.commandQueueSourceRules = j.at("command_sources").get<std::vector<int>>();
    next.calculateRowColMasks();
    auto restart = board(j.at("restart_target"), ld.width, ld.height, STRIDE_OBJ_);
    require(j.at("undo").is_array(), "Invalid Undo history");
    std::vector<LevelBackup> history;
    for (const auto& entry : j.at("undo"))
        history.push_back(board(entry, ld.width, ld.height, STRIDE_OBJ_));
    RNG nextRng;
    const auto& rng = j.at("rng");
    const auto permutation = words(rng.at("s"), 256);
    std::array<bool, 256> seen{};
    for (size_t k = 0; k < permutation.size(); ++k) {
        const auto n = permutation[k];
        require(n >= 0 && n <= 255 && !seen[n], "Invalid RC4 permutation");
        seen[n] = true;
        nextRng.rc4.s[k] = static_cast<uint8_t>(n);
    }
    nextRng.rc4.i = static_cast<int>(integer(rng.at("i"), 0, 255));
    nextRng.rc4.j = static_cast<int>(integer(rng.at("j"), 0, 255));
    require(nextRng.rc4.i >= 0 && nextRng.rc4.i <= 255 &&
            nextRng.rc4.j >= 0 && nextRng.rc4.j <= 255, "Invalid RC4 index");
    nextRng.has_normal = rng.at("has_normal").get<bool>();
    nextRng.cached_normal = rng.at("cached_normal").get<double>();
    require(std::isfinite(nextRng.cached_normal), "Invalid cached RNG sample");
    const bool win = j.at("winning").get<bool>(), again = j.at("againing").get<bool>();
    const auto& telemetry = j.at("telemetry");
    const bool track = telemetry.at("enabled").get<bool>();
    const size_t nwords = (allRules_.size() + 31) / 32;
    const size_t savedWords = telemetry.at("current").size();
    require(savedWords == nwords || (!track && savedWords == 0),
            "Invalid telemetry size");
    auto current = bits(telemetry.at("current"), savedWords);
    auto accumulated = bits(telemetry.at("accumulated"), savedWords);
    auto create = bits(j.at("sfx_create"), STRIDE_OBJ_);
    auto destroy = bits(j.at("sfx_destroy"), STRIDE_OBJ_);

    executionBudget_ = budget;
    executionWork_ = 0;
    level_ = std::move(next); curLevel_ = index;
    restartTarget_ = std::move(restart); undoHistory_ = std::move(history);
    rng_ = nextRng; winning_ = win; againing_ = again;
    trackRulesFired_ = track;
    firedThisCall_ = std::move(current); firedAccumulated_ = std::move(accumulated);
    sfxCreateMask_ = std::move(create); sfxDestroyMask_ = std::move(destroy);
    restarting_ = runRulesOnLevelStart_ = false;
}
#endif
