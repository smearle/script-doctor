#include "engine.h"
#include <exception>
#include <limits>
#include <sstream>

// History grows with play. Ordinary steps snapshot its SIZE only; copy the
// old prefix lazily if a nested Undo or level reload will destroy entries.
struct Engine::ExecutionSnapshot {
    Level level;
    RNG rng;
    int curLevel;
    bool winning, againing, restarting, startup;
    BitVec create, destroy, current, accumulated;
#if PUZZLESCRIPT_CPP_HISTORY
    size_t undoSize;
    LevelBackup restartTarget;
    std::unique_ptr<std::vector<LevelBackup>> undo;
#endif
    explicit ExecutionSnapshot(const Engine& e)
        : level(e.level_), rng(e.rng_), curLevel(e.curLevel_),
          winning(e.winning_), againing(e.againing_), restarting(e.restarting_),
          startup(e.runRulesOnLevelStart_), create(e.sfxCreateMask_),
          destroy(e.sfxDestroyMask_), current(e.firedThisCall_),
          accumulated(e.firedAccumulated_)
#if PUZZLESCRIPT_CPP_HISTORY
          , undoSize(e.undoHistory_.size()), restartTarget(e.restartTarget_)
#endif
    {}
};

Engine::ExecutionScope::ExecutionScope(Engine& e)
    : engine_(e), exceptions_(std::uncaught_exceptions()) {
    if (e.activeExecution_) return;
    // Complete the allocation before installing the active scope.
    snapshot_ = std::make_unique<ExecutionSnapshot>(e);
    e.executionWork_ = 0;
    e.executionRuleLine_ = -1;
    e.activeExecution_ = this;
}

Engine::ExecutionScope::~ExecutionScope() noexcept {
    if (!snapshot_) return;
    auto& e = engine_;
    if (std::uncaught_exceptions() > exceptions_) {
        auto& s = *snapshot_;
        e.level_ = std::move(s.level); e.rng_ = s.rng;
        e.curLevel_ = s.curLevel; e.winning_ = s.winning;
        e.againing_ = s.againing; e.restarting_ = s.restarting;
        e.runRulesOnLevelStart_ = s.startup;
        e.sfxCreateMask_ = std::move(s.create);
        e.sfxDestroyMask_ = std::move(s.destroy);
        e.firedThisCall_ = std::move(s.current);
        e.firedAccumulated_ = std::move(s.accumulated);
#if PUZZLESCRIPT_CPP_HISTORY
        if (s.undo) e.undoHistory_ = std::move(*s.undo);
        else e.undoHistory_.resize(s.undoSize);  // shrink appended entries only
        e.restartTarget_ = std::move(s.restartTarget);
#endif
    }
    e.activeExecution_ = nullptr;
}

void Engine::preserveUndoHistory() {
#if PUZZLESCRIPT_CPP_HISTORY
    if (!activeExecution_) return;
    auto& s = *activeExecution_->snapshot_;
    if (!s.undo) {
        s.undo = std::make_unique<std::vector<LevelBackup>>(
            undoHistory_.begin(), undoHistory_.begin() + s.undoSize);
    }
#endif
}

void Engine::setExecutionBudget(uint64_t units) {
    if (!units || units > static_cast<uint64_t>(INT64_MAX))
        throw std::invalid_argument("Execution budget must be in [1, INT64_MAX]");
    if (activeExecution_) throw std::logic_error("Cannot change an active execution budget");
    executionBudget_ = units;
}

[[noreturn]] void Engine::budgetExceeded(const char* stage) const {
    std::ostringstream out;
    out << "PuzzleScript execution budget exceeded during " << stage
        << " (rule line " << executionRuleLine_
        << ", used " << executionWork_ << ", limit " << executionBudget_ << ")";
    throw ExecutionBudgetExceeded(out.str());
}

void Engine::spendWork(uint64_t units, const char* stage) {
    if (units > executionBudget_ - executionWork_) budgetExceeded(stage);
    executionWork_ += units;
}

bool Engine::matchCell(const CellPattern& cell, int index) {
    spendWork(1, "cell matching");
    return cell.matches(index, level_.objects.data(), level_.movements.data(),
                        STRIDE_OBJ_, STRIDE_MOV_);
}

uint64_t Engine::reserveTupleWork(const MatchLists& lists) {
    if (lists.empty()) return 0;
    // Each tuple binds one match per row. Check BEFORE multiplying, allocating
    // or applying anything: a product such as 2^88 never needs to be represented.
    const uint64_t remaining = executionBudget_ - executionWork_;
    const uint64_t maxTuples = remaining / lists.size();
    uint64_t count = 1;
    for (const auto& row : lists) {
        if (row.empty()) return 0;
        if (count > maxTuples / row.size()) budgetExceeded("tuple bindings");
        count *= row.size();
    }
    spendWork(count * lists.size(), "tuple bindings");
    return count;
}

std::vector<std::vector<int>> Engine::tupleAt(const MatchLists& lists, uint64_t index) {
    std::vector<std::vector<int>> result;
    result.reserve(lists.size());
    // The reference engine's first pattern row varies fastest.
    for (const auto& row : lists) {
        result.push_back(row[index % row.size()]);
        index /= row.size();
    }
    return result;
}
