#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "engine.h"
#include "batched_engine.h"
#include "renderer.h"
#include "solver.h"

namespace py = pybind11;

PYBIND11_MODULE(_puzzlescript_cpp, m) {
    m.doc() = "C++ PuzzleScript engine with pybind11 bindings";

    py::class_<RandomRolloutResult>(m, "RandomRolloutResult")
        .def_readonly("iterations", &RandomRolloutResult::iterations)
        .def_readonly("time", &RandomRolloutResult::time)
        .def_readonly("timeout", &RandomRolloutResult::timeout);

    py::class_<SolverResult>(m, "SolverResult")
        .def_readonly("won", &SolverResult::won)
        .def_readonly("actions", &SolverResult::actions)
        .def_readonly("iterations", &SolverResult::iterations)
        .def_readonly("time", &SolverResult::time)
        .def_readonly("score", &SolverResult::score)
        .def_readonly("state", &SolverResult::state)
        .def_readonly("timeout", &SolverResult::timeout)
        .def_readonly("id_dict", &SolverResult::idDict);

    py::class_<MCTSOptions>(m, "MCTSOptions")
        .def(py::init<>())
        .def_readwrite("max_sim_length", &MCTSOptions::maxSimLength)
        .def_readwrite("use_score", &MCTSOptions::useScore)
        .def_readwrite("explore_deadends", &MCTSOptions::exploreDeadends)
        .def_readwrite("deadend_bonus", &MCTSOptions::deadendBonus)
        .def_readwrite("win_bonus", &MCTSOptions::winBonus)
        .def_readwrite("most_visited", &MCTSOptions::mostVisited)
        .def_readwrite("exploration_constant", &MCTSOptions::explorationConstant)
        .def_readwrite("max_iterations", &MCTSOptions::maxIterations);

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("load_from_json", &Engine::loadFromJSON,
             py::arg("json_str"),
             "Load a compiled game state from JSON string")
        .def("load_level", &Engine::loadLevel,
             py::arg("level_index"),
             "Load a specific level by index (0-based, skipping message levels)")
        .def("process_input", &Engine::processInput,
             py::arg("direction"),
             "Process input. dir: 0=up, 1=left, 2=down, 3=right, 4=action, -1=tick. Returns True if anything changed.")
        .def("check_win", &Engine::checkWin,
             "Check win conditions. Returns True if won.")
        .def("get_score", &Engine::getScore,
             "Get the JS-native heuristic score for the current state (lower is better)")
        .def("get_score_normalized", &Engine::getScoreNormalized,
             "Get the normalized JS-native heuristic score for the current state")
        .def("is_winning", &Engine::isWinning,
             "Whether the game is in a winning state")
        .def("is_againing", &Engine::isAgaining,
             "Whether the game wants an 'again' tick")
        .def("get_objects", [](const Engine& e) {
            const auto& objs = e.getObjects();
            return py::array_t<int32_t>(
                {static_cast<py::ssize_t>(objs.size())},
                objs.data()
            );
        }, "Get the current level objects as a numpy int32 array")
        .def("get_objects_2d", [](const Engine& e) {
            const auto& objs = e.getObjects();
            int w = e.getWidth();
            int h = e.getHeight();
            int stride = static_cast<int>(objs.size()) / (w * h);
            // Return as (width, height, stride) array — column-major like JS
            return py::array_t<int32_t>(
                {static_cast<py::ssize_t>(w), static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(stride)},
                objs.data()
            );
        }, "Get objects as (width, height, stride_obj) numpy array")
        .def("get_width", &Engine::getWidth)
        .def("get_height", &Engine::getHeight)
        .def("get_object_count", &Engine::getObjectCount)
        .def("get_num_levels", &Engine::getNumLevels)
        .def("get_id_dict", &Engine::getIdDict,
             "Get the object ID → name mapping")
        .def("restart", &Engine::restart,
             "Restart the current level")
        .def("backup_level", &Engine::backupLevel,
             "Create a backup of the current level state")
        .def("restore_level", &Engine::restoreLevel,
             py::arg("backup"),
             "Restore level from a backup");

    py::class_<LevelBackup>(m, "LevelBackup")
        .def_readonly("dat", &LevelBackup::dat)
        .def_readonly("width", &LevelBackup::width)
        .def_readonly("height", &LevelBackup::height);

    // ---- BatchedEngine (vectorized gym-style interface) ---------------
    py::class_<BatchedEngine>(m, "BatchedEngine")
        .def(py::init<int>(), py::arg("batch_size"))
        .def("load_from_json", &BatchedEngine::loadFromJSON,
             py::arg("json_str"),
             "Load compiled game JSON into all engines")
        .def("set_levels", &BatchedEngine::setLevels,
             py::arg("level_indices"),
             "Assign level index per environment")
        .def("reset", &BatchedEngine::reset,
             py::arg("env_indices"),
             "Reset specific environments (empty list = reset all)")
        .def("reset_all", &BatchedEngine::resetAll,
             "Reset all environments")
        .def("step", [](BatchedEngine& be, py::array_t<int32_t> actions) {
            auto buf = actions.request();
            if (buf.ndim != 1 || buf.shape[0] != be.batchSize()) {
                throw std::runtime_error("actions must be 1-d array of length batch_size");
            }
            std::vector<int> acts(be.batchSize());
            const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
            for (int i = 0; i < be.batchSize(); ++i) acts[i] = ptr[i];
            be.step(acts);
        }, py::arg("actions"),
           "Step all envs. actions: int32 array of shape (batch,)")
        .def("get_obs", [](const BatchedEngine& be) {
            auto shape = be.getObsShape();
            const auto& obs = be.getObs();
            return py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(shape[0]),
                 static_cast<py::ssize_t>(shape[1]),
                 static_cast<py::ssize_t>(shape[2]),
                 static_cast<py::ssize_t>(shape[3])},
                obs.data()
            );
        }, "Get observations as uint8 array of shape (batch, n_objs, height, width)")
        .def("get_rewards", [](const BatchedEngine& be) {
            const auto& r = be.getRewards();
            return py::array_t<float>(
                {static_cast<py::ssize_t>(r.size())},
                r.data()
            );
        }, "Get rewards as float32 array of shape (batch,)")
        .def("get_scores", [](const BatchedEngine& be) {
            const auto& s = be.getScores();
            return py::array_t<float>(
                {static_cast<py::ssize_t>(s.size())},
                s.data()
            );
        }, "Get heuristic scores as float32 array of shape (batch,)")
        .def("get_score_deltas", [](const BatchedEngine& be) {
            const auto& s = be.getScoreDeltas();
            return py::array_t<float>(
                {static_cast<py::ssize_t>(s.size())},
                s.data()
            );
        }, "Get heuristic score deltas as float32 array of shape (batch,)")
        .def("get_dones", [](const BatchedEngine& be) {
            const auto& d = be.getDones();
            // Convert vector<bool> to numpy bool array
            py::array_t<bool> arr({static_cast<py::ssize_t>(d.size())});
            auto ptr = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < d.size(); ++i) ptr(i) = d[i];
            return arr;
        }, "Get dones as bool array of shape (batch,)")
        .def("get_wins", [](const BatchedEngine& be) {
            const auto& w = be.getWins();
            py::array_t<bool> arr({static_cast<py::ssize_t>(w.size())});
            auto ptr = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < w.size(); ++i) ptr(i) = w[i];
            return arr;
        }, "Get wins as bool array of shape (batch,)")
        .def("get_obs_shape", &BatchedEngine::getObsShape)
        .def_property_readonly("batch_size", &BatchedEngine::batchSize)
        .def_property_readonly("num_objects", &BatchedEngine::numObjects)
        .def_property_readonly("level_width", &BatchedEngine::levelWidth)
        .def_property_readonly("level_height", &BatchedEngine::levelHeight)
        .def_property_readonly("num_levels", &BatchedEngine::numLevels)
        .def("get_objects", [](const BatchedEngine& be, int env_idx) {
            const auto& objs = be.getObjects(env_idx);
            return py::array_t<int32_t>(
                {static_cast<py::ssize_t>(objs.size())},
                objs.data()
            );
        }, py::arg("env_idx"),
           "Get raw objects array for a single environment")
        .def("get_width", &BatchedEngine::getWidth, py::arg("env_idx"))
        .def("get_height", &BatchedEngine::getHeight, py::arg("env_idx"));

    m.def("random_rollout_raw", &randomRolloutRaw,
          py::arg("engine"), py::arg("max_iters") = 100000, py::arg("timeout_ms") = -1);
    m.def("solve_random", &solveRandom,
          py::arg("engine"), py::arg("max_length") = 100, py::arg("max_iters") = 100000, py::arg("timeout_ms") = 60000);
    m.def("solve_bfs", &solveBFS,
          py::arg("engine"), py::arg("max_iters") = 100000, py::arg("timeout_ms") = -1);
    m.def("solve_astar", &solveAStar,
          py::arg("engine"), py::arg("max_iters") = 100000, py::arg("timeout_ms") = -1);
    m.def("solve_gbfs", &solveGBFS,
          py::arg("engine"), py::arg("max_iters") = 100000, py::arg("timeout_ms") = -1);
    m.def("solve_mcts", &solveMCTS,
          py::arg("engine"), py::arg("options") = MCTSOptions());

    // ---- Renderer (sprite-based frame rendering) ---------------------
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<>())
        .def("load_sprite_data", &Renderer::loadSpriteData,
             py::arg("json_str"),
             "Load sprite data from JSON produced by serializeSpriteDataJSON()")
        .def("ready", &Renderer::ready,
             "Whether sprite data has been loaded")
        .def_property_readonly("cell_width", &Renderer::cellWidth)
        .def_property_readonly("cell_height", &Renderer::cellHeight)
        .def_property_readonly("num_sprites", &Renderer::numSprites)
        .def("render_engine", [](const Renderer& r, const Engine& e) {
            const auto& objs = e.getObjects();
            int w = e.getWidth();
            int h = e.getHeight();
            int n_objs = e.getObjectCount();
            auto frame = r.renderFromObjects(objs.data(), w, h, n_objs);
            int fh = h * r.cellHeight();
            int fw = w * r.cellWidth();
            return py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(fh),
                 static_cast<py::ssize_t>(fw),
                 static_cast<py::ssize_t>(3)},
                frame.data()
            );
        }, py::arg("engine"),
           "Render current state of an Engine as (H_px, W_px, 3) uint8 array")
        .def("render_obs", [](const Renderer& r,
                              py::array_t<uint8_t> obs,
                              int n_objs, int height, int width) {
            auto buf = obs.request();
            const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
            auto frame = r.renderFromObs(ptr, n_objs, height, width);
            int fh = height * r.cellHeight();
            int fw = width  * r.cellWidth();
            return py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(fh),
                 static_cast<py::ssize_t>(fw),
                 static_cast<py::ssize_t>(3)},
                frame.data()
            );
        }, py::arg("obs"), py::arg("n_objs"), py::arg("height"), py::arg("width"),
           "Render from multihot obs (n_objs, H, W) as (H_px, W_px, 3) uint8 array")
        .def("render_objects", [](const Renderer& r,
                                  py::array_t<int32_t> objects,
                                  int width, int height, int n_objs) {
            auto buf = objects.request();
            const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
            auto frame = r.renderFromObjects(ptr, width, height, n_objs);
            int fh = height * r.cellHeight();
            int fw = width  * r.cellWidth();
            return py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(fh),
                 static_cast<py::ssize_t>(fw),
                 static_cast<py::ssize_t>(3)},
                frame.data()
            );
        }, py::arg("objects"), py::arg("width"), py::arg("height"), py::arg("n_objs"),
           "Render from raw column-major int32 objects array as (H_px, W_px, 3) uint8 array")
        .def("render_batched_env", [](const Renderer& r, const BatchedEngine& be,
                                      int env_idx) {
            const auto& objs = be.getObjects(env_idx);
            int w = be.getWidth(env_idx);
            int h = be.getHeight(env_idx);
            int n_objs = be.numObjects();
            auto frame = r.renderFromObjects(objs.data(), w, h, n_objs);
            int fh = h * r.cellHeight();
            int fw = w * r.cellWidth();
            return py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(fh),
                 static_cast<py::ssize_t>(fw),
                 static_cast<py::ssize_t>(3)},
                frame.data()
            );
        }, py::arg("batched_engine"), py::arg("env_idx"),
           "Render one env of a BatchedEngine as (H_px, W_px, 3) uint8 array");
}
