#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(_puzzlescript_cpp, m) {
    m.doc() = "C++ PuzzleScript engine with pybind11 bindings";

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
        .def_readonly("width", &LevelBackup::width)
        .def_readonly("height", &LevelBackup::height);
}
