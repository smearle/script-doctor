#include "engine.h"
#include <cassert>
#include <fstream>
#include <iterator>
#include <utility>
#include <iostream>

int main(int argc, char** argv) {
    assert(argc == 2);
    std::ifstream file(argv[1]);
    std::string game((std::istreambuf_iterator<char>(file)), {});
    Engine first;
    assert(first.loadFromJSON(game));
    first.loadLevel(0, "move-test");
    first.setTrackRulesFired(true);
    first.processInput(3);
#if PUZZLESCRIPT_CPP_HISTORY
    const auto expected = first.serializeState();
    Engine second(std::move(first));
    assert(second.serializeState() == expected);
    Engine third;
    third = std::move(second);
    assert(third.serializeState() == expected);
    assert(third.undo());
    std::cout << "move constructor/assignment preserve complete runtime\n";
#else
    std::cout << "history-free Engine size: " << sizeof(Engine) << "\n";
#endif
}
