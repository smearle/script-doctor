import os

import random
from timeit import default_timer as timer

from javascript import require
# ps = require('puzzlescript')
ps = require('./script-doctor-nodejs/node_modules/puzzlescript/lib/index.js')
# terminal = require('./node_modules/puzzlescript-cli/src/cli/terminal.ts')

# game = open("games/sokoban_match3.txt").read()
game = open("data/scraped_games/sokoban_basic.txt").read()
game_data = ps.Parser.parse(game).data
engine = ps.GameEngine(game_data, ps.EmptyGameEngineHandler())
n_steps = 20_000
level_i = 0

def profile_rand_actions():
    # print(engine.solveLevelBFS(0))
    start_time = timer()
    ret = engine.profileRandActions(level_i, n_steps)
    print(f"FPS: {n_steps / (timer() - start_time)}")

def profile_rand_actions_from_python():
    actions = ["UP", "DOWN", "LEFT", "RIGHT", "ACTION"]
    engine.setLevel(level_i, None)
    start_time = timer()
    for i in range(n_steps):
        action_i = random.randint(0, 4)
        action = actions[action_i]
        # engine.press(action)
        # ret = engine.tick()
        ret = engine.takeAction(action)
    print(ret)
    print(f"FPS: {n_steps / (timer() - start_time)}")


if __name__ == "__main__":
    profile_rand_actions_from_python()