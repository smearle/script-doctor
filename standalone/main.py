import os
import javascript

ps = javascript.require('./puzzlescript/engine.js')
solver = javascript.require('./puzzlescript/solver.js')
game_path = os.path.join('games', 'sokoban.ps')
with open(game_path, 'r') as f:
    game_text = f.read() 

ps.compile(game_text, 0)
ret = solver.solveBFS(ps)

print(ret)