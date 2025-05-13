const puzzlescript = require('./puzzlescript/engine.js');
const solver = require('./puzzlescript/solver.js');
const fs = require('node:fs');

try {
  const gameText = fs.readFileSync(process.argv[2], 'utf8');
  let targetLevel = 0;
  let algorithm = solver.solveBFS;
  let type = ""
  for(let i=3; i<process.argv.length; i++) {
    if(i%2 == 1){
      type = process.argv[i];
    }
    else{
      if(type == "--level" || type == "-l"){
        targetLevel = parseInt(process.argv[i]);
      }
      if(type == "--algo" || type == "-a" || type == "--algorithm"){
        if(process.argv[i] == "bfs" || process.argv[i] == "BFS"){
          algorithm = solver.solveBFS;
        }
        else if(process.argv[i] == "AStar" || process.argv[i] == "ASTAR" || process.argv[i] == "A*" || process.argv[i] == "a*"){
          algorithm = solver.solveAStar;
        }
        else if(process.argv[i] == "mcts" || process.argv[i] == "MCTS"){
          algorithm = solver.solveMCTS;
        }
        else{
          console.log("Unknown algorithm: ", process.argv[i]);
          return;
        }
      }
    }
  }
  puzzlescript.compile(gameText, targetLevel);
  console.log(algorithm(puzzlescript));
} catch (err) {
  console.error(err);
}