const puzzlescript = require('./puzzlescript/engine.js');
const solver = require('./puzzlescript/solver.js');
const fs = require('node:fs');

try {
  const gameText = fs.readFileSync(process.argv[2], 'utf8');
  let targetLevel = 0;
  let algorithm = solver.solveBFS;
  let outputFile = "";
  let numRuns = 1;

  let type = "";
  for(let i=3; i<process.argv.length; i++) {
    if(i%2 == 1){
      type = process.argv[i].toLowerCase();
    }
    else{
      if(type == "--level" || type == "-l"){
        targetLevel = parseInt(process.argv[i]);
      }
      if(type == "--algo" || type == "-a" || type == "--algorithm"){
        if(process.argv[i].toLowerCase() == "bfs"){
          algorithm = solver.solveBFS;
        }
        else if(process.argv[i].toLowerCase() == "astar"){
          algorithm = solver.solveAStar;
        }
        else if(process.argv[i].toLowerCase() == "mcts"){
          algorithm = solver.solveMCTS;
        }
        else{
          console.log("Unknown algorithm: ", process.argv[i]);
          return;
        }
      }
      if(type == "--output" || type == "-o"){
        outputFile = process.argv[i];
      }
    }
  }
  puzzlescript.compile(gameText, targetLevel);
  let result = algorithm(puzzlescript);
  result = {
      "game": process.argv[2],
      "level": targetLevel,
      "algorithm": algorithm.name,
      "isSolved": result[0],
      "actions": result[1],
      "iterations": result[2],
      "fps": result[3],
  };
  if(outputFile.length == 0){
    console.log(result);
  }
  else{
    fs.writeFileSync(outputFile, JSON.stringify(result));
  }
} catch (err) {
  console.error(err);
}