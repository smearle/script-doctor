// const { Deque } = import('./collections'); // Use a deque for efficient pop/push

function getConsoleText() {
  // This probably exists somewhere else already?
  var consoleOut = document.getElementById('consoletextarea');

  // Initialize an empty array to store the extracted text
  var textContentArray = [];

  // Iterate over all child divs inside the consoletextarea
  consoleOut.querySelectorAll('div').forEach(function(div) {
      // Push the plain text content of each div into the array
      textContentArray.push(div.textContent.trim());
  });

  // Join the array elements with line breaks (or other delimiter)
  var plainTextOutput = textContentArray.join('\n');

  return plainTextOutput
}

class GameIndividual {
  constructor(code, minCode, fitness, maxMeanSolComplexity, compiledIters, solvedIters, anySolvedIters, skipped) {
    this.code = code;
    this.minCode = minCode;
    this.fitness = fitness;
    this.maxMeanSolComplexity = maxMeanSolComplexity;
    this.compiledIters = compiledIters;
    this.solvedIters = solvedIters;
    this.anySolvedIters = anySolvedIters;
    this.skipped = skipped;
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function playTest() {
  // const game = 'sokoban_match3';
  const game = 'sokoban_basic';
  const n_level = 0;

  response = await fetch('/load_game_from_file', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      'game': game,
    }),
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  code = await response.text();
  loadFile(code);
  console.log(code)

  editor.setValue(code);
  editor.clearHistory();
  clearConsole();
  setEditorClean();
  unloadGame();
  compile(['restart'], code);

  console.log('Playtesting...');
  compile(['loadLevel', n_level], editor.getValue());
  console.log('Solving level:', n_level, ' with A*');
  var [sol_a, n_search_iters_a] = await solveLevelBestFirst(level_i=n_level);


  editor.setValue(code);
  editor.clearHistory();
  clearConsole();
  setEditorClean();
  unloadGame();
  console.log('Solving level:', n_level, ' with BFS');
  [sol_a, n_search_iters_a] = await solveLevelBFS(n_level);
  // const [sol, n_search_iters] = await solveLevelBFS(n_level);
  // gameToLoad = '/demo/sokoban_match3.txt';
  // gameToLoad = '/misc/3d_sokoban.txt';
  // sol = await solveLevel(0);

  // Load the the text file demo/sokoban_match3.txt
  // tryLoadFile('sokoban_match3');
  // var client = new XMLHttpRequest();
  // client.open('GET', gameToLoad);
  // client.onreadystatechange = async function() {
  //   console.log('Ready state:', client.readyState);
  //   console.log('Response', client.responseText);
  //   editor.setValue(client.responseText);
  //   sol = await solveLevel(0);
  //   console.log('Solution:', sol);
  // }
  // await client.send();
  // console.log('Loaded level:', editor.getValue());
}


function serialize(val) {
  return JSON.stringify(val);
}

class Queue {
  constructor() {
    this.inStack = [];
    this.outStack = [];
  }

  enqueue(value) {
    this.inStack.push(value);
  }

  dequeue() {
    if (this.outStack.length === 0) {
      while (this.inStack.length > 0) {
        this.outStack.push(this.inStack.pop());
      }
    }
    return this.outStack.pop();
  }

  isEmpty() {
    return this.inStack.length === 0 && this.outStack.length === 0;
  }

  size() {
    return this.inStack.length + this.outStack.length;
  }
}

function byScoreAndLength2(a, b) {
	// if (a[2] != b[2]) {
	// 	return a[2] < b[2];
	// } else {
	// 	return a[0] < b[0];
	// }
	
	if (a[0] != b[0]) {
		return a[0] < b[0];
	} else {
		return a[2].length < b[2].length;
	}
}


function hashStateObjects(state) {
  return JSON.stringify(state).split('').reduce((hash, char) => {
    return (hash * 31 + char.charCodeAt(0)) % 1_000_000_003; // Simple hash
  }, 0);
}


async function solveLevelBFS(levelIdx, captureStates=false, maxIters=1_000_000) {
  console.log('max iters:', maxIters);
	precalcDistances();

  // Load the level
  compile(['loadLevel', levelIdx], editor.getValue());
  init_level = backupLevel();
  init_level_map = init_level['dat'];

  // frontier = [init_level];
  // action_seqs = [[]];
  // frontier = new Queue();
  // action_seqs = new Queue();

  frontier = new Queue();

  frontier.enqueue([init_level, []]);
  // action_seqs.enqueue([]);

  var sol = [];
  var bestScore = -10000;
  console.log(sol.length);
  // visited = new Set([hashState(init_level_map)]);
  visited = {};
  // visited[level.objects] = true;
  i = 0;
  start_time = Date.now();
  console.log(frontier.size())
  while (frontier.size() > 0) {
    backups = [];

    // const level = frontier.shift();
    // const action_seq = action_seqs.shift();
    const [parent_level, action_seq] = frontier.dequeue();
    // const action_seq = action_seqs.dequeue();

    if (!action_seq) {
      console.log(`Action sequence is empty. Length of frontier: ${frontier.size()}`);
    }
    for (const move of Array(5).keys()) {
      if (i > maxIters) {
        console.log('Exceeded ' + maxIters + ' iterations. Exiting.');
        return [-1, i];
      }
      restoreLevel(parent_level);

      new_action_seq = action_seq.slice();
      new_action_seq.push(move);
      // try {
      //   changed = processInputSearch(move);
      // } catch (e) {
      //   console.log('Error while processing input:', e);
      //   return [-2, i];
      // }
			var changed = processInputSearch(move);
			while (againing) {
				changed = processInputSearch(-1) || changedSomething;
			}
      if (winning) {
        console.log(`Winning! Solution:, ${new_action_seq}\n Iterations: ${i}`);
        console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
        return [new_action_seq, i];
      }
      else if (changed) {
        new_level = backupLevel();
        // new_level_map = new_level['dat'];
        // const newHash = hashState(new_level_map);
        // if (!visited.has(newHash)) {
        if (!(level.objects in visited)) {
          // console.log('New state found:', level.objects);

          // UNCOMMENT THESE LINES FOR VISUAL DEBUGGING
          // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
          // redraw();
          ///////////////////////////////////////////////

          frontier.enqueue([new_level, new_action_seq]);
          // frontier.enqueue(new_level);
          if (!new_action_seq) {
            console.log(`New action sequence is undefined when pushing.`);
          }
          // action_seqs.enqueue(new_action_seq);

          // visited.add(newHash);
          visited[level.objects] = true;
        } 
        // console.log('State already visited:', level.objects);
      }
    }
    if (i % 10000 == 0) {
      now = Date.now();
      console.log('Iteration:', i);
      console.log('FPS:', (i / (now - start_time) * 1000).toFixed(2));
      console.log(`Size of frontier: ${frontier.size()}`);
      console.log(`Visited states: ${Object.keys(visited).length}`);
      // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
      // redraw();
    }
    i++;
  }
  return [sol, i];
}

class MCTSNode{
  constructor(action, parent, max_children) {
    this.parent = parent;
    this.action = action;
    this.children = [];
    for(let i=0; i<max_children; i++){
      this.children.push(null);
    }
    this.visits = 0;
    this.score = 0;
  }

  ucb_score(c) {
    if(this.parent == null){
      return this.score / this.visits;
    }
    return this.score / this.visits + c * Math.sqrt(Math.log(this.parent.visits) / this.visits);
  }

  select(c){
    if(!this.is_fully_expanded()){
      return null;
    }
    let index = 0;
    for(let i=0; i<this.children.length; i++){
      if(this.children[i].ucb_score(c) > this.children[index].ucb_score(c)){
        index = i;
      }
    }
    return this.children[index];
  }

  is_fully_expanded(){
    for(let child of this.children){
      if(child == null){
        return false;
      }
    }
    return true;
  }

  expand(){
    if(this.is_fully_expanded()){
      return null;
    }
    for(let i=0; i<this.children.length; i++){
      if(this.children[i] == null){
        let changed = processInputSearch(i);
        let level = this.level;
        if(changed){
          level = backupLevel();
        }
        this.children[i] = new MCTSNode(i, this, this.children.length)
        return this.children[i];
      }
    }
    return null;
  }

  backup(score){
    this.score += score;
    this.visits += 1;
    if(this.parent != null){
      this.parent.backup(score);
    }
  }

  simulate(max_length, score_fn, win_bonus){
    let changes = 0;
    for(let i=0; i<max_length; i++){
      let changed = processInputSearch(Math.min(5, Math.floor(Math.random() * 6)));
      if(changed){
        changes += 1;
      }
      if(winning){
        return win_bonus;
      }
    }
    if(score_fn){
      return score_fn();
    }
    return (changes / max_length);
  }

  get_actions(){
    let sol = [];
    let current = this;
    while(current.parent != null){
      sol.push(current.action);
      current = current.parent;
    }
    return sol.reverse();
  }

  get_most_visited_action(){
    let max_action = 0;
    for(let i=0; i<this.children.length; i++){
      if(this.children[i].visits > this.children[max_action].visits){
        max_action = i;
      }
    }
    return max_action;
  }

  get_best_action(){
    let max_action = 0;
    for(let i=0; i<this.children.length; i++){
      if(this.children[i].score / this.children[i].visited > this.children[max_action].score / this.children[max_action].visited){
        max_action = i;
      }
    }
    return max_action;
  }
}

// level: is the starting level
// max_sim_length: maximum number of random simulation before stopping and backpropagate
// score_fn: if you want to use heuristic function which is advisable and make sure the values are always between 0 and 1
// explore_deadends: if you want to explore deadends by default, the search don't continue in deadends
// deadend_bonus: bonus when you find a deadend node (usually negative number to avoid)
// most_visited: decide to return most visited action or best value action
// win_bonus: bonus when you find a winning node
// c: is the MCTS constant that balance between exploitation and exploration
// max_iterations: max number of iterations before you consider the solution is not available
async function solveLevelMCTS(level, options = {}) {
  // Load the level
  if(options == null){
    options = {};
  }
  let defaultOptions = {
    "max_sim_length": 1000,
    "score_fn": null, 
    "explore_deadends": false, 
    "deadend_bonus": -100, 
    "win_bonus": 100,
    "most_visited": true,
    "c": Math.sqrt(2), 
    "max_iterations": -1
  };
  for(let key in defaultOptions){
    if(!options.hasOwnProperty(key)){
      options[key] = defaultOptions[key];
    }
  }
  compile(['loadLevel', level], editor.getValue());
  init_level = backupLevel();
  init_level_map = init_level['dat'];
  let rootNode = new MCTSNode(-1, null, 5);
  let i = 0;
  let deadend_nodes = 1;
  let start_time = Date.now();
  while(options.max_iterations <= 0 || (options.max_iterations > 0 && i < options.max_iterations)){
    // start from th root
    currentNode = rootNode;
    restoreLevel(init_level);
    let changed = true;
    // selecting next node
    while(currentNode.is_fully_expanded()){
      currentNode = currentNode.select(options.c);
      changed = processInputSearch(currentNode.action);
      if(winning){
        let sol = current.get_actions();
        console.log(`Winning! Solution:, ${sol}\n Iterations: ${i}`);
        console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
        return [sol, i];
      }
      if(!options.explore_deadends && !changed){
        break;
      }
    }

    // if node is deadend, punish it
    if(!options.explore_deadends && !changed){
      currentNode.score += options.deadend_bonus;
      currentNode.backup(0);
      deadend_nodes += 1;
    }
    //otherwise expand
    else{
      currentNode = currentNode.expand();
      changed = processInputSearch(currentNode.action);
      if(winning){
        let sol = current.get_actions();
        console.log(`Winning! Solution:, ${sol}\n Iterations: ${i}`);
        console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
        return [sol, i];
      }
      // if node is deadend, punish it
      if(!options.explore_deadends && !changed){
        currentNode.score += options.deadend_bonus;
        currentNode.backup(0);
        deadend_nodes += 1;
        
      }
      //otherwise simulate then backup
      else{
        let value = currentNode.simulate(options.max_sim_length, options.score_fn, options.win_bonus);
        currentNode.backup(value);
      }
    }
    // print progress
    if (i % 10000 == 0) {
      now = Date.now();
      console.log('Iteration:', i);
      console.log('FPS:', (i / (now - start_time) * 1000).toFixed(2));
      console.log(`Visited Deadends: ${deadend_nodes}`);
      // console.log(`Visited states: ${visited.size}`);
      // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
      // redraw();
    }
    i+= 1;
  }
  let actions = [];
  currentNode = rootNode;
  while(currentNode.is_fully_expanded()){
    let action = -1;
    if(options.most_visited){
      action = currentNode.get_most_visited_action();
    }
    else{
      action = currentNode.get_best_action();
    }
    actions.push(action);
    currentNode = currentNode.children[action];
  }
  return [actions, options.max_iterations];
}

async function testBFS() {
  console.log('Testing BFS...');
  // Get title of the game
  const title = state.metadata.title;
  console.log('Title:', title);
  // Determine how many levels are available in this game
  const n_levels = state.levels.length;
  console.log('Number of levels:', n_levels);
  // Iterate through levels
  for (let i = 0; i < n_levels; i++) {
    compile(['loadLevel', i], editor.getValue());
    console.log('Solving level:', i, ' with BFS');
    var [sol, nSearchIters] = await solveLevelBFS(i);
    console.log('Solution:', sol);
    console.log('Iterations:', nSearchIters);
    inputHistory = sol;
    const solDir = `sols/${title}/level_${i}`;
    const [dataURL, filename] = makeGIFDoctor();
    await fetch ('/save_sol', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        levelIdx: i,
        sol: sol,
        solDir: solDir,
        dataURL: dataURL,
      })
    });
  }
}

async function testMCTS() {
  console.log('Testing MCTS...');
  const n_level = 0;
  compile(['loadLevel', n_level], editor.getValue());
  console.log('Solving level:', n_level, ' with MCTS');
  let heuristic = getScoreNormalized;
  if(heuristic != null){
    precalcDistances();
  }
  var [sol_a, n_search_iters_a] = await solveLevelMCTS(level_i=n_level, {"score_fn": heuristic, "max_iterations": 100000});
  console.log('Solution:', sol_a);
}


async function solveLevelBestFirst(captureStates=false, gameHash=0, levelI=0, maxIters=100_000_000) {
	// if (levelEditorOpened) return;
	// if (showingSolution) return;
	// if (solving) return;
	// if (textMode || state.levels.length === 0) return;

  start_time = Date.now();
	precalcDistances();
	abortSolver = false;
	muted = true;
	solving = true;
	// restartTarget = backupLevel();
	DoRestartSearch();
	hasUsedCheckpoint = false;
	backups = [];
	var oldDT = deltatime;
	deltatime = 0;
	var actions = [0, 1, 2, 3, 4];
	if ('noaction' in state.metadata) {
		actions = [0, 1, 2, 3];
	}
	exploredStates = {};
	exploredStates[level.objects] = [level.objects.slice(0), -1];
	var queue;
	queue = new FastPriorityQueue(byScoreAndLength);
	queue.add([0, level.objects.slice(0), 0]);
	consolePrint("searching...");
	// var solvingProgress = document.getElementById("solvingProgress");
	// var cancelLink = document.getElementById("cancelClickLink");
	// cancelLink.hidden = false;
	// console.log("searching...");
  var totalIters = 0
	var iters = 0;
	var size = 1;

	var startTime = performance.now();

  if (captureStates) {
    const canvas = document.getElementById('gameCanvas');
    const imageData = canvas.toDataURL('image/png');
    const stateText = backupLevel()['dat'].map(row => row.join('')).join('\n');
    await fetch('/save_init_state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        state_repr: stateText,
        game_hash: gameHash,
        game_level: levelI,
        state_hash: hashStateObjects(level.objects),
        im_data: imageData,
      })
    });
  }

	while (!queue.isEmpty() && totalIters < maxIters) {

		if (abortSolver) {
			console.log("solver aborted");
			// cancelLink.hidden = true;
			break;
		}
    if (totalIters > maxIters) {
      console.log('Exceeded max iterations. Exiting.');
      break;
    }
		iters++;
		if (iters > 500) {
			iters = 0;
			// console.log(size);
			// solvingProgress.innerHTML = "searched: " + size;
			// redraw();
			// await timeout(1);
		}
		var temp = queue.poll();
		var parentState = temp[1];
		var numSteps = temp[2];
		// console.log(numSteps);
		shuffleALittle(actions);
		for (var i = 0, len = actions.length; i < len; i++) {
			for (var k = 0, len2 = parentState.length; k < len2; k++) {
				level.objects[k] = parentState[k];
			}
			var changedSomething = processInputSearch(actions[i]);
			while (againing) {
				changedSomething = processInputSearch(-1) || changedSomething;
			}

			if (changedSomething) {
				if (level.objects in exploredStates) {
					continue;
				}
        if (captureStates) {
          await processStateTransition(gameHash, parentState, level.objects, actions[i]);
          console.log(winning);
        }

        // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
        // redraw();

				exploredStates[level.objects] = [parentState, actions[i]];
				if (winning || hasUsedCheckpoint) {
          console.log('Winning!');
					muted = false;
					solving = false;
					winning = false;
					hasUsedCheckpoint = false;
					var solution = MakeSolution(level.objects);
					// var chunks = chunkString(solution, 5).join(" ");
					var totalTime = (performance.now() - startTime) / 1000;
					consolePrint("solution found: (" + solution.length + " steps, " + size + " positions explored in " + totalTime + " seconds)");
					// console.log("solution found:\n" + chunks + "\nin " + totalIters + " steps");
					console.log("solution found:\n" + solution + "\nin " + totalIters + " steps");
					// solvingProgress.innerHTML = "";
					deltatime = oldDT;
					// playSound(13219900);
					DoRestartSearch();
					redraw();
					// cancelLink.hidden = true;
					// consolePrint("<a href=\"javascript:ShowSolution('" + solution + "');\">" + chunks + "</a>");
					// consolePrint("<br>");
					// consolePrint("<a href=\"javascript:StopSolution();\"> stop showing solution </a>");
					// consolePrint("<br>");
					// ShowSolution(solution);
					return [solution, totalIters];
				}
				size++;
				queue.add([getScore(), level.objects.slice(0), numSteps + 1]);
			}
		}
    if (totalIters % 10000 == 0) {
      now = Date.now();
      console.log('Iteration:', totalIters);
      console.log('FPS:', (totalIters / (now - start_time) * 1000).toFixed(2));
      console.log(`Size of frontier: ${queue.size}`);
      console.log(`Visited states: ${Object.keys(visited).length}`);
      // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
      // redraw();
    }
    totalIters++;
	}
	muted = false;
	solving = false;
	DoRestartSearch();
	console.log("no solution found (" + totalIters + " iterations, " + size + " positions explored, frontier size: " + queue.size + ")");
	console.log("no solution found");
	// solvingProgress.innerHTML = "";
	deltatime = oldDT;
	playSound(52291704);
	redraw();
	// cancelLink.hidden = true;
  return ['', totalIters];
}


async function captureGameState() {
  // Capture current game state as PNG
  const canvas = document.getElementById('gameCanvas');
  const imageData = canvas.toDataURL('image/png');
  return imageData;
}

async function processStateTransition(gameHash, parentState, childState, action) {
  // const img1 = await captureGameState(); 
  const hash1 = hashStateObjects(parentState);

  for (var k = 0, len2 = parentState.length; k < len2; k++) {
    level.objects[k] = parentState[k];
  }
  redraw(); 
  
  // Apply action and capture result
  processInput(action);
  while (againing) {
    processInput(-1);
  }
  
  const img2 = await captureGameState();
  const hash2 = hashStateObjects(childState);
  // const hash2 = state2
  const stateText = backupLevel()['dat'].map(row => row.join('')).join('\n');
  // Save transition
  await fetch('/save_transition', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      state_repr: stateText,
      game_hash: gameHash,
      game_level: curlevel,
      state1_hash: hash1,
      state2_hash: hash2, 
      state2_img: img2,
      action: action
    })
  });
}


async function genGame(config) {
  /* This function will recursively call itself to iterate on broken
   * (uncompilable or unsolvable (or too simply solvable)) games.
   */

  consoleText = '';
  larkError = '';
  nGenAttempts = 0;
  code = '';
  compilationSuccess = false;
  solvable = false;
  solverText = '';
  compiledIters = [];
  solvedIters = [];
  anySolvedIters = [];
  maxMeanSolComplexity = 0;

  bestIndividual = new GameIndividual('', null, -Infinity, 0, [], [], true);
  while (nGenAttempts < config.maxGenAttempts & (nGenAttempts == 0 | !compilationSuccess | !solvable)) {
    console.log(`Game ${config.saveDir}, attempt ${nGenAttempts}.`);

    var response;

    if (config.fromPlan & nGenAttempts == 0) {
      response = await fetch('/gen_game_from_plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          seed: config.expSeed,
          save_dir: config.saveDir,
          game_idea: config.idea,
          n_iter: nGenAttempts,
        }),
      });
    } else {
      // Get our GPT completion from python
      response = await fetch('/gen_game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          seed: config.expSeed,
          fewshot: config.fewshot,
          cot: config.cot,
          save_dir: config.saveDir,
          gen_mode: config.genMode,
          parents: config.parents,
          code: code,
          from_idea: config.fromIdea,
          game_idea: config.idea,
          lark_error: larkError,
          console_text: consoleText,
          solver_text: solverText,
          compilation_success: compilationSuccess,
          n_iter: nGenAttempts,
          meta_parents: config.metaParents,
        }),
      });
    }
  
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
  
    const data = await response.json();

    code = data.code;
    vizFeedback = data.viz_feedback;
    minCode = null;
    // if min_code is not None, then use this
    if (data.min_code) {
      minCode = data.min_code;
    }
    sols = data.sols;
    larkError = data.lark_error
    if (data.skip) {
      return new GameIndividual(code, minCode, -1, 0, [], [], true);
    }
    errorLoadingLevel = false;
    try {
      codeToCompile = minCode ? minCode : code;
      editor.setValue(codeToCompile);
      editor.clearHistory();
      clearConsole();
      setEditorClean();
      unloadGame();
    } catch (e) {
      console.log('Error while loading code:', e);
      errorLoadingLevel = true;
      consoleText = `Error while loading code into editor: ${e}.`;
      errorCount = 10;
    }
    if (!errorLoadingLevel) {
      try {
        compile(['restart'], codeToCompile);
      } catch (e) {
        console.log('Error while compiling code:', e);
      }
      consoleText = getConsoleText();
    }

    if (errorCount > 0) {
      compilationSuccess = false;
      solvable = false;
      solverText = '';
      // console.log(`Errors: ${errorCount}. Iterating on the game code. Attempt ${nGenAttempts}.`);
      fitness = -errorCount;
      dataURLs = [];
    } else {
      compiledIters.push(nGenAttempts);
      compilationSuccess = true;
      solverText = '';
      solvable = true;
      dataURLs = [];
      var anySolvable = false;
      var sol;
      var nSearchIters;
      // console.log('No compilation errors. Performing playtest.');
      fitness = 0
      solComplexities = []
      for (level_i in state.levels) {
        // console.log('Levels:', state.levels);
        // Check if type `Level` or dict
        if (!state.levels[level_i].hasOwnProperty('height')) {
          // console.log(`Skipping level ${level_i} as it does not appear to be a map (just a message?): ${state.levels[level_i]}.`);
          continue;
        }
        // try {
          // Check if level_i is in sols
        if (sols.hasOwnProperty(level_i)) {
          // console.log('Using cached solution.');
          [sol, nSearchIters] = sols[level_i];
        } else {
          clearConsole();
          console.log(`Solving level ${level_i}...`);
          [sol, nSearchIters] = await solveLevelBFS(level_i);
          if (sol.length > 0) {
            console.log(`Solution for level ${level_i}:`, sol);
            console.log(`Saving gif for level ${level_i}.`);
            curlevel = level_i;
            compile(['loadLevel', level_i], editor.getValue());
            inputHistory = sol;
            const [ dataURL, filename ] = makeGIFDoctor();
            dataURLs.push([dataURL, level_i]);
          }
        }
        // } catch (e) {
        //   console.log('Error while solving level:', e);
        //   sol = [];
        //   n_search_iters = -1;
        //   solverText += ` Level ${level_i} resulted in error: ${e}. Please repair it.`;
        // }
        if (!sol) {
          console.log(`sol undefined`);
        }
        sols[level_i] = [sol, nSearchIters];
        // console.log('Solution:', sol);
        // check if sol is undefined
        solComplexity = 0;
        if (sol.length > 0) {
          fitness += nSearchIters;
          solComplexity = nSearchIters;
          // console.log('Level is solvable.');
          // solverText += `Found solution for level ${level_i} in ${n_search_iters} iterations: ${sol}.\n`
          solverText += `Found solution for level ${level_i} in ${nSearchIters} iterations. Solution is ${sol.length} moves long.\n`
          if (sol.length > 1) {
            anySolvable = true;
          }
          if (sol.length < 10) {
            solverText += `Solution is very short. Please make it a bit more complex.\n`
            solvable = false;
          }
        } else if (sol == -1) {
          solvable = false;
          solverText += `Hit maximum search depth of ${i} while attempting to solve ${level_i}. Are you sure it's solvable? If so, please make it a bit simpler.\n`
        } else if (sol == -2) {
          solvable = false;
          consoleText = getConsoleText();
          solverText += `Error while solving level ${level_i}. Please repair it.\nThe PuzzleScript console output was:\n${consoleText}\n`
        } else {
          // console.log(`Level ${level_i} is not solvable.`);
          solvable = false;
          solverText += ` Level ${level_i} is not solvable. Please repair it.\n`
        }
        solComplexities.push(solComplexity);
      }
      if (solComplexities.length == 0) {
        solComplexities = [0];
      }
      meanSolComplexity = solComplexities.reduce((a, b) => a + b, 0) / solComplexities.length;
      maxMeanSolComplexity = Math.max(maxMeanSolComplexity, meanSolComplexity);
      if (solvable) {
        // If all levels are solvable
        solvedIters.push(nGenAttempts)
      }
      if (anySolvable) {
        anySolvedIters.push(nGenAttempts)
      }
    }
    response = await fetch('/log_gen_results', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        save_dir: config.saveDir,
        sols: sols,
        n_iter: nGenAttempts,
        gif_urls: dataURLs,
        console_text: consoleText,
        solver_text: solverText,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    nGenAttempts++;
    individual = new GameIndividual(code, minCode, fitness, maxMeanSolComplexity, compiledIters, solvedIters, anySolvedIters, false);
    bestIndividual = bestIndividual.fitness < individual.fitness ? individual : bestIndividual;
  }
  return bestIndividual;
}

async function interactiveEvo() {
  /** The main function to initiate interactive evolution. Currently triggered by clicking the "GARDEN" button
   * in the top-left of the PS Editor page (doctor.html), but should happen automatically on a separate garden.html.
   */
  const response = await fetch('/get_evo_args', {
    method: 'GET',
  });
  const data = await response.json();
  const evoSeed = data.seed;
  const seed_games = await displayGameSeeds();
  // const metaParents = await userSelectParentsDummy(seed_games);
  const metaParents = ['midas', 'i am two']
  await evolve(evoSeed, metaParents);
}

async function displayGameSeeds(){
  /** Fetch all games in our dataset, to display as "seeds" for potential recombination or mutation by the user. */
  const response = await fetch('/list_scraped_games', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      target_dir: 'scraped_games',
    }),
  });
  const data = await response.json();
  // TODO: Display the seed garden in the UI
  // For now, log the length of the seed garden
  console.log(`Seed garden length: ${data.length}`);
  return data
}

async function userSelectParentsDummy(seed_games) {
  /** Dummy function for user to select parents from seed garden.
  For now, just return to random games. Make sure the games are different.
  */
  const firstIndex = Math.floor(Math.random() * seed_games.length); 
  const first = seed_games.splice(firstIndex, 1)[0];
  const secondIndex = Math.floor(Math.random() * seed_games.length);
  const second = seed_games.splice(secondIndex, 1)[0];  
  return [first, second];
}


const popSize = 5;
const nGens = 10;

async function evolve2() {
  // Get a candidate game from the server
  const response = await fetch('/evo_ask', {
    method: 'GET',
  });

  // Evaluate the game
  const data = await response.json();
  const game = data.game;
  code = data.code;
  vizFeedback = data.viz_feedback;
  minCode = null;
  // if min_code is not None, then use this
  if (data.min_code) {
    minCode = data.min_code;
  }
  sols = data.sols;
  larkError = data.lark_error
  if (data.skip) {
    return new GameIndividual(code, minCode, -1, 0, [], [], true);
  }
  errorLoadingLevel = false;
  try {
    codeToCompile = minCode ? minCode : code;
    editor.setValue(codeToCompile);
    editor.clearHistory();
    clearConsole();
    setEditorClean();
    unloadGame();
  } catch (e) {
    console.log('Error while loading code:', e);
    errorLoadingLevel = true;
    consoleText = `Error while loading code into editor: ${e}.`;
    errorCount = 10;
  }
  if (!errorLoadingLevel) {
    try {
      compile(['restart'], codeToCompile);
    } catch (e) {
      console.log('Error while compiling code:', e);
    }
    consoleText = getConsoleText();
  }

  if (errorCount > 0) {
    compilationSuccess = false;
    solvable = false;
    solverText = '';
    // console.log(`Errors: ${errorCount}. Iterating on the game code. Attempt ${nGenAttempts}.`);
    fitness = -errorCount;
    dataURLs = [];
  } else {
    compiledIters.push(nGenAttempts);
    compilationSuccess = true;
    solverText = '';
    solvable = true;
    dataURLs = [];
    var anySolvable = false;
    var sol;
    var nSearchIters;
    // console.log('No compilation errors. Performing playtest.');
    fitness = 0
    solComplexities = []
    for (level_i in state.levels) {
      // console.log('Levels:', state.levels);
      // Check if type `Level` or dict
      if (!state.levels[level_i].hasOwnProperty('height')) {
        // console.log(`Skipping level ${level_i} as it does not appear to be a map (just a message?): ${state.levels[level_i]}.`);
        continue;
      }
      // try {
        // Check if level_i is in sols
      if (sols.hasOwnProperty(level_i)) {
        // console.log('Using cached solution.');
        [sol, nSearchIters] = sols[level_i];
      } else {
        clearConsole();
        console.log(`Solving level ${level_i}...`);
        [sol, nSearchIters] = await solveLevelBFS(level_i);
        if (sol.length > 0) {
          console.log(`Solution for level ${level_i}:`, sol);
          console.log(`Saving gif for level ${level_i}.`);
          curlevel = level_i;
          compile(['loadLevel', level_i], editor.getValue());
          inputHistory = sol;
          const [ dataURL, filename ] = makeGIFDoctor();
          dataURLs.push([dataURL, level_i]);
        }
      }
      // } catch (e) {
      //   console.log('Error while solving level:', e);
      //   sol = [];
      //   n_search_iters = -1;
      //   solverText += ` Level ${level_i} resulted in error: ${e}. Please repair it.`;
      // }
      if (!sol) {
        console.log(`sol undefined`);
      }
      sols[level_i] = [sol, nSearchIters];
      // console.log('Solution:', sol);
      // check if sol is undefined
      solComplexity = 0;
      if (sol.length > 0) {
        fitness += nSearchIters;
        solComplexity = nSearchIters;
        // console.log('Level is solvable.');
        // solverText += `Found solution for level ${level_i} in ${n_search_iters} iterations: ${sol}.\n`
        solverText += `Found solution for level ${level_i} in ${nSearchIters} iterations. Solution is ${sol.length} moves long.\n`
        if (sol.length > 1) {
          anySolvable = true;
        }
        if (sol.length < 10) {
          solverText += `Solution is very short. Please make it a bit more complex.\n`
          solvable = false;
        }
      } else if (sol == -1) {
        solvable = false;
        solverText += `Hit maximum search depth of ${i} while attempting to solve ${level_i}. Are you sure it's solvable? If so, please make it a bit simpler.\n`
      } else if (sol == -2) {
        solvable = false;
        consoleText = getConsoleText();
        solverText += `Error while solving level ${level_i}. Please repair it.\nThe PuzzleScript console output was:\n${consoleText}\n`
      } else {
        // console.log(`Level ${level_i} is not solvable.`);
        solvable = false;
        solverText += ` Level ${level_i} is not solvable. Please repair it.\n`
      }
      solComplexities.push(solComplexity);
    }
    if (solComplexities.length == 0) {
      solComplexities = [0];
    }
    meanSolComplexity = solComplexities.reduce((a, b) => a + b, 0) / solComplexities.length;
    maxMeanSolComplexity = Math.max(maxMeanSolComplexity, meanSolComplexity);
    if (solvable) {
      // If all levels are solvable
      solvedIters.push(nGenAttempts)
    }
    if (anySolvable) {
      anySolvedIters.push(nGenAttempts)
    }
  }

  // Now send the results to the server
  response = await fetch('/evo_tell', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      sols: sols,
      gif_urls: dataURLs,
      console_text: consoleText,
      solver_text: solverText,
    }),
  });
}

async function evolve(evoSeed, metaParents=null) {
  /** The main loop for evolving games.
   * Maybe move this to python eventually?
   */
  // Create an initial population of 10 games
  pop = [];
  gen = 0
  if (!metaParents) {
    evoDir = `evo-${evoSeed}`;
  }
  else {
    evoDir = `evo-${evoSeed}_meta`
  }
  
  // Create a base config for this experiment
  const baseConfig = new GameConfig({
    expSeed: evoSeed,
    fewshot: true,
    cot: true,
    metaParents: metaParents
  });
  
  for (indIdx = 0; indIdx < (popSize*2); indIdx++) {
    genDir = `${evoDir}/gen${gen}`;
    saveDir = `${genDir}/game${indIdx}`;
    
    // Create a config for this specific game
    const gameConfig = baseConfig.extend({
      genMode: 'init',
      saveDir: saveDir,
      maxGenAttempts: 10
    });
    
    game_i = await genGame(gameConfig);
    pop.push(game_i);
  }
  
  for (gen = 1; gen < nGens; gen++) {
    // Sort the population by fitness, in descending order
    pop = pop.sort((a, b) => b.fitness - a.fitness);
    // Get rid of the bottom half
    pop = pop.slice(0, popSize);
    // Print list of fitnesses
    popFits = pop.map(game => game.fitness);
    meanPopFit = popFits.reduce((acc, fit) => acc + fit, 0) / popFits.length;
    genDir = `${evoDir}/gen${gen}`;
    stats = {
      'pop_fits': popFits,
      'mean_pop_fit': meanPopFit,
    }
    await fetch('/save_evo_gen_stats', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        save_dir: `${evoDir}/stats/gen-${gen}.json`,
        stats: stats,
      }),
    });
    console.log(`Generation ${gen}. Fitnesses: ${popFits}`);
    console.log(`Generation ${gen}. Mean fitness: ${meanPopFit}`);
    // Select the top half of the population as parents
    ancestors = pop;
    // Get mean fitness of elites
    eliteFits =  ancestors.map(game => game.fitness);
    meanEliteFit = eliteFits.reduce((acc, fit) => acc + fit, 0) / eliteFits.length;
    console.log(`Generation ${gen}. Elite fitnesses: ${eliteFits}`);
    console.log(`Generation ${gen}. Mean elite fitness: ${meanEliteFit}`);
    // Generate the next generation
    newPop = [];
    for (indIdx = 0; indIdx < popSize; indIdx++) {
      doCrossOver = Math.random() < 0.5;
      let gameConfig;
      
      if (doCrossOver) {
        // Get two random games from list without replacement
        parent1 = ancestors[Math.floor(Math.random() * popSize)];
        // Create copy of array without parent1
        remainingAncestors = ancestors.filter(parent => parent != parent1);
        parent2 = remainingAncestors[Math.floor(Math.random() * (popSize - 1))];
        parents = [parent1, parent2];
        
        gameConfig = baseConfig.extend({
          genMode: 'crossover',
          parents: parents,
          saveDir: `${genDir}/game${indIdx}`
        });
      } else {
        parents = [ancestors[Math.floor(Math.random() * popSize)]];
        
        gameConfig = baseConfig.extend({
          genMode: 'mutate',
          parents: parents,
          saveDir: `${genDir}/game${indIdx}`
        });
      }
      
      newPop.push(await genGame(gameConfig));
    }
    pop = pop.concat(newPop);
  }
}

async function saveStats(saveDir, results) {
  const response = await fetch('/save_sweep_stats', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      save_dir: saveDir,
      results: results,
    }),
  });
}

async function sweepGeneral() {
  await fetch('/reset_sweep', {
    method: 'POST',
  });
  isDone = false;
  while (!isDone) {
    response = await fetch('/get_sweep_args', {
      method: 'GET',
    });
    args = await response.json();
    isDone = args.done;
    if (!isDone) {
      const gameConfig = new GameConfig({
        genMode: 'init',
        saveDir: args.gameDir,
        expSeed: args.gameIdx,
        fewshot: args.fewshot,
        cot: args.cot,
        fromIdea: args.fromIdea,
        idea: args.gameIdea,
        fromPlan: args.fromPlan
      });
      
      gameInd = await genGame(gameConfig);
      await fetch('/save_game_stats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          gameDir: args.gameDir,
          expDir: args.expDir,
          gameInd: gameInd,
        }),
      });
    }
  }
}

async function sweep() {
  saveDir = `sweep-${expSeed}`
  results = {};
  for (var gameIdx = 0; gameIdx < 20; gameIdx++) {
    for (var fewshot_i = 0; fewshot_i < 2; fewshot_i++) {
      for (var cot_i = 0; cot_i < 2; cot_i++) {
        expName = `fewshot-${fewshot_i}_cot-${cot_i}`;
        if (!results.hasOwnProperty(expName)) {
          results[expName] = [];
        }
        gameStr = `${saveDir}/${expName}/game-${gameIdx}`;
        const gameConfig = new GameConfig({
          genMode: 'init',
          saveDir: gameStr,
          expSeed: gameIdx,
          fewshot: fewshot_i == 1,
          cot: cot_i == 1
        });
        
        console.log(`Generating game ${gameStr}`);
        gameInd = await genGame(gameConfig);
        results[expName].push(gameInd);
      }
    }
  }
  saveStats(saveDir, results);
}

brainstormSeed = 0;

async function fromIdeaSweep() {
  // Open the ideas json
  const response = await fetch('/load_ideas', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ brainstorm_seed: brainstormSeed }),
  });
  ideas = await response.json()
  results = {};
  fewshot_i = 1;
  fromIdea_i = 1;
  for (var cot_i = 0; cot_i < 2; cot_i++) {
    hypStr = `fromIdea-${fromIdea_i}_fewshot-${fewshot_i}_cot-${cot_i}`;
    results[hypStr] = [];
    for (var gameIdx = 0; gameIdx < 20; gameIdx++) {
      saveDir = `sweep-${expSeed}`
      gameStr = `${saveDir}/${hypStr}/game-${gameIdx}`;
      ideaIdx = gameIdx % ideas.length;
      idea = ideas[ideaIdx];
      
      const gameConfig = new GameConfig({
        genMode: 'init',
        saveDir: gameStr,
        expSeed: gameIdx,
        fewshot: fewshot_i == 1,
        cot: cot_i == 1,
        fromIdea: fromIdea_i == 1,
        idea: idea
      });
      
      console.log(`Generating game ${gameStr}`);
      gameInd = await genGame(gameConfig);
      results[hypStr].push(gameInd);
    }
  }
  saveStats(saveDir + '/fromIdea', results);
}

async function fromPlanSweep() {
  // Open the ideas json
  const response = await fetch('/load_ideas', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ brainstorm_seed: brainstormSeed }),
  });
  ideas = await response.json()
  results = {};
  cot_i = 1;
  fewshot_i = 1;
  fromIdea_i = 1;
  fromPlan_i = 1;
  hypStr = `fromPlan-${fromPlan_i}`;
  results[hypStr] = [];
  for (var gameIdx = 0; gameIdx < 20; gameIdx++) {
    saveDir = `sweep-${expSeed}`
    gameStr = `${saveDir}/${hypStr}/game-${gameIdx}`;
    ideaIdx = gameIdx % ideas.length;
    idea = ideas[ideaIdx];
    
    const gameConfig = new GameConfig({
      genMode: 'init',
      saveDir: gameStr,
      expSeed: gameIdx,
      fewshot: fewshot_i == 1,
      cot: cot_i == 1,
      fromIdea: fromIdea_i == 1,
      idea: idea,
      fromPlan: fromPlan_i == 1
    });
    
    console.log(`Generating game ${gameStr}`);
    gameInd = await genGame(gameConfig);
    results[hypStr].push(gameInd);
  }
  saveStats(saveDir + '/fromPlan', results);
}

async function collectGameData(gamePath, captureStates=true) {
  // Load game
  const response = await fetch('/load_game_from_file', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ game: gamePath })
  });
  
  const code = await response.text();
  
  // Initialize game
  editor.setValue(code);
  clearConsole();
  setEditorClean();
  unloadGame();
  compile(['restart'], code);

  // Process each level
  for (let level = 0; level < state.levels.length; level++) {
    if (!state.levels[level].hasOwnProperty('height')) {
      continue;
    }
    
    console.log(`Processing level ${level} of game ${gamePath}`);
    compile(['loadLevel', level], code);
    // const [sol, n_iters] = await solveLevelAStar(captureStates=captureStates, gameHash=gamePath, level_i=level, maxIters=1_000_000);
    const solDir = `sols/${gamePath}`;
    const solPath = `${solDir}/level-${level}.json`;
    // If the solution exists, skip it
    const solExists = await fetch('/file_exists', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filePath: solPath })
    });
    const solExistsData = await solExists.json();
    if (solExistsData.exists) {
      console.log(`Solution for level ${level} already exists. Skipping.`);
      continue;
    }
    // If the solution does not exist, solve it
    const [sol, n_iters] = await solveLevelBFS(level, captureStates=captureStates, maxIters=100_000);
    console.log(`Finished processing level ${level}`);
    if (sol.length > 0) {
      console.log(`Solution for level ${level}:`, sol);
      console.log(`Saving gif for level ${level}.`);
      inputHistory = sol;
      const [ dataURL, filename ] = makeGIFDoctor();
      await fetch ('/save_sol', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          levelIdx: level,
          sol: sol,
          solDir: solDir,
          dataURL: dataURL,
        })
      });
    }
  }
}

async function processAllGames() {
  const response = await fetch('/list_scraped_games', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      target_dir: 'min_games',
    }),
  });
  const games = await response.json();

  // Shuffle the games
  // games.sort(() => Math.random() - 0.5);
  
  for (const game of games) {
    console.log(`Processing game: ${game}`);
    await collectGameData(game, captureStates=false);
  }
}
// var experimentDropdown = document.getElementById("experimentDropdown");
// experimentDropdown.addEventListener("change", experimentDropdownChange, false);

var sweepClickLink = document.getElementById("sweepClickLink");
sweepClickLink.addEventListener("click", sweepClick, false);

var BFSClickLink = document.getElementById("BFSClickLink");
BFSClickLink.addEventListener("click", testBFS, false);

var MCTSClickLink = document.getElementById("MCTSClickLink");
MCTSClickLink.addEventListener("click", testMCTS, false);

var genDataClickLink = document.getElementById("genDataClickLink");
genDataClickLink.addEventListener("click", processAllGames, false);

var solveClickLink = document.getElementById("solveClickLink");
solveClickLink.addEventListener("click", playTest, false);

var evolveClickLink = document.getElementById("evolveClickLink");
evolveClickLink.addEventListener("click", evolve, false);

var interactiveEvoClickLink = document.getElementById("interactiveEvoClickLink");
interactiveEvoClickLink.addEventListener("click", interactiveEvo, false);

var expFn = evolve;

function experimentDropdownChange() {
  console.log('Experiment changed');
  var experiment = experimentDropdown.value;
  if (experiment == 'evolve') {
    expFn = evolve;
  } else if (experiment == 'fewshot_cot') {
    expFn = sweep;
  } else if (experiment == 'from_idea') {
    expFn = fromIdeaSweep;
  } else if (experiment == 'from_plan') {
    expFn = fromPlanSweep;
  }
  else {
    console.log('Unknown experiment:', experiment);
  }
}

function sweepClick() {
  console.log('Sweep clicked');
  // expFn();
  sweepGeneral();
}

 const expSeed = 0;

// sweepGeneral();
// sweep();
// fromIdeaSweep();
// fromPlanSweep();
// playTest();
// evolve(expSeed);
// evolve2();
processAllGames();

// genGame('init', [], 'test_99', 99, fewshot=true, cot=true, maxGenAttempts=20);