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

/**
 * FastPriorityQueue.js : a fast heap-based priority queue  in JavaScript.
 * (c) the authors
 * Licensed under the Apache License, Version 2.0.
 *
 * Speed-optimized heap-based priority queue for modern browsers and JavaScript engines.
 *
 * Usage :
         Installation (in shell, if you use node):
         $ npm install fastpriorityqueue

         Running test program (in JavaScript):

         // var FastPriorityQueue = require("fastpriorityqueue");// in node
         var x = new FastPriorityQueue();
         x.add(1);
         x.add(0);
         x.add(5);
         x.add(4);
         x.add(3);
         x.peek(); // should return 0, leaves x unchanged
         x.size; // should return 5, leaves x unchanged
         while(!x.isEmpty()) {
           console.log(x.poll());
         } // will print 0 1 3 4 5
         x.trim(); // (optional) optimizes memory usage
 */
         'use strict';

         var defaultcomparator = function(a, b) {
           return a < b;
         };
         
         // the provided comparator function should take a, b and return *true* when a < b
         function FastPriorityQueue(comparator) {
           if (!(this instanceof FastPriorityQueue)) return new FastPriorityQueue(comparator);
           this.array = [];
           this.size = 0;
           this.compare = comparator || defaultcomparator;
         }
         
         FastPriorityQueue.prototype.clone = function() {
           var fpq = new FastPriorityQueue(this.compare);
           fpq.size = this.size;
           for (var i = 0; i < this.size; i++) {
             fpq.array.push(this.array[i]);
           }
           return fpq;
         };
         
         // Add an element into the queue
         // runs in O(log n) time
         FastPriorityQueue.prototype.add = function(myval) {
           var i = this.size;
           this.array[this.size] = myval;
           this.size += 1;
           var p;
           var ap;
           while (i > 0) {
             p = (i - 1) >> 1;
             ap = this.array[p];
             if (!this.compare(myval, ap)) {
               break;
             }
             this.array[i] = ap;
             i = p;
           }
           this.array[i] = myval;
         };
         
         // replace the content of the heap by provided array and "heapifies it"
         FastPriorityQueue.prototype.heapify = function(arr) {
           this.array = arr;
           this.size = arr.length;
           var i;
           for (i = this.size >> 1; i >= 0; i--) {
             this._percolateDown(i);
           }
         };
         
         // for internal use
         FastPriorityQueue.prototype._percolateUp = function(i, force) {
           var myval = this.array[i];
           var p;
           var ap;
           while (i > 0) {
             p = (i - 1) >> 1;
             ap = this.array[p];
             // force will skip the compare
             if (!force && !this.compare(myval, ap)) {
               break;
             }
             this.array[i] = ap;
             i = p;
           }
           this.array[i] = myval;
         };
         
         // for internal use
         FastPriorityQueue.prototype._percolateDown = function(i) {
           var size = this.size;
           var hsize = this.size >>> 1;
           var ai = this.array[i];
           var l;
           var r;
           var bestc;
           while (i < hsize) {
             l = (i << 1) + 1;
             r = l + 1;
             bestc = this.array[l];
             if (r < size) {
               if (this.compare(this.array[r], bestc)) {
                 l = r;
                 bestc = this.array[r];
               }
             }
             if (!this.compare(bestc, ai)) {
               break;
             }
             this.array[i] = bestc;
             i = l;
           }
           this.array[i] = ai;
         };
         
         // internal
         // _removeAt(index) will delete the given index from the queue,
         // retaining balance. returns true if removed.
         FastPriorityQueue.prototype._removeAt = function(index) {
           if (this.isEmpty() || index > this.size - 1 || index < 0) return false;
         
           // impl1:
           //this.array.splice(index, 1);
           //this.heapify(this.array);
           // impl2:
           this._percolateUp(index, true);
           this.poll();
           return true;
         };
         
         // remove(myval[, comparator]) will remove the given item from the
         // queue, checked for equality by using compare if a new comparator isn't provided.
         // (for exmaple, if you want to remove based on a seperate key value, not necessarily priority).
         // return true if removed.
         FastPriorityQueue.prototype.remove = function(myval, comparator) {
           if (!comparator) {
             comparator = this.compare;
           }
           if (this.isEmpty()) return false;
           for (var i = 0; i < this.size; i++) {
             if (comparator(this.array[i], myval) || comparator(myval, this.array[i])) {
               continue;
             }
             // items are equal, remove
             return this._removeAt(i);
           }
           return false;
         };
         
         // Look at the top of the queue (a smallest element)
         // executes in constant time
         //
         // Calling peek on an empty priority queue returns
         // the "undefined" value.
         // https://developer.mozilla.org/en/docs/Web/JavaScript/Reference/Global_Objects/undefined
         //
         FastPriorityQueue.prototype.peek = function() {
           if (this.size == 0) return undefined;
           return this.array[0];
         };
         
         // remove the element on top of the heap (a smallest element)
         // runs in logarithmic time
         //
         // If the priority queue is empty, the function returns the
         // "undefined" value.
         // https://developer.mozilla.org/en/docs/Web/JavaScript/Reference/Global_Objects/undefined
         //
         // For long-running and large priority queues, or priority queues
         // storing large objects, you may  want to call the trim function
         // at strategic times to recover allocated memory.
         FastPriorityQueue.prototype.poll = function() {
           if (this.size == 0) return undefined;
           var ans = this.array[0];
           if (this.size > 1) {
             this.array[0] = this.array[--this.size];
             this._percolateDown(0 | 0);
           } else {
             this.size -= 1;
           }
           return ans;
         };
         
         // This function adds the provided value to the heap, while removing
         //  and returning the peek value (like poll). The size of the priority
         // thus remains unchanged.
         FastPriorityQueue.prototype.replaceTop = function(myval) {
           if (this.size == 0) return undefined;
           var ans = this.array[0];
           this.array[0] = myval;
           this._percolateDown(0 | 0);
           return ans;
         };
         
         // recover unused memory (for long-running priority queues)
         FastPriorityQueue.prototype.trim = function() {
           this.array = this.array.slice(0, this.size);
         };
         
         // Check whether the heap is empty
         FastPriorityQueue.prototype.isEmpty = function() {
           return this.size === 0;
         };
         
         // iterate over the items in order, pass a callback that receives (item, index) as args.
         // TODO once we transpile, uncomment
         // if (Symbol && Symbol.iterator) {
         //   FastPriorityQueue.prototype[Symbol.iterator] = function*() {
         //     if (this.isEmpty()) return;
         //     var fpq = this.clone();
         //     while (!fpq.isEmpty()) {
         //       yield fpq.poll();
         //     }
         //   };
         // }
         FastPriorityQueue.prototype.forEach = function(callback) {
           if (this.isEmpty() || typeof callback != 'function') return;
           var i = 0;
           var fpq = this.clone();
           while (!fpq.isEmpty()) {
             callback(fpq.poll(), i++);
           }
         };
         
         // return the k 'smallest' elements of the queue
         // runs in O(k log k) time
         // this is the equivalent of repeatedly calling poll, but
         // it has a better computational complexity, which can be
         // important for large data sets.
         FastPriorityQueue.prototype.kSmallest = function(k) {
           if (this.size == 0) return [];
           var comparator = this.compare;
           var arr = this.array
           var fpq = new FastPriorityQueue(function(a,b){
            return comparator(arr[a],arr[b]);
           });
           k = Math.min(this.size, k);
           var smallest = new Array(k);
           var j = 0;
           fpq.add(0);
           while (j < k) {
             var small = fpq.poll();
             smallest[j++] = this.array[small];
             var l = (small << 1) + 1;
             var r = l + 1;
             if (l < this.size) fpq.add(l);
             if (r < this.size) fpq.add(r);
           }
           return smallest;
         }

var distanceTable;

var act2str = "uldrx";
var exploredStates;

function precalcDistances(engine) {
  function distance(index1, index2) {
    return Math.abs(Math.floor(index1 / engine.getLevel().height) - Math.floor(index2 / engine.getLevel().height)) + 
      Math.abs((index1 % engine.getLevel().height) - (index2 % engine.getLevel().height));
  }

	distanceTable = [];
	for (var i = 0; i < engine.getLevel().n_tiles; i++) {
		ds = [];
		for (var j = 0; j < engine.getLevel().n_tiles; j++) {
			ds.push(distance(i, j));
		}
		distanceTable.push(ds);
	}
}

function getScore(engine) {
	var score = 0.0;
	var maxDistance = engine.getLevel().width + engine.getLevel().height;
	if (engine.getState().winconditions.length > 0) {
		for (var wcIndex = 0; wcIndex < engine.getState().winconditions.length; wcIndex++) {
			var wincondition = engine.getState().winconditions[wcIndex];
			var filter1 = wincondition[1];
			var filter2 = wincondition[2];
			if (wincondition[0] == -1) {
				// "no" conditions
				for (var i = 0; i < engine.getLevel().n_tiles; i++) {
					var cell = engine.getLevel().getCellInto(i, engine.get_o10());
					if ((!filter1.bitsClearInArray(cell.data)) && (!filter2.bitsClearInArray(cell.data))) {
						score += 1.0; // penalization for each case
					}
				}
			} else {
				// "some" or "all" conditions
				for (var i = 0; i < engine.getLevel().n_tiles; i++) {
					if (!filter1.bitsClearInArray(engine.getLevel().getCellInto(i, engine.get_o10()).data)) {
						var minDistance = maxDistance;
						for (var j = 0; j < engine.getLevel().n_tiles; j++) {
							if (!filter2.bitsClearInArray(engine.getLevel().getCellInto(j, engine.get_o10()).data)) {
								var dist = distanceTable[i][j];
								if (dist < minDistance) {
									minDistance = dist;
								}
							}
						}
						score += minDistance;
					}
				}
			}
		}
	}
	// console.log(score);
	return score;
}

function getScoreNormalized(engine) {
	var score = 0.0;
	var maxDistance = engine.getLevel().width + engine.getLevel().height;
	var normal_value = 0.0;
	if (engine.getState().winconditions.length > 0) {
		for (var wcIndex = 0; wcIndex < engine.getState().winconditions.length; wcIndex++) {
			var wincondition = engine.getState().winconditions[wcIndex];
			var filter1 = wincondition[1];
			var filter2 = wincondition[2];
			if (wincondition[0] == -1) {
				// "no" conditions
				for (var i = 0; i < engine.getLevel().n_tiles; i++) {
					var cell = engine.getLevel().getCellInto(i, _o10);
					if ((!filter1.bitsClearInArray(cell.data)) && (!filter2.bitsClearInArray(cell.data))) {
						score += 1.0; // penalization for each case
						normal_value += maxDistance;
					}
					
				}
			} else {
				// "some" or "all" conditions
				for (var i = 0; i < engine.getLevel().n_tiles; i++) {
					if (!filter1.bitsClearInArray(engine.getLevel().getCellInto(i, engine.get_o10()).data)) {
						var minDistance = maxDistance;
						for (var j = 0; j < engine.getLevel().n_tiles; j++) {
							if (!filter2.bitsClearInArray(engine.getLevel().getCellInto(j, engine.get_o10()).data)) {
								var dist = distanceTable[i][j];
								if (dist < minDistance) {
									minDistance = dist;
								}
							}
						}
						score += minDistance;
						normal_value += maxDistance;
					}
				}
			}
		}
	}
	// console.log(score);
	return 1 - score / normal_value;
}

function takeAction(engine, action) {
  // let changed = engine.processInput(action);
  let changed = processInputSearch(engine, action);
  level_map = engine.backupLevel()['dat'];
  score = getScore(engine);
  if (engine.getWinning()) {
    DoRestartSearch(engine);
    // console.log('Winning!');
    // return true;
  }
  // Dummy values for winning, solution, iterations, and elapsed time
  return [false, [], 0, 0, score, level_map];
}

function randomRollout(engine, maxIters=100_000) {
  precalcDistances(engine);
  let i = 0;
  let start_time = Date.now();
  const timeout_ms = 60 * 1000;
  var score = getScore(engine);
  var level_map = engine.backupLevel()['dat'];
  while (i < maxIters) {
    // if (i % 1000 == 0) {
    elapsed_time = Date.now() - start_time;
    if (elapsed_time > timeout_ms) {
      console.log(`Timeout after ${elapsed_time / 1000} seconds. Returning.`);
      return [false, [], i, ((Date.now() - start_time) / 1000), score, level_map];
    }
    // }
    // let changed = engine.processInput(Math.min(5, Math.floor(Math.random() * 6)));
    let changed = processInputSearch(engine, Math.min(5, Math.floor(Math.random() * 6)));
    if (changed) {
      score = getScore(engine);
      new_level = engine.backupLevel()['dat'];
      if (engine.getWinning()) {
        // console.log(`Winning! Solution:, ${new_action_seq}\n Iterations: ${i}`);
        // console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
        // return [true, [], i, ((Date.now() - start_time) / 1000)];
        DoRestartSearch(engine);
      }
    }
    i++;
  }
  if(i >= maxIters) {
    // console.log('Exceeded max iterations. Exiting.');
    return [false, [], i, ((Date.now() - start_time) / 1000), score, level_map];
  }
  // Dummy values for winning and solution
  return [false, [], i, ((Date.now() - start_time) / 1000), score, level_map];
}

function processInputSearch(engine, action){
  var changedSomething = engine.processInput(action);
  while (engine.getAgaining()) {
    changedSomething = engine.processInput(-1) || changedSomething;
  }
 return changedSomething;
}

function solveRandom(engine, maxLength=100, maxIters=100_000) {
  let i = 0;
  let start_time = Date.now();
  const timeout_ms = 60 * 1000;
  let solution = [];
  while (i < maxIters) {
    if (i % maxLength == 0){
      DoRestartSearch(engine);
      solution = [];
    }
    // if (i % 1000 == 0) {
    elapsed_time = Date.now() - start_time;
    if (elapsed_time > timeout_ms) {
      console.log(`Timeout after ${elapsed_time / 1000} seconds. Returning.`);
      return [false, [], i, ((Date.now() - start_time) / 1000)];
    }
    // }
    // let changed = engine.processInput(Math.min(5, Math.floor(Math.random() * 6)));
    let action = Math.min(5, Math.floor(Math.random() * 6));
    solution.push(action);
    let changed = processInputSearch(engine, action);
    if (changed) {
      if (engine.getWinning()) {
        // console.log(`Winning! Solution:, ${new_action_seq}\n Iterations: ${i}`);
        // console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
        return [true, solution, i, ((Date.now() - start_time) / 1000)];
        
      }
    }
    i++;
  }
  if(i >= maxIters) {
    // console.log('Exceeded max iterations. Exiting.');
    return [false, [], i, ((Date.now() - start_time) / 1000)];
  }
  return [false, [], i, ((Date.now() - start_time) / 1000)];
}

function solveBFS(engine, maxIters, timeoutJS) {
  precalcDistances(engine);
  timeout_ms = timeoutJS;
  function hashState(state) {
    return JSON.stringify(state).split('').reduce((hash, char) => {
      return (hash * 31 + char.charCodeAt(0)) % 1_000_000_003; // Simple hash
    }, 0);
  }
  
  init_level = engine.backupLevel();
  init_level_map = init_level['dat'];

  // frontier = [init_level];
  // action_seqs = [[]];
  // frontier = new Queue();
  // action_seqs = new Queue();

  frontier = new Queue();

  frontier.enqueue([init_level, []]);
  // action_seqs.enqueue([]);

  var sol = [];
  var bestState = init_level;
  var bestScore = getScore(engine);
  // console.log(sol.length);
  visited = new Set([hashState(init_level_map)]);
  i = 0;
  start_time = Date.now();
  // console.log(frontier.size())
  while (frontier.size() > 0 && i < maxIters) {
    if (i % 1000 == 0) {
      elapsed_time = Date.now() - start_time;
      if ((timeout_ms > 0) && (elapsed_time > timeout_ms)) {
        console.log(`Timeout after ${elapsed_time / 1000} seconds. Returning best result found so far.`);
        return [false, sol, i, ((Date.now() - start_time) / 1000), bestScore, bestState];
      }
    }
    backups = [];

    // const level = frontier.shift();
    // const action_seq = action_seqs.shift();
    const [level, action_seq] = frontier.dequeue();
    // const action_seq = action_seqs.dequeue();

    if (!action_seq) {
      // console.log(`Action sequence is empty. Length of frontier: ${frontier.size()}`);
    }
    for (const move of Array(5).keys()) {
      if (i > maxIters) {
        // console.log('Exceeded 1M iterations. Exiting.');
        return [false, sol, i, ((Date.now() - start_time) / 1000), bestScore, bestState];
      }
      engine.restoreLevel(level);

      new_action_seq = action_seq.slice();
      new_action_seq.push(move);
      try {
        changed = processInputSearch(engine, move);
      } catch (e) {
        // console.log('Error while processing input:', e);
        return [false, sol, i, ((Date.now() - start_time) / 1000), bestScore, bestState];
      }
      if (changed) {
        new_level = engine.backupLevel();
        new_level_map = new_level['dat'];
        if (engine.getWinning()) {
          // console.log(`Winning! Solution:, ${new_action_seq}\n Iterations: ${i}`);
          // console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
          score = getScore(engine);
          return [true, new_action_seq, i, ((Date.now() - start_time) / 1000), score, new_level_map];
        }
        const newHash = hashState(new_level_map);
        if (!visited.has(newHash)) {
          
          // UNCOMMENT THESE LINES FOR VISUAL DEBUGGING
          // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
          // redraw();

          frontier.enqueue([new_level, new_action_seq]);
          // frontier.enqueue(new_level);
          if (!new_action_seq) {
            console.log(`New action sequence is undefined when pushing.`);
          }
          // action_seqs.enqueue(new_action_seq);
          visited.add(newHash);
          score = getScore(engine);
          // Use this condition if we want short and maximlly good sequences
          // if ((score < bestScore) | (score == bestScore && new_action_seq.length > sol.length)) {
          // Use this condition if we want maximally long sequences to validate the jax engine, for example
          if ((score < bestScore) | (score == bestScore && new_action_seq.length > sol.length)) {
            bestScore = score;
            bestState = new_level_map;
            sol = new_action_seq;
          }
        } 
      }
    }
    if (i % 10000 == 0) {
      now = Date.now();
      console.log('Iteration:', i);
      console.log('FPS:', (i / (now - start_time) * 1000).toFixed(2));
      console.log(`Size of frontier: ${frontier.size()}`);
      console.log(`Visited states: ${visited.size}`);
    }
    i++;
  }
  if(i >= maxIters) {
    return [false, sol, i, ((Date.now() - start_time) / 1000), bestScore, bestState];
  }
  return [false, sol, i, ((Date.now() - start_time) / 1000), bestScore, bestState];
}

function DoRestartSearch(engine, force) {
  if (engine.getRestarting()){
    return;
  }
  if (force!==true && ('norestart' in engine.getState().metadata)) {
    return;
  }
  engine.setRestarting(true);
  if (force!==true) {
    engine.addUndoState(engine.backupLevel());
  }

  engine.restoreLevel(engine.getRestartTarget());
  // tryPlayRestartSound();

  if ('run_rules_on_level_start' in engine.getState().metadata) {
    engine.processInput(-1,true);
  }
  
  engine.getLevel().commandQueue=[];
  engine.getLevel().commandQueueSourceRules=[];
  engine.setRestarting(false);
}

function solveAStar(engine, maxIters=100_000) {
  function MakeSolution(state) {
    var sol = [];
    while (true) {
      var p = exploredStates[state];
      if (p[1] == -1) {
        break;
      } else {
        sol = [p[1]].concat(sol);
        state = p[0];
      }
    }
    return sol;
  }

  function byScoreAndLength(a, b) {
    if (a[0] != b[0]) {
      return a[0] < b[0];
    } else {
      return a[2].length < b[2].length;
    }
  }

  function shuffleALittle(array) {
    randomIndex = 1 + Math.floor(Math.random() * (array.length - 1));
    temporaryValue = array[0];
    array[0] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

	precalcDistances(engine);
	abortSolver = false;
	muted = true;
	solving = true;
	// restartTarget = backupLevel();
	DoRestartSearch(engine, 0);
	hasUsedCheckpoint = false;
	backups = [];
	var oldDT = engine.getDeltaTime();
	engine.setDeltaTime(0);
	var actions = [0, 1, 2, 3, 4];
	if ('noaction' in engine.getState().metadata) {
		actions = [0, 1, 2, 3];
	}
	exploredStates = {};
	exploredStates[engine.getLevel().objects] = [engine.getLevel().objects.slice(0), -1];
	var queue;
	queue = new FastPriorityQueue(byScoreAndLength);
	queue.add([0, engine.getLevel().objects.slice(0), 0]);
	// var solvingProgress = document.getElementById("solvingProgress");
	// var cancelLink = document.getElementById("cancelClickLink");
	// cancelLink.hidden = false;
	// console.log("searching...");
  var totalIters = 0
	var iters = 0;
	var size = 1;

	var start_time = Date.now();

	while (!queue.isEmpty() && totalIters < maxIters) {
    if (totalIters > maxIters) {
      // console.log('Exceeded max iterations. Exiting.');
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
				engine.getLevel().objects[k] = parentState[k];
			}
			// var changedSomething = engine.processInput(actions[i]);
			// while (engine.getAgaining()) {
			// 	changedSomething = engine.processInput(-1) || changedSomething;
			// }
      var changedSomething = processInputSearch(engine, actions[i]);

			if (changedSomething) {
				if (engine.getLevel().objects in exploredStates) {
					continue;
				}

        // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
        // redraw();

				exploredStates[engine.getLevel().objects] = [parentState, actions[i]];
				if (engine.getWinning() || engine.getHasUsedCheckpoint()) {
          // console.log('Winning!');
					muted = false;
					solving = false;
					winning = false;
					engine.setHasUsedCheckpoint(false);
					var solution = MakeSolution(engine.getLevel().objects);
					// var chunks = chunkString(solution, 5).join(" ");
					// var totalTime = (performance.now() - startTime) / 1000;
					// console.log("solution found:\n" + chunks + "\nin " + totalIters + " steps");
					// solvingProgress.innerHTML = "";
					engine.setDeltaTime(oldDT);
					DoRestartSearch(engine);
					// redraw();
					return [true, solution, totalIters, ((Date.now() - start_time) / 1000)];
				}
				size++;
				queue.add([getScore(engine), engine.getLevel().objects.slice(0), numSteps + 1]);
			}
		}
    totalIters++;
	}
	muted = false;
	solving = false;
	DoRestartSearch(engine);
	// console.log("no solution found");
	// solvingProgress.innerHTML = "";
	deltatime = oldDT;
	// redraw();
	// cancelLink.hidden = true;
  return [false, [], totalIters, ((Date.now() - start_time) / 1000)];
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

  simulate(engine, max_length, score_fn, win_bonus){
    let changes = 0;
    for(let i=0; i<max_length; i++){
      // let changed = engine.processInput(Math.min(5, Math.floor(Math.random() * 6)));
      let changed = processInputSearch(engine, Math.min(5, Math.floor(Math.random() * 6)));
      if(changed){
        changes += 1;
      }
      if(engine.getWinning()){
        return win_bonus;
      }
    }
    if(score_fn){
      return score_fn(engine);
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

  tree_size(){
    let size = 1;
    for(let child of this.children){
      if(child != null){
        size += child.tree_size();
      }
    }
    return size;
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
      if(this.children[i].score / this.children[i].visited > 
          this.children[max_action].score / this.children[max_action].visited){
        max_action = i;
      }
    }
    return max_action;
  }
}

// level: is the starting level
// max_sim_length: maximum number of random simulation before stopping and backpropagate
// score_fn: boolean to use the heuristic function
// explore_deadends: if you want to explore deadends by default, the search don't continue in deadends
// deadend_bonus: bonus when you find a deadend node (usually negative number to avoid)
// most_visited: decide to return most visited action or best value action
// win_bonus: bonus when you find a winning node
// c: is the MCTS constant that balance between exploitation and exploration
// max_iterations: max number of iterations before you consider the solution is not available
function solveMCTS(engine, options = {}) {
  // Load the level
  if(options == null){
    options = {};
  }
  let defaultOptions = {
    "max_sim_length": 100,
    "score_fn": true, 
    "explore_deadends": false, 
    "deadend_bonus": -25, 
    "win_bonus": 100,
    "most_visited": true,
    "c": Math.sqrt(2), 
    "max_iterations": 100_000
  };
  for(let key in defaultOptions){
    if(!options.hasOwnProperty(key)){
      options[key] = defaultOptions[key];
    }
  }
  if(options.score_fn){
    precalcDistances(engine);
    options.score_fn = getScoreNormalized;
  }

  let init_level = engine.backupLevel();
  let rootNode = new MCTSNode(-1, null, 5);
  let i = 0;
  let deadend_nodes = 1;
  let start_time = Date.now();
  while(options.max_iterations <= 0 || (options.max_iterations > 0 && i < options.max_iterations)){
    // start from th root
    currentNode = rootNode;
    engine.restoreLevel(init_level);
    let changed = true;
    // selecting next node
    while(currentNode.is_fully_expanded()){
      currentNode = currentNode.select(options.c);
      // changed = engine.processInput(currentNode.action);
      changed = processInputSearch(engine, currentNode.action);
      if(engine.getWinning()){
        let sol = currentNode.get_actions();
        // console.log(`Winning! Solution:, ${sol}\n Iterations: ${i}\n Tree size: ${rootNode.tree_size()}`);
        return [true, sol, i, ((Date.now() - start_time) / 1000)];
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
      // changed = engine.processInput(currentNode.action);
      changed = processInputSearch(engine, currentNode.action);
      if(engine.getWinning()){
        let sol = currentNode.get_actions();
        // console.log(`Winning! Solution:, ${sol}\n Iterations: ${i}`);
        // console.log('FPS:', (i / (Date.now() - start_time) * 1000).toFixed(2));
        return [true, sol, i, ((Date.now() - start_time) / 1000)];
      }
      // if node is deadend, punish it
      if(!options.explore_deadends && !changed){
        currentNode.score += options.deadend_bonus;
        currentNode.backup(0);
        deadend_nodes += 1;
        
      }
      //otherwise simulate then backup
      else{
        let value = currentNode.simulate(engine, options.max_sim_length, options.score_fn, options.win_bonus);
        currentNode.backup(value);
      }
    }
    // print progress
    // if (i % 10000 == 0) {
      // now = Date.now();
      // console.log('Iteration:', i);
      // console.log('FPS:', (i / (now - start_time) * 1000).toFixed(2));
      // console.log(`Visited Deadends: ${deadend_nodes}`);
      // console.log(`Visited states: ${visited.size}`);
      // await new Promise(resolve => setTimeout(resolve, 1)); // Small delay for live feedback
      // redraw();
    // }
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
  return [false, actions, options.max_iterations, ((Date.now() - start_time) / 1000)];
}

function getNLevels(engine) {
  let n_levels = 0;
  for (let i = 0; i < engine.getLevel().n_tiles; i++) {
    if (engine.getLevel().getCellInto(i, engine.get_o10()).data != 0) {
      n_levels++;
    }
  }
  return n_levels;
}

module.exports = {
  solveMCTS,
  solveAStar,
  solveBFS,
  solveRandom,
  randomRollout,
  takeAction,
}