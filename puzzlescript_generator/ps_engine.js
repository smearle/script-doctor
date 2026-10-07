// Load script-doctor's headless PuzzleScript engine for batch use.
//
//   const { loadEngine } = require('./ps_engine.js');
//   const engine = loadEngine(ENGINE_DIR);   // ENGINE_DIR holds puzzlescript_nodejs/ and PuzzleScript/
//
// Two patches are applied inside the engine's own VM realm; code built with that realm's
// Function constructor (engine.inEngine) sees the engine's globals:
//   IDE = false      the source bundle declares a lexical IDE that is true; headless, the
//                    editor and debugger hooks it guards are undefined and throw;
//   processOutputCommands is a no-op: a shown message would set textMode, after which the
//                    headless engine never checks for a win again (a browser checks when the
//                    player dismisses the message). Sounds are not needed either.
// Both are idempotent, so they are harmless where the wrapper already sets IDE = false.
// Distinct ENGINE_DIRs load as separate module instances, each with its own realm.
'use strict';
const path = require('path');

function loadEngine(engineDir) {
  const engine = require(path.join(path.resolve(engineDir), 'puzzlescript_nodejs/puzzlescript/engine.js'));
  engine.inEngine = engine.processInput.constructor;
  engine.inEngine('IDE = false; processOutputCommands = function () {};')();
  return engine;
}

module.exports = { loadEngine };
