// Check PuzzleScript sources with the reference engine (script-doctor's headless wrapper).
//
//   node ps_check.js ENGINE_DIR SHARD.jsonl START [--dynamics]
//
// SHARD lines are {"id", "text"}; one JSON result per game is written to stdout, in order,
// starting at line START (so a supervisor can restart past a game that hangs the engine).
// A game passes when the engine reports "Successful Compilation" with no error message
// and the game has at least one playable level. With --dynamics, the first playable
// level also gets a seeded random rollout and a budgeted BFS.
'use strict';
const fs = require('fs');
const path = require('path');

const [engineDir, shardPath, startArg, ...flags] = process.argv.slice(2);
const { loadEngine } = require('./ps_engine.js');
const engine = loadEngine(engineDir);
const solver = require(path.join(engineDir, 'puzzlescript_nodejs/puzzlescript/solver.js'));
const dynamics = flags.includes('--dynamics');
const ROLLOUT_STEPS = 200;
const BFS_ITERS = 20000;
const BFS_TIMEOUT_MS = 10000;

const origLog = console.log;
console.log = () => {};  // the solver prints progress; stdout carries results only
const SEED = 'ps-check';  // engine RNG seed for `random` rules: rollouts and BFS are reproducible

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function levelKey() {
  return JSON.stringify(Array.from(engine.backupLevel().dat));
}

// The solver steps the engine itself: give it an engine that clears the message log before
// each turn as step() does (ps_engine.js).
const solverEngine = Object.assign({}, engine, {
  processInput: (...a) => { engine.clearLog(); return engine.processInput(...a); },
});

function step(action) {
  engine.setWinning(false);
  engine.setHasUsedCheckpoint(false);
  engine.clearLog();
  engine.processInput(action);
  while (engine.getAgaining()) engine.processInput(-1);
  return engine.getWinning();
}

function probe(text, levelIndex) {
  const out = {};
  engine.compile(['loadLevel', levelIndex], text, SEED);
  const meta = engine.getState().metadata || {};
  const actions = ('noaction' in meta) ? [0, 1, 2, 3] : [0, 1, 2, 3, 4];
  const rand = mulberry32(12345);
  const start = levelKey();
  const seen = new Set([start]);
  let prev = start, changed = 0, won = false;
  for (let t = 0; t < ROLLOUT_STEPS; t++) {
    const w = step(actions[Math.floor(rand() * actions.length)]);
    const k = levelKey();
    if (k !== prev) changed++;
    seen.add(k);
    prev = k;
    if (w) { won = true; break; }
  }
  out.rollout = { steps: ROLLOUT_STEPS, changed, distinct: seen.size, won };
  engine.compile(['loadLevel', levelIndex], text, SEED);
  const t0 = Date.now();
  const r = solver.solveBFS(solverEngine, BFS_ITERS, BFS_TIMEOUT_MS);
  out.bfs = { solved: !!r[0], sol_len: r[0] ? r[1].length : null, iters: r[2],
              timeout: !!r[6], ms: Date.now() - t0 };
  return out;
}

const lines = fs.readFileSync(shardPath, 'utf8').split('\n').filter((l) => l.trim());
for (let i = Number(startArg || 0); i < lines.length; i++) {
  const { id, text } = JSON.parse(lines[i]);
  const res = { i, id };
  try {
    engine.unloadGame();
    engine.clearCapturedErrors();
    const t0 = Date.now();
    engine.compile(['restart'], text);
    res.compile_ms = Date.now() - t0;
    const msgs = engine.getCapturedErrors();
    const success = msgs.some((m) => m.includes('Successful Compilation'));
    const errors = msgs.filter((m) => !m.includes('Successful Compilation'));
    let levels = [];
    try { levels = engine.getLevelInfo().filter((l) => l.type === 'level'); } catch (e) {}
    res.compiled = success && !msgs.some((m) => m.includes('Errors detected'));
    res.n_levels = res.compiled ? levels.length : 0;
    res.errors = errors.slice(0, 4).map((m) => m.slice(0, 200));
    res.ok = res.compiled && levels.length > 0;
    if (dynamics && res.ok) Object.assign(res, probe(text, levels[0].index));
  } catch (e) {
    res.ok = false;
    res.exception = String(e).slice(0, 300);
  }
  process.stdout.write(JSON.stringify(res) + '\n');
}
