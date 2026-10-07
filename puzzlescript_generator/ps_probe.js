// Behaviour fingerprints of one-level PuzzleScript games, so that sampled mechanics for
// the same level can be grouped by how they behave (level_eval.py).
//
//   node ps_probe.js ENGINE_DIR SHARD.jsonl START
//
// SHARD lines are {"id", "text"}: standard PuzzleScript with one playable level, whose object
// names are shared by the games being compared (level-first names, level_first.py), so a
// cell's contents mean the same observation in every game. One JSON result per game goes to
// stdout, in order, from line START (check_games.py restarts past a game that hangs).
//
// The same PROBES seeded random sequences of STEPS actions (0-4, also in noaction games, so
// every game sees the same inputs) are played from the level's start; each probe's sequence
// of cell contents, by object name, after every action is hashed. A win ends a probe, and
// so does a turn whose again-chain exceeds AGAIN_CAP ticks (flagged). Probe 0 is replayed
// under a second engine seed: a difference means `random` rules make the game stochastic.
// Record: ok, probes (hashes), distinct_states (over all probes), won, again_capped, stochastic.
'use strict';
const fs = require('fs');
const crypto = require('crypto');

const [engineDir, shardPath, startArg] = process.argv.slice(2);
const { loadEngine } = require('./ps_engine.js');
const engine = loadEngine(engineDir);
console.log = () => {};
const PROBES = 16;
const STEPS = 48;
const AGAIN_CAP = 500;
const SEEDS = ['ps-probe', 'ps-probe-replay'];
// Load level k of the compiled game as compile(['loadLevel', k], text, seed) does (ps_equiv.js).
const loadLevel = engine.inEngine('k', 'seed', `
  curlevel = k; curlevelTarget = null; winning = false; againing = false; timer = 0;
  titleScreen = false; textMode = false; tick_lazy_function_generation(false);
  titleSelected = false; quittingMessageScreen = false; quittingTitleScreen = false;
  messageselected = false; titleMode = 0;
  loadLevelFromState(state, k, seed);`);

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SEQS = Array.from({ length: PROBES }, (_, j) => {
  const rand = mulberry32(7000 + j);
  return Array.from({ length: STEPS }, () => Math.floor(rand() * 5));
});

function cells() {
  const dat = engine.backupLevel().dat;
  const lvl = engine.getLevel();
  const n = lvl.width * lvl.height;
  const words = dat.length / n;
  const ids = engine.getState().idDict;
  const out = [];
  for (let c = 0; c < n; c++) {
    const names = [];
    for (let w = 0; w < words; w++) {
      const v = dat[c * words + w];
      if (!v) continue;
      for (let b = 0; b < 32; b++) if (v & (1 << b)) names.push(ids[w * 32 + b]);
    }
    out.push(names.sort().join(','));
  }
  return out.join('|');
}

// One probe from the level's start: the cell states after each action, and how it ended.
function probe(k, seed, actions, seen) {
  loadLevel(k, seed);
  const states = [cells()];
  let won = false;
  let capped = false;
  for (const a of actions) {
    engine.setWinning(false);
    engine.setHasUsedCheckpoint(false);
    engine.clearLog();
    engine.processInput(a);
    let ticks = 0;
    while (engine.getAgaining() && ticks < AGAIN_CAP) { engine.processInput(-1); ticks++; }
    capped = engine.getAgaining();
    won = engine.getWinning();
    states.push(cells() + (won ? '#won' : '') + (capped ? '#capped' : ''));
    if (won || capped) break;
  }
  if (seen) states.forEach((s) => seen.add(s));
  return { hash: crypto.createHash('sha1').update(states.join('\n')).digest('hex').slice(0, 16), won, capped };
}

const lines = fs.readFileSync(shardPath, 'utf8').split('\n').filter((l) => l.trim());
for (let i = Number(startArg || 0); i < lines.length; i++) {
  const { id, text } = JSON.parse(lines[i]);
  const res = { i, id };
  try {
    engine.unloadGame();
    engine.clearCapturedErrors();
    engine.compile(['restart'], text, SEEDS[0]);
    const msgs = engine.getCapturedErrors();
    const compiled = msgs.some((m) => m.includes('Successful Compilation')) &&
      !msgs.some((m) => m.includes('Errors detected'));
    const levels = compiled ? engine.getLevelInfo().filter((l) => l.type === 'level').map((l) => l.index) : [];
    res.ok = levels.length === 1;
    if (res.ok) {
      const seen = new Set();
      const runs = SEQS.map((actions) => probe(levels[0], SEEDS[0], actions, seen));
      res.probes = runs.map((r) => r.hash);
      res.distinct_states = seen.size;
      res.won = runs.some((r) => r.won);
      res.again_capped = runs.some((r) => r.capped);
      res.stochastic = probe(levels[0], SEEDS[1], SEQS[0], null).hash !== runs[0].hash;
    }
  } catch (e) {
    res.ok = false;
    res.exception = String(e).slice(0, 300);
  }
  process.stdout.write(JSON.stringify(res) + '\n');
}
