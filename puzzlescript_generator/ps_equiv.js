// Check that a canonical PuzzleScript game behaves like its original (reference engine).
//
//   node ps_equiv.js ENGINE_DIR SHARD.jsonl START
//
// SHARD lines are {"id", "orig", "canon", "map"}; map sends original object names to
// canonical ones. One JSON result per game goes to stdout, in order, from line START (the
// supervisor in check_games.py restarts past a game that hangs). Checks:
//   init: every playable level starts with the same objects in every cell (under map);
//   dyn:  on the first two playable levels, 60 seeded random actions give the same cell
//         contents and win flag after every step.
// Every level is loaded with the same engine RNG seed, so `random` rules draw the same
// numbers (nondet: the original still does not replay itself, so dyn is moot for it).
// Message and sound output commands are no-ops in both games (see below).
'use strict';
const fs = require('fs');

const [engineDir, shardPath, startArg] = process.argv.slice(2);
const { loadEngine } = require('./ps_engine.js');
const engine = loadEngine(engineDir);
console.log = () => {};
const STEPS = 60;
const SEED = 'ps-equiv';
const inEngine = engine.inEngine;  // see ps_engine.js (message output is a no-op in both games)
// Load level k of the compiled game as compile(['loadLevel', k], text, seed) does after
// compiling (setGameState's 'loadLevel' case), without compiling again.
const loadLevel = inEngine('k', 'seed', `
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

function compileGame(text) {
  engine.unloadGame();
  engine.clearCapturedErrors();
  engine.compile(['restart'], text, SEED);
  const msgs = engine.getCapturedErrors();
  const ok = msgs.some((m) => m.includes('Successful Compilation')) && !msgs.some((m) => m.includes('Errors detected'));
  const levels = ok ? engine.getLevelInfo().filter((l) => l.type === 'level').map((l) => l.index) : [];
  return { ok, levels, errors: msgs.filter((m) => !m.includes('Successful Compilation')).slice(0, 3) };
}

function cells(map) {
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
      for (let b = 0; b < 32; b++) {
        if (v & (1 << b)) {
          const name = ids[w * 32 + b];
          names.push(map ? (map[name] || ('?' + name)) : name);
        }
      }
    }
    names.sort();
    out.push(names.join(','));
  }
  return lvl.width + 'x' + lvl.height + ':' + out.join('|');
}

function step(action) {
  engine.setWinning(false);
  engine.setHasUsedCheckpoint(false);
  engine.processInput(action);
  while (engine.getAgaining()) engine.processInput(-1);
  return engine.getWinning();
}

function trajectory(k, map, actions) {
  loadLevel(k, SEED);
  const states = [cells(map)];
  for (const a of actions) {
    const won = step(a);
    states.push(cells(map) + (won ? '#won' : ''));
    if (won) break;
  }
  return states.join('\n');
}

// For the game just compiled: the initial cells of every playable level, then `replays`
// trajectories on each of the first two playable levels (one action sequence per level).
function measure(g, map, actionSeqs, replays) {
  if (!g.ok) return g;
  g.init = g.levels.map((k) => { loadLevel(k, SEED); return cells(map); });
  g.traj = actionSeqs.slice(0, g.levels.length).map((actions, j) =>
    Array.from({ length: replays }, () => trajectory(g.levels[j], map, actions)));
  return g;
}

const lines = fs.readFileSync(shardPath, 'utf8').split('\n').filter((l) => l.trim());
for (let i = Number(startArg || 0); i < lines.length; i++) {
  const { id, orig, canon, map } = JSON.parse(lines[i]);
  const res = { i, id };
  try {
    const o = compileGame(orig);
    const meta = o.ok ? (engine.getState().metadata || {}) : {};
    const choices = ('noaction' in meta) ? [0, 1, 2, 3] : [0, 1, 2, 3, 4];
    const actionSeqs = [0, 1].map((j) => {
      const rand = mulberry32(1000 + j);
      return Array.from({ length: STEPS }, () => choices[Math.floor(rand() * choices.length)]);
    });
    measure(o, map, actionSeqs, 2);
    const c = measure(compileGame(canon), null, actionSeqs, 1);
    res.canon_compiled = c.ok;
    res.canon_errors = c.errors;
    res.n_levels = [o.levels.length, c.levels.length];
    if (!o.ok || !c.ok || o.levels.length !== c.levels.length || !c.levels.length) {
      res.init_ok = false;
    } else {
      const k = o.init.findIndex((s, j) => s !== c.init[j]);
      res.init_ok = k < 0;
      if (k >= 0) res.init_mismatch_level = k;
      if (res.init_ok) {
        res.dyn_ok = true;
        o.traj.forEach((runs, j) => {
          if (runs[0] !== runs[1]) res.nondet = true;  // the original does not replay itself
          if (res.dyn_ok && runs[0] !== c.traj[j][0]) {
            const a = runs[0].split('\n');
            const b = c.traj[j][0].split('\n');
            let t = 0;
            while (t < Math.max(a.length, b.length) && a[t] === b[t]) t++;
            res.dyn_ok = false;
            res.dyn_mismatch = [j, t];
          }
        });
      }
    }
  } catch (e) {
    res.exception = String(e).slice(0, 300);
    res.init_ok = false;
  }
  process.stdout.write(JSON.stringify(res) + '\n');
}
