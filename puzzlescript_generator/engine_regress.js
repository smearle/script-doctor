// Compare two versions of the reference engine on the same games (submodule updates).
//
//   PS_NEW_ENGINE=DIR node engine_regress.js OLD_ENGINE_DIR SHARD.jsonl START
//
// SHARD lines are {"id", "text"}; one JSON result per game goes to stdout, in order, from
// line START (check_games.py restarts past a game that hangs). Each engine dir holds the
// headless wrapper (puzzlescript_nodejs/) and PuzzleScript/src; the two load into separate
// VM realms. Per game: the compile verdict of each (as ps_check.js), whether the parse
// (objects, layers, legend, rules, win conditions) and every playable level's compiled
// cells agree, and whether 60 seeded random actions on the first two playable levels give
// the same cells and win flags (same engine RNG seed; message and sound output are no-ops
// in both, see ps_equiv.js).
'use strict';
const fs = require('fs');

const [oldDir, shardPath, startArg] = process.argv.slice(2);
const newDir = process.env.PS_NEW_ENGINE;
const { loadEngine } = require('./ps_engine.js');
const engines = [oldDir, newDir].map(loadEngine);
console.log = () => {};
const STEPS = 60;
const SEED = 'ps-regress';
const loaders = engines.map((engine) => {
  return engine.inEngine('k', 'seed', `
    curlevel = k; curlevelTarget = null; winning = false; againing = false; timer = 0;
    titleScreen = false; textMode = false; tick_lazy_function_generation(false);
    titleSelected = false; quittingMessageScreen = false; quittingTitleScreen = false;
    messageselected = false; titleMode = 0;
    loadLevelFromState(state, k, seed);`);
});

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function cells(engine) {
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
  return lvl.width + 'x' + lvl.height + ':' + out.join('|');
}

function run(e, text) {
  const engine = engines[e];
  const r = {};
  engine.unloadGame();
  engine.clearCapturedErrors();
  engine.compile(['restart'], text, SEED);
  const msgs = engine.getCapturedErrors();
  const compiled = msgs.some((m) => m.includes('Successful Compilation')) && !msgs.some((m) => m.includes('Errors detected'));
  const levels = compiled ? engine.getLevelInfo().filter((l) => l.type === 'level').map((l) => l.index) : [];
  r.ok = compiled && levels.length > 0;
  r.errors = msgs.filter((m) => !m.includes('Successful Compilation')).slice(0, 2);
  if (!r.ok) return r;
  const p = engine.serializeParsedState();
  r.parse = JSON.stringify([p.idDict, p.collisionLayers, p.legend_synonyms, p.legend_aggregates,
    p.legend_properties, p.rules.map((x) => x[0]), p.winconditions.map((w) => w.filter((t) => typeof t === 'string'))]);
  r.init = levels.map((k) => { loaders[e](k, SEED); return cells(engine); }).join('\n');
  const meta = engine.getState().metadata || {};
  const choices = ('noaction' in meta) ? [0, 1, 2, 3] : [0, 1, 2, 3, 4];
  r.traj = levels.slice(0, 2).map((k, j) => {
    const rand = mulberry32(1000 + j);
    loaders[e](k, SEED);
    const states = [cells(engine)];
    for (let t = 0; t < STEPS; t++) {
      engine.setWinning(false);
      engine.setHasUsedCheckpoint(false);
      engine.processInput(choices[Math.floor(rand() * choices.length)]);
      while (engine.getAgaining()) engine.processInput(-1);
      const won = engine.getWinning();
      states.push(cells(engine) + (won ? '#won' : ''));
      if (won) break;
    }
    return states.join('\n');
  }).join('\n\n');
  return r;
}

const lines = fs.readFileSync(shardPath, 'utf8').split('\n').filter((l) => l.trim());
for (let i = Number(startArg || 0); i < lines.length; i++) {
  const { id, text } = JSON.parse(lines[i]);
  const res = { i, id };
  try {
    const [a, b] = [run(0, text), run(1, text)];
    res.ok = [a.ok, b.ok];
    res.errors = [a.errors, b.errors];
    if (a.ok && b.ok) {
      res.same_parse = a.parse === b.parse;
      res.same_init = a.init === b.init;
      res.same_dyn = a.traj === b.traj;
    }
  } catch (e) {
    res.exception = String(e).slice(0, 300);
  }
  process.stdout.write(JSON.stringify(res) + '\n');
}
