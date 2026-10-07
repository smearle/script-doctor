// Dump the reference engine's own parse of PuzzleScript games, for canonicalize.py.
//
//   node ps_extract.js ENGINE_DIR SHARD.jsonl START
//
// SHARD lines are {"id", "text"}; one JSON result per game goes to stdout, in order, from
// line START (check_games.py restarts past a game that hangs). A game is ok under the same
// verdict as ps_check.js: "Successful Compilation", no error, at least one playable level.
// For an ok game the record also gives
//   objects     object names in id order (the compiler assigns ids in collision-layer order);
//   layers      collision layers, expanded to objects, in the compiler's order;
//   synonyms, aggregates, properties   the legend, as [name, member, ...];
//   rules       raw rule lines, comment-stripped by the parser, startloop/endloop included;
//   wins        win conditions as token lists;
//   metadata    prelude flags (strings, numbers or number pairs);
//   levels      each playable level's initial cells as the compiler builds them, before
//               any rule runs (ragged rows padded, background filled, one object per
//               layer): {w, h, sets: [[object ids]], grid: [set index per cell, row-major]}.
'use strict';
const fs = require('fs');

const [engineDir, shardPath, startArg] = process.argv.slice(2);
const { loadEngine } = require('./ps_engine.js');
const engine = loadEngine(engineDir);
console.log = () => {};

function levelCells(lv) {
  // Level.objects is column-major (cell = x * height + y) with `words` 32-bit words per cell.
  const words = lv.objects.length / lv.n_tiles;
  const sets = [];
  const index = new Map();
  const grid = [];
  for (let y = 0; y < lv.height; y++) {
    for (let x = 0; x < lv.width; x++) {
      const c = x * lv.height + y;
      const ids = [];
      for (let w = 0; w < words; w++) {
        const v = lv.objects[c * words + w];
        if (!v) continue;
        for (let b = 0; b < 32; b++) if (v & (1 << b)) ids.push(w * 32 + b);
      }
      const key = ids.join(',');
      if (!index.has(key)) { index.set(key, sets.length); sets.push(ids); }
      grid.push(index.get(key));
    }
  }
  return { w: lv.width, h: lv.height, sets, grid };
}

const lines = fs.readFileSync(shardPath, 'utf8').split('\n').filter((l) => l.trim());
for (let i = Number(startArg || 0); i < lines.length; i++) {
  const { id, text } = JSON.parse(lines[i]);
  const res = { i, id };
  try {
    engine.unloadGame();
    engine.clearCapturedErrors();
    engine.compile(['restart'], text);
    const msgs = engine.getCapturedErrors();
    const compiled = msgs.some((m) => m.includes('Successful Compilation')) &&
      !msgs.some((m) => m.includes('Errors detected'));
    const levels = compiled ? engine.getState().levels.filter((l) => l.objects) : [];
    res.ok = compiled && levels.length > 0;
    if (res.ok) {
      const p = engine.serializeParsedState();
      res.objects = p.idDict;
      res.layers = p.collisionLayers;
      res.synonyms = p.legend_synonyms;
      res.aggregates = p.legend_aggregates;
      res.properties = p.legend_properties;
      res.rules = p.rules.map((r) => r[0]);
      res.wins = p.winconditions.map((w) => w.filter((t) => typeof t === 'string'));
      res.metadata = p.metadata;
      res.levels = levels.map(levelCells);
    }
  } catch (e) {
    res.ok = false;
    res.exception = String(e).slice(0, 300);
  }
  process.stdout.write(JSON.stringify(res) + '\n');
}
