// Compile PuzzleScript games with the reference engine and export them in the JSON format
// script-doctor's C++ engine loads (serializeCompiledStateJSON of the headless wrapper).
//
//   node ps_export.js ENGINE_DIR SHARD.jsonl START
//
// SHARD lines are {"id", "text"} (standard PuzzleScript). One line per game goes to stdout, in
// order, from line START (check_games.py restarts past a game that hangs): {i, id, ok, json} or
// {i, id, ok: false, error}. A game is exported only if it compiles without errors and has at
// least one playable level; the compile uses the probe's seed, so levels built by
// run_rules_on_level_start match the stage-3 checks.
'use strict';
const fs = require('fs');

const [engineDir, shardPath, startArg] = process.argv.slice(2);
const { loadEngine } = require('./ps_engine.js');
const engine = loadEngine(engineDir);
const write = process.stdout.write.bind(process.stdout);
console.log = () => {};

const lines = fs.readFileSync(shardPath, 'utf8').split('\n').filter((l) => l.trim());
for (let i = Number(startArg || 0); i < lines.length; i++) {
  const { id, text } = JSON.parse(lines[i]);
  const res = { i, id };
  try {
    engine.unloadGame();
    engine.clearLog();
    engine.compile(['restart'], text, 'ps-probe');
    const msgs = engine.getCapturedErrors();
    const compiled = msgs.some((m) => m.includes('Successful Compilation')) &&
      !msgs.some((m) => m.includes('Errors detected'));
    const levels = compiled ? engine.getLevelInfo().filter((l) => l.type === 'level') : [];
    res.ok = levels.length > 0;
    if (res.ok) res.json = engine.serializeCompiledStateJSON();
    else res.error = compiled ? 'no playable level' : 'compile errors';
  } catch (e) {
    res.ok = false;
    res.error = String(e).slice(0, 300);
  }
  write(JSON.stringify(res) + '\n');
}
