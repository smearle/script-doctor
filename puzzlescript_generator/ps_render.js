// Render the first playable level of each PuzzleScript game with the reference engine.
//
//   node ps_render.js ENGINE_DIR IN.jsonl > OUT.jsonl
//
// IN lines are {"id", "text"}; each OUT line is {"id", "width", "height", "rgb_b64"} (or
// {"id", "error"}), the initial state of the first level as the engine would draw it.
'use strict';
const fs = require('fs');

const [engineDir, inPath] = process.argv.slice(2);
const { loadEngine } = require('./ps_engine.js');
const engine = loadEngine(engineDir);
const origLog = console.log;
console.log = () => {};

for (const line of fs.readFileSync(inPath, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const { id, text } = JSON.parse(line);
  let out = { id };
  try {
    engine.unloadGame();
    engine.clearCapturedErrors();
    engine.compile(['restart'], text);
    const level = engine.getLevelInfo().find((l) => l.type === 'level');
    if (!level) throw new Error('no playable level');
    engine.compile(['loadLevel', level.index], text);
    const f = engine.renderFrame();
    if (!f) throw new Error('renderFrame returned null');
    out = { id, width: f.width, height: f.height, rgb_b64: f.dataBase64 };
  } catch (e) {
    out.error = String(e).slice(0, 200);
  }
  process.stdout.write(JSON.stringify(out) + '\n');
}
