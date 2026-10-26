'use strict';
// Reference traces for test_cpp_history.py. No custom Undo/Restart emulation.
const fs = require('fs');
console.log = () => {};
const api = require('../puzzlescript_nodejs/puzzlescript/engine.js');
const fixtures = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = [];
for (const f of fixtures) {
    const e = api.createFreshApi();
    e.compile(['loadLevel', 0], f.text, f.seed);
    const errors = e.getCapturedErrors();
    if (e.processInput.constructor('return errorCount;')() > 0) throw new Error(f.name + ': ' + errors.join('\n'));
    const compiled = e.serializeCompiledStateJSON();
    const extra = e.processInput.constructor('return backups.length;');
    const snap = () => ({
        objects: Array.from(e.backupLevel().dat),
        winning: e.getWinning(), againing: e.getAgaining(), depth: extra()
    });
    const trace = [snap()];
    for (const a of f.actions) {
        if (a === 'undo') e.DoUndo(false, false);
        else if (a === 'force_undo') e.DoUndo(true, false);
        else if (a === 'undo_distinct') e.DoUndo(false, true);
        else if (a === 'restart') e.DoRestart(false);
        else if (a === 'force_restart') e.DoRestart(true);
        else e.processInput(a);
        trace.push(snap());
    }
    result.push({name: f.name, compiled, trace});
}
process.stdout.write(JSON.stringify(result));
