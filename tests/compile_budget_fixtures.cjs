'use strict';
const fs = require('fs');
console.log = () => {};
const api = require('../puzzlescript_nodejs/puzzlescript/engine.js');
const games = JSON.parse(fs.readFileSync(0, 'utf8'));
const results = [];
for (const game of games) {
    const e = api.createFreshApi();
    e.processInput.constructor('IDE = false; processOutputCommands = function () {};')();
    // Compile only: never execute startup rules in the unbounded reference
    // when the fixture deliberately contains an explosive product.
    e.compile(['rebuild'], game.text, 'budget-tests');
    if (e.processInput.constructor('return errorCount;')() > 0)
        throw new Error(game.name + ': ' + e.getCapturedErrors().join('\n'));
    results.push({name: game.name, compiled: e.serializeCompiledStateJSON()});
}
process.stdout.write(JSON.stringify(results));
