'use strict';

const fs = require('fs');
const engine = require('./engine.js');
const solver = require('./solver.js');

let gameText = null;
let levelI = 0;

function loadLevel() {
    engine.compile(['loadLevel', levelI], gameText);
}

process.on('message', (message) => {
    const cmd = message && message.cmd;

    try {
        if (cmd === 'init') {
            gameText = fs.readFileSync(message.gamePath, 'utf8');
            levelI = message.levelI;
            process.send({ ok: true, cmd: 'ready' });
            return;
        }

        if (cmd === 'run') {
            loadLevel();
            const result = solver.randomRolloutRaw(
                engine,
                message.nSteps,
                message.timeoutMs,
            );
            process.send({ ok: true, cmd: 'result', result });
            return;
        }

        if (cmd === 'close') {
            process.exit(0);
        }

        process.send({
            ok: false,
            error_type: 'ValueError',
            error: `Unknown worker command: ${cmd}`,
        });
    } catch (error) {
        process.send({
            ok: false,
            error_type: error.name,
            error: error.message,
            error_stack: error.stack,
        });
    }
});
