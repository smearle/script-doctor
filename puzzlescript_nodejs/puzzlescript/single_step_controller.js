'use strict';

const readline = require('readline');
const engine = require('./engine.js');

const MAX_AGAIN = 50;

let levelBackup = null;

function respond(payload) {
    process.stdout.write(`${JSON.stringify(payload)}\n`);
}

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        crlfDelay: Infinity,
    });

    for await (const line of rl) {
        if (!line.trim()) {
            continue;
        }

        const message = JSON.parse(line);
        const cmd = message.cmd;

        try {
            if (cmd === 'init') {
                engine.compile(['loadLevel', message.levelI], message.gameText);
                levelBackup = engine.backupLevel();
                respond({ ok: true });
                continue;
            }

            if (cmd === 'step') {
                engine.processInput(message.action);
                let againCount = 0;
                while (engine.getAgaining() && againCount < MAX_AGAIN) {
                    engine.processInput(-1);
                    againCount += 1;
                }
                const won = engine.getWinning();
                if (won) {
                    engine.restoreLevel(levelBackup);
                }
                respond({ ok: true, won });
                continue;
            }

            if (cmd === 'reset') {
                engine.restoreLevel(levelBackup);
                respond({ ok: true });
                continue;
            }

            if (cmd === 'close') {
                respond({ ok: true });
                process.exit(0);
            }

            respond({ ok: false, error: `Unknown command: ${cmd}` });
        } catch (error) {
            respond({ ok: false, error: error.message, error_stack: error.stack });
        }
    }
}

main().catch((error) => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exit(1);
});
