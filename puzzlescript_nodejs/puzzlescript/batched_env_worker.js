'use strict';

const engine = require('./engine.js');
const solver = require('./solver.js');

let gameText = null;
let levelI = 0;
let maxEpisodeSteps = 0;
let objectNames = [];
let objectCount = 0;
let width = 0;
let height = 0;
let strideObj = 0;
let stepCount = 0;
let currentScore = 0;

function doRestart(force) {
    engine.clearBackups();
    if (engine.getRestarting()) {
        return;
    }
    if (force !== true && ('norestart' in engine.getState().metadata)) {
        return;
    }
    engine.setRestarting(true);
    if (force !== true) {
        engine.addUndoState(engine.backupLevel());
    }
    engine.restoreLevel(engine.getRestartTarget());
    if ('run_rules_on_level_start' in engine.getState().metadata) {
        engine.processInput(-1, true);
    }
    engine.getLevel().commandQueue = [];
    engine.getLevel().commandQueueSourceRules = [];
    engine.setRestarting(false);
}

function loadLevel() {
    engine.compile(['loadLevel', levelI], gameText);
    const level = engine.backupLevel();
    objectNames = Array.from(engine.getState().idDict);
    objectCount = objectNames.length;
    width = level.width;
    height = level.height;
    strideObj = Math.ceil(objectCount / 32);
    stepCount = 0;
    solver.precalcDistances(engine);
    currentScore = Number(solver.getScore(engine));
    return level;
}

function levelToObservation(level) {
    const obs = Buffer.alloc(objectCount * width * height, 0);
    for (let x = 0; x < width; x += 1) {
        for (let y = 0; y < height; y += 1) {
            const cellOffset = x * height + y;
            const flatIdx = cellOffset * strideObj;
            for (let chunkI = 0; chunkI < strideObj; chunkI += 1) {
                const raw = Number(level.dat[String(flatIdx + chunkI)] || 0) >>> 0;
                for (let bit = 0; bit < 32; bit += 1) {
                    const objI = chunkI * 32 + bit;
                    if (objI >= objectCount) {
                        break;
                    }
                    if ((raw & (1 << bit)) !== 0) {
                        obs[objI * width * height + cellOffset] = 1;
                    }
                }
            }
        }
    }
    return obs;
}

function snapshot() {
    const level = engine.backupLevel();
    return {
        obs: levelToObservation(level),
        score: currentScore,
        steps: stepCount,
        won: false,
    };
}

function resetEnv() {
    loadLevel();
    return snapshot();
}

function stepEnv(action) {
    const prevScore = currentScore;
    engine.processInput(action);
    while (engine.getAgaining()) {
        engine.processInput(-1);
    }

    const scoreBeforeReset = Number(solver.getScore(engine));
    const won = Boolean(engine.getWinning());
    stepCount += 1;
    const truncated = stepCount >= maxEpisodeSteps;
    const reward = (scoreBeforeReset - prevScore) + (won ? 1.0 : 0.0) - 0.01;

    if (won) {
        doRestart(true);
        stepCount = 0;
    } else if (truncated) {
        loadLevel();
    }

    currentScore = Number(solver.getScore(engine));
    const level = engine.backupLevel();
    return {
        obs: levelToObservation(level),
        reward,
        done: won || truncated,
        truncated,
        won,
        score: scoreBeforeReset,
        steps: stepCount,
    };
}

process.on('message', (message) => {
    const cmd = message && message.cmd;

    try {
        if (cmd === 'init') {
            gameText = String(message.gameText);
            levelI = Number(message.levelI);
            maxEpisodeSteps = Number(message.maxEpisodeSteps);
            const initial = resetEnv();
            process.send({
                ok: true,
                cmd: 'ready',
                width,
                height,
                objectCount,
                objectNames,
                obs: initial.obs,
                score: initial.score,
                steps: initial.steps,
            });
            return;
        }

        if (cmd === 'reset') {
            const result = resetEnv();
            process.send({
                ok: true,
                cmd: 'reset_result',
                obs: result.obs,
                score: result.score,
                steps: result.steps,
            });
            return;
        }

        if (cmd === 'step') {
            const result = stepEnv(Number(message.action));
            process.send({
                ok: true,
                cmd: 'step_result',
                ...result,
            });
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
