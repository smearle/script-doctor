'use strict';

const path = require('path');
const readline = require('readline');
const { fork } = require('child_process');

const workerPath = path.join(__dirname, 'batched_env_worker.js');
const INIT_REQUEST_TIMEOUT_MS = 60_000;
const RESET_REQUEST_TIMEOUT_MS = 15_000;
const STEP_REQUEST_TIMEOUT_MS = 15_000;
const CLOSE_GRACE_TIMEOUT_MS = 1_000;

function waitForMessage(worker, timeoutMs) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            cleanup();
            reject(new Error(`Worker timed out after ${timeoutMs}ms`));
        }, timeoutMs);
        const onMessage = (message) => {
            cleanup();
            resolve(message);
        };
        const onExit = (code, signal) => {
            cleanup();
            reject(new Error(`Worker exited before responding (code=${code}, signal=${signal})`));
        };
        const onError = (error) => {
            cleanup();
            reject(error);
        };
        const cleanup = () => {
            clearTimeout(timer);
            worker.off('message', onMessage);
            worker.off('exit', onExit);
            worker.off('error', onError);
        };

        worker.once('message', onMessage);
        worker.once('exit', onExit);
        worker.once('error', onError);
    });
}

function request(worker, payload, timeoutMs) {
    worker.send(payload);
    return waitForMessage(worker, timeoutMs);
}

function formatWorkerError(result, envI) {
    return `Node batched worker ${envI} failed: ${result.error_type || 'Error'}: ${result.error || 'unknown error'}${result.error_stack ? `\n${result.error_stack}` : ''}`;
}

function ensureWorkerResultsOkay(results) {
    for (let envI = 0; envI < results.length; envI += 1) {
        const result = results[envI];
        if (!result || result.ok !== true) {
            throw new Error(formatWorkerError(result || {}, envI));
        }
        if (typeof result.obs === 'undefined') {
            throw new Error(`Node batched worker ${envI} returned no observation payload.`);
        }
    }
}

function writeResponse(payload, obsBuffer) {
    const header = {
        ...payload,
        obs_bytes: obsBuffer ? obsBuffer.length : 0,
    };
    process.stdout.write(`${JSON.stringify(header)}\n`);
    if (obsBuffer && obsBuffer.length > 0) {
        process.stdout.write(obsBuffer);
    }
}

async function closeWorkers(state) {
    const workers = state.workers || [];
    await Promise.all(workers.map(async (worker) => {
        try {
            if (worker.connected) {
                worker.send({ cmd: 'close' });
            }
        } catch (_) {
            // Best-effort shutdown.
        }
        await new Promise((resolve) => setTimeout(resolve, CLOSE_GRACE_TIMEOUT_MS));
        try {
            if (worker.exitCode === null && worker.signalCode === null) {
                worker.kill('SIGTERM');
            }
        } catch (_) {
            // Best-effort shutdown.
        }
    }));
}

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        crlfDelay: Infinity,
    });

    const state = {
        workers: [],
        obsBuffers: [],
        width: 0,
        height: 0,
        objectCount: 0,
        objectNames: [],
        nEnvs: 0,
        maxEpisodeSteps: 0,
    };

    try {
        for await (const line of rl) {
            if (!line.trim()) {
                continue;
            }

            const message = JSON.parse(line);
            const cmd = message.cmd;

            if (cmd === 'init') {
                state.nEnvs = Number(message.nEnvs);
                state.maxEpisodeSteps = Number(message.maxEpisodeSteps);
                state.workers = [];
                state.obsBuffers = [];

                for (let workerI = 0; workerI < state.nEnvs; workerI += 1) {
                    state.workers.push(fork(workerPath, [], {
                        stdio: ['ignore', 'ignore', 'ignore', 'ipc'],
                    }));
                }

                const ready = await Promise.all(state.workers.map((worker) => request(worker, {
                    cmd: 'init',
                    gameText: message.gameText,
                    levelI: message.levelI,
                    maxEpisodeSteps: state.maxEpisodeSteps,
                    autoReset: message.autoReset !== false,
                }, INIT_REQUEST_TIMEOUT_MS)));
                ensureWorkerResultsOkay(ready);

                const first = ready[0];
                state.width = Number(first.width);
                state.height = Number(first.height);
                state.objectCount = Number(first.objectCount);
                state.objectNames = Array.from(first.objectNames || []);
                state.obsBuffers = ready.map((item) => Buffer.from(item.obs));

                writeResponse({
                    ok: true,
                    cmd: 'ready',
                    width: state.width,
                    height: state.height,
        object_count: state.objectCount,
        object_names: state.objectNames,
        n_envs: state.nEnvs,
        num_levels: Number(first.numLevels),
    }, Buffer.concat(state.obsBuffers));
                continue;
            }

            if (cmd === 'reset') {
                const requested = Array.isArray(message.indices)
                    ? message.indices.map((idx) => Number(idx))
                    : [...Array(state.nEnvs).keys()];

                const results = await Promise.all(requested.map((envI) => request(state.workers[envI], {
                    cmd: 'reset',
                }, RESET_REQUEST_TIMEOUT_MS)));
                ensureWorkerResultsOkay(results);

                for (let i = 0; i < requested.length; i += 1) {
                    state.obsBuffers[requested[i]] = Buffer.from(results[i].obs);
                }

                writeResponse({
                    ok: true,
                    cmd: 'reset_result',
                }, Buffer.concat(state.obsBuffers));
                continue;
            }

            if (cmd === 'step') {
                const actions = Array.from(message.actions || []);
                if (actions.length !== state.nEnvs) {
                    throw new Error(`Expected ${state.nEnvs} actions, received ${actions.length}.`);
                }

                const results = await Promise.all(actions.map((action, envI) => request(state.workers[envI], {
                    cmd: 'step',
                    action: Number(action),
                }, STEP_REQUEST_TIMEOUT_MS).catch((error) => ({
                    ok: false,
                    error_type: error.name,
                    error: error.message,
                }))));
                ensureWorkerResultsOkay(results);

                for (let envI = 0; envI < state.nEnvs; envI += 1) {
                    state.obsBuffers[envI] = Buffer.from(results[envI].obs);
                }

                writeResponse({
                    ok: true,
                    cmd: 'step_result',
                    rewards: results.map((item) => Number(item.reward)),
                    dones: results.map((item) => Boolean(item.done)),
                    truncated: results.map((item) => Boolean(item.truncated)),
                    won: results.map((item) => Boolean(item.won)),
                    score: results.map((item) => Number(item.score)),
                    steps: results.map((item) => Number(item.steps)),
                    level_i: results.map((item) => Number(item.level_i)),
                    next_level_i: results.map((item) => Number(item.next_level_i)),
                }, Buffer.concat(state.obsBuffers));
                continue;
            }

            if (cmd === 'close') {
                await closeWorkers(state);
                writeResponse({
                    ok: true,
                    cmd: 'closed',
                });
                process.exit(0);
            }

            throw new Error(`Unknown controller command: ${cmd}`);
        }
    } catch (error) {
        writeResponse({
            ok: false,
            cmd: 'error',
            error: error.message,
        });
    } finally {
        await closeWorkers(state);
    }
}

main().catch((error) => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exit(1);
});
