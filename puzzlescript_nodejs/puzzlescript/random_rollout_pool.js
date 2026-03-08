'use strict';

const path = require('path');
const { performance } = require('perf_hooks');
const { fork } = require('child_process');

const workerPath = path.join(__dirname, 'random_rollout_worker.js');

function summarizeRun(results, nSteps) {
    const workerResults = results.filter((result) => result.ok);
    const failedResults = results.filter((result) => !result.ok);

    if (workerResults.length === 0) {
        const sampleError = failedResults[0] || {};
        throw new Error(
            `All ${results.length} Node workers failed. Sample ${sampleError.error_type || 'error'}: ${sampleError.error || ''}`,
        );
    }

    const totalIterations = workerResults.reduce((sum, result) => sum + result.result.iterations, 0);
    const totalReportedWorkerTime = workerResults.reduce((sum, result) => sum + result.result.time, 0);
    const timeouts = workerResults.reduce((sum, result) => sum + Number(result.result.timeout), 0);
    const requestedIterations = results.length * nSteps;

    return {
        n_envs: results.length,
        total_iterations: totalIterations,
        requested_iterations: requestedIterations,
        completed_ratio: requestedIterations > 0 ? totalIterations / requestedIterations : 0,
        successful_workers: workerResults.length,
        failed_workers: failedResults.length,
        timeouts,
        timed_out: timeouts > 0,
        had_worker_failures: failedResults.length > 0,
        sample_worker_error: failedResults.length > 0 ? {
            error_type: failedResults[0].error_type,
            error: failedResults[0].error,
        } : null,
        mean_worker_fps: totalReportedWorkerTime > 0 ? totalIterations / totalReportedWorkerTime : 0,
    };
}

function waitForMessage(worker) {
    return new Promise((resolve, reject) => {
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
            worker.off('message', onMessage);
            worker.off('exit', onExit);
            worker.off('error', onError);
        };

        worker.once('message', onMessage);
        worker.once('exit', onExit);
        worker.once('error', onError);
    });
}

async function main() {
    const config = JSON.parse(process.argv[2]);
    const workers = [];

    try {
        for (let workerI = 0; workerI < config.nEnvs; workerI += 1) {
            const worker = fork(workerPath, [], {
                stdio: ['inherit', 'inherit', 'inherit', 'ipc'],
            });
            workers.push(worker);
        }

        await Promise.all(workers.map(async (worker) => {
            worker.send({
                cmd: 'init',
                gamePath: config.gamePath,
                levelI: config.levelI,
            });
            const message = await waitForMessage(worker);
            if (!message.ok || message.cmd !== 'ready') {
                throw new Error(message.error || 'Node worker failed during init.');
            }
        }));

        const runs = [];
        for (let repeatI = 0; repeatI < config.repeats; repeatI += 1) {
            const start = performance.now();
            const resultPromises = workers.map((worker) => {
                worker.send({
                    cmd: 'run',
                    nSteps: config.nSteps,
                    timeoutMs: config.timeoutMs,
                });
                return waitForMessage(worker);
            });

            const results = await Promise.all(resultPromises);
            const wallTime = (performance.now() - start) / 1000;
            const summary = summarizeRun(results, config.nSteps);
            runs.push({
                ...summary,
                wall_time: wallTime,
                fps: wallTime > 0 ? summary.total_iterations / wallTime : 0,
            });
        }

        process.stdout.write(`${JSON.stringify({ runs })}\n`);
    } finally {
        for (const worker of workers) {
            if (worker.connected) {
                worker.send({ cmd: 'close' });
            }
        }
    }
}

main().catch((error) => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exit(1);
});
