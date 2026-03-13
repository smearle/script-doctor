'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const engine = require('./engine.js');
const solver = require('./solver.js');

const jsgifRoot = path.resolve(__dirname, '../../PuzzleScript/src/js/jsgif');
const gifSourceFiles = [
    'LZWEncoder.js',
    'NeuQuant.js',
    'GIFEncoder.js',
];

let gifEncoderLoaded = false;

function ensureGifEncoderLoaded() {
    if (gifEncoderLoaded && typeof GIFEncoder !== 'undefined') {
        return;
    }

    for (const file of gifSourceFiles) {
        const absPath = path.join(jsgifRoot, file);
        const code = fs.readFileSync(absPath, 'utf8');
        vm.runInThisContext(code, { filename: absPath });
    }

    gifEncoderLoaded = true;
}

function scaleRgbFrame(rgbData, width, height, scale) {
    if (scale <= 1) {
        return { width, height, data: rgbData };
    }

    const scaledWidth = width * scale;
    const scaledHeight = height * scale;
    const scaled = new Uint8Array(scaledWidth * scaledHeight * 3);

    for (let y = 0; y < scaledHeight; y++) {
        const srcY = Math.floor(y / scale);
        for (let x = 0; x < scaledWidth; x++) {
            const srcX = Math.floor(x / scale);
            const srcIdx = (srcY * width + srcX) * 3;
            const dstIdx = (y * scaledWidth + x) * 3;
            scaled[dstIdx] = rgbData[srcIdx];
            scaled[dstIdx + 1] = rgbData[srcIdx + 1];
            scaled[dstIdx + 2] = rgbData[srcIdx + 2];
        }
    }

    return { width: scaledWidth, height: scaledHeight, data: scaled };
}

function rgbToRgba(rgbData) {
    const rgba = new Uint8Array((rgbData.length / 3) * 4);
    let src = 0;
    let dst = 0;
    while (src < rgbData.length) {
        rgba[dst] = rgbData[src];
        rgba[dst + 1] = rgbData[src + 1];
        rgba[dst + 2] = rgbData[src + 2];
        rgba[dst + 3] = 255;
        src += 3;
        dst += 4;
    }
    return rgba;
}

function addRenderedFrame(encoder, levelObj, scale) {
    const frame = engine.renderFrame(levelObj);
    if (!frame) {
        throw new Error('renderFrame returned null while encoding GIF');
    }

    const rgbData = new Uint8Array(Buffer.from(frame.dataBase64, 'base64'));
    const scaled = scaleRgbFrame(rgbData, frame.width, frame.height, scale);
    encoder.setSize(scaled.width, scaled.height);
    encoder.addFrame(rgbToRgba(scaled.data), true);
}

function renderSolutionGif({
    gameText,
    levelIndex,
    actions,
    gifPath,
    frameDurationMs = 50,
    scale = 1,
}) {
    ensureGifEncoderLoaded();

    const outDir = path.dirname(gifPath);
    if (outDir) {
        fs.mkdirSync(outDir, { recursive: true });
    }

    engine.compile(['loadLevel', levelIndex], gameText);
    solver.precalcDistances(engine);

    const encoder = new GIFEncoder();
    encoder.setRepeat(0);
    encoder.setDelay(frameDurationMs);
    encoder.start();

    addRenderedFrame(encoder, engine.backupLevel(), scale);

    for (const action of actions) {
        const result = solver.takeAction(engine, action);
        addRenderedFrame(encoder, result[5], scale);
    }

    encoder.finish();
    fs.writeFileSync(gifPath, Buffer.from(encoder.stream().bin));
    return gifPath;
}

module.exports = {
    renderSolutionGif,
};
