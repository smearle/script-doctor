'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { performance } = require('perf_hooks');

const srcRoot = path.resolve(__dirname, '../../PuzzleScript/src');

const sourceFiles = [
    'js/storagewrapper.js',
    'js/bitvec.js',
    'js/level.js',
    'js/languageConstants.js',
    'js/globalVariables.js',
    'js/debug.js',
    'js/font.js',
    'js/rng.js',
    'js/riffwave.js',
    'js/sfxr.js',
    'js/codemirror/stringstream.js',
    'js/colors.js',
    'js/engine.js',
    'js/parser.js',
    'js/compiler.js',
    'js/soundbar.js',
];

let cachedApi = null;
let cachedContext = null;

function createSandbox() {
    const storage = {};
    const sandbox = {
        console,
        setTimeout,
        clearTimeout,
        setInterval,
        clearInterval,
        Int32Array,
        Uint8Array,
        Uint32Array,
        Float32Array,
        Float64Array,
        ArrayBuffer,
        Math,
        Date,
        JSON,
        performance,
    };

    sandbox.localStorage = {
        getItem(key) {
            return Object.prototype.hasOwnProperty.call(storage, key) ? storage[key] : null;
        },
        setItem(key, value) {
            storage[key] = String(value);
        },
        removeItem(key) {
            delete storage[key];
        },
    };

    const makeElement = () => ({
        style: {},
        innerHTML: '',
        textContent: '',
        addEventListener() {},
        removeEventListener() {},
        appendChild() {},
        setAttribute() {},
        getContext() {
            return null;
        },
    });

    sandbox.document = {
        URL: 'node://standalone',
        body: {
            classList: {
                contains() {
                    return false;
                },
            },
            addEventListener() {},
            removeEventListener() {},
            appendChild() {},
        },
        createElement() {
            return makeElement();
        },
        getElementById() {
            return null;
        },
        addEventListener() {},
        removeEventListener() {},
    };

    sandbox.window = sandbox;
    sandbox.globalThis = sandbox;
    sandbox.self = sandbox;
    sandbox.canvas = null;
    sandbox.lastDownTarget = null;
    sandbox.input = makeElement();
    sandbox.IDE = false;

    sandbox.canvasResize = function() {};
    sandbox.redraw = function() {};
    sandbox.forceRegenImages = function() {};
    sandbox.consolePrintFromRule = function() {};
    sandbox.consolePrint = function() {};
    sandbox.console_print_raw = sandbox.console.log.bind(sandbox.console);
    sandbox.consoleError = function() {};
    sandbox.consoleCacheDump = function() {};
    sandbox.addToDebugTimeline = function() {
        return 0;
    };
    sandbox.killAudioButton = function() {};
    sandbox.showAudioButton = function() {};
    sandbox.regenSpriteImages = function() {};
    sandbox.jumpToLine = function() {};
    sandbox.printLevel = function() {};
    sandbox.playSound = function() {};
    sandbox.clearInputHistory = function() {};
    sandbox.pushInput = function() {};
    sandbox.resetCommands = function() {};
    sandbox.tryPlaySimpleSound = function() {};
    sandbox.tryLoadCustomFont = function() {};
    sandbox.storage_quota_infinite = true;

    vm.createContext(sandbox);
    return sandbox;
}

function buildSourceBundle() {
    let allCode = `
var module = undefined;
var exports = undefined;
var require = undefined;
`;

    for (const file of sourceFiles) {
        const absolutePath = path.join(srcRoot, file);
        const code = fs.readFileSync(absolutePath, 'utf8');
        allCode += `\n// ---- ${file} ----\n${code}\n`;
    }

    allCode += `
globalThis.__PS_NODE_API__ = {
    compile,
    backupLevel,
    restoreLevel,
    processInput,
    addUndoState,
    getWinning: () => winning,
    setWinning: (value) => { winning = value; },
    getLevel: () => level,
    getState: () => state,
    getRestarting: () => restarting,
    setRestarting: (value) => { restarting = value; },
    getRestartTarget: () => restartTarget,
    getDeltaTime: () => deltatime,
    setDeltaTime: (value) => { deltatime = value; },
    getAgaining: () => againing,
    getHasUsedCheckpoint: () => hasUsedCheckpoint,
    setHasUsedCheckpoint: (value) => { hasUsedCheckpoint = value; },
    get_o10: () => _o10,
    getNumLevels: () => state.levels.length,
    unloadGame,
    clearBackups: () => { backups = []; },
};
`;

    return allCode;
}

function loadApi() {
    if (cachedApi !== null) {
        return cachedApi;
    }

    cachedContext = createSandbox();
    vm.runInContext(buildSourceBundle(), cachedContext, {
        filename: 'standalone_puzzlescript_sources.js',
    });
    cachedApi = cachedContext.__PS_NODE_API__;
    return cachedApi;
}

module.exports = loadApi();
