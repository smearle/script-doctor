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
    sandbox.Audio = function Audio() {
        return {
            currentTime: 0,
            volume: 0,
            paused: true,
            play() {},
            pause() {},
            load() {},
            addEventListener() {},
            removeEventListener() {},
            setAttribute() {},
            cloneNode() {
                return this;
            },
            canPlayType() {
                return '';
            },
        };
    };
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
function _bvToArr(bv) {
    if (!bv || !bv.data) return [];
    return Array.from(bv.data);
}

function _serializeCellPattern(cp) {
    let rep = null;
    if (cp.replacement) {
        rep = {
            objectsClear: _bvToArr(cp.replacement.objectsClear),
            objectsSet: _bvToArr(cp.replacement.objectsSet),
            movementsClear: _bvToArr(cp.replacement.movementsClear),
            movementsSet: _bvToArr(cp.replacement.movementsSet),
            movementsLayerMask: _bvToArr(cp.replacement.movementsLayerMask),
            randomEntityMask: _bvToArr(cp.replacement.randomEntityMask),
            randomDirMask: _bvToArr(cp.replacement.randomDirMask),
        };
    }
    return {
        objectsPresent: _bvToArr(cp.objectsPresent),
        objectsMissing: _bvToArr(cp.objectsMissing),
        anyObjectsPresent: cp.anyObjectsPresent.map(bv => _bvToArr(bv)),
        movementsPresent: _bvToArr(cp.movementsPresent),
        movementsMissing: _bvToArr(cp.movementsMissing),
        replacement: rep,
    };
}

function _serializeRule(rule) {
    const patterns = [];
    for (let i = 0; i < rule.patterns.length; i++) {
        const row = [];
        for (let j = 0; j < rule.patterns[i].length; j++) {
            const cell = rule.patterns[i][j];
            if (cell === ellipsisPattern || (Array.isArray(cell) && cell[0] === 'ellipsis')) {
                row.push('ellipsis');
            } else {
                row.push(_serializeCellPattern(cell));
            }
        }
        patterns.push(row);
    }
    return {
        direction: rule.direction,
        patterns: patterns,
        hasReplacements: rule.hasReplacements,
        lineNumber: rule.lineNumber,
        ellipsisCount: rule.ellipsisCount,
        groupNumber: rule.groupNumber,
        rigid: rule.rigid,
        commands: rule.commands,
        isRandom: rule.isRandom,
        cellRowMasks: rule.cellRowMasks.map(m => _bvToArr(m)),
        cellRowMasks_Movements: rule.cellRowMasks_Movements.map(m => _bvToArr(m)),
    };
}

function serializeCompiledState() {
    if (!state) return null;
    const s = state;
    const OBJECT_SIZE = Math.ceil(s.objectCount / 32);
    const MOVEMENT_SIZE = Math.ceil(s.collisionLayers.length / 5);

    const rules = s.rules.map(group => group.map(r => _serializeRule(r)));
    const lateRules = s.lateRules.map(group => group.map(r => _serializeRule(r)));

    const winconditions = s.winconditions.map(wc => ({
        num: wc[0],
        mask1: _bvToArr(wc[1]),
        mask2: typeof wc[2] === 'string' ? null : _bvToArr(wc[2]),
        lineNumber: wc[3],
        aggr1: wc[4],
        aggr2: wc[5],
    }));

    const levels = [];
    for (let i = 0; i < s.levels.length; i++) {
        const lv = s.levels[i];
        if (lv.objects) {
            levels.push({
                type: 'level',
                index: i,
                lineNumber: lv.lineNumber,
                width: lv.width,
                height: lv.height,
                layerCount: lv.layerCount,
                objects: Array.from(lv.objects),
            });
        } else {
            levels.push({type: 'message', index: i});
        }
    }

    const layerMasks = s.layerMasks.map(m => _bvToArr(m));
    const playerMask = [s.playerMask[0], _bvToArr(s.playerMask[1])];

    const loopPoint = {};
    if (s.loopPoint) {
        for (const k in s.loopPoint) {
            loopPoint[k] = s.loopPoint[k];
        }
    }
    const lateLoopPoint = {};
    if (s.lateLoopPoint) {
        for (const k in s.lateLoopPoint) {
            lateLoopPoint[k] = s.lateLoopPoint[k];
        }
    }

    const result = {
        objectCount: s.objectCount,
        layerCount: s.collisionLayers.length,
        STRIDE_OBJ: OBJECT_SIZE,
        STRIDE_MOV: MOVEMENT_SIZE,
        idDict: s.idDict,
        playerMask: playerMask,
        layerMasks: layerMasks,
        rules: rules,
        lateRules: lateRules,
        winconditions: winconditions,
        levels: levels,
        loopPoint: loopPoint,
        lateLoopPoint: lateLoopPoint,
        rigid: !!s.rigid,
        rigidGroupIndex_to_GroupIndex: s.rigidGroupIndex_to_GroupIndex || [],
        groupNumber_to_RigidGroupIndex: s.groupNumber_to_RigidGroupIndex || {},
        metadata: {},
        backgroundid: s.backgroundid,
        backgroundlayer: s.backgroundlayer,
        collisionLayers: s.collisionLayers.map(layer => layer.map(obj => obj)),
    };

    if (s.metadata) {
        for (const k_meta in s.metadata) {
            const v = s.metadata[k_meta];
            if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
                result.metadata[k_meta] = v;
            }
        }
    }

    return result;
}

function serializeLevel(levelIndex) {
    if (!state || !state.levels || levelIndex >= state.levels.length) return null;
    const lv = state.levels[levelIndex];
    if (!lv.objects) return null;
    return {
        width: lv.width,
        height: lv.height,
        layerCount: lv.layerCount,
        objects: Array.from(lv.objects),
    };
}

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
    serializeCompiledState,
    serializeCompiledStateJSON: () => JSON.stringify(serializeCompiledState()),
    serializeLevel,
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
