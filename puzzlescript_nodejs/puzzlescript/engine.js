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
        Buffer,
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
// ---- Monkey-patch to capture pre-compilation state ----
var _parsedSnapshot = null;
var _origLevelsToArray = levelsToArray;
levelsToArray = function(st) {
    // Snapshot rules, winconditions, and levels before destructive transforms.
    // At this point generateExtraMembers and generateMasks have run,
    // so state.objects, legend_*, collisionLayers, idDict are populated.
    try {
        _parsedSnapshot = {
            rules: JSON.parse(JSON.stringify(st.rules)),
            winconditions: JSON.parse(JSON.stringify(st.winconditions)),
            levels: JSON.parse(JSON.stringify(st.levels)),
            loops: st.loops ? JSON.parse(JSON.stringify(st.loops)) : [],
        };
    } catch(e) {
        _parsedSnapshot = null;
    }
    return _origLevelsToArray(st);
};

function serializeParsedState() {
    if (!state || !_parsedSnapshot) return null;
    var s = state;

    var objects = {};
    for (var name in s.objects) {
        var obj = s.objects[name];
        objects[name] = {
            colors: obj.colors.map(function(c) { return String(c); }),
            spritematrix: obj.spritematrix.map(function(row) {
                return Array.isArray(row) ? Array.from(row) : String(row);
            }),
        };
    }

    // legend entries: pass through as-is (no line numbers to strip)
    var legend_synonyms = s.legend_synonyms.map(function(e) { return Array.from(e); });
    var legend_aggregates = s.legend_aggregates.map(function(e) { return Array.from(e); });
    var legend_properties = s.legend_properties.map(function(e) { return Array.from(e); });

    var collisionLayers = s.collisionLayers.map(function(layer) {
        return layer.map(function(o) { return String(o); });
    });

    var metadata = {};
    if (s.metadata) {
        for (var k in s.metadata) {
            var v = s.metadata[k];
            if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
                metadata[k] = v;
            } else if (Array.isArray(v) && v.every(function(x) { return typeof x === 'number'; })) {
                metadata[k] = Array.from(v);
            }
        }
    }

    return {
        objects: objects,
        idDict: s.idDict.map(function(n) { return String(n); }),
        legend_synonyms: legend_synonyms,
        legend_aggregates: legend_aggregates,
        legend_properties: legend_properties,
        collisionLayers: collisionLayers,
        rules: _parsedSnapshot.rules,
        winconditions: _parsedSnapshot.winconditions,
        levels: _parsedSnapshot.levels,
        loops: _parsedSnapshot.loops,
        metadata: metadata,
        original_case_names: s.original_case_names || {},
    };
}

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
            levels.push({type: 'message', index: i, message: lv.message || ''});
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
            } else if (Array.isArray(v) && v.every(item => typeof item === 'number')) {
                result.metadata[k_meta] = Array.from(v);
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

function _playerMatchesCell(cellWords) {
    const playerMask = state.playerMask[1];
    if (state.playerMask[0]) {
        for (let i = 0; i < playerMask.data.length; i++) {
            if ((playerMask.data[i] & cellWords[i]) !== playerMask.data[i]) {
                return false;
            }
        }
        return true;
    }
    for (let i = 0; i < playerMask.data.length; i++) {
        if (playerMask.data[i] & cellWords[i]) {
            return true;
        }
    }
    return false;
}

function _getVisibleBounds(lv, objArr) {
    let mini = 0;
    let minj = 0;
    let maxi = lv.width;
    let maxj = lv.height;
    const metadata = state.metadata || {};
    const flickscreen = metadata.flickscreen;
    const zoomscreen = metadata.zoomscreen;

    if (flickscreen === undefined && zoomscreen === undefined) {
        return { mini, minj, maxi, maxj };
    }

    const rawBounds = Array.isArray(lv.oldflickscreendat) ? lv.oldflickscreendat : [];
    const prevBounds = rawBounds.length === 4 ? rawBounds.map(v => v | 0) : null;

    let playerPosition = -1;
    const cellWords = new Array(STRIDE_OBJ);
    for (let x = 0; x < lv.width && playerPosition < 0; x++) {
        for (let y = 0; y < lv.height; y++) {
            const posIndex = y + x * lv.height;
            const baseIdx = posIndex * STRIDE_OBJ;
            for (let word = 0; word < STRIDE_OBJ; word++) {
                cellWords[word] = objArr[baseIdx + word];
            }
            if (_playerMatchesCell(cellWords)) {
                playerPosition = posIndex;
                break;
            }
        }
    }

    if (playerPosition >= 0) {
        const px = (playerPosition / lv.height) | 0;
        const py = playerPosition % lv.height;
        if (flickscreen !== undefined) {
            const screenWidth = flickscreen[0] | 0;
            const screenHeight = flickscreen[1] | 0;
            mini = (((px / screenWidth) | 0) * screenWidth);
            minj = (((py / screenHeight) | 0) * screenHeight);
            maxi = Math.min(mini + screenWidth, lv.width);
            maxj = Math.min(minj + screenHeight, lv.height);
        } else {
            const screenWidth = zoomscreen[0] | 0;
            const screenHeight = zoomscreen[1] | 0;
            mini = Math.max(Math.min(px - ((screenWidth / 2) | 0), lv.width - screenWidth), 0);
            minj = Math.max(Math.min(py - ((screenHeight / 2) | 0), lv.height - screenHeight), 0);
            maxi = Math.min(mini + screenWidth, lv.width);
            maxj = Math.min(minj + screenHeight, lv.height);
        }
    } else if (prevBounds !== null) {
        [mini, minj, maxi, maxj] = prevBounds;
    }

    return { mini, minj, maxi, maxj };
}

globalThis.__PS_NODE_API__ = {
    compile,
    backupLevel,
    restoreLevel,
    processInput,
    addUndoState,
    DoUndo,
    DoRestart,
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
    getLevelInfo: () => state.levels.map((lv, i) => {
        if (lv.objects) return { type: 'level', index: i };
        return { type: 'message', index: i, message: lv.message || '' };
    }),
    unloadGame,
    clearBackups: () => { backups = []; },
    drainLazyGeneration: () => { tick_lazy_function_generation(false); },
    serializeCompiledState,
    serializeCompiledStateJSON: () => JSON.stringify(serializeCompiledState()),
    serializeParsedState,
    serializeParsedStateJSON: () => JSON.stringify(serializeParsedState()),
    serializeLevel,
    serializeSpriteDataJSON: () => {
        if (!state) return '{}';
        const sprites = [];
        for (const name of state.idDict) {
            const obj = state.objects[name];
            const colors = obj.colors.map(c => String(c));
            const matrix = [];
            for (let r = 0; r < obj.spritematrix.length; r++) {
                matrix.push(Array.from(obj.spritematrix[r]));
            }
            sprites.push({ name, colors, spritematrix: matrix });
        }
        const bgcolor = (state.bgcolor || '#000000').toString();
        return JSON.stringify({ sprites, bgcolor });
    },

    // ---------------------------------------------------------------
    // JS-native frame rendering (mirrors graphics.js redraw logic)
    // Returns { width, height, cellWidth, cellHeight, data: [...] }
    // where data is a flat array of RGB uint8 values (h*ch, w*cw, 3).
    // If levelObj is provided, render that; otherwise render current level.
    // ---------------------------------------------------------------
    renderFrame: (levelObj) => {
        if (!state) return null;
        const lv = levelObj || level;
        if (!lv) return null;
        // backupLevel() stores objects in 'dat'; Level objects use 'objects'
        const objArr = lv.objects || lv.dat;
        if (!objArr) return null;

        const w = lv.width;
        const h = lv.height;
        const objectCount = state.objectCount;
        const { mini, minj, maxi, maxj } = _getVisibleBounds(lv, objArr);
        const viewW = Math.max(maxi - mini, 0);
        const viewH = Math.max(maxj - minj, 0);

        // Parse hex color to [r,g,b]
        function hexToRGB(hex) {
            if (hex == null) return null;
            hex = String(hex).trim();
            if (hex.toLowerCase() === 'transparent') return null;
            hex = hex.replace('#', '');
            if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
            if (!/^[0-9a-fA-F]{6}$/.test(hex)) return null;
            return [
                parseInt(hex.substring(0, 2), 16),
                parseInt(hex.substring(2, 4), 16),
                parseInt(hex.substring(4, 6), 16),
            ];
        }

        // Pre-render sprites into pixel arrays (matching createSprite logic)
        // Each sprite: { pixels: Uint8Array(ch*cw*3), mask: Uint8Array(ch*cw) }
        const spriteData = state.idDict.map(name => {
            const obj = state.objects[name];
            const grid = obj.spritematrix;
            const colors = obj.colors;
            const gh = grid.length;
            const gw = gh > 0 ? grid[0].length : 0;
            // Cell size = sprite grid size (typically 5x5)
            // In graphics.js: cw = ~~(cellwidth / w), ch = ~~(cellheight / h)
            // But without a canvas, cellwidth/cellheight are the sprite dimensions.
            // The original uses cellwidth = canvas.width / screenwidth, and
            // each sprite is drawn at grid[j][k] -> fillRect(k*cw, j*ch, cw, pixh).
            // Without a canvas, the natural cell size = sprite grid dimensions.
            const pixels = new Uint8Array(gh * gw * 3);
            const mask = new Uint8Array(gh * gw);
            for (let j = 0; j < gh; j++) {
                for (let k = 0; k < gw; k++) {
                    const val = grid[j][k];
                    if (val >= 0) {
                        const rgb = hexToRGB(colors[val]);
                        if (rgb === null) continue;
                        const idx = (j * gw + k);
                        pixels[idx * 3    ] = rgb[0];
                        pixels[idx * 3 + 1] = rgb[1];
                        pixels[idx * 3 + 2] = rgb[2];
                        mask[idx] = 1;
                    }
                }
            }
            return { pixels, mask, cw: gw, ch: gh };
        });

        // Determine cell size from first sprite
        const cellW = spriteData.length > 0 ? spriteData[0].cw : 5;
        const cellH = spriteData.length > 0 ? spriteData[0].ch : 5;

        // Allocate output frame
        const frameW = viewW * cellW;
        const frameH = viewH * cellH;
        const data = new Uint8Array(frameH * frameW * 3);

        // Fill background
        const bg = hexToRGB(state.bgcolor || '#000000') || [0, 0, 0];
        for (let p = 0; p < frameH * frameW; p++) {
            data[p * 3    ] = bg[0];
            data[p * 3 + 1] = bg[1];
            data[p * 3 + 2] = bg[2];
        }

        // Render cells — matching redraw() iteration order exactly:
        //   for i (x/col) in [0, width), for j (y/row) in [0, height)
        //     posIndex = j + i * height  (column-major)
        //     for each object k, if bit set, composite sprite
        for (let i = mini; i < maxi; i++) {
            for (let j = minj; j < maxj; j++) {
                const posIndex = j + i * h;
                const baseIdx = posIndex * STRIDE_OBJ;
                // Check if any bits are set (skip empty cells)
                let anySet = false;
                for (let s = 0; s < STRIDE_OBJ; s++) {
                    if (objArr[baseIdx + s] !== 0) { anySet = true; break; }
                }
                if (!anySet) continue;

                const px = (i - mini) * cellW;
                const py = (j - minj) * cellH;
                for (let k = 0; k < objectCount; k++) {
                    const word = k >> 5;    // k / 32
                    const bit = k & 31;     // k % 32
                    if ((objArr[baseIdx + word] & (1 << bit)) === 0) continue;
                    if (k >= spriteData.length) continue;
                    const sp = spriteData[k];
                    // Composite sprite pixels onto frame
                    for (let sj = 0; sj < sp.ch; sj++) {
                        for (let sk = 0; sk < sp.cw; sk++) {
                            const si = sj * sp.cw + sk;
                            if (!sp.mask[si]) continue;
                            const fx = px + sk;
                            const fy = py + sj;
                            const fi = (fy * frameW + fx) * 3;
                            data[fi    ] = sp.pixels[si * 3    ];
                            data[fi + 1] = sp.pixels[si * 3 + 1];
                            data[fi + 2] = sp.pixels[si * 3 + 2];
                        }
                    }
                }
            }
        }

        return {
            width: frameW,
            height: frameH,
            cellWidth: cellW,
            cellHeight: cellH,
            gridWidth: viewW,
            gridHeight: viewH,
            visibleBounds: [mini, minj, maxi, maxj],
            dataBase64: Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString('base64'),
        };
    },

    // Render from a raw objects array (e.g. from a saved state / backup)
    // objects: flat Int32Array or Array of column-major bitfield words
    // gridW, gridH: level dimensions
    renderFrameFromObjects: (objects, gridW, gridH) => {
        if (!state) return null;
        // Create a minimal level-like object
        const fakeLv = { width: gridW, height: gridH, objects: objects };
        return globalThis.__PS_NODE_API__.renderFrame(fakeLv);
    },
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

// Creates an isolated VM sandbox with its own compiled-game state.
// Use this instead of the shared loadApi() singleton when multiple independent
// engine instances are needed (e.g. NodeJS + CPP backends running concurrently).
function createFreshApi() {
    const freshContext = createSandbox();
    vm.runInContext(buildSourceBundle(), freshContext, {
        filename: 'standalone_puzzlescript_sources.js',
    });
    return freshContext.__PS_NODE_API__;
}

const _sharedApi = loadApi();
_sharedApi.createFreshApi = createFreshApi;
module.exports = _sharedApi;
