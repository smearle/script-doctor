"use strict";
var solution = [];
function storage_has(e) {
    return null !== localStorage.getItem(e);
}
function storage_get(e) {
    return localStorage.getItem(e);
}
function storage_get_int(e, n) {
    const t = parseInt(localStorage.getItem(e), 10);
    return isNaN(t) ? n : t;
}
function storage_set(e, n) {
    return localStorage.setItem(e, n);
}
function storage_remove(e) {
    localStorage.removeItem(e);
}
function BitVec(e) {
    this.data = new Int32Array(e);
}
function IBITSET(e, n) {
    return `${e}.data[${n}>>5] |= 1 << (${n} & 31);`;
}
function GET(e, n) {
    return `((${e}.data[${n >> 5}] & ${1 << (31 & n)}) !== 0)`;
}
function GETSHIFTOR(e, n, t) {
    const o = 31 & t,
        r = t >> 5;
    return o ? `${n}&((${e}.data[${r}] >>> ${o}) | (${e}.data[${r}+1] << (32-${o})))` : `${n}&(${e}.data[${r}] >>> ${o})`;
}
function ISHIFTOR(e, n, t) {
    return `{\n\t\tlet toshift = ${t}&31;\n\t\tlet low = ${n} << toshift;\n\t\t${e}.data[${t}>>5] |= low;\n\t\tif (toshift) {\n\t\t\tlet high = ${n} >> (32 - toshift);\n\t\t\t${e}.data[(${t}>>5)+1] |= high;\n\t\t}\n\t}`;
}
function ISHIFTCLEAR(e, n, t) {
    const o = 31 & t,
        r = t >> 5;
    let l = `${e}.data[${r}] &= ~(${n + "<<" + o});\n`;
    if (o) {
        l += `${e}.data[${r + 1}] &= ~(${n + ">>>" + (32 - o)});\n`;
    }
    return l;
}
function EQUALS(e, n, t) {
    let o = "(true";
    for (let r = 0; r < t; r++) o += `&&(${e}.data[${r}] === ${n}.data[${r}])`;
    return o + ")";
}
function EQUALS_TOK_REAL(e, n) {
    let t = "(true";
    for (let o = 0; o < n.data.length; o++) t += `&&(${e}.data[o] === ${n.data[o]})`;
    return t + ")";
}
function NOT_EQUALS(e, n, t) {
    let o = "(false";
    for (let r = 0; r < t; r++) o += `||(${e}.data[${r}] !== ${n}.data[${r}])`;
    return o + ")";
}
function ARRAY_SET_ZERO(e) {
    return e + ".fill(0);\n";
}
function SET_ZERO(e) {
    return e + ".data.fill(0);\n";
}
function IS_ZERO(e, n) {
    let t = "(true";
    for (let o = 0; o < n; o++) t += `&&(${e}.data[${o}]===0)`;
    return t + ")";
}
function IS_NONZERO(e, n) {
    let t = "(false";
    for (let o = 0; o < n; o++) t += `||(${e}.data[${o}]!==0)`;
    return t + ")";
}
function BITS_SET_IN_ARRAY(e, n, t) {
    let o = "(true";
    for (let r = 0; r < t; r++) o += `&&((${e}.data[${r}] & ${n}[${r}]) === ${e}.data[${r}])`;
    return o + ")";
}
function NOT_BITS_SET_IN_ARRAY(e, n, t) {
    let o = "(false";
    for (let r = 0; r < t; r++) o += `||((${e}.data[${r}] & ${n}[${r}]) !== ${e}.data[${r}])`;
    return o + ")";
}
function BITS_CLEAR_IN_ARRAY(e, n, t) {
    if (0 === t) return "true";
    let o = "(true";
    for (let r = 0; r < t; r++) o += `&&((${e}.data[${r}] & ${n}.data[${r}]) === 0)`;
    return o + ")";
}
function ANY_BITS_IN_COMMON(e, n, t) {
    if (0 === t) return "false";
    let o = "(false";
    for (let r = 0; r < t; r++) o += `||((${e}.data[${r}] & ${n}.data[${r}]) !== 0)`;
    return o + ")";
}
function ANY_BITS_IN_COMMON_TOK_REAL(e, n) {
    if (0 === n.length) return "false";
    let t = "(false";
    for (let o = 0; o < n.length; o++) t += `||((${e}.data[${o}] & ${n[o]}) !== 0)`;
    return t + ")";
}
function UNROLL(e, n) {
    const t = e.split(" ");
    let o = "";
    for (let e = 0; e < n; e++) o += `${t[0]}.data[${e}] ${t[1]} ${t[2]}.data[${e}];\n`;
    return o;
}
function UNROLL_TOK_REAL(e, n, t, o) {
    let r = "";
    for (let l = 0; l < o; l++) r += e + ".data[" + l + "]" + n + t.data[l] + ";\n";
    return r;
}
function LEVEL_GET_CELL_INTO(e, n, t, o) {
    let r = "";
    for (let e = 0; e < o; e++) r += t + `.data[${e}]=level.objects[${n}*${o}+${e}];\n`;
    return r;
}
function LEVEL_GET_MOVEMENTS_INTO(e, n, t) {
    let o = "";
    for (let r = 0; r < t; r++) o += n + `.data[${r}]=level.movements[${e}*${t}+${r}];\n`;
    return o;
}
function LEVEL_SET_CELL(e, n, t, o) {
    let r = "";
    for (let l = 0; l < o; l++) r += `\t${e}.objects[${n}*${o}+${l}]=${t}.data[${l}];\n`;
    return r;
}
function IMPORT_COMPILE_TIME_ARRAY(e, n, t) {
    let o = "";
    for (let r = 0; r < t; r++) o += `${e}.data[${r}]=${n.data[r]};\n`;
    return o;
}
function Level(e, n, t, o, r) {
    (this.lineNumber = e), (this.width = n), (this.height = t), (this.n_tiles = n * t), (this.objects = r), (this.layerCount = o), (this.commandQueue = []), (this.commandQueueSourceRules = []);
}
let _movementVecs;
(BitVec.prototype.cloneInto = function (e) {
    for (let n = 0; n < this.data.length; ++n) e.data[n] = this.data[n];
    return e;
}),
    (BitVec.prototype.clone = function () {
        return new BitVec(this.data);
    }),
    (BitVec.prototype.iand = function (e) {
        for (let n = 0; n < this.data.length; ++n) this.data[n] &= e.data[n];
    }),
    (BitVec.prototype.inot = function () {
        for (let e = 0; e < this.data.length; ++e) this.data[e] = ~this.data[e];
    }),
    (BitVec.prototype.ior = function (e) {
        for (let n = 0; n < this.data.length; ++n) this.data[n] |= e.data[n];
    }),
    (BitVec.prototype.iclear = function (e) {
        for (let n = 0; n < this.data.length; ++n) this.data[n] &= ~e.data[n];
    }),
    (BitVec.prototype.ibitset = function (e) {
        this.data[e >> 5] |= 1 << (31 & e);
    }),
    (BitVec.prototype.ibitclear = function (e) {
        this.data[e >> 5] &= ~(1 << (31 & e));
    }),
    (BitVec.prototype.get = function (e) {
        return 0 != (this.data[e >> 5] & (1 << (31 & e)));
    }),
    (BitVec.prototype.getshiftor = function (e, n) {
        const t = 31 & n;
        let o = this.data[n >> 5] >>> t;
        return t && (o |= this.data[1 + (n >> 5)] << (32 - t)), o & e;
    }),
    (BitVec.prototype.ishiftor = function (e, n) {
        const t = 31 & n,
            o = n >> 5;
        let r = e << t;
        if (((this.data[o] |= r), t)) {
            let n = e >> (32 - t);
            this.data[o + 1] |= n;
        }
    }),
    (BitVec.prototype.ishiftclear = function (e, n) {
        const t = 31 & n,
            o = n >> 5,
            r = e << t;
        if (((this.data[o] &= ~r), t)) {
            let t = e >> (32 - (31 & n));
            this.data[o + 1] &= ~t;
        }
    }),
    (BitVec.prototype.equals = function (e) {
        if (this.data.length !== e.data.length) return !1;
        for (let n = 0; n < this.data.length; ++n) if (this.data[n] !== e.data[n]) return !1;
        return !0;
    }),
    (BitVec.prototype.setZero = function () {
        this.data.fill(0);
    }),
    (BitVec.prototype.iszero = function () {
        for (let e = 0; e < this.data.length; ++e) if (0 !== this.data[e]) return !1;
        return !0;
    }),
    (BitVec.prototype.bitsSetInArray = function (e) {
        for (let n = 0; n < this.data.length; ++n) if ((this.data[n] & e[n]) !== this.data[n]) return !1;
        return !0;
    }),
    (BitVec.prototype.bitsClearInArray = function (e) {
        for (let n = 0; n < this.data.length; ++n) if (this.data[n] & e[n]) return !1;
        return !0;
    }),
    (BitVec.prototype.anyBitsInCommon = function (e) {
        for (let n = 0; n < this.data.length; ++n) if (this.data[n] & e.data[n]) return !0;
        return !1;
    }),
    (BitVec.prototype.prettyPrint = function () {
        var e = "";
        for (let n = 0; n < this.data.length; n++) {
            for (let t = 0; t < 32; t++) e += this.data[n] & (1 << t) ? "1" : "0";
            e += " ";
        }
        return e;
    }),
    (Level.prototype.delta_index = function (e) {
        const [n, t] = dirMasksDelta[e];
        return n * this.height + t;
    }),
    (Level.prototype.clone = function () {
        let e = new Level(this.lineNumber, this.width, this.height, this.layerCount, null);
        return (e.objects = new Int32Array(this.objects)), e;
    }),
    (Level.prototype.getCell = function (e) {
        return new BitVec(this.objects.subarray(e * STRIDE_OBJ, e * STRIDE_OBJ + STRIDE_OBJ));
    }),
    (Level.prototype.getCellInto = function (e, n) {
        for (let t = 0; t < STRIDE_OBJ; t++) n.data[t] = this.objects[e * STRIDE_OBJ + t];
        return n;
    }),
    (Level.prototype.setCell = function (e, n) {
        for (let t = 0; t < n.data.length; ++t) this.objects[e * STRIDE_OBJ + t] = n.data[t];
    });
let _movementVecIndex = 0;
function LEVEL_SET_MOVEMENTS(e, n, t) {
    var o = "{";
    for (let r = 0; r < t; r++) o += `\tlevel.movements[${e}*${t}+${r}]=${n}.data[${r}];\n`;
    return (o += `\n\tconst colIndex=(${e}/level.height)|0;\n\tconst rowIndex=(${e}%level.height);\n\n\t${UNROLL(`level.colCellContents_Movements[colIndex] |= ${n}`, t)}\n\t${UNROLL(
        `level.rowCellContents_Movements[rowIndex] |= ${n}`,
        t
    )}\n\t${UNROLL(`level.mapCellContents_Movements |= ${n}`, t)}\n}`);
}
(Level.prototype.getMovements = function (e) {
    let n = _movementVecs[_movementVecIndex];
    _movementVecIndex = (_movementVecIndex + 1) % _movementVecs.length;
    for (let t = 0; t < STRIDE_MOV; t++) n.data[t] = this.movements[e * STRIDE_MOV + t];
    return n;
}),
    (Level.prototype.getRigids = function (e) {
        return this.rigidMovementAppliedMask[e].clone();
    }),
    (Level.prototype.getMovementsInto = function (e, n) {
        let t = n;
        for (let n = 0; n < STRIDE_MOV; n++) t.data[n] = this.movements[e * STRIDE_MOV + n];
        return t;
    }),
    (Level.prototype.setMovements = function (e, n) {
        for (let t = 0; t < n.data.length; ++t) this.movements[e * STRIDE_MOV + t] = n.data[t];
        let t = (e / this.height) | 0,
            o = e % this.height;
        level.colCellContents_Movements[t].ior(n), level.rowCellContents_Movements[o].ior(n), level.mapCellContents_Movements.ior(n);
    }),
    (Level.prototype.calcBackgroundMask = function (e) {
        void 0 === e.backgroundlayer && logError("you have to have a background layer");
        let n = e.layerMasks[e.backgroundlayer];
        for (let e = 0; e < this.n_tiles; e++) {
            let t = this.getCell(e);
            if ((t.iand(n), !t.iszero())) return t;
        }
        const t = new BitVec(STRIDE_OBJ);
        return t.ibitset(e.backgroundid), t;
    });
const relativedirs = ["^", "v", "<", ">", "moving", "stationary", "parallel", "perpendicular", "no"],
    sectionNames = ["objects", "legend", "sounds", "collisionlayers", "rules", "winconditions", "levels"],
    commandwords = ["sfx0", "sfx1", "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10", "cancel", "checkpoint", "restart", "win", "message", "again"],
    commandwords_sfx = ["sfx0", "sfx1", "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10"],
    reg_name = /[\p{L}\p{N}_]+[\p{Z}\s]*/u,
    reg_soundseed = /\d+\b/u,
    reg_sectionNames = /(objects|collisionlayers|legend|sounds|rules|winconditions|levels)(?![\p{L}\p{N}_])[\p{Z}\s]*/u,
    reg_equalsrow = /[\=]+/,
    reg_notcommentstart = /[^\(]+/,
    reg_match_until_commentstart_or_whitespace = /[^\p{Z}\s\()]+[\p{Z}\s]*/u,
    reg_csv_separators = /[ \,]*/,
    reg_soundverbs = /(move|action|create|destroy|cantmove)\b[\p{Z}\s]*/u,
    soundverbs_directional = ["move", "cantmove"],
    reg_soundevents = /(undo|restart|titlescreen|startgame|cancel|endgame|startlevel|endlevel|showmessage|closemessage|sfx0|sfx1|sfx2|sfx3|sfx4|sfx5|sfx6|sfx7|sfx8|sfx9|sfx10)\b[\p{Z}\s]*/u,
    reg_directions = /^(action|up|down|left|right|\^|v|\<|\>|moving|stationary|parallel|perpendicular|horizontal|orthogonal|vertical|no|randomdir|random)$/,
    reg_loopmarker = /^(startloop|endloop)$/,
    reg_ruledirectionindicators = /^(up|down|left|right|horizontal|vertical|orthogonal|late|rigid)$/,
    reg_sounddirectionindicators = /[\p{Z}\s]*(up|down|left|right|horizontal|vertical|orthogonal)(?![\p{L}\p{N}_])[\p{Z}\s]*/u,
    reg_winconditionquantifiers = /^(all|any|no|some)$/,
    keyword_array = [
        "checkpoint",
        "objects",
        "collisionlayers",
        "legend",
        "sounds",
        "rules",
        "...",
        "winconditions",
        "levels",
        "|",
        "[",
        "]",
        "up",
        "down",
        "left",
        "right",
        "late",
        "rigid",
        "^",
        "v",
        ">",
        "<",
        "no",
        "randomdir",
        "random",
        "horizontal",
        "vertical",
        "any",
        "all",
        "no",
        "some",
        "moving",
        "stationary",
        "parallel",
        "perpendicular",
        "action",
        "message",
        "move",
        "action",
        "create",
        "destroy",
        "cantmove",
        "sfx0",
        "sfx1",
        "sfx2",
        "sfx3",
        "Sfx4",
        "sfx5",
        "sfx6",
        "sfx7",
        "sfx8",
        "sfx9",
        "sfx10",
        "cancel",
        "checkpoint",
        "restart",
        "win",
        "message",
        "again",
        "undo",
        "restart",
        "titlescreen",
        "startgame",
        "cancel",
        "endgame",
        "startlevel",
        "endlevel",
        "showmessage",
        "closemessage",
    ],
    directionaggregates = {
        horizontal: ["left", "right"],
        horizontal_par: ["left", "right"],
        horizontal_perp: ["left", "right"],
        vertical: ["up", "down"],
        vertical_par: ["up", "down"],
        vertical_perp: ["up", "down"],
        moving: ["up", "down", "left", "right", "action"],
        orthogonal: ["up", "down", "left", "right"],
        perpendicular: ["^", "v"],
        parallel: ["<", ">"],
    },
    relativeDirections = ["^", "v", "<", ">", "perpendicular", "parallel"],
    simpleAbsoluteDirections = ["up", "down", "left", "right"],
    simpleRelativeDirections = ["^", "v", "<", ">"],
    reg_directions_only = /^(\>|\<|\^|v|up|down|left|right|moving|stationary|no|randomdir|random|horizontal|vertical|orthogonal|perpendicular|parallel|action)$/,
    REGEX_HEX = /^#([0-9A-F]{3}){1,2}$/i,
    reg_color = /(black|white|gray|darkgray|lightgray|grey|darkgrey|lightgrey|red|darkred|lightred|brown|darkbrown|lightbrown|orange|yellow|green|darkgreen|lightgreen|blue|lightblue|darkblue|purple|pink|transparent|#(?:[0-9a-f]{3}){1,2})\s*/;
let unitTesting = !1,
    lazyFunctionGeneration = !0,
    curlevel = 0,
    curlevelTarget = null,
    hasUsedCheckpoint = !1,
    levelEditorOpened = !1,
    muted = 0,
    runrulesonlevelstart_phase = !1,
    ignoreNotJustPressedAction = !0,
    textMode = !0;
function doSetupTitleScreenLevelContinue() {
    try {
        if (storage_has(document.URL)) {
            if (storage_has(document.URL + "_checkpoint")) {
                let e = storage_get(document.URL + "_checkpoint");
                curlevelTarget = JSON.parse(e);
                let n = [];
                for (let e in Object.keys(curlevelTarget.dat)) n[e] = curlevelTarget.dat[e];
                curlevelTarget.dat = new Int32Array(n);
            }
            curlevel = storage_get_int(document.URL, 0);
        }
    } catch (e) {}
}
doSetupTitleScreenLevelContinue();
let verbose_logging = !1,
    throttle_movement = !1,
    suppress_all_console_output = !1,
    cache_console_messages = !1,
    quittingTitleScreen = !1,
    quittingMessageScreen = !1,
    deltatime = 17,
    timer = 0,
    repeatinterval = 150,
    autotick = 0,
    autotickinterval = 0,
    winning = !1,
    againing = !1,
    againinterval = 150,
    norepeat_action = !1,
    oldflickscreendat = [],
    keybuffer = [],
    restarting = !1,
    messageselected = !1,
    textImages = {},
    level = new Level();
function get_title_animation_frame() {
    return Math.floor((timer / 1e3 / 0.3) * 10);
}
var WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR = [];
function tick_lazy_function_generation(e = !1) {
    if (0 === WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.length) return;
    let n = performance.now();
    for (; (performance.now() - n < 20 || !e) && WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.length > 0; ) {
        const e = WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.shift();
        if (e instanceof CellPattern) e.matches = e.generateMatchFunction();
        else {
            if (!(e instanceof Rule)) throw new Error("Unknown object type: " + e);
            e.generate_all_MatchFunctions();
        }
        0;
    }
}
function lazy_function_generation_clear_backlog() {
    WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR = [];
}
let canSetHTMLColors = !0,
    canDump = !1,
    canOpenEditor = !1,
    IDE = !1;
const diffToVisualize = null;
function stripTags(e) {
    let n = document.createElement("div");
    return (n.innerHTML = e), n.textContent || n.innerText || "";
}
function consolePrint(e, n) {}
function consolePrintFromRule(e, n, t) {}
function consoleCacheDump(e) {}
function UnitTestingThrow(e) {}
function consoleError(e, n) {
    let t = document.getElementById("errormessage");
    (e = stripTags(e)), (t.innerHTML += e + "<br>");
}
function logErrorNoLine(e) {
    let n = document.getElementById("errormessage");
    (e = stripTags(e)), (n.innerHTML += e + "<br>");
}
function clearInputHistory() {}
function pushInput(e) {}
function pushSoundToHistory(e) {}
let font = {
    0: "\n00000\n00000\n00000\n01110\n10001\n10011\n10101\n11001\n10001\n01110\n00000\n00000",
    1: "\n00000\n00000\n00000\n11100\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    2: "\n00000\n00000\n00000\n11110\n00001\n00001\n01110\n10000\n10000\n11111\n00000\n00000",
    3: "\n00000\n00000\n00000\n11110\n00001\n00110\n00001\n00001\n00001\n11110\n00000\n00000",
    4: "\n00000\n00000\n00000\n10000\n10000\n10000\n10010\n11111\n00010\n00010\n00000\n00000",
    5: "\n00000\n00000\n00000\n11111\n10000\n11110\n00001\n00001\n00001\n11110\n00000\n00000",
    6: "\n00000\n00000\n00000\n01110\n10000\n11110\n10001\n10001\n10001\n01110\n00000\n00000",
    7: "\n00000\n00000\n00000\n11111\n00001\n00010\n00100\n00100\n00100\n00100\n00000\n00000",
    8: "\n00000\n00000\n00000\n01110\n10001\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    9: "\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n01111\n00001\n01110\n00000\n00000",
    a: "\n00000\n00000\n00000\n00000\n00000\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    b: "\n00000\n00000\n00000\n10000\n10000\n11110\n10001\n10001\n10001\n01110\n00000\n00000",
    c: "\n00000\n00000\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n01111\n00000\n00000",
    d: "\n00000\n00000\n00000\n00001\n00001\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    e: "\n00000\n00000\n00000\n00000\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    f: "\n00000\n00000\n00000\n00011\n00100\n11111\n00100\n00100\n00100\n00100\n00000\n00000",
    g: "\n00000\n00000\n00000\n00000\n00000\n01111\n10001\n10001\n10001\n01111\n00001\n01110",
    h: "\n00000\n00000\n00000\n10000\n10000\n11110\n10001\n10001\n10001\n10001\n00000\n00000",
    i: "\n00000\n00000\n00000\n00100\n00000\n01100\n00100\n00100\n00100\n01110\n00000\n00000",
    j: "\n00000\n00000\n00000\n00100\n00000\n01100\n00100\n00100\n00100\n00100\n10100\n01000",
    k: "\n00000\n00000\n00000\n10000\n10000\n10001\n10010\n11100\n10010\n10001\n00000\n00000",
    l: "\n00000\n00000\n00000\n01100\n00100\n00100\n00100\n00100\n00100\n01110\n00000\n00000",
    m: "\n00000\n00000\n00000\n00000\n00000\n01010\n10101\n10101\n10101\n10101\n00000\n00000",
    n: "\n00000\n00000\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n10001\n00000\n00000",
    o: "\n00000\n00000\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    p: "\n00000\n00000\n00000\n00000\n00000\n11110\n10001\n10001\n10001\n11110\n10000\n10000",
    q: "\n00000\n00000\n00000\n00000\n00000\n01111\n10001\n10001\n10001\n01111\n00001\n00001",
    r: "\n00000\n00000\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n10000\n00000\n00000",
    s: "\n00000\n00000\n00000\n00000\n00000\n01111\n10000\n01110\n00001\n11110\n00000\n00000",
    t: "\n00000\n00000\n00000\n00100\n00100\n11111\n00100\n00100\n00100\n00011\n00000\n00000",
    u: "\n00000\n00000\n00000\n00000\n00000\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    v: "\n00000\n00000\n00000\n00000\n00000\n10001\n10010\n10100\n11000\n10000\n00000\n00000",
    w: "\n00000\n00000\n00000\n00000\n00000\n10101\n10101\n10101\n10101\n01010\n00000\n00000",
    x: "\n00000\n00000\n00000\n00000\n00000\n10001\n01010\n00100\n01010\n10001\n00000\n00000",
    "×": "\n00000\n00000\n00000\n00000\n00000\n10001\n01010\n00100\n01010\n10001\n00000\n00000",
    y: "\n00000\n00000\n00000\n00000\n00000\n10001\n10001\n10001\n10001\n01111\n00001\n11110",
    z: "\n00000\n00000\n00000\n00000\n00000\n11111\n00010\n00100\n01000\n11111\n00000\n00000",
    A: "\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    B: "\n00000\n00000\n00000\n11110\n10001\n11110\n10001\n10001\n10001\n11110\n00000\n00000",
    C: "\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n10000\n10000\n01111\n00000\n00000",
    D: "\n00000\n00000\n00000\n11110\n10001\n10001\n10001\n10001\n10001\n11110\n00000\n00000",
    E: "\n00000\n00000\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11111\n00000\n00000",
    F: "\n00000\n00000\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n10000\n00000\n00000",
    G: "\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n10011\n10001\n01111\n00000\n00000",
    H: "\n00000\n00000\n00000\n10001\n10001\n11111\n10001\n10001\n10001\n10001\n00000\n00000",
    I: "\n00000\n00000\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    J: "\n00000\n00000\n00000\n01111\n00001\n00001\n00001\n00001\n00001\n01110\n00000\n00000",
    K: "\n00000\n00000\n00000\n10001\n10010\n10100\n11000\n10100\n10010\n10001\n00000\n00000",
    L: "\n00000\n00000\n00000\n10000\n10000\n10000\n10000\n10000\n10000\n11111\n00000\n00000",
    M: "\n00000\n00000\n00000\n11111\n10101\n10101\n10101\n10101\n10101\n10101\n00000\n00000",
    N: "\n00000\n00000\n00000\n10001\n11001\n10101\n10011\n10001\n10001\n10001\n00000\n00000",
    O: "\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    P: "\n00000\n00000\n00000\n11110\n10001\n10001\n10001\n11110\n10000\n10000\n00000\n00000",
    Q: "\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n10001\n10101\n01110\n00100\n00000",
    R: "\n00000\n00000\n00000\n11110\n10001\n10001\n11110\n10001\n10001\n10001\n00000\n00000",
    S: "\n00000\n00000\n00000\n01111\n10000\n01110\n00001\n00001\n00001\n11110\n00000\n00000",
    T: "\n00000\n00000\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n00100\n00000\n00000",
    U: "\n00000\n00000\n00000\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    V: "\n00000\n00000\n00000\n10001\n10001\n10001\n10001\n10001\n01010\n00100\n00000\n00000",
    W: "\n00000\n00000\n00000\n10101\n10101\n10101\n10101\n10101\n10101\n01010\n00000\n00000",
    X: "\n00000\n00000\n00000\n10001\n10001\n01010\n00100\n01010\n10001\n10001\n00000\n00000",
    Y: "\n00000\n00000\n00000\n10001\n10001\n01010\n00100\n00100\n00100\n00100\n00000\n00000",
    Z: "\n00000\n00000\n00000\n11111\n00001\n00010\n00100\n01000\n10000\n11111\n00000\n00000",
    ".": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00100\n00000\n00000",
    "·": "\n00000\n00000\n00000\n00000\n00000\n00000\n00100\n00000\n00000\n00000\n00000\n00000",
    "•": "\n00000\n00000\n00000\n00000\n00000\n01110\n01110\n01110\n00000\n00000\n00000\n00000",
    "…": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n10101\n00000\n00000",
    "†": "\n00000\n00100\n00100\n01110\n00100\n00100\n00100\n00100\n00100\n00100\n00000\n00000",
    "‡": "\n00000\n00100\n00100\n01110\n00100\n00100\n00100\n00100\n01110\n00100\n00000\n00000",
    ƒ: "\n00000\n00000\n00000\n00011\n00100\n11111\n00100\n00100\n00100\n00100\n01000\n00000",
    "‚": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00100\n01100\n00000\n00000",
    "„": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n01001\n11011\n00000\n00000",
    ",": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00100\n01100\n00000\n00000",
    ";": "\n00000\n00000\n00000\n00000\n00000\n00100\n00000\n00000\n00100\n01100\n00000\n00000",
    ":": "\n00000\n00000\n00000\n00000\n00000\n00100\n00000\n00000\n00000\n00100\n00000\n00000",
    "?": "\n00000\n00000\n00000\n01110\n10001\n00001\n00001\n00110\n00000\n00100\n00000\n00000",
    "¿": "\n00000\n00000\n00000\n00100\n00000\n01100\n10000\n10000\n10001\n01110\n00000\n00000",
    "!": "\n00000\n00000\n00000\n00100\n00100\n00100\n00100\n00100\n00000\n00100\n00000\n00000",
    "¡": "\n00000\n00000\n00000\n00100\n00000\n00100\n00100\n00100\n00100\n00100\n00000\n00000",
    "@": "\n00000\n00000\n00000\n00000\n00000\n01110\n10001\n10111\n10000\n01110\n00000\n00000",
    "£": "\n00000\n00000\n00000\n00000\n00000\n01110\n01001\n11100\n01000\n11111\n00000\n00000",
    $: "\n00000\n00000\n00000\n00000\n00100\n01111\n10100\n01110\n00101\n11110\n00100\n00000",
    "%": "\n00000\n00000\n00000\n00000\n00000\n11001\n11010\n00100\n01011\n10011\n00000\n00000",
    "‰": "\n00000\n00000\n00000\n00000\n11001\n11010\n00100\n01011\n10011\n00000\n00011\n00011",
    "^": "\n00000\n00000\n00000\n00100\n01010\n00000\n00000\n00000\n00000\n00000\n00000\n00000",
    "&": "\n00000\n00000\n00000\n00000\n00000\n01100\n10000\n01011\n10010\n01100\n00000\n00000",
    "*": "\n00000\n00000\n00000\n00000\n00000\n01010\n00100\n01010\n00000\n00000\n00000\n00000",
    "(": "\n00000\n00000\n00000\n00010\n00100\n00100\n00100\n00100\n00100\n00010\n00000\n00000",
    ")": "\n00000\n00000\n00000\n01000\n00100\n00100\n00100\n00100\n00100\n01000\n00000\n00000",
    "+": "\n00000\n00000\n00000\n00000\n00000\n00100\n00100\n11111\n00100\n00100\n00000\n00000",
    "÷": "\n00000\n00000\n00000\n00000\n00000\n00100\n00000\n11111\n00000\n00100\n00000\n00000",
    "±": "\n00000\n00000\n00000\n00000\n00000\n00100\n00100\n11111\n00100\n11111\n00000\n00000",
    "-": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n01110\n00000\n00000\n00000\n00000",
    "–": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n11110\n00000\n00000\n00000\n00000",
    "—": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n11111\n00000\n00000\n00000\n00000",
    _: "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n11111\n00000\n00000",
    "=": "\n00000\n00000\n00000\n00000\n00000\n00000\n11111\n00000\n11111\n00000\n00000\n00000",
    " ": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000",
    "{": "\n00000\n00000\n00000\n00110\n00100\n00100\n01100\n00100\n00100\n00110\n00000\n00000",
    "}": "\n00000\n00000\n00000\n01100\n00100\n00100\n00110\n00100\n00100\n01100\n00000\n00000",
    "[": "\n00000\n00000\n00000\n00110\n00100\n00100\n00100\n00100\n00100\n00110\n00000\n00000",
    "]": "\n00000\n00000\n00000\n01100\n00100\n00100\n00100\n00100\n00100\n01100\n00000\n00000",
    "'": "\n00000\n00000\n00000\n00100\n00100\n00100\n00000\n00000\n00000\n00000\n00000\n00000",
    "‘": "\n00000\n00000\n00000\n00110\n00100\n00000\n00000\n00000\n00000\n00000\n00000\n00000",
    "’": "\n00000\n00000\n00000\n00100\n01100\n00000\n00000\n00000\n00000\n00000\n00000\n00000",
    "“": "\n00000\n00000\n00000\n11011\n10010\n00000\n00000\n00000\n00000\n00000\n00000\n00000",
    "”": "\n00000\n00000\n00000\n01001\n11011\n00000\n00000\n00000\n00000\n00000\n00000\n00000",
    '"': "\n00000\n00000\n00000\n01010\n01010\n01010\n00000\n00000\n00000\n00000\n00000\n00000",
    "/": "\n00000\n00000\n00000\n00000\n00000\n00001\n00010\n00100\n01000\n10000\n00000\n00000",
    "\\": "\n00000\n00000\n00000\n00000\n00000\n10000\n01000\n00100\n00010\n00001\n00000\n00000",
    "|": "\n00000\n00000\n00000\n00000\n00000\n00100\n00100\n00100\n00100\n00100\n00000\n00000",
    "¦": "\n00000\n00000\n00000\n00000\n00100\n00100\n00000\n00100\n00100\n00100\n00000\n00000",
    "<": "\n00000\n00000\n00000\n00000\n00000\n00010\n00100\n01000\n00100\n00010\n00000\n00000",
    "‹": "\n00000\n00000\n00000\n00000\n00000\n00000\n00100\n01000\n00100\n00000\n00000\n00000",
    "«": "\n00000\n00000\n00000\n00000\n00000\n00000\n01001\n10010\n01001\n00000\n00000\n00000",
    ">": "\n00000\n00000\n00000\n00000\n00000\n01000\n00100\n00010\n00100\n01000\n00000\n00000",
    "›": "\n00000\n00000\n00000\n00000\n00000\n00000\n00100\n00010\n00100\n00000\n00000\n00000",
    "»": "\n00000\n00000\n00000\n00000\n00000\n00000\n10010\n01001\n10010\n00000\n00000\n00000",
    "~": "\n00000\n00000\n00000\n00000\n00000\n00000\n01000\n10101\n00010\n00000\n00000\n00000",
    "˜": "\n00000\n00000\n00000\n00000\n00000\n01010\n10100\n00000\n00000\n00000\n00000\n00000",
    "`": "\n00000\n00000\n00000\n00000\n00000\n01000\n00100\n00000\n00000\n00000\n00000\n00000",
    "#": "\n00000\n00000\n00000\n00000\n00000\n01010\n11111\n01010\n11111\n01010\n00000\n00000",
    À: "\n01000\n00100\n00000\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    Á: "\n00010\n00100\n00000\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    Â: "\n00100\n01010\n00000\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    Ã: "\n01000\n10101\n00010\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    Ä: "\n00000\n01010\n00000\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    Å: "\n00100\n01010\n00100\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    Æ: "\n00000\n00000\n00000\n01111\n10100\n10100\n10100\n11111\n10100\n10111\n00000\n00000",
    Ç: "\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n10000\n10000\n01111\n00100\n01000",
    È: "\n01000\n00100\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11111\n00000\n00000",
    É: "\n00010\n00100\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11111\n00000\n00000",
    Ê: "\n00100\n01010\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11111\n00000\n00000",
    Ë: "\n00000\n01010\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11111\n00000\n00000",
    Ì: "\n01000\n00100\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    Í: "\n00010\n00100\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    Î: "\n00100\n01010\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    Ï: "\n00000\n01010\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    Ð: "\n00000\n00000\n00000\n01110\n01001\n01001\n11101\n01001\n01001\n01110\n00000\n00000",
    Ñ: "\n01001\n10110\n00000\n10001\n11001\n10101\n10011\n10001\n10001\n10001\n00000\n00000",
    Ò: "\n01000\n00100\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    Ó: "\n00010\n00100\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    Ô: "\n00100\n01010\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    Õ: "\n01001\n10110\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    Ö: "\n00000\n01010\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    Ø: "\n00000\n00010\n00100\n01110\n10101\n10101\n10101\n10101\n10101\n01110\n00100\n01000",
    Ù: "\n00000\n01000\n00100\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    Ú: "\n00000\n00010\n00100\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    Û: "\n00100\n01010\n00000\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    Ü: "\n00000\n01010\n00000\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    Ý: "\n00000\n00000\n00100\n10001\n10001\n01010\n00100\n00100\n00100\n00100\n00000\n00000",
    Þ: "\n00000\n00000\n10000\n11110\n10001\n10001\n10001\n10001\n10001\n11110\n10000\n00000",
    ß: "\n00000\n00000\n00000\n01110\n10001\n10110\n10001\n10001\n10001\n10110\n10000\n00000",
    ẞ: "\n00000\n00000\n00000\n01110\n10001\n10110\n10001\n10001\n10001\n10110\n00000\n00000",
    à: "\n00000\n00000\n01000\n00100\n00000\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    á: "\n00000\n00000\n00010\n00100\n00000\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    â: "\n00000\n00000\n00100\n01010\n00000\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    ã: "\n00000\n00000\n01001\n10110\n00000\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    ä: "\n00000\n00000\n00000\n01010\n00000\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    å: "\n00000\n00100\n01010\n00100\n00000\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    æ: "\n00000\n00000\n00000\n00000\n00000\n01110\n10101\n10110\n10100\n01111\n00000\n00000",
    ç: "\n00000\n00000\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n01111\n00100\n01000",
    è: "\n00000\n00000\n01000\n00100\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    é: "\n00000\n00000\n00010\n00100\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    ê: "\n00000\n00000\n00100\n01010\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    ë: "\n00000\n00000\n00000\n01010\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    ì: "\n00000\n00000\n01000\n00100\n00000\n01100\n00100\n00100\n00100\n01110\n00000\n00000",
    í: "\n00000\n00000\n00010\n00100\n00000\n01100\n00100\n00100\n00100\n01110\n00000\n00000",
    î: "\n00000\n00000\n00100\n01010\n00000\n01100\n00100\n00100\n00100\n01110\n00000\n00000",
    ï: "\n00000\n00000\n00000\n01010\n00000\n01100\n00100\n00100\n00100\n01110\n00000\n00000",
    ð: "\n00000\n00000\n00010\n00111\n00010\n01110\n10010\n10010\n10010\n01110\n00000\n00000",
    ñ: "\n00000\n00000\n01001\n10110\n00000\n01110\n10001\n10001\n10001\n10001\n00000\n00000",
    ò: "\n00000\n00000\n01000\n00100\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    ó: "\n00000\n00000\n00010\n00100\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    ô: "\n00000\n00000\n00100\n01010\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    õ: "\n00000\n00000\n01001\n10110\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    ö: "\n00000\n00000\n00000\n01010\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    ø: "\n00000\n00000\n00000\n00010\n00100\n01110\n10101\n10101\n10101\n01110\n00100\n01000",
    ù: "\n00000\n00000\n00000\n01000\n00100\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    ú: "\n00000\n00000\n00000\n00010\n00100\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    û: "\n00000\n00000\n00100\n01010\n00000\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    ü: "\n00000\n00000\n00000\n01010\n00000\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    ý: "\n00000\n00000\n00000\n00010\n00100\n10001\n10001\n10001\n10001\n01111\n00001\n11110",
    þ: "\n00000\n00000\n00000\n10000\n10000\n11110\n10001\n10001\n10001\n11110\n10000\n10000",
    ÿ: "\n00000\n00000\n00000\n01010\n00000\n10001\n10001\n10001\n10001\n01111\n00001\n11110",
    Ā: "\n00000\n01110\n00000\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    ā: "\n00000\n00000\n00000\n01110\n00000\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    Ă: "\n01010\n00100\n00000\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    ă: "\n00000\n00000\n01010\n00100\n00000\n01111\n10001\n10001\n10001\n01111\n00000\n00000",
    Ą: "\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00010\n00001",
    ą: "\n00000\n00000\n00000\n00000\n00000\n01111\n10001\n10001\n10001\n01111\n00010\n00001",
    Ć: "\n00010\n00100\n00000\n01111\n10000\n10000\n10000\n10000\n10000\n01111\n00000\n00000",
    ć: "\n00000\n00000\n00010\n00100\n00000\n01111\n10000\n10000\n10000\n01111\n00000\n00000",
    Ĉ: "\n00100\n01010\n00000\n01111\n10000\n10000\n10000\n10000\n10000\n01111\n00000\n00000",
    ĉ: "\n00000\n00000\n00100\n01010\n00000\n01111\n10000\n10000\n10000\n01111\n00000\n00000",
    Ċ: "\n00000\n00100\n00000\n01111\n10000\n10000\n10000\n10000\n10000\n01111\n00000\n00000",
    ċ: "\n00000\n00000\n00000\n00100\n00000\n01111\n10000\n10000\n10000\n01111\n00000\n00000",
    Č: "\n01010\n00100\n00000\n01111\n10000\n10000\n10000\n10000\n10000\n01111\n00000\n00000",
    č: "\n00000\n00000\n01010\n00100\n00000\n01111\n10000\n10000\n10000\n01111\n00000\n00000",
    Ď: "\n01010\n00100\n00000\n11110\n10001\n10001\n10001\n10001\n10001\n11110\n00000\n00000",
    ď: "\n00000\n00000\n00000\n00101\n00101\n01100\n10100\n10100\n10100\n01100\n00000\n00000",
    Đ: "\n00000\n00000\n00000\n01110\n01001\n01001\n11101\n01001\n01001\n01110\n00000\n00000",
    đ: "\n00000\n00000\n00010\n00111\n00010\n01110\n10010\n10010\n10010\n01110\n00000\n00000",
    Ē: "\n00000\n01110\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11111\n00000\n00000",
    ē: "\n00000\n00000\n00000\n01110\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    Ĕ: "\n01010\n00100\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11111\n00000\n00000",
    ĕ: "\n00000\n00000\n01010\n00100\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    Ė: "\n00000\n00100\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11111\n00000\n00000",
    ė: "\n00000\n00000\n00000\n00100\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    Ę: "\n00000\n00000\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11111\n00010\n00001",
    ę: "\n00000\n00000\n00000\n00000\n00000\n01110\n10001\n11111\n10000\n01110\n00010\n00001",
    Ě: "\n01010\n00100\n00000\n11111\n10000\n11111\n10000\n10000\n10000\n11110\n00000\n00000",
    ě: "\n00000\n00000\n01010\n00100\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    Ĝ: "\n00100\n01010\n00000\n01111\n10000\n10000\n10000\n10011\n10001\n01111\n00000\n00000",
    ĝ: "\n00000\n00000\n00100\n01010\n00000\n01111\n10001\n10001\n10001\n01111\n00001\n01110",
    Ğ: "\n01010\n00100\n00000\n01111\n10000\n10000\n10000\n10011\n10001\n01111\n00000\n00000",
    ğ: "\n00000\n00000\n01010\n00100\n00000\n01111\n10001\n10001\n10001\n01111\n00001\n01110",
    Ġ: "\n00000\n00100\n00000\n01111\n10000\n10000\n10000\n10011\n10001\n01111\n00000\n00000",
    ġ: "\n00000\n00000\n00000\n00100\n00000\n01111\n10001\n10001\n10001\n01111\n00001\n01110",
    Ģ: "\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n10011\n10001\n01111\n00000\n01100",
    ģ: "\n00010\n00100\n00000\n01111\n10000\n10000\n10000\n10011\n10001\n01111\n00000\n00000",
    Ĥ: "\n00100\n01010\n00000\n10001\n10001\n11111\n10001\n10001\n10001\n10001\n00000\n00000",
    ĥ: "\n00100\n01010\n00000\n10000\n10000\n11110\n10001\n10001\n10001\n10001\n00000\n00000",
    Ħ: "\n00000\n00000\n01010\n11111\n01010\n01110\n01010\n01010\n01010\n01010\n00000\n00000",
    ħ: "\n00000\n00000\n01000\n11100\n01000\n01110\n01001\n01001\n01001\n01001\n00000\n00000",
    Ĩ: "\n01001\n10110\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    ĩ: "\n01010\n10100\n00000\n00100\n00000\n01100\n00100\n00100\n00100\n01110\n00000\n00000",
    Ī: "\n00000\n01110\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    ī: "\n00000\n00000\n00000\n01110\n00000\n01100\n00100\n00100\n00100\n01110\n00000\n00000",
    Ĭ: "\n01010\n00100\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    ĭ: "\n00000\n00000\n01010\n00100\n00000\n01100\n00100\n00100\n00100\n01110\n00000\n00000",
    Į: "\n00000\n00000\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00010\n00001",
    į: "\n00000\n00000\n00000\n00100\n00000\n01100\n00100\n00100\n00100\n01110\n00010\n00001",
    İ: "\n00000\n00100\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n11111\n00000\n00000",
    ı: "\n00000\n00000\n00000\n00000\n00000\n01100\n00100\n00100\n00100\n01110\n00000\n00000",
    Ĳ: "\n00000\n00000\n00000\n10010\n10010\n10010\n10010\n10010\n10010\n10110\n00000\n00000",
    ĳ: "\n00000\n00000\n00000\n01001\n00000\n11001\n01001\n01001\n01001\n11101\n00001\n00010",
    Ĵ: "\n00010\n00101\n00000\n01111\n00001\n00001\n00001\n00001\n00001\n01110\n00000\n00000",
    ĵ: "\n00000\n00000\n00100\n01010\n00000\n01100\n00100\n00100\n00100\n00100\n10100\n01000",
    Ķ: "\n00000\n00000\n00000\n10001\n10010\n10100\n11000\n10100\n10010\n10001\n00100\n01000",
    ķ: "\n00000\n00000\n00000\n10000\n10000\n10001\n10010\n11100\n10010\n10001\n00100\n01000",
    ĸ: "\n00000\n00000\n00000\n00000\n00000\n10001\n10010\n11100\n10010\n10001\n00000\n00000",
    Ĺ: "\n00000\n00010\n00100\n10000\n10000\n10000\n10000\n10000\n10000\n11111\n00000\n00000",
    ĺ: "\n00010\n00100\n00000\n01100\n00100\n00100\n00100\n00100\n00100\n01110\n00000\n00000",
    Ļ: "\n00000\n00000\n00000\n10000\n10000\n10000\n10000\n10000\n10000\n11111\n00000\n00100",
    ļ: "\n00000\n00000\n00000\n01100\n00100\n00100\n00100\n00100\n00100\n01110\n00000\n00100",
    Ľ: "\n00000\n00000\n00000\n10010\n10010\n10000\n10000\n10000\n10000\n11111\n00000\n00000",
    ľ: "\n00000\n00000\n00000\n01101\n00101\n00100\n00100\n00100\n00100\n01110\n00000\n00000",
    Ŀ: "\n00000\n00000\n00000\n10000\n10000\n10100\n10000\n10000\n10000\n11111\n00000\n00000",
    ŀ: "\n00000\n00000\n00000\n01100\n00100\n00100\n00101\n00100\n00100\n01110\n00000\n00000",
    Ł: "\n00000\n00000\n00000\n01000\n01010\n01100\n11000\n01000\n01000\n01111\n00000\n00000",
    ł: "\n00000\n00000\n00000\n01100\n00100\n00100\n00110\n01100\n00100\n01110\n00000\n00000",
    Ń: "\n00000\n00010\n00100\n10001\n11001\n10101\n10011\n10001\n10001\n10001\n00000\n00000",
    ń: "\n00000\n00000\n00010\n00100\n00000\n01110\n10001\n10001\n10001\n10001\n00000\n00000",
    Ņ: "\n00000\n00000\n00000\n10001\n11001\n10101\n10011\n10001\n10001\n10001\n00100\n01000",
    ņ: "\n00000\n00000\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n10001\n00100\n01000",
    Ň: "\n00000\n01010\n00100\n10001\n11001\n10101\n10011\n10001\n10001\n10001\n00000\n00000",
    ň: "\n00000\n00000\n01010\n00100\n00000\n01110\n10001\n10001\n10001\n10001\n00000\n00000",
    ŉ: "\n00000\n00000\n00000\n10000\n10000\n00110\n01001\n01001\n01001\n01001\n00000\n00000",
    Ŋ: "\n00000\n00000\n00000\n10001\n11001\n10101\n10011\n10001\n10001\n10001\n00001\n00010",
    ŋ: "\n00000\n00000\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n10001\n00001\n00010",
    Ō: "\n00000\n01110\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    ō: "\n00000\n00000\n00000\n01110\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    Ŏ: "\n01010\n00100\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    ŏ: "\n00000\n00000\n01010\n00100\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    Ő: "\n01001\n10010\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    ő: "\n00000\n00000\n01001\n10010\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    Œ: "\n00000\n00000\n00000\n01111\n10100\n10100\n10111\n10100\n10100\n01111\n00000\n00000",
    œ: "\n00000\n00000\n00000\n00000\n00000\n01110\n10101\n10110\n10100\n01111\n00000\n00000",
    Ŕ: "\n00010\n00100\n00000\n11110\n10001\n10001\n11110\n10001\n10001\n10001\n00000\n00000",
    ŕ: "\n00000\n00000\n00010\n00100\n00000\n01111\n10000\n10000\n10000\n10000\n00000\n00000",
    Ŗ: "\n00000\n00000\n00000\n11110\n10001\n10001\n11110\n10001\n10001\n10001\n00100\n01000",
    ŗ: "\n00000\n00000\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n10000\n00100\n01000",
    Ř: "\n01010\n00100\n00000\n11110\n10001\n10001\n11110\n10001\n10001\n10001\n00000\n00000",
    ř: "\n00000\n00000\n01010\n00100\n00000\n01111\n10000\n10000\n10000\n10000\n00000\n00000",
    Ś: "\n00010\n00100\n00000\n01111\n10000\n01110\n00001\n00001\n00001\n11110\n00000\n00000",
    ś: "\n00000\n00000\n00010\n00100\n00000\n01111\n10000\n01110\n00001\n11110\n00000\n00000",
    Ŝ: "\n00100\n01010\n00000\n01111\n10000\n01110\n00001\n00001\n00001\n11110\n00000\n00000",
    ŝ: "\n00000\n00000\n00100\n01010\n00000\n01111\n10000\n01110\n00001\n11110\n00000\n00000",
    Ş: "\n00000\n00000\n00000\n01111\n10000\n01110\n00001\n00001\n00001\n11110\n00100\n00000",
    ş: "\n00000\n00000\n00000\n00000\n00000\n01111\n10000\n01110\n00001\n11110\n00100\n01000",
    Š: "\n01010\n00100\n00000\n01111\n10000\n01110\n00001\n00001\n00001\n11110\n00000\n00000",
    š: "\n00000\n00000\n01010\n00100\n00000\n01111\n10000\n01110\n00001\n11110\n00000\n00000",
    Ţ: "\n00000\n00000\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n00100\n00010\n00100",
    ţ: "\n00000\n00000\n00000\n00100\n00100\n11111\n00100\n00100\n00100\n00011\n00000\n01100",
    Ť: "\n01010\n00100\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n00100\n00000\n00000",
    ť: "\n00000\n00000\n00001\n00101\n00100\n11111\n00100\n00100\n00100\n00011\n00000\n00000",
    Ŧ: "\n00000\n00000\n00000\n11111\n00100\n00100\n01110\n00100\n00100\n00100\n00000\n00000",
    ŧ: "\n00000\n00000\n00000\n00100\n00100\n11111\n00100\n01110\n00100\n00011\n00000\n00000",
    Ũ: "\n01001\n10110\n00000\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    ũ: "\n00000\n00000\n01001\n10110\n00000\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    Ū: "\n00000\n01110\n00000\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    ū: "\n00000\n00000\n00000\n01110\n00000\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    Ŭ: "\n01010\n00100\n00000\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    ŭ: "\n00000\n00000\n01010\n00100\n00000\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    Ů: "\n00100\n01010\n00100\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    ů: "\n00000\n00000\n00100\n01010\n00100\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    Ű: "\n01001\n10010\n00000\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    ű: "\n00000\n00000\n01001\n10010\n00000\n10001\n10001\n10001\n10001\n01111\n00000\n00000",
    Ų: "\n00000\n00000\n00000\n10001\n10001\n10001\n10001\n10001\n10001\n01110\n00100\n00010",
    ų: "\n00000\n00000\n00000\n00000\n00000\n10001\n10001\n10001\n10001\n01111\n00010\n00001",
    Ŵ: "\n00100\n01010\n00000\n10101\n10101\n10101\n10101\n10101\n10101\n01010\n00000\n00000",
    ŵ: "\n00000\n00000\n00100\n01010\n00000\n10101\n10101\n10101\n10101\n01010\n00000\n00000",
    Ŷ: "\n00100\n01010\n00000\n10001\n10001\n01010\n00100\n00100\n00100\n00100\n00000\n00000",
    ŷ: "\n00000\n00000\n00100\n01010\n00000\n10001\n10001\n10001\n10001\n01111\n00001\n11110",
    Ÿ: "\n00000\n01010\n00000\n10001\n10001\n01010\n00100\n00100\n00100\n00100\n00000\n00000",
    Ź: "\n00010\n00100\n00000\n11111\n00001\n00010\n00100\n01000\n10000\n11111\n00000\n00000",
    ź: "\n00000\n00000\n00010\n00100\n00000\n11111\n00010\n00100\n01000\n11111\n00000\n00000",
    Ż: "\n00000\n00100\n00000\n11111\n00001\n00010\n00100\n01000\n10000\n11111\n00000\n00000",
    ż: "\n00000\n00000\n00000\n00100\n00000\n11111\n00010\n00100\n01000\n11111\n00000\n00000",
    Ž: "\n01010\n00100\n00000\n11111\n00001\n00010\n00100\n01000\n10000\n11111\n00000\n00000",
    ž: "\n00000\n00000\n01010\n00100\n00000\n11111\n00010\n00100\n01000\n11111\n00000\n00000",
    "€": "\n00000\n00000\n00000\n00111\n01000\n11110\n01000\n11110\n01000\n00111\n00000\n00000",
    "™": "\n00000\n11111\n00100\n00100\n00100\n00000\n01010\n10101\n10101\n10101\n00000\n00000",
    "¢": "\n00000\n00000\n00000\n00010\n00100\n01111\n10100\n10100\n10100\n01111\n00100\n01000",
    "¤": "\n00000\n00000\n00000\n00000\n10001\n01110\n10001\n10001\n01110\n10001\n00000\n00000",
    "¥": "\n00000\n00000\n10001\n01010\n00100\n01110\n00100\n01110\n00100\n00000\n00000",
    "§": "\n00000\n00000\n00000\n01110\n10000\n01110\n10001\n01110\n00001\n01110\n00000\n00000",
    "¨": "\n00000\n00000\n00000\n01010\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000",
    "©": "\n00000\n00000\n00000\n01110\n10001\n10111\n10101\n10111\n10001\n01110\n00000\n00000",
    "®": "\n00000\n00000\n00000\n01110\n10001\n10111\n10101\n10101\n10001\n01110\n00000\n00000",
    ª: "\n00000\n01110\n00010\n01110\n01010\n01110\n00000\n00000\n00000\n00000\n00000\n00000",
    º: "\n00000\n00100\n01010\n01010\n01010\n00100\n00000\n00000\n00000\n00000\n00000\n00000",
    "¬": "\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n01110\n00010\n00000\n00000\n00000",
    "¯": "\n00000\n00000\n00000\n01110\n00000\n00000\n00000\n00000\n00000\n00000\n00000\n00000",
    "°": "\n00000\n00000\n00100\n01010\n00100\n00000\n00000\n00000\n00000\n00000\n00000\n00000",
    А: "\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    Б: "\n00000\n00000\n00000\n11111\n10000\n11110\n10001\n10001\n10001\n11110\n00000\n00000",
    В: "\n00000\n00000\n00000\n11110\n10001\n11110\n10001\n10001\n10001\n11110\n00000\n00000",
    Г: "\n00000\n00000\n00000\n11111\n10000\n10000\n10000\n10000\n10000\n10000\n00000\n00000",
    Д: "\n00000\n00000\n00000\n00011\n00101\n01001\n01001\n01001\n01001\n11111\n10001\n00000",
    Е: "\n00000\n00000\n00000\n11111\n10000\n11110\n10000\n10000\n10000\n11111\n00000\n00000",
    Ё: "\n00000\n01010\n00000\n11111\n10000\n11110\n10000\n10000\n10000\n11111\n00000\n00000",
    Ж: "\n00000\n00000\n00000\n10101\n10101\n01110\n10101\n10101\n10101\n10101\n00000\n00000",
    З: "\n00000\n00000\n00000\n11110\n00001\n01110\n00001\n00001\n00001\n11110\n00000\n00000",
    И: "\n00000\n00000\n00000\n10001\n10011\n10101\n11001\n10001\n10001\n10001\n00000\n00000",
    Й: "\n00010\n00100\n00000\n10001\n10011\n10101\n11001\n10001\n10001\n10001\n00000\n00000",
    К: "\n00000\n00000\n00000\n10001\n10010\n10100\n11000\n10100\n10010\n10001\n00000\n00000",
    Л: "\n00000\n00000\n00000\n00001\n00001\n00011\n00101\n01001\n10001\n10001\n00000\n00000",
    М: "\n00000\n00000\n00000\n10001\n11011\n10101\n10101\n10001\n10001\n10001\n00000\n00000",
    Н: "\n00000\n00000\n00000\n10001\n10001\n11111\n10001\n10001\n10001\n10001\n00000\n00000",
    О: "\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n10001\n10001\n01110\n00000\n00000",
    П: "\n00000\n00000\n00000\n11111\n10001\n10001\n10001\n10001\n10001\n10001\n00000\n00000",
    Р: "\n00000\n00000\n00000\n11110\n10001\n10001\n11110\n10000\n10000\n10000\n00000\n00000",
    С: "\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n10000\n10000\n01111\n00000\n00000",
    Т: "\n00000\n00000\n00000\n11111\n00100\n00100\n00100\n00100\n00100\n00100\n00000\n00000",
    У: "\n00000\n00000\n00000\n10001\n10001\n10001\n01111\n00001\n00001\n11110\n00000\n00000",
    Ф: "\n00000\n00000\n00000\n01110\n10101\n10101\n10101\n01110\n00100\n00100\n00000\n00000",
    Х: "\n00000\n00000\n00000\n10001\n10001\n01010\n00100\n01010\n10001\n10001\n00000\n00000",
    Ц: "\n00000\n00000\n00000\n10010\n10010\n10010\n10010\n10010\n10010\n11111\n00001\n00000",
    Ч: "\n00000\n00000\n00000\n10001\n10001\n10001\n01111\n00001\n00001\n00001\n00000\n00000",
    Ш: "\n00000\n00000\n00000\n10001\n10001\n10001\n10001\n10101\n10101\n11111\n00000\n00000",
    Щ: "\n00000\n00000\n00000\n10001\n10001\n10001\n10001\n10101\n10101\n11111\n00001\n00010",
    Ъ: "\n00000\n00000\n00000\n11000\n01000\n01110\n01001\n01001\n01001\n01110\n00000\n00000",
    Ы: "\n00000\n00000\n00000\n10001\n10001\n11001\n10101\n10101\n10101\n11001\n00000\n00000",
    Ь: "\n00000\n00000\n00000\n10000\n10000\n11110\n10001\n10001\n10001\n11110\n00000\n00000",
    Э: "\n00000\n00000\n00000\n11110\n00001\n00001\n11111\n00001\n00001\n11110\n00000\n00000",
    Ю: "\n00000\n00000\n00000\n10010\n10101\n10101\n11101\n10101\n10101\n10010\n00000\n00000",
    Я: "\n00000\n00000\n00000\n01111\n10001\n10001\n01111\n00101\n01001\n10001\n00000\n00000",
    а: "\n00000\n00000\n00000\n00000\n00000\n01110\n00001\n01111\n10011\n01101\n00000\n00000",
    б: "\n00000\n00000\n00000\n00001\n01110\n10000\n11110\n10001\n10001\n01110\n00000\n00000",
    в: "\n00000\n00000\n00000\n00000\n00000\n11110\n10001\n11110\n10001\n11110\n00000\n00000",
    г: "\n00000\n00000\n00000\n00000\n00000\n11111\n10000\n10000\n10000\n10000\n00000\n00000",
    д: "\n00000\n00000\n00000\n00000\n00000\n00111\n00101\n00101\n01001\n11111\n10001\n00000",
    е: "\n00000\n00000\n00000\n00000\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    ё: "\n00000\n00000\n00000\n01010\n00000\n01110\n10001\n11111\n10000\n01110\n00000\n00000",
    ж: "\n00000\n00000\n00000\n00000\n00000\n10101\n10101\n01110\n10101\n10101\n00000\n00000",
    з: "\n00000\n00000\n00000\n00000\n00000\n11110\n00001\n01110\n00001\n11110\n00000\n00000",
    и: "\n00000\n00000\n00000\n00000\n00000\n10001\n10011\n10101\n11001\n10001\n00000\n00000",
    й: "\n00000\n00000\n00010\n00100\n00000\n10001\n10011\n10101\n11001\n10001\n00000\n00000",
    к: "\n00000\n00000\n00000\n00000\n00000\n10001\n10010\n11100\n10010\n10001\n00000\n00000",
    л: "\n00000\n00000\n00000\n00000\n00000\n00001\n00011\n00101\n01001\n10001\n00000\n00000",
    м: "\n00000\n00000\n00000\n00000\n00000\n10001\n11011\n10101\n10001\n10001\n00000\n00000",
    н: "\n00000\n00000\n00000\n00000\n00000\n10001\n10001\n11111\n10001\n10001\n00000\n00000",
    о: "\n00000\n00000\n00000\n00000\n00000\n01110\n10001\n10001\n10001\n01110\n00000\n00000",
    п: "\n00000\n00000\n00000\n00000\n00000\n11111\n10001\n10001\n10001\n10001\n00000\n00000",
    р: "\n00000\n00000\n00000\n00000\n00000\n11110\n10001\n10001\n10001\n11110\n10000\n10000",
    с: "\n00000\n00000\n00000\n00000\n00000\n01111\n10000\n10000\n10000\n01111\n00000\n00000",
    т: "\n00000\n00000\n00000\n00000\n00000\n11111\n00100\n00100\n00100\n00100\n00000\n00000",
    у: "\n00000\n00000\n00000\n00000\n00000\n10001\n10001\n10001\n01111\n00001\n00001\n11110",
    ф: "\n00000\n00000\n00000\n00000\n00000\n01110\n10101\n10101\n10101\n01110\n00100\n00100",
    х: "\n00000\n00000\n00000\n00000\n00000\n10001\n01010\n00100\n01010\n10001\n00000\n00000",
    ц: "\n00000\n00000\n00000\n00000\n00000\n10010\n10010\n10010\n10010\n11111\n00001\n00000",
    ч: "\n00000\n00000\n00000\n00000\n00000\n10001\n10001\n01111\n00001\n00001\n00000\n00000",
    ш: "\n00000\n00000\n00000\n00000\n00000\n10001\n10001\n10101\n10101\n11111\n00000\n00000",
    щ: "\n00000\n00000\n00000\n00000\n00000\n10001\n10001\n10101\n10101\n11111\n00001\n00000",
    ъ: "\n00000\n00000\n00000\n00000\n00000\n11000\n01000\n01110\n01001\n01110\n00000\n00000",
    ы: "\n00000\n00000\n00000\n00000\n00000\n10001\n10001\n11001\n10101\n11001\n00000\n00000",
    ь: "\n00000\n00000\n00000\n00000\n00000\n10000\n10000\n11110\n10001\n11110\n00000\n00000",
    э: "\n00000\n00000\n00000\n00000\n00000\n11110\n00001\n11111\n00001\n11110\n00000\n00000",
    ю: "\n00000\n00000\n00000\n00000\n00000\n10010\n10101\n11101\n10101\n10010\n00000\n00000",
    я: "\n00000\n00000\n00000\n00000\n00000\n01111\n10001\n01111\n01001\n10001\n00000\n00000",
    "☺": "\n00000\n00000\n00000\n00000\n00000\n01110\n10101\n11111\n10001\n01110\n00000\n00000",
};
function RC4(e) {
    (this.s = new Array(256)), (this.i = 0), (this.j = 0);
    for (let e = 0; e < 256; e++) this.s[e] = e;
    e && this.mix(e);
}
function print_call_stack() {
    let e = new Error().stack;
    console.log(e);
}
function RNG(e) {
    (this.seed = e),
        null == e
            ? (e = (Math.random() + Date.now()).toString())
            : "function" == typeof e
            ? ((this.uniform = e),
              (this.nextByte = function () {
                  return ~~(256 * this.uniform());
              }),
              (e = null))
            : "[object String]" !== Object.prototype.toString.call(e) && (e = JSON.stringify(e)),
        (this._normal = null),
        (this._state = e ? new RC4(e) : null);
}
/**
 * Seedable random number generator functions.
 * @version 1.0.0
 * @license Public Domain
 *
 * @example
 * let rng = new RNG('Example');
 * rng.random(40, 50);  // =>  42
 * rng.uniform();       // =>  0.7972798995050903
 * rng.normal();        // => -0.6698504543216376
 * rng.exponential();   // =>  1.0547367609131555
 * rng.poisson(4);      // =>  2
 * rng.gamma(4);        // =>  2.781724687386858
 */
(String.prototype.getBytes = function () {
    let e = [];
    for (let n = 0; n < this.length; n++) {
        let t = this.charCodeAt(n),
            o = [];
        do {
            o.push(255 & t), (t >>= 8);
        } while (t > 0);
        e = e.concat(o.reverse());
    }
    return e;
}),
    (RC4.prototype._swap = function (e, n) {
        let t = this.s[e];
        (this.s[e] = this.s[n]), (this.s[n] = t);
    }),
    (RC4.prototype.mix = function (e) {
        let n = e.getBytes(),
            t = 0;
        for (let e = 0; e < this.s.length; e++) (t += this.s[e] + n[e % n.length]), (t %= 256), this._swap(e, t);
    }),
    (RC4.prototype.next = function () {
        return (this.i = (this.i + 1) % 256), (this.j = (this.j + this.s[this.i]) % 256), this._swap(this.i, this.j), this.s[(this.s[this.i] + this.s[this.j]) % 256];
    }),
    (RNG.prototype.nextByte = function () {
        return this._state.next();
    }),
    (RNG.prototype.uniform = function () {
        let e = 0;
        for (let n = 0; n < 7; n++) (e *= 256), (e += this.nextByte());
        return e / (Math.pow(2, 56) - 1);
    }),
    (RNG.prototype.random = function (e, n) {
        return null == e ? this.uniform() : (null == n && ((n = e), (e = 0)), e + Math.floor(this.uniform() * (n - e)));
    }),
    (RNG.prototype.normal = function () {
        if (null !== this._normal) {
            let e = this._normal;
            return (this._normal = null), e;
        }
        {
            let e = this.uniform() || Math.pow(2, -53),
                n = this.uniform();
            return (this._normal = Math.sqrt(-2 * Math.log(e)) * Math.sin(2 * Math.PI * n)), Math.sqrt(-2 * Math.log(e)) * Math.cos(2 * Math.PI * n);
        }
    }),
    (RNG.prototype.exponential = function () {
        return -Math.log(this.uniform() || Math.pow(2, -53));
    }),
    (RNG.prototype.poisson = function (e) {
        let n = Math.exp(-(e || 1)),
            t = 0,
            o = 1;
        do {
            t++, (o *= this.uniform());
        } while (o > n);
        return t - 1;
    }),
    (RNG.prototype.gamma = function (e) {
        let n = (e < 1 ? 1 + e : e) - 1 / 3,
            t = 1 / Math.sqrt(9 * n);
        do {
            do {
                let e = this.normal();
                Math.pow(t * e + 1, 3);
            } while (v <= 0);
            this.uniform(), Math.pow(x, 2);
        } while (u >= 1 - 0.0331 * x2 * x2 && Math.log(u) >= 0.5 * x2 + n * (1 - v + Math.log(v)));
        return e < 1 ? n * v * Math.exp(this.exponential() / -e) : n * v;
    }),
    (RNG.roller = function (e, n) {
        let t = e.split(/(\d+)?d(\d+)([+-]\d+)?/).slice(1),
            o = parseFloat(t[0]) || 1,
            r = parseFloat(t[1]),
            l = parseFloat(t[2]) || 0;
        return (
            (n = n || new RNG()),
            function () {
                let e = o + l;
                for (let t = 0; t < o; t++) e += n.random(r);
                return e;
            }
        );
    });
var FastBase64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=",
    FastBase64_encLookup = [];
function FastBase64_Init() {
    for (var e = 0; e < 4096; e++) FastBase64_encLookup[e] = FastBase64_chars[e >> 6] + FastBase64_chars[63 & e];
}
function FastBase64_Encode(e) {
    for (var t = e.length, o = "", r = 0; t > 2; ) (n = (e[r] << 16) | (e[r + 1] << 8) | e[r + 2]), (o += FastBase64_encLookup[n >> 12] + FastBase64_encLookup[4095 & n]), (t -= 3), (r += 3);
    if (t > 0) {
        var l = (252 & e[r]) >> 2,
            i = (3 & e[r]) << 4;
        if ((t > 1 && (i |= (240 & e[++r]) >> 4), (o += FastBase64_chars[l]), (o += FastBase64_chars[i]), 2 == t)) {
            var s = (15 & e[r++]) << 2;
            (s |= (192 & e[r]) >> 6), (o += FastBase64_chars[s]);
        }
        1 == t && (o += "="), (o += "=");
    }
    return o;
}
function u32ToArray(e) {
    return [255 & e, (e >> 8) & 255, (e >> 16) & 255, (e >> 24) & 255];
}
function u16ToArray(e) {
    return [255 & e, (e >> 8) & 255];
}
function MakeRiff(e, n, t) {
    var o,
        r = {
            chunkId: [82, 73, 70, 70],
            chunkSize: 0,
            format: [87, 65, 86, 69],
            subChunk1Id: [102, 109, 116, 32],
            subChunk1Size: 16,
            audioFormat: 1,
            numChannels: 1,
            sampleRate: e,
            byteRate: 0,
            blockAlign: 0,
            bitsPerSample: n,
            subChunk2Id: [100, 97, 116, 97],
            subChunk2Size: 0,
        };
    return (
        (r.byteRate = (r.sampleRate * r.numChannels * r.bitsPerSample) >> 3),
        (r.blockAlign = (r.numChannels * r.bitsPerSample) >> 3),
        (r.subChunk2Size = t.length),
        (r.chunkSize = 36 + r.subChunk2Size),
        {
            dat: [],
            wav: (o = r.chunkId.concat(
                u32ToArray(r.chunkSize),
                r.format,
                r.subChunk1Id,
                u32ToArray(r.subChunk1Size),
                u16ToArray(r.audioFormat),
                u16ToArray(r.numChannels),
                u32ToArray(r.sampleRate),
                u32ToArray(r.byteRate),
                u16ToArray(r.blockAlign),
                u16ToArray(r.bitsPerSample),
                r.subChunk2Id,
                u32ToArray(r.subChunk2Size),
                t
            )),
            header: r,
            dataURI: "data:audio/wav;base64," + FastBase64_Encode(o),
        }
    );
}
FastBase64_Init(), "undefined" != typeof exports && (exports.RIFFWAVE = RIFFWAVE);
const SOUND_VOL = 0.25,
    SAMPLE_RATE = 5512,
    BIT_DEPTH = 8,
    SQUARE = 0,
    SAWTOOTH = 1,
    SINE = 2,
    NOISE = 3,
    TRIANGLE = 4,
    BREAKER = 5,
    SHAPES = ["square", "sawtooth", "sine", "noise", "triangle", "breaker"];
let AUDIO_CONTEXT;
function checkAudioContextExists() {
    try {
        null == AUDIO_CONTEXT && ("undefined" != typeof AudioContext ? (AUDIO_CONTEXT = new AudioContext()) : "undefined" != typeof webkitAudioContext && (AUDIO_CONTEXT = new webkitAudioContext()));
    } catch (e) {
        window.console.log(e);
    }
}
checkAudioContextExists();
let rng,
    masterVolume = 1;
function Params() {
    const e = {};
    return (
        (e.wave_type = SQUARE),
        (e.p_env_attack = 0),
        (e.p_env_sustain = 0.3),
        (e.p_env_punch = 0),
        (e.p_env_decay = 0.4),
        (e.p_base_freq = 0.3),
        (e.p_freq_limit = 0),
        (e.p_freq_ramp = 0),
        (e.p_freq_dramp = 0),
        (e.p_vib_strength = 0),
        (e.p_vib_speed = 0),
        (e.p_arp_mod = 0),
        (e.p_arp_speed = 0),
        (e.p_duty = 0),
        (e.p_duty_ramp = 0),
        (e.p_repeat_speed = 0),
        (e.p_pha_offset = 0),
        (e.p_pha_ramp = 0),
        (e.p_lpf_freq = 1),
        (e.p_lpf_ramp = 0),
        (e.p_lpf_resonance = 0),
        (e.p_hpf_freq = 0),
        (e.p_hpf_ramp = 0),
        (e.sound_vol = 0.5),
        (e.sample_rate = 44100),
        (e.bit_depth = 8),
        e
    );
}
let seeded = !1;
function frnd(e) {
    return seeded ? rng.uniform() * e : Math.random() * e;
}
function rnd(e) {
    return seeded ? Math.floor(rng.uniform() * (e + 1)) : Math.floor(Math.random() * (e + 1));
}
function pickupCoin() {
    const e = Params();
    if (
        ((e.wave_type = Math.floor(frnd(SHAPES.length))),
        3 === e.wave_type && (e.wave_type = 0),
        (e.p_base_freq = 0.4 + frnd(0.5)),
        (e.p_env_attack = 0),
        (e.p_env_sustain = frnd(0.1)),
        (e.p_env_decay = 0.1 + frnd(0.4)),
        (e.p_env_punch = 0.3 + frnd(0.3)),
        rnd(1))
    ) {
        e.p_arp_speed = 0.5 + frnd(0.2);
        let n = 1 + (1 | frnd(7)),
            t = n + (1 | frnd(7)) + 2;
        e.p_arp_mod = +n / +t;
    }
    return e;
}
function laserShoot() {
    const e = Params();
    return (
        (e.wave_type = rnd(2)),
        e.wave_type === SINE && rnd(1) && (e.wave_type = rnd(1)),
        (e.wave_type = Math.floor(frnd(SHAPES.length))),
        3 === e.wave_type && (e.wave_type = SQUARE),
        (e.p_base_freq = 0.5 + frnd(0.5)),
        (e.p_freq_limit = e.p_base_freq - 0.2 - frnd(0.6)),
        e.p_freq_limit < 0.2 && (e.p_freq_limit = 0.2),
        (e.p_freq_ramp = -0.15 - frnd(0.2)),
        0 === rnd(2) && ((e.p_base_freq = 0.3 + frnd(0.6)), (e.p_freq_limit = frnd(0.1)), (e.p_freq_ramp = -0.35 - frnd(0.3))),
        rnd(1) ? ((e.p_duty = frnd(0.5)), (e.p_duty_ramp = frnd(0.2))) : ((e.p_duty = 0.4 + frnd(0.5)), (e.p_duty_ramp = -frnd(0.7))),
        (e.p_env_attack = 0),
        (e.p_env_sustain = 0.1 + frnd(0.2)),
        (e.p_env_decay = frnd(0.4)),
        rnd(1) && (e.p_env_punch = frnd(0.3)),
        0 === rnd(2) && ((e.p_pha_offset = frnd(0.2)), (e.p_pha_ramp = -frnd(0.2))),
        rnd(1) && (e.p_hpf_freq = frnd(0.3)),
        e
    );
}
function explosion() {
    const e = Params();
    return (
        rnd(1) ? ((e.p_base_freq = 0.1 + frnd(0.4)), (e.p_freq_ramp = -0.1 + frnd(0.4))) : ((e.p_base_freq = 0.2 + frnd(0.7)), (e.p_freq_ramp = -0.2 - frnd(0.2))),
        (e.p_base_freq *= e.p_base_freq),
        0 === rnd(4) && (e.p_freq_ramp = 0),
        0 === rnd(2) && (e.p_repeat_speed = 0.3 + frnd(0.5)),
        (e.p_env_attack = 0),
        (e.p_env_sustain = 0.1 + frnd(0.3)),
        (e.p_env_decay = frnd(0.5)),
        0 === rnd(1) && ((e.p_pha_offset = -0.3 + frnd(0.9)), (e.p_pha_ramp = -frnd(0.3))),
        (e.p_env_punch = 0.2 + frnd(0.6)),
        rnd(1) && ((e.p_vib_strength = frnd(0.7)), (e.p_vib_speed = frnd(0.6))),
        0 === rnd(2) && ((e.p_arp_speed = 0.6 + frnd(0.3)), (e.p_arp_mod = 0.8 - frnd(1.6))),
        e
    );
}
function birdSound() {
    const e = Params();
    return frnd(10) < 1
        ? ((e.wave_type = Math.floor(frnd(SHAPES.length))),
          3 === e.wave_type && (e.wave_type = SQUARE),
          (e.p_env_attack = 0.4304400932967592 + frnd(0.2) - 0.1),
          (e.p_env_sustain = 0.15739346034252394 + frnd(0.2) - 0.1),
          (e.p_env_punch = 0.004488201744871758 + frnd(0.2) - 0.1),
          (e.p_env_decay = 0.07478075528212291 + frnd(0.2) - 0.1),
          (e.p_base_freq = 0.9865265720147687 + frnd(0.2) - 0.1),
          (e.p_freq_limit = 0 + frnd(0.2) - 0.1),
          (e.p_freq_ramp = -0.2995018224359539 + frnd(0.2) - 0.1),
          frnd(1) < 0.5 && (e.p_freq_ramp = 0.1 + frnd(0.15)),
          (e.p_freq_dramp = 0.004598608156964473 + frnd(0.1) - 0.05),
          (e.p_vib_strength = -0.2202799497929496 + frnd(0.2) - 0.1),
          (e.p_vib_speed = 0.8084998703158364 + frnd(0.2) - 0.1),
          (e.p_arp_mod = 0),
          (e.p_arp_speed = 0),
          (e.p_duty = -0.9031808754347107 + frnd(0.2) - 0.1),
          (e.p_duty_ramp = -0.8128699999808343 + frnd(0.2) - 0.1),
          (e.p_repeat_speed = 0.601486018931999 + frnd(0.2) - 0.1),
          (e.p_pha_offset = -0.9424902314367765 + frnd(0.2) - 0.1),
          (e.p_pha_ramp = -0.1055482222272056 + frnd(0.2) - 0.1),
          (e.p_lpf_freq = 0.9989765717851521 + frnd(0.2) - 0.1),
          (e.p_lpf_ramp = -0.25051720626043017 + frnd(0.2) - 0.1),
          (e.p_lpf_resonance = 0.32777871505494693 + frnd(0.2) - 0.1),
          (e.p_hpf_freq = 0.0023548750981756753 + frnd(0.2) - 0.1),
          (e.p_hpf_ramp = -0.002375673204842568 + frnd(0.2) - 0.1),
          e)
        : frnd(10) < 1
        ? ((e.wave_type = Math.floor(frnd(SHAPES.length))),
          3 === e.wave_type && (e.wave_type = SQUARE),
          (e.p_env_attack = 0.5277795946672003 + frnd(0.2) - 0.1),
          (e.p_env_sustain = 0.18243733568468432 + frnd(0.2) - 0.1),
          (e.p_env_punch = -0.020159754546840117 + frnd(0.2) - 0.1),
          (e.p_env_decay = 0.1561353422051903 + frnd(0.2) - 0.1),
          (e.p_base_freq = 0.9028855606533718 + frnd(0.2) - 0.1),
          (e.p_freq_limit = -0.008842787837148716),
          (e.p_freq_ramp = -0.1),
          (e.p_freq_dramp = -0.012891241489551925),
          (e.p_vib_strength = -0.17923136138403065 + frnd(0.2) - 0.1),
          (e.p_vib_speed = 0.908263385610142 + frnd(0.2) - 0.1),
          (e.p_arp_mod = 0.41690153355414894 + frnd(0.2) - 0.1),
          (e.p_arp_speed = 0.0010766233195860704 + frnd(0.2) - 0.1),
          (e.p_duty = -0.8735363011184684 + frnd(0.2) - 0.1),
          (e.p_duty_ramp = -0.7397985366747507 + frnd(0.2) - 0.1),
          (e.p_repeat_speed = 0.0591789344172107 + frnd(0.2) - 0.1),
          (e.p_pha_offset = -0.9961184222777699 + frnd(0.2) - 0.1),
          (e.p_pha_ramp = -0.08234769395850523 + frnd(0.2) - 0.1),
          (e.p_lpf_freq = 0.9412475115697335 + frnd(0.2) - 0.1),
          (e.p_lpf_ramp = -0.18261358925834958 + frnd(0.2) - 0.1),
          (e.p_lpf_resonance = 0.24541438107389477 + frnd(0.2) - 0.1),
          (e.p_hpf_freq = -0.01831940280978611 + frnd(0.2) - 0.1),
          (e.p_hpf_ramp = -0.03857383633171346 + frnd(0.2) - 0.1),
          e)
        : frnd(10) < 1
        ? ((e.wave_type = Math.floor(frnd(SHAPES.length))),
          3 === e.wave_type && (e.wave_type = SQUARE),
          (e.p_env_attack = 0.4304400932967592 + frnd(0.2) - 0.1),
          (e.p_env_sustain = 0.15739346034252394 + frnd(0.2) - 0.1),
          (e.p_env_punch = 0.004488201744871758 + frnd(0.2) - 0.1),
          (e.p_env_decay = 0.07478075528212291 + frnd(0.2) - 0.1),
          (e.p_base_freq = 0.9865265720147687 + frnd(0.2) - 0.1),
          (e.p_freq_limit = 0 + frnd(0.2) - 0.1),
          (e.p_freq_ramp = -0.2995018224359539 + frnd(0.2) - 0.1),
          (e.p_freq_dramp = 0.004598608156964473 + frnd(0.2) - 0.1),
          (e.p_vib_strength = -0.2202799497929496 + frnd(0.2) - 0.1),
          (e.p_vib_speed = 0.8084998703158364 + frnd(0.2) - 0.1),
          (e.p_arp_mod = -0.46410459213693644 + frnd(0.2) - 0.1),
          (e.p_arp_speed = -0.10955361249587248 + frnd(0.2) - 0.1),
          (e.p_duty = -0.9031808754347107 + frnd(0.2) - 0.1),
          (e.p_duty_ramp = -0.8128699999808343 + frnd(0.2) - 0.1),
          (e.p_repeat_speed = 0.7014860189319991 + frnd(0.2) - 0.1),
          (e.p_pha_offset = -0.9424902314367765 + frnd(0.2) - 0.1),
          (e.p_pha_ramp = -0.1055482222272056 + frnd(0.2) - 0.1),
          (e.p_lpf_freq = 0.9989765717851521 + frnd(0.2) - 0.1),
          (e.p_lpf_ramp = -0.25051720626043017 + frnd(0.2) - 0.1),
          (e.p_lpf_resonance = 0.32777871505494693 + frnd(0.2) - 0.1),
          (e.p_hpf_freq = 0.0023548750981756753 + frnd(0.2) - 0.1),
          (e.p_hpf_ramp = -0.002375673204842568 + frnd(0.2) - 0.1),
          e)
        : frnd(5) > 1
        ? ((e.wave_type = Math.floor(frnd(SHAPES.length))),
          3 === e.wave_type && (e.wave_type = SQUARE),
          rnd(1)
              ? ((e.p_arp_mod = 0.2697849293151393 + frnd(0.2) - 0.1),
                (e.p_arp_speed = -0.3131172257760948 + frnd(0.2) - 0.1),
                (e.p_base_freq = 0.8090588299313949 + frnd(0.2) - 0.1),
                (e.p_duty = -0.6210022920964955 + frnd(0.2) - 0.1),
                (e.p_duty_ramp = -0.00043441813553182567 + frnd(0.2) - 0.1),
                (e.p_env_attack = 0.004321877246874195 + frnd(0.2) - 0.1),
                (e.p_env_decay = 0.1 + frnd(0.2) - 0.1),
                (e.p_env_punch = 0.061737781504416146 + frnd(0.2) - 0.1),
                (e.p_env_sustain = 0.4987252564798832 + frnd(0.2) - 0.1),
                (e.p_freq_dramp = 0.31700340314222614 + frnd(0.2) - 0.1),
                (e.p_freq_limit = 0 + frnd(0.2) - 0.1),
                (e.p_freq_ramp = -0.163380391341416 + frnd(0.2) - 0.1),
                (e.p_hpf_freq = 0.4709005021145149 + frnd(0.2) - 0.1),
                (e.p_hpf_ramp = 0.6924667290539194 + frnd(0.2) - 0.1),
                (e.p_lpf_freq = 0.8351398631384511 + frnd(0.2) - 0.1),
                (e.p_lpf_ramp = 0.36616557192873134 + frnd(0.2) - 0.1),
                (e.p_lpf_resonance = -0.08685777111664439 + frnd(0.2) - 0.1),
                (e.p_pha_offset = -0.036084571580025544 + frnd(0.2) - 0.1),
                (e.p_pha_ramp = -0.014806445085568108 + frnd(0.2) - 0.1),
                (e.p_repeat_speed = -0.8094368475518489 + frnd(0.2) - 0.1),
                (e.p_vib_speed = 0.4496665457171294 + frnd(0.2) - 0.1),
                (e.p_vib_strength = 0.23413762515532424 + frnd(0.2) - 0.1))
              : ((e.p_arp_mod = -0.35697118026766184 + frnd(0.2) - 0.1),
                (e.p_arp_speed = 0.3581140690559588 + frnd(0.2) - 0.1),
                (e.p_base_freq = 1.3260897696157528 + frnd(0.2) - 0.1),
                (e.p_duty = -0.30984900436710694 + frnd(0.2) - 0.1),
                (e.p_duty_ramp = -0.0014374759133411626 + frnd(0.2) - 0.1),
                (e.p_env_attack = 0.3160357835682254 + frnd(0.2) - 0.1),
                (e.p_env_decay = 0.1 + frnd(0.2) - 0.1),
                (e.p_env_punch = 0.24323114016870148 + frnd(0.2) - 0.1),
                (e.p_env_sustain = 0.4 + frnd(0.2) - 0.1),
                (e.p_freq_dramp = 0.2866475886237244 + frnd(0.2) - 0.1),
                (e.p_freq_limit = 0 + frnd(0.2) - 0.1),
                (e.p_freq_ramp = -0.10956352368742976 + frnd(0.2) - 0.1),
                (e.p_hpf_freq = 0.20772718017889846 + frnd(0.2) - 0.1),
                (e.p_hpf_ramp = 0.1564090637378835 + frnd(0.2) - 0.1),
                (e.p_lpf_freq = 0.6021372770637031 + frnd(0.2) - 0.1),
                (e.p_lpf_ramp = 0.24016227139979027 + frnd(0.2) - 0.1),
                (e.p_lpf_resonance = -0.08787383821160144 + frnd(0.2) - 0.1),
                (e.p_pha_offset = -0.381597686151701 + frnd(0.2) - 0.1),
                (e.p_pha_ramp = -0.0002481687661373495 + frnd(0.2) - 0.1),
                (e.p_repeat_speed = 0.07812112809425686 + frnd(0.2) - 0.1),
                (e.p_vib_speed = -0.13648848579133943 + frnd(0.2) - 0.1),
                (e.p_vib_strength = 0.0018874158972302657 + frnd(0.2) - 0.1)),
          e)
        : ((e.wave_type = Math.floor(frnd(SHAPES.length))),
          (1 !== e.wave_type && 3 !== e.wave_type) || (e.wave_type = 2),
          (e.p_base_freq = 0.85 + frnd(0.15)),
          (e.p_freq_ramp = 0.3 + frnd(0.15)),
          (e.p_env_attack = 0 + frnd(0.09)),
          (e.p_env_sustain = 0.2 + frnd(0.3)),
          (e.p_env_decay = 0 + frnd(0.1)),
          (e.p_duty = frnd(2) - 1),
          (e.p_duty_ramp = Math.pow(frnd(2) - 1, 3)),
          (e.p_repeat_speed = 0.5 + frnd(0.1)),
          (e.p_pha_offset = -0.3 + frnd(0.9)),
          (e.p_pha_ramp = -frnd(0.3)),
          (e.p_arp_speed = 0.4 + frnd(0.6)),
          (e.p_arp_mod = 0.8 + frnd(0.1)),
          (e.p_lpf_resonance = frnd(2) - 1),
          (e.p_lpf_freq = 1 - Math.pow(frnd(1), 3)),
          (e.p_lpf_ramp = Math.pow(frnd(2) - 1, 3)),
          e.p_lpf_freq < 0.1 && e.p_lpf_ramp < -0.05 && (e.p_lpf_ramp = -e.p_lpf_ramp),
          (e.p_hpf_freq = Math.pow(frnd(1), 5)),
          (e.p_hpf_ramp = Math.pow(frnd(2) - 1, 5)),
          e);
}
function pushSound() {
    const e = Params();
    return (
        (e.wave_type = Math.floor(frnd(SHAPES.length))),
        2 === e.wave_type && e.wave_type++,
        0 === e.wave_type && (e.wave_type = NOISE),
        (e.p_base_freq = 0.1 + frnd(0.4)),
        (e.p_freq_ramp = 0.05 + frnd(0.2)),
        (e.p_env_attack = 0.01 + frnd(0.09)),
        (e.p_env_sustain = 0.01 + frnd(0.09)),
        (e.p_env_decay = 0.01 + frnd(0.09)),
        (e.p_repeat_speed = 0.3 + frnd(0.5)),
        (e.p_pha_offset = -0.3 + frnd(0.9)),
        (e.p_pha_ramp = -frnd(0.3)),
        (e.p_arp_speed = 0.6 + frnd(0.3)),
        (e.p_arp_mod = 0.8 - frnd(1.6)),
        e
    );
}
function powerUp() {
    const e = Params();
    return (
        rnd(1) ? (e.wave_type = SAWTOOTH) : (e.p_duty = frnd(0.6)),
        (e.wave_type = Math.floor(frnd(SHAPES.length))),
        3 === e.wave_type && (e.wave_type = SQUARE),
        rnd(1)
            ? ((e.p_base_freq = 0.2 + frnd(0.3)), (e.p_freq_ramp = 0.1 + frnd(0.4)), (e.p_repeat_speed = 0.4 + frnd(0.4)))
            : ((e.p_base_freq = 0.2 + frnd(0.3)), (e.p_freq_ramp = 0.05 + frnd(0.2)), rnd(1) && ((e.p_vib_strength = frnd(0.7)), (e.p_vib_speed = frnd(0.6)))),
        (e.p_env_attack = 0),
        (e.p_env_sustain = frnd(0.4)),
        (e.p_env_decay = 0.1 + frnd(0.4)),
        e
    );
}
function hitHurt() {
    const e = Params();
    return (
        (e.wave_type = rnd(2)),
        e.wave_type === SINE && (e.wave_type = NOISE),
        e.wave_type === SQUARE && (e.p_duty = frnd(0.6)),
        (e.wave_type = Math.floor(frnd(SHAPES.length))),
        (e.p_base_freq = 0.2 + frnd(0.6)),
        (e.p_freq_ramp = -0.3 - frnd(0.4)),
        (e.p_env_attack = 0),
        (e.p_env_sustain = frnd(0.1)),
        (e.p_env_decay = 0.1 + frnd(0.2)),
        rnd(1) && (e.p_hpf_freq = frnd(0.3)),
        e
    );
}
function jump() {
    const e = Params();
    return (
        (e.wave_type = SQUARE),
        (e.wave_type = Math.floor(frnd(SHAPES.length))),
        3 === e.wave_type && (e.wave_type = SQUARE),
        (e.p_duty = frnd(0.6)),
        (e.p_base_freq = 0.3 + frnd(0.3)),
        (e.p_freq_ramp = 0.1 + frnd(0.2)),
        (e.p_env_attack = 0),
        (e.p_env_sustain = 0.1 + frnd(0.3)),
        (e.p_env_decay = 0.1 + frnd(0.2)),
        rnd(1) && (e.p_hpf_freq = frnd(0.3)),
        rnd(1) && (e.p_lpf_freq = 1 - frnd(0.6)),
        e
    );
}
function blipSelect() {
    const e = Params();
    return (
        (e.wave_type = rnd(1)),
        (e.wave_type = Math.floor(frnd(SHAPES.length))),
        3 === e.wave_type && (e.wave_type = rnd(1)),
        e.wave_type === SQUARE && (e.p_duty = frnd(0.6)),
        (e.p_base_freq = 0.2 + frnd(0.4)),
        (e.p_env_attack = 0),
        (e.p_env_sustain = 0.1 + frnd(0.1)),
        (e.p_env_decay = frnd(0.2)),
        (e.p_hpf_freq = 0.1),
        e
    );
}
function random() {
    const e = Params();
    return (
        (e.wave_type = Math.floor(frnd(SHAPES.length))),
        (e.p_base_freq = Math.pow(frnd(2) - 1, 2)),
        rnd(1) && (e.p_base_freq = Math.pow(frnd(2) - 1, 3) + 0.5),
        (e.p_freq_limit = 0),
        (e.p_freq_ramp = Math.pow(frnd(2) - 1, 5)),
        e.p_base_freq > 0.7 && e.p_freq_ramp > 0.2 && (e.p_freq_ramp = -e.p_freq_ramp),
        e.p_base_freq < 0.2 && e.p_freq_ramp < -0.05 && (e.p_freq_ramp = -e.p_freq_ramp),
        (e.p_freq_dramp = Math.pow(frnd(2) - 1, 3)),
        (e.p_duty = frnd(2) - 1),
        (e.p_duty_ramp = Math.pow(frnd(2) - 1, 3)),
        (e.p_vib_strength = Math.pow(frnd(2) - 1, 3)),
        (e.p_vib_speed = frnd(2) - 1),
        (e.p_env_attack = Math.pow(frnd(2) - 1, 3)),
        (e.p_env_sustain = Math.pow(frnd(2) - 1, 2)),
        (e.p_env_decay = frnd(2) - 1),
        (e.p_env_punch = Math.pow(frnd(0.8), 2)),
        e.p_env_attack + e.p_env_sustain + e.p_env_decay < 0.2 && ((e.p_env_sustain += 0.2 + frnd(0.3)), (e.p_env_decay += 0.2 + frnd(0.3))),
        (e.p_lpf_resonance = frnd(2) - 1),
        (e.p_lpf_freq = 1 - Math.pow(frnd(1), 3)),
        (e.p_lpf_ramp = Math.pow(frnd(2) - 1, 3)),
        e.p_lpf_freq < 0.1 && e.p_lpf_ramp < -0.05 && (e.p_lpf_ramp = -e.p_lpf_ramp),
        (e.p_hpf_freq = Math.pow(frnd(1), 5)),
        (e.p_hpf_ramp = Math.pow(frnd(2) - 1, 5)),
        (e.p_pha_offset = Math.pow(frnd(2) - 1, 3)),
        (e.p_pha_ramp = Math.pow(frnd(2) - 1, 3)),
        (e.p_repeat_speed = frnd(2) - 1),
        (e.p_arp_speed = frnd(2) - 1),
        (e.p_arp_mod = frnd(2) - 1),
        e
    );
}
const generators = [pickupCoin, laserShoot, explosion, powerUp, hitHurt, jump, blipSelect, pushSound, random, birdSound],
    generatorNames = ["pickupCoin", "laserShoot", "explosion", "powerUp", "hitHurt", "jump", "blipSelect", "pushSound", "random", "birdSound"];
function generateFromSeed(e) {
    rng = new RNG((e / 100) | 0);
    const n = generators[(e % 100) % generators.length];
    seeded = !0;
    const t = n();
    return (t.seed = e), (seeded = !1), t;
}
function SoundEffect(e, n) {
    this._buffer = AUDIO_CONTEXT.createBuffer(1, e, n);
}
function ULBS() {
    if ("suspended" === AUDIO_CONTEXT.state) {
        const e = function () {
            AUDIO_CONTEXT.resume().then(function () {
                document.body.removeEventListener("touchstart", e),
                    document.body.removeEventListener("touchend", e),
                    document.body.removeEventListener("mousedown", e),
                    document.body.removeEventListener("mouseup", e),
                    document.body.removeEventListener("keydown", e),
                    document.body.removeEventListener("keyup", e);
            });
        };
        document.body.addEventListener("touchstart", e, !1),
            document.body.addEventListener("touchend", e, !1),
            document.body.addEventListener("mousedown", e, !1),
            document.body.addEventListener("mouseup", e, !1),
            document.body.addEventListener("keydown", e, !1),
            document.body.addEventListener("keyup", e, !1);
    }
}
if (
    ((SoundEffect.prototype.getBuffer = function () {
        return this._buffer.getChannelData(0);
    }),
    (SoundEffect.prototype.play = function () {
        ULBS();
        const e = AUDIO_CONTEXT.createBufferSource(),
            n = AUDIO_CONTEXT.createBiquadFilter(),
            t = AUDIO_CONTEXT.createBiquadFilter(),
            o = AUDIO_CONTEXT.createBiquadFilter();
        (e.buffer = this._buffer), e.connect(n), (n.frequency.value = 1600), (t.frequency.value = 1600), (o.frequency.value = 1600), n.connect(t), t.connect(o), o.connect(AUDIO_CONTEXT.destination);
        const r = AUDIO_CONTEXT.currentTime;
        void 0 !== e.start ? e.start(r) : e.noteOn(r),
            (e.onended = function () {
                o.disconnect();
            });
    }),
    (SoundEffect.MIN_SAMPLE_RATE = 22050),
    void 0 === AUDIO_CONTEXT &&
        (((SoundEffect = function (e, n) {
            (this._sample_rate = n), (this._buffer = new Array(e)), (this._audioElement = null);
        }).prototype.getBuffer = function () {
            return (this._audioElement = null), this._buffer;
        }),
        (SoundEffect.prototype.play = function () {
            if (this._audioElement) this._audioElement.cloneNode(!1).play();
            else {
                for (let e = 0; e < this._buffer.length; e++) this._buffer[e] = 255 & Math.floor(128 * Math.max(0, Math.min(this._buffer[e] + 1, 2)));
                let e = MakeRiff(this._sample_rate, BIT_DEPTH, this._buffer);
                (this._audioElement = new Audio()), (this._audioElement.src = e.dataURI), this._audioElement.play();
            }
        }),
        (SoundEffect.MIN_SAMPLE_RATE = 1)),
    (SoundEffect.generate = function (e) {
        function n() {
            (t = 0),
                (o = 100 / (e.p_base_freq * e.p_base_freq + 0.001)),
                (r = Math.floor(o)),
                (l = 100 / (e.p_freq_limit * e.p_freq_limit + 0.001)),
                (i = 1 - 0.01 * Math.pow(e.p_freq_ramp, 3)),
                (s = 1e-6 * -Math.pow(e.p_freq_dramp, 3)),
                (a = 0.5 - 0.5 * e.p_duty),
                (c = 5e-5 * -e.p_duty_ramp),
                (d = e.p_arp_mod >= 0 ? 1 - 0.9 * Math.pow(e.p_arp_mod, 2) : 1 + 10 * Math.pow(e.p_arp_mod, 2)),
                (u = 0),
                (h = Math.floor(2e4 * Math.pow(1 - e.p_arp_speed, 2) + 32)),
                1 == e.p_arp_speed && (h = 0);
        }
        let t, o, r, l, i, s, a, c, d, u, h;
        n();
        let g = 0,
            p = 0,
            f = 0.1 * Math.pow(e.p_lpf_freq, 3),
            m = 1 + 1e-4 * e.p_lpf_ramp,
            _ = (5 / (1 + 20 * Math.pow(e.p_lpf_resonance, 2))) * (0.01 + f);
        _ > 0.8 && (_ = 0.8);
        let v = 0,
            y = 0.1 * Math.pow(e.p_hpf_freq, 2),
            b = 1 + 3e-4 * e.p_hpf_ramp,
            w = 0,
            E = 0.01 * Math.pow(e.p_vib_speed, 2),
            C = 0.5 * e.p_vib_strength,
            k = 0,
            R = 0,
            M = 0,
            S = [Math.floor(e.p_env_attack * e.p_env_attack * 1e5), Math.floor(e.p_env_sustain * e.p_env_sustain * 1e5), Math.floor(e.p_env_decay * e.p_env_decay * 1e5)],
            I = S[0] + S[1] + S[2],
            T = 0,
            O = 1020 * Math.pow(e.p_pha_offset, 2);
        e.p_pha_offset < 0 && (O = -O);
        let A = 1 * Math.pow(e.p_pha_ramp, 2);
        e.p_pha_ramp < 0 && (A = -A);
        let x = Math.abs(Math.floor(O)),
            N = 0,
            L = [];
        for (let e = 0; e < 1024; ++e) L[e] = 0;
        let D = [];
        for (let e = 0; e < 32; ++e) D[e] = 2 * Math.random() - 1;
        let j = Math.floor(2e4 * Math.pow(1 - e.p_repeat_speed, 2) + 32);
        0 == e.p_repeat_speed && (j = 0);
        let F,
            B = Math.exp(e.sound_vol) - 1,
            P = 0,
            $ = 0,
            U = Math.floor(44100 / e.sample_rate),
            V = 0,
            H = Math.ceil(I / U),
            q = !1;
        F = e.sample_rate < SoundEffect.MIN_SAMPLE_RATE ? new SoundEffect(4 * H, SoundEffect.MIN_SAMPLE_RATE) : new SoundEffect(H, e.sample_rate);
        let z = F.getBuffer(),
            W = 0;
        for (let u = 0; ; ++u) {
            0 != j && ++t >= j && n(), 0 != h && u >= h && ((h = 0), (o *= d)), (i += s), (o *= i), o > l && ((o = l), e.p_freq_limit > 0 && (q = !0));
            let I = o;
            if ((C > 0 && ((w += E), (I = o * (1 + Math.sin(w) * C))), (r = Math.floor(I)), r < 8 && (r = 8), (a += c), a < 0 && (a = 0), a > 0.5 && (a = 0.5), M++, M > S[R])) {
                for (M = 1, R++; R < 3 && 0 === S[R]; ) R++;
                if (3 === R) break;
            }
            (k = 0 === R ? M / S[0] : 1 === R ? 1 + 2 * Math.pow(1 - M / S[1], 1) * e.p_env_punch : 1 - M / S[2]),
                (O += A),
                (x = Math.abs(Math.floor(O))),
                x > 1023 && (x = 1023),
                0 != b && ((y *= b), y < 1e-5 && (y = 1e-5), y > 0.1 && (y = 0.1)),
                (W = 0);
            for (let n = 0; n < 8; ++n) {
                let n = 0;
                if ((T++, T >= r && ((T %= r), e.wave_type === NOISE))) for (let e = 0; e < 32; ++e) D[e] = 2 * Math.random() - 1;
                let t = T / r;
                if (e.wave_type === SQUARE) n = t < a ? 0.5 : -0.5;
                else if (e.wave_type === SAWTOOTH) n = 1 - 2 * t;
                else if (e.wave_type === SINE) n = Math.sin(2 * t * Math.PI);
                else if (e.wave_type === NOISE) n = D[Math.floor((32 * T) / r)];
                else if (4 === e.wave_type) n = Math.abs(1 - 2 * t) - 1;
                else {
                    if (5 !== e.wave_type) throw new Exception("bad wave type! " + e.wave_type);
                    n = Math.abs(1 - t * t * 2) - 1;
                }
                let o = g;
                (f *= m),
                    f < 0 && (f = 0),
                    f > 0.1 && (f = 0.1),
                    1 != e.p_lpf_freq ? ((p += (n - g) * f), (p -= p * _)) : ((g = n), (p = 0)),
                    (g += p),
                    (v += g - o),
                    (v -= v * y),
                    (n = v),
                    (L[1023 & N] = n),
                    (n += L[(N - x + 1024) & 1023]),
                    (N = (N + 1) & 1023),
                    (W += n * k);
            }
            (P += W), ++$ >= U && (($ = 0), (W = P / U), (P = 0), (W = (W / 8) * masterVolume), (W *= B), (z[V++] = W), e.sample_rate < SoundEffect.MIN_SAMPLE_RATE && ((z[V++] = W), (z[V++] = W), (z[V++] = W)));
        }
        return U > 0 && ((W = P / U), (W = (W / 8) * masterVolume), (W *= B), (z[V++] = W), e.sample_rate < SoundEffect.MIN_SAMPLE_RATE && ((z[V++] = W), (z[V++] = W), (z[V++] = W))), F;
    }),
    "undefined" != typeof exports)
) {
    require("./riffwave").RIFFWAVE;
    (exports.Params = Params), (exports.generate = generate);
}
const sfxCache = {};
let cachedSeeds = [];
const CACHE_MAX = 50;
function cacheSeed(e) {
    if (e in sfxCache) return sfxCache[e];
    const n = generateFromSeed(e);
    (n.sound_vol = SOUND_VOL), (n.sample_rate = SAMPLE_RATE), (n.bit_depth = BIT_DEPTH);
    const t = SoundEffect.generate(n);
    for (sfxCache[e] = t, cachedSeeds.push(e); cachedSeeds.length > CACHE_MAX; ) {
        const e = cachedSeeds[0];
        (cachedSeeds = cachedSeeds.slice(1)), delete sfxCache[e];
    }
    return t;
}
function playSound(e, n) {
    if ((!0 !== n && pushSoundToHistory(e), muted)) return;
    if ((checkAudioContextExists(), unitTesting)) return;
    cacheSeed(e).play();
}
function killAudioButton() {
    const e = document.getElementById("muteButton"),
        n = document.getElementById("unMuteButton");
    e && (e.remove(), n.remove());
}
function showAudioButton() {
    const e = document.getElementById("muteButton"),
        n = document.getElementById("unMuteButton");
    e && ((e.style.display = "block"), (n.style.display = "none"));
}
function toggleMute() {
    0 === muted ? muteAudio() : unMuteAudio();
}
function muteAudio() {
    muted = 1;
    const e = document.getElementById("muteButton"),
        n = document.getElementById("unMuteButton");
    e && ((e.style.display = "none"), (n.style.display = "block"));
}
function unMuteAudio() {
    muted = 0;
    const e = document.getElementById("muteButton"),
        n = document.getElementById("unMuteButton");
    e && ((e.style.display = "block"), (n.style.display = "none"));
}
function CodeMirror(e, n) {}
CodeMirror.defineMode = function (e, n) {};
var StringStream = (CodeMirror.StringStream = function (e, n) {
    (this.pos = this.start = 0), (this.string = e), (this.tabSize = n || 8), (this.lastColumnPos = this.lastColumnValue = 0), (this.lineStart = 0);
});
StringStream.prototype = {
    eol: function () {
        return this.pos >= this.string.length;
    },
    sol: function () {
        return this.pos == this.lineStart;
    },
    peek: function () {
        return this.string.charAt(this.pos) || void 0;
    },
    next: function () {
        if (this.pos < this.string.length) return this.string.charAt(this.pos++);
    },
    eat: function (e) {
        var n = this.string.charAt(this.pos);
        if ("string" == typeof e) var t = n == e;
        else t = n && (e.test ? e.test(n) : e(n));
        if (t) return ++this.pos, n;
    },
    eatWhile: function (e) {
        for (var n = this.pos; this.eat(e); );
        return this.pos > n;
    },
    eatSpace: function () {
        for (var e = this.pos; /[\s\u00a0]/.test(this.string.charAt(this.pos)); ) ++this.pos;
        return this.pos > e;
    },
    skipToEnd: function () {
        this.pos = this.string.length;
    },
    skipTo: function (e) {
        var n = this.string.indexOf(e, this.pos);
        if (n > -1) return (this.pos = n), !0;
    },
    backUp: function (e) {
        this.pos -= e;
    },
    column: function () {
        return (
            this.lastColumnPos < this.start && ((this.lastColumnValue = countColumn(this.string, this.start, this.tabSize, this.lastColumnPos, this.lastColumnValue)), (this.lastColumnPos = this.start)),
            this.lastColumnValue - (this.lineStart ? countColumn(this.string, this.lineStart, this.tabSize) : 0)
        );
    },
    indentation: function () {
        return countColumn(this.string, null, this.tabSize) - (this.lineStart ? countColumn(this.string, this.lineStart, this.tabSize) : 0);
    },
    match: function (e, n, t) {
        if ("string" != typeof e) {
            var o = this.string.slice(this.pos).match(e);
            return o && o.index > 0 ? null : (o && !1 !== n && (this.pos += o[0].length), o);
        }
        var r = function (e) {
            return t ? e.toLowerCase() : e;
        };
        if (r(this.string.substr(this.pos, e.length)) == r(e)) return !1 !== n && (this.pos += e.length), !0;
    },
    current: function () {
        return this.string.slice(this.start, this.pos);
    },
    hideFirstChars: function (e, n) {
        this.lineStart += e;
        try {
            return n();
        } finally {
            this.lineStart -= e;
        }
    },
};
const colorPalettesAliases = {
        1: "mastersystem",
        2: "gameboycolour",
        3: "amiga",
        4: "arnecolors",
        5: "famicom",
        6: "atari",
        7: "pastel",
        8: "ega",
        9: "amstrad",
        10: "proteus_mellow",
        11: "proteus_rich",
        12: "proteus_night",
        13: "c64",
        14: "whitingjp",
    },
    colorPalettes = {
        mastersystem: {
            black: "#000000",
            white: "#FFFFFF",
            grey: "#555555",
            darkgrey: "#555500",
            lightgrey: "#AAAAAA",
            gray: "#555555",
            darkgray: "#555500",
            lightgray: "#AAAAAA",
            red: "#FF0000",
            darkred: "#AA0000",
            lightred: "#FF5555",
            brown: "#AA5500",
            darkbrown: "#550000",
            lightbrown: "#FFAA00",
            orange: "#FF5500",
            yellow: "#FFFF55",
            green: "#55AA00",
            darkgreen: "#005500",
            lightgreen: "#AAFF00",
            blue: "#5555AA",
            lightblue: "#AAFFFF",
            darkblue: "#000055",
            purple: "#550055",
            pink: "#FFAAFF",
        },
        gameboycolour: {
            black: "#000000",
            white: "#FFFFFF",
            grey: "#7F7F7C",
            darkgrey: "#3E3E44",
            lightgrey: "#BAA7A7",
            gray: "#7F7F7C",
            darkgray: "#3E3E44",
            lightgray: "#BAA7A7",
            red: "#A7120C",
            darkred: "#880606",
            lightred: "#BA381F",
            brown: "#57381F",
            darkbrown: "#3E2519",
            lightbrown: "#8E634B",
            orange: "#BA4B32",
            yellow: "#C0BA6F",
            green: "#517525",
            darkgreen: "#385D12",
            lightgreen: "#6F8E44",
            blue: "#5D6FA7",
            lightblue: "#8EA7A7",
            darkblue: "#4B575D",
            purple: "#3E3E44",
            pink: "#BA381F",
        },
        amiga: {
            black: "#000000",
            white: "#FFFFFF",
            grey: "#BBBBBB",
            darkgrey: "#333333",
            lightgrey: "#FFEEDD",
            gray: "#BBBBBB",
            darkgray: "#333333",
            lightgray: "#FFEEDD",
            red: "#DD1111",
            darkred: "#990000",
            lightred: "#FF4422",
            brown: "#663311",
            darkbrown: "#331100",
            lightbrown: "#AA6644",
            orange: "#FF6644",
            yellow: "#FFDD66",
            green: "#448811",
            darkgreen: "#335500",
            lightgreen: "#88BB77",
            blue: "#8899DD",
            lightblue: "#BBDDEE",
            darkblue: "#666688",
            purple: "#665555",
            pink: "#997788",
        },
        arnecolors: {
            black: "#000000",
            white: "#FFFFFF",
            grey: "#9d9d9d",
            darkgrey: "#697175",
            lightgrey: "#cccccc",
            gray: "#9d9d9d",
            darkgray: "#697175",
            lightgray: "#cccccc",
            red: "#be2633",
            darkred: "#732930",
            lightred: "#e06f8b",
            brown: "#a46422",
            darkbrown: "#493c2b",
            lightbrown: "#eeb62f",
            orange: "#eb8931",
            yellow: "#f7e26b",
            green: "#44891a",
            darkgreen: "#2f484e",
            lightgreen: "#a3ce27",
            blue: "#1d57f7",
            lightblue: "#B2DCEF",
            darkblue: "#1B2632",
            purple: "#342a97",
            pink: "#de65e2",
        },
        famicom: {
            black: "#000000",
            white: "#ffffff",
            grey: "#7c7c7c",
            darkgrey: "#080808",
            lightgrey: "#bcbcbc",
            gray: "#7c7c7c",
            darkgray: "#080808",
            lightgray: "#bcbcbc",
            red: "#f83800",
            darkred: "#881400",
            lightred: "#f87858",
            brown: "#AC7C00",
            darkbrown: "#503000",
            lightbrown: "#FCE0A8",
            orange: "#FCA044",
            yellow: "#F8B800",
            green: "#00B800",
            darkgreen: "#005800",
            lightgreen: "#B8F8B8",
            blue: "#0058F8",
            lightblue: "#3CBCFC",
            darkblue: "#0000BC",
            purple: "#6644FC",
            pink: "#F878F8",
        },
        atari: {
            black: "#000000",
            white: "#FFFFFF",
            grey: "#909090",
            darkgrey: "#404040",
            lightgrey: "#b0b0b0",
            gray: "#909090",
            darkgray: "#404040",
            lightgray: "#b0b0b0",
            red: "#A03C50",
            darkred: "#700014",
            lightred: "#DC849C",
            brown: "#805020",
            darkbrown: "#703400",
            lightbrown: "#CB9870",
            orange: "#CCAC70",
            yellow: "#ECD09C",
            green: "#58B06C",
            darkgreen: "#006414",
            lightgreen: "#70C484",
            blue: "#1C3C88",
            lightblue: "#6888C8",
            darkblue: "#000088",
            purple: "#3C0080",
            pink: "#B484DC",
        },
        pastel: {
            black: "#000000",
            white: "#FFFFFF",
            grey: "#3e3e3e",
            darkgrey: "#313131",
            lightgrey: "#9cbcbc",
            gray: "#3e3e3e",
            darkgray: "#313131",
            lightgray: "#9cbcbc",
            red: "#f56ca2",
            darkred: "#a63577",
            lightred: "#ffa9cf",
            brown: "#b58c53",
            darkbrown: "#787562",
            lightbrown: "#B58C53",
            orange: "#EB792D",
            yellow: "#FFe15F",
            green: "#00FF4F",
            darkgreen: "#2b732c",
            lightgreen: "#97c04f",
            blue: "#0f88d3",
            lightblue: "#00fffe",
            darkblue: "#293a7b",
            purple: "#ff6554",
            pink: "#eb792d",
        },
        ega: {
            black: "#000000",
            white: "#ffffff",
            grey: "#555555",
            darkgrey: "#555555",
            lightgrey: "#aaaaaa",
            gray: "#555555",
            darkgray: "#555555",
            lightgray: "#aaaaaa",
            red: "#ff5555",
            darkred: "#aa0000",
            lightred: "#ff55ff",
            brown: "#aa5500",
            darkbrown: "#aa5500",
            lightbrown: "#ffff55",
            orange: "#ff5555",
            yellow: "#ffff55",
            green: "#00aa00",
            darkgreen: "#00aaaa",
            lightgreen: "#55ff55",
            blue: "#5555ff",
            lightblue: "#55ffff",
            darkblue: "#0000aa",
            purple: "#aa00aa",
            pink: "#ff55ff",
        },
        proteus_mellow: {
            black: "#3d2d2e",
            white: "#ddf1fc",
            grey: "#9fb2d4",
            darkgrey: "#7b8272",
            lightgrey: "#a4bfda",
            gray: "#9fb2d4",
            darkgray: "#7b8272",
            lightgray: "#a4bfda",
            red: "#9d5443",
            darkred: "#8c5b4a",
            lightred: "#94614c",
            brown: "#89a78d",
            darkbrown: "#829e88",
            lightbrown: "#aaae97",
            orange: "#d1ba86",
            yellow: "#d6cda2",
            green: "#75ac8d",
            darkgreen: "#8fa67f",
            lightgreen: "#8eb682",
            blue: "#88a3ce",
            lightblue: "#a5adb0",
            darkblue: "#5c6b8c",
            purple: "#d39fac",
            pink: "#c8ac9e",
        },
        proteus_night: {
            black: "#010912",
            white: "#fdeeec",
            grey: "#051d40",
            darkgrey: "#091842",
            lightgrey: "#062151",
            gray: "#051d40",
            darkgray: "#091842",
            lightgray: "#062151",
            red: "#ad4576",
            darkred: "#934765",
            lightred: "#ab6290",
            brown: "#61646b",
            darkbrown: "#3d2d2d",
            lightbrown: "#8393a0",
            orange: "#0a2227",
            yellow: "#0a2541",
            green: "#75ac8d",
            darkgreen: "#0a2434",
            lightgreen: "#061f2e",
            blue: "#0b2c79",
            lightblue: "#809ccb",
            darkblue: "#08153b",
            purple: "#666a87",
            pink: "#754b4d",
        },
        proteus_rich: {
            black: "#6f686f",
            white: "#d1b1e2",
            grey: "#b9aac1",
            darkgrey: "#8e8b84",
            lightgrey: "#c7b5cd",
            gray: "#b9aac1",
            darkgray: "#8e8b84",
            lightgray: "#c7b5cd",
            red: "#a11f4f",
            darkred: "#934765",
            lightred: "#c998ad",
            brown: "#89867d",
            darkbrown: "#797f75",
            lightbrown: "#ab9997",
            orange: "#ce8c5c",
            yellow: "#f0d959",
            green: "#75bc54",
            darkgreen: "#599d79",
            lightgreen: "#90cf5c",
            blue: "#8fd0ec",
            lightblue: "#bcdce7",
            darkblue: "#0b2c70",
            purple: "#9b377f",
            pink: "#cd88e5",
        },
        amstrad: {
            black: "#000000",
            white: "#ffffff",
            grey: "#7f7f7f",
            darkgrey: "#636363",
            lightgrey: "#afafaf",
            gray: "#7f7f7f",
            darkgray: "#636363",
            lightgray: "#afafaf",
            red: "#ff0000",
            darkred: "#7f0000",
            lightred: "#ff7f7f",
            brown: "#ff7f00",
            darkbrown: "#7f7f00",
            lightbrown: "#ffff00",
            orange: "#ff007f",
            yellow: "#ffff7f",
            green: "#01ff00",
            darkgreen: "#007f00",
            lightgreen: "#7fff7f",
            blue: "#0000ff",
            lightblue: "#7f7fff",
            darkblue: "#00007f",
            purple: "#7f007f",
            pink: "#ff7fff",
        },
        c64: {
            black: "#000000",
            white: "#ffffff",
            grey: "#6C6C6C",
            darkgrey: "#444444",
            lightgrey: "#959595",
            gray: "#6C6C6C",
            darkgray: "#444444",
            lightgray: "#959595",
            red: "#68372B",
            darkred: "#3f1e17",
            lightred: "#9A6759",
            brown: "#433900",
            darkbrown: "#221c02",
            lightbrown: "#6d5c0d",
            orange: "#6F4F25",
            yellow: "#B8C76F",
            green: "#588D43",
            darkgreen: "#345129",
            lightgreen: "#9AD284",
            blue: "#6C5EB5",
            lightblue: "#70A4B2",
            darkblue: "#352879",
            purple: "#6F3D86",
            pink: "#b044ac",
        },
        whitingjp: {
            black: "#202527",
            white: "#eff8fd",
            grey: "#7b7680",
            darkgrey: "#3c3b44",
            lightgrey: "#bed0d7",
            gray: "#7b7680",
            darkgray: "#3c3b44",
            lightgray: "#bed0d7",
            red: "#bd194b",
            darkred: "#6b1334",
            lightred: "#ef2358",
            brown: "#b52e1c",
            darkbrown: "#681c12",
            lightbrown: "#e87b45",
            orange: "#ff8c10",
            yellow: "#fbd524",
            green: "#36bc3c",
            darkgreen: "#317610",
            lightgreen: "#8ce062",
            blue: "#3f62c6",
            lightblue: "#57bbe0",
            darkblue: "#2c2fa0",
            purple: "#7037d9",
            pink: "#ec2b8f",
        },
    };
function createSprite(e, n, t, o) {
    void 0 === t && (t = [state.bgcolor, state.fgcolor]);
    var r = makeSpriteCanvas(e),
        l = r.getContext("2d");
    l.clearRect(0, 0, cellwidth, cellheight);
    var i = n[0].length,
        s = n.length,
        a = ~~(cellwidth / (i + (0 | o))),
        c = ~~(cellheight / (s + (0 | o))),
        d = c;
    "scanline" in state.metadata && (d = Math.ceil(c / 2)), (l.fillStyle = state.fgcolor);
    for (var u = 0; u < s; u++)
        for (var h = 0; h < i; h++) {
            var g = n[u][h];
            if (g >= 0) {
                var p = (u * c) | 0,
                    f = (h * a) | 0;
                const e = t[g];
                (l.fillStyle = e), l.fillRect(f, p, a, d);
            }
        }
    return r;
}
function renderCJKCharacter(e, n) {
    state.bgcolor, state.fgcolor;
    var t = makeSpriteCanvas("specialchar_" + e);
    t.width = 2 * cellwidth;
    var o = t.getContext("2d");
    o.clearRect(0, 0, 2 * cellwidth, cellheight), (o.fillStyle = state.fgcolor);
    const r = cellheight - 4 * n;
    return (o.font = r + "px Arial"), (o.textAlign = "center"), o.fillText(e, cellwidth, r + n), t;
}
function regenText(e, n) {
    for (var t in ((textImages = {}), titleImage)) {
        var o = titleImage[t];
        for (var r in o) {
            var l = o[r];
            if (!textImages.hasOwnProperty(l))
                if (font.hasOwnProperty(l)) {
                    const e = font[l].split("\n").map((e) =>
                        e
                            .trim()
                            .split("")
                            .map((e) => parseInt(e))
                    );
                    e.shift(), (textImages[l] = createSprite("char" + l, e, ["#00000000", state.fgcolor], 1));
                } else textImages[l] = renderCJKCharacter(l, 1);
        }
    }
}
var spriteimages,
    glyphImagesCorrespondance,
    glyphImages,
    glyphHighlight,
    glyphHighlightDiff,
    glyphHighlightResize,
    glyphPrintButton,
    glyphMouseOver,
    editor_s_grille = [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ];
function regenSpriteImages() {
    if (textMode) return (spriteimages = []), void regenText();
    if ((!0 === IDE && (textImages.editor_s = createSprite("chars", editor_s_grille, void 0)), 0 !== state.levels.length)) {
        spriteimages = [];
        for (var e = 0; e < sprites.length; e++) null != sprites[e] && (spriteimages[e] = createSprite(e.toString(), sprites[e].dat, sprites[e].colors));
        canOpenEditor && generateGlyphImages();
    }
}
var canvas,
    ctx,
    x,
    y,
    cellwidth,
    cellheight,
    xoffset,
    yoffset,
    lastDownTarget,
    glyphSelectedIndex = 0,
    editorRowCount = 1,
    editorGlyphMovements = [],
    canvasdict = {};
function makeSpriteCanvas(e) {
    var n;
    return e in canvasdict ? (n = canvasdict[e]) : ((n = document.createElement("canvas")), (canvasdict[e] = n)), (n.width = cellwidth), (n.height = cellheight), n;
}
function generateGlyphImages() {
    if (0 === cellwidth || 0 === cellheight) return;
    (glyphImagesCorrespondance = []), (glyphImages = []);
    const e = {};
    for (var n of state.glyphOrder)
        if (1 == n.length && state.glyphDict.hasOwnProperty(n)) {
            var t = state.glyphDict[n],
                o = t.join(",");
            if (e.hasOwnProperty(o)) continue;
            var r = makeSpriteCanvas("C" + n),
                l = r.getContext("2d");
            glyphImagesCorrespondance.push(n), (e[o] = !0);
            for (var i = 0; i < t.length; i++) {
                var s = t[i];
                -1 !== s && l.drawImage(spriteimages[s], 0, 0);
            }
            glyphImages.push(r);
        }
    if (IDE) {
        ((l = (glyphHighlight = makeSpriteCanvas("highlight")).getContext("2d")).fillStyle = "#FFFFFF"),
            l.fillRect(0, 0, cellwidth, 1),
            l.fillRect(0, 0, 1, cellheight),
            l.fillRect(0, cellheight - 1, cellwidth, 1),
            l.fillRect(cellwidth - 1, 0, 1, cellheight),
            (glyphPrintButton = textImages.editor_s),
            ((l = (glyphHighlightDiff = makeSpriteCanvas("glyphHighlightDiff")).getContext("2d")).fillStyle = state.bgcolor),
            l.fillRect(0, 0, cellwidth, 2),
            l.fillRect(0, 0, 2, cellheight),
            l.fillRect(0, cellheight - 2, cellwidth, 2),
            l.fillRect(cellwidth - 2, 0, 2, cellheight),
            (l.fillStyle = state.fgcolor),
            l.fillRect(0, 0, cellwidth, 1),
            l.fillRect(0, 0, 1, cellheight),
            l.fillRect(0, cellheight - 1, cellwidth, 1),
            l.fillRect(cellwidth - 1, 0, 1, cellheight),
            (glyphPrintButton = textImages.editor_s),
            ((l = (glyphHighlightResize = makeSpriteCanvas("highlightresize")).getContext("2d")).fillStyle = "#FFFFFF");
        var a = (cellwidth / 2 - 1) | 0,
            c = cellwidth - a - 1 - a,
            d = (cellheight / 2 - 1) | 0,
            u = cellheight - d - 1 - a;
        l.fillRect(a, 0, c, cellheight),
            l.fillRect(0, d, cellwidth, u),
            ((l = (glyphMouseOver = makeSpriteCanvas("glyphMouseOver")).getContext("2d")).fillStyle = "yellow"),
            l.fillRect(0, 0, cellwidth, 2),
            l.fillRect(0, 0, 2, cellheight),
            l.fillRect(0, cellheight - 2, cellwidth, 2),
            l.fillRect(cellwidth - 2, 0, 2, cellheight);
        const e = [
            [
                [3, 2],
                [5, 0],
                [7, 2],
            ],
            [
                [3, 8],
                [5, 10],
                [7, 8],
            ],
            [
                [2, 3],
                [0, 5],
                [2, 7],
            ],
            [
                [7, 3],
                [10, 5],
                [7, 7],
            ],
            [
                [3, 5],
                [5, 7],
                [7, 5],
                [5, 3],
            ],
            [
                [3, 3],
                [5, 3],
                [5, 4],
                [4, 4],
                [4, 5],
                [3, 5],
            ],
        ];
        for (i = 0; i < e.length; i++) {
            editorGlyphMovements[i] = makeSpriteCanvas("editorGlyphMovements" + i);
            var h = e[i];
            ((l = editorGlyphMovements[i].getContext("2d")).lineWidth = 1), (l.fillStyle = state.bgcolor), (l.strokeStyle = state.fgcolor), l.beginPath(), l.moveTo((h[0][0] * cellwidth) / 10, (h[0][1] * cellheight) / 10);
            for (var g = 1; g < h.length; g++) l.lineTo((h[g][0] * cellwidth) / 10, (h[g][1] * cellheight) / 10);
            l.closePath(), l.fill(), l.stroke();
        }
    }
}
function glyphCount() {
    var e = 0;
    for (var n of state.glyphOrder) 1 == n.length && state.glyphDict.hasOwnProperty(n) && e++;
    return e;
}
function redraw() {
    if (0 !== cellwidth && 0 !== cellheight)
        if ((void 0 === spriteimages && regenSpriteImages(), textMode)) {
            (ctx.fillStyle = state.bgcolor), ctx.fillRect(0, 0, canvas.width, canvas.height);
            for (var e = 0; e < TERMINAL_WIDTH; e++)
                for (var n = 0; n < TERMINAL_HEIGHT; n++) {
                    var t = titleImage[n].charAt(e);
                    if (t in textImages) {
                        var o = textImages[t];
                        ctx.drawImage(o, xoffset + e * cellwidth, yoffset + n * cellheight);
                    }
                }
        } else {
            var r = level;
            null !== diffToVisualize &&
                (((r = new Level(-1, diffToVisualize.width, diffToVisualize.height, diffToVisualize.layerCount, diffToVisualize.objects)).movements = diffToVisualize.movements),
                (r.rigidMovementAppliedMask = diffToVisualize.rigidMovementAppliedMask)),
                (ctx.fillStyle = state.bgcolor),
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            var l = 0,
                i = screenwidth,
                s = 0,
                a = screenheight;
            if (levelEditorOpened) {
                var c = glyphCount();
                (i -= 2), (a -= 2 + (editorRowCount = Math.ceil(c / (screenwidth - 1))));
            } else if (flickscreen) {
                if ((h = getPlayerPositions()).length > 0) {
                    var d = ((g = h[0]) / r.height) | 0,
                        u = g % r.height | 0;
                    (l = ((d / screenwidth) | 0) * screenwidth), (s = ((u / screenheight) | 0) * screenheight), (i = Math.min(l + screenwidth, r.width)), (a = Math.min(s + screenheight, r.height)), (oldflickscreendat = [l, s, i, a]);
                } else oldflickscreendat.length > 0 && ((l = oldflickscreendat[0]), (s = oldflickscreendat[1]), (i = oldflickscreendat[2]), (a = oldflickscreendat[3]));
            } else if (zoomscreen) {
                var h;
                if ((h = getPlayerPositions()).length > 0) {
                    var g;
                    (d = ((g = h[0]) / r.height) | 0), (u = g % r.height | 0);
                    (l = Math.max(Math.min(d - ((screenwidth / 2) | 0), r.width - screenwidth), 0)),
                        (s = Math.max(Math.min(u - ((screenheight / 2) | 0), r.height - screenheight), 0)),
                        (i = Math.min(l + screenwidth, r.width)),
                        (a = Math.min(s + screenheight, r.height)),
                        (oldflickscreendat = [l, s, i, a]);
                } else oldflickscreendat.length > 0 && ((l = oldflickscreendat[0]), (s = oldflickscreendat[1]), (i = oldflickscreendat[2]), (a = oldflickscreendat[3]));
            }
            for (e = l; e < i; e++)
                for (n = s; n < a; n++)
                    for (var p = n + e * r.height, f = r.getCellInto(p, _o12), m = 0; m < state.objectCount; m++)
                        if (0 != f.get(m)) {
                            o = spriteimages[m];
                            ctx.drawImage(o, xoffset + (e - l) * cellwidth, yoffset + (n - s) * cellheight);
                        }
            if (null !== diffToVisualize) {
                for (var _ = diffToVisualize.lineNumber - 1; _ >= -1 && !debug_visualisation_array[diffToVisualize.turnIndex].hasOwnProperty(_); _--);
                var v = debug_visualisation_array[diffToVisualize.turnIndex][_],
                    y = new Level(-1, v.width, v.height, v.layerCount, v.objects);
                (y.movements = v.movements), (y.rigidMovementAppliedMask = v.rigidMovementAppliedMask);
                for (e = l; e < i; e++)
                    for (n = s; n < a; n++) {
                        p = n + e * r.height;
                        var b = y.getMovements(p),
                            w = r.getMovements(p),
                            E = y.getCellInto(p, _o11);
                        f = r.getCellInto(p, _o12);
                        (w.equals(b) && f.equals(E)) || ctx.drawImage(glyphHighlightDiff, xoffset + (e - l) * cellwidth, yoffset + (n - s) * cellheight);
                    }
                for (e = l; e < i; e++)
                    for (n = s; n < a; n++) {
                        (p = n + e * r.height), (w = r.getMovements(p));
                        for (var C = 0; C < r.layerCount; C++) {
                            var k = w.getshiftor(31, 5 * C);
                            for (m = 0; m < 5; m++) 0 != (k & Math.pow(2, m)) && ctx.drawImage(editorGlyphMovements[m], xoffset + (e - l) * cellwidth, yoffset + (n - s) * cellheight);
                        }
                    }
                if (state.rigid)
                    for (e = l; e < i; e++)
                        for (n = s; n < a; n++) {
                            p = n + e * r.height;
                            var R = r.getRigids(p);
                            for (C = 0; C < r.layerCount; C++) {
                                0 !== R.getshiftor(31, 5 * C) && ctx.drawImage(editorGlyphMovements[5], xoffset + (e - l) * cellwidth, yoffset + (n - s) * cellheight);
                            }
                        }
            }
            levelEditorOpened && drawEditorIcons(l, s);
        }
}
function drawEditorIcons(e, n) {
    glyphImages.length;
    var t = glyphImages.length - 0;
    ctx.drawImage(glyphPrintButton, xoffset - cellwidth, yoffset - cellheight * (1 + editorRowCount)),
        mouseCoordY === -1 - editorRowCount && -1 === mouseCoordX && ctx.drawImage(glyphMouseOver, xoffset - cellwidth, yoffset - cellheight * (1 + editorRowCount));
    for (var o = mouseCoordX + (screenwidth - 1) * (s = editorRowCount - (-mouseCoordY - 2) - 1), r = 0; r < t; r++) {
        var l = glyphImages[0 + r],
            i = r % (screenwidth - 1),
            s = (r / (screenwidth - 1)) | 0;
        ctx.drawImage(l, xoffset + i * cellwidth, yoffset + s * cellheight - cellheight * (1 + editorRowCount)),
            mouseCoordX >= 0 && mouseCoordX < screenwidth - 1 && o === r && ctx.drawImage(glyphMouseOver, xoffset + i * cellwidth, yoffset + s * cellheight - cellheight * (1 + editorRowCount)),
            r === glyphSelectedIndex && ctx.drawImage(glyphHighlight, xoffset + i * cellwidth, yoffset + s * cellheight - cellheight * (1 + editorRowCount));
    }
    var a = "",
        c = null;
    if (mouseCoordX >= 0 && mouseCoordX < screenwidth && o >= 0 && o < t) {
        const e = glyphImagesCorrespondance[0 + o];
        (a = e), e in state.synonymsDict ? (a += " = " + state.synonymsDict[e]) : e in state.aggregatesDict && (a += " = " + state.aggregatesDict[e].join(" and "));
    }
    if (-1 === o) a = "print level to console";
    else if (mouseCoordX >= 0 && mouseCoordY >= 0 && mouseCoordX < screenwidth && mouseCoordY < screenheight - editorRowCount - 2) {
        const t = level.getCellInto(mouseCoordY + n + (mouseCoordX + e) * level.height, _o12);
        null !== (c = state.idDict.filter((e, n) => 0 != t.get(n))) && (a = c.join(", "));
    }
    a.length > 0 && ((ctx.fillStyle = state.fgcolor), (ctx.font = '16px "Source Sans Pro", Helvetica, Arial, sans-serif'), ctx.fillText(a, xoffset, yoffset - 0.4 * cellheight)),
        mouseCoordX >= -1 &&
            mouseCoordY >= -1 &&
            mouseCoordX < screenwidth - 1 &&
            mouseCoordY < screenheight - 1 - editorRowCount &&
            (-1 == mouseCoordX || -1 == mouseCoordY || mouseCoordX == screenwidth - 2 || mouseCoordY === screenheight - 2 - editorRowCount
                ? ctx.drawImage(glyphHighlightResize, xoffset + mouseCoordX * cellwidth, yoffset + mouseCoordY * cellheight)
                : ctx.drawImage(glyphHighlight, xoffset + mouseCoordX * cellwidth, yoffset + mouseCoordY * cellheight));
}
window.addEventListener(
    "resize",
    function () {
        canvasResize();
    },
    !1
),
    (canvas = document.getElementById("gameCanvas")),
    (ctx = canvas.getContext("2d")),
    (x = 0),
    (y = 0);
var oldcellwidth = 0,
    oldcellheight = 0,
    oldtextmode = -1,
    oldfgcolor = -1,
    forceRegenImages = !1;
function canvasResize(e) {
    if (((e ||= level), (canvas.width = canvas.parentNode.clientWidth), (canvas.height = canvas.parentNode.clientHeight), (screenwidth = e.width), (screenheight = e.height), void 0 !== state))
        if (((flickscreen = void 0 !== state.metadata.flickscreen), (zoomscreen = void 0 !== state.metadata.zoomscreen), levelEditorOpened)) {
            screenwidth += 2;
            var n = glyphCount();
            (editorRowCount = Math.ceil(n / (screenwidth - 1))), (screenheight += 2 + editorRowCount);
        } else flickscreen ? ((screenwidth = state.metadata.flickscreen[0]), (screenheight = state.metadata.flickscreen[1])) : zoomscreen && ((screenwidth = state.metadata.zoomscreen[0]), (screenheight = state.metadata.zoomscreen[1]));
    textMode && ((screenwidth = TERMINAL_WIDTH), (screenheight = TERMINAL_HEIGHT)), (cellwidth = canvas.width / screenwidth), (cellheight = canvas.height / screenheight);
    var t = 5,
        o = 5;
    textMode && ((t = 6), (o = font.X.split("\n").map((e) => e.trim()).length));
    (cellwidth = t * Math.max(~~(cellwidth / t), 1)),
        (cellheight = o * Math.max(~~(cellheight / o), 1)),
        (xoffset = 0),
        (yoffset = 0),
        cellwidth / t > cellheight / o
            ? ((cellwidth = (cellheight * t) / o), (xoffset = (canvas.width - cellwidth * screenwidth) / 2), (yoffset = (canvas.height - cellheight * screenheight) / 2))
            : ((cellheight = (cellwidth * o) / t), (yoffset = (canvas.height - cellheight * screenheight) / 2), (xoffset = (canvas.width - cellwidth * screenwidth) / 2)),
        levelEditorOpened && !textMode && ((xoffset += cellwidth), (yoffset += cellheight * (1 + editorRowCount))),
        (cellheight |= 0),
        (xoffset |= 0),
        (yoffset |= 0),
        (oldcellwidth != (cellwidth |= 0) || oldcellheight != cellheight || oldtextmode != textMode || textMode || oldfgcolor != state.fgcolor || forceRegenImages) && ((forceRegenImages = !1), regenSpriteImages()),
        (oldcellheight = cellheight),
        (oldcellwidth = cellwidth),
        (oldtextmode = textMode),
        (oldfgcolor = state.fgcolor),
        redraw();
}
let RandomGen = new RNG();
const intro_template = [
        "..................................",
        "..................................",
        "..................................",
        "......Puzzle Script Terminal......",
        "..............v 1.8...............",
        "..................................",
        "..................................",
        "..................................",
        ".........insert cartridge.........",
        "..................................",
        "..................................",
        "..................................",
        "..................................",
    ],
    messagecontainer_template = [
        "..................................",
        "..................................",
        "..................................",
        "..................................",
        "..................................",
        "..................................",
        "..................................",
        "..................................",
        "..................................",
        "..................................",
        "..........X to continue...........",
        "..................................",
        "..................................",
    ],
    blank_row = "..................................",
    titletemplate_menu_startgame = ["............start game............", "..........#.start game.#..........", "-----------.start game.-----------"],
    titletemplate_menu_newgame = [".............new game.............", "...........#.new game.#...........", "------------.new game.------------"],
    titletemplate_menu_continue = [".............continue.............", "...........#.continue.#...........", "------------.continue.------------"],
    titletemplate_controls = {
        arrows: ".arrow keys to move...............",
        action: ".X to action......................",
        undorestart: ".Z to undo, R to restart..........",
        undo: ".Z to undo........................",
        restart: ".R to restart.....................",
    },
    loading_line = "------------ loading  ------------";
let titleImage = [];
const TERMINAL_WIDTH = intro_template[0].length,
    TERMINAL_HEIGHT = intro_template.length;
let titleScreen = !0,
    titleMode = 0,
    titleSelection = 0,
    titleSelected = !1;
function showContinueOptionOnTitleScreen() {
    return (curlevel > 0 || null !== curlevelTarget) && curlevel in state.levels;
}
function unloadGame() {
    levelEditorOpened && printLevel(), (levelEditorOpened = !1), (state = introstate), (level = new Level(0, 5, 5, 2, null)), (level.objects = new Int32Array(0)), generateTitleScreen(), canvasResize(), redraw();
}
function align_centre(e, n) {
    if (e.length >= n) return e;
    let t = n - e.length,
        o = Math.floor(t / 2),
        r = t - o;
    return ".".repeat(o) + e + ".".repeat(r);
}
function align_right(e, n) {
    return e.length >= n ? e : (e.length < n - 1 && (e += "."), e.padStart(n, "."));
}
function generateTitleScreen() {
    if (((titleMode = showContinueOptionOnTitleScreen() ? 1 : 0), 0 === state.levels.length))
        return document.body.classList.contains("light-theme") ? ((state.fgcolor = "#000000"), (state.bgcolor = "#FFFFFF")) : ((state.fgcolor = "#FFFFFF"), (state.bgcolor = "#000000")), void (titleImage = intro_template);
    let e = "PuzzleScript Game";
    void 0 !== state.metadata.title && (e = state.metadata.title);
    let n = wordwrap(expandCJKCharacters(e), TERMINAL_WIDTH);
    n = n.map((e) => align_centre(e, TERMINAL_WIDTH));
    let t = [];
    void 0 !== state.metadata.author && ((t = wordwrap(expandCJKCharacters("by " + state.metadata.author), TERMINAL_WIDTH)), (t = t.map((e) => align_right(e, TERMINAL_WIDTH))));
    let o = { header: null, menu_options: [], controls: [titletemplate_controls.arrows] };
    0 === titleMode
        ? titleSelected
            ? (o.menu_options.push(blank_row), o.menu_options.push(titletemplate_menu_startgame[2]), o.menu_options.push(blank_row))
            : (o.menu_options.push(blank_row), o.menu_options.push(titletemplate_menu_startgame[1]), o.menu_options.push(blank_row))
        : 0 === titleSelection
        ? (o.menu_options.push(titletemplate_menu_newgame[titleSelected ? 2 : 1]), o.menu_options.push(blank_row), o.menu_options.push(titletemplate_menu_continue[0]))
        : (o.menu_options.push(titletemplate_menu_newgame[0]), o.menu_options.push(blank_row), o.menu_options.push(titletemplate_menu_continue[titleSelected ? 2 : 1]));
    const r = !("noaction" in state.metadata),
        l = !("noundo" in state.metadata),
        i = !("norestart" in state.metadata);
    let s = 0;
    r ? o.controls.push(titletemplate_controls.action) : s++,
        l && i ? o.controls.push(titletemplate_controls.undorestart) : i ? o.controls.push(titletemplate_controls.restart) : l ? o.controls.push(titletemplate_controls.undo) : s++,
        s > 1 && (s--, o.controls.push(blank_row));
    let a = 5 + s,
        c = 0,
        d = n.length,
        u = 0,
        h = t.length,
        g = 0;
    if ((c + d + u + h + g < a && g++, c + d + u + h + g < a && u++, c + d + u + h + g < a && c++, c + d + u + h + g < a && c++, c + d + u + h + g > a)) {
        let e = !1,
            o = !1;
        for (; c + d + u + h + g > a; ) t.length > 1 ? (t.pop(), (o = !0), h--) : n.length > 1 && (n.pop(), (e = !0), d--);
        e && logWarning("Game title is too long to fit on screen; truncating to fit.", state.metadata_lines.title, !0), o && logWarning("Author name is too long to fit on screen; truncating to fit.", state.metadata_lines.author, !0);
    }
    let p = [];
    for (let e = 0; e < c; e++) p.push(blank_row);
    for (let e = 0; e < d; e++) p.push(n[e]);
    for (let e = 0; e < u; e++) p.push(blank_row);
    for (let e = 0; e < h; e++) p.push(t[e]);
    for (let e = 0; e < g; e++) p.push(blank_row);
    o.header = p;
    let f = 0;
    titleImage = [];
    for (let e = 0; e < o.header.length; e++) titleImage.push(o.header[e]);
    for (let e = 0; e < o.menu_options.length; e++) {
        let n = o.menu_options[e];
        -1 !== n.indexOf("---") && (f = titleImage.length), titleImage.push(n);
    }
    titleImage.push(blank_row);
    for (let e = 0; e < o.controls.length; e++) titleImage.push(o.controls[e]);
    for (titleImage.push(blank_row); titleImage.length < TERMINAL_HEIGHT; ) titleImage.push(blank_row);
    titleImage.push(blank_row);
    let m = !1;
    if (titleSelected) {
        let e = get_title_animation_frame() + 2;
        const n = e > 12;
        n && ((titleImage[f] = loading_line), 13 === e && (m = !0)), (e %= 23), e > 11 && (e = 11 - (e % 12));
        const t = 11 - e,
            o = 22 + e;
        if (n) titleImage[f] = titleImage[f].slice(0, t) + "#" + titleImage[f].slice(t + 1, o) + "#" + titleImage[f].slice(o + 1);
        else {
            const e = " ".repeat(t);
            titleImage[f] = e + "#" + titleImage[f].slice(t + 1, o) + "#" + e;
        }
        2 === e && (m = !0);
    }
    for (let e = 0; e < titleImage.length; e++) titleImage[e] = titleImage[e].replace(/\./g, " ");
    m && regenSpriteImages();
}
const introstate = { title: "EMPTY GAME", attribution: "increpare", objectCount: 2, metadata: [], levels: [], bgcolor: "#000000", fgcolor: "#FFFFFF" };
let state = introstate;
function deepClone(e) {
    if (!e) return e;
    let n;
    if (
        ([Number, String, Boolean].forEach(function (t) {
            e instanceof t && (n = t(e));
        }),
        void 0 === n)
    )
        if ("[object Array]" === Object.prototype.toString.call(e))
            (n = []),
                e.forEach(function (e, t, o) {
                    n[t] = deepClone(e);
                });
        else if ("object" == typeof e)
            if (e.nodeType && "function" == typeof e.cloneNode) {
                e.cloneNode(!0);
            } else if (e.prototype) n = e;
            else if (e instanceof Date) n = new Date(e);
            else {
                n = {};
                for (let t in e) n[t] = deepClone(e[t]);
            }
        else n = e;
    return n;
}
function wordwrap(e, n) {
    if (!e) return e;
    let t = ".{1," + (n = n || 75) + "}(\\s|$)|.{" + n + "}|.+$";
    return e.match(RegExp(t, "g"));
}
let splitMessage = [];
function expandCJKCharacters(e) {
    let n = "";
    for (let t = 0; t < e.length; t++) {
        let o = e[t];
        (n += o), font.hasOwnProperty(o) || (n += " ");
    }
    return n;
}
function drawMessageScreen() {
    (titleMode = 0), (textMode = !0), (titleImage = deepClone(messagecontainer_template));
    for (let e = 0; e < titleImage.length; e++) titleImage[e] = titleImage[e].replace(/\./g, " ");
    let e = titleImage[9],
        n = titleImage[10];
    titleImage[10] = e;
    let t,
        o = titleImage[0].length;
    if ("" === messagetext) {
        t = state.levels[curlevel].message.trim();
    } else t = messagetext;
    splitMessage = wordwrap(expandCJKCharacters(t), titleImage[0].length);
    let r = 5 - ((splitMessage.length / 2) | 0);
    r < 0 && (r = 0);
    let l = Math.min(splitMessage.length, 12);
    for (let e = 0; e < l; e++) {
        let n = splitMessage[e],
            t = r + e,
            l = n.length,
            i = ((o - l) / 2) | 0,
            s = titleImage[t];
        titleImage[t] = s.slice(0, i) + n + s.slice(i + n.length);
    }
    let i = 10;
    l >= 10 && (i = l < 12 ? l + 1 : 12), (titleImage[i] = quittingMessageScreen ? e : n), canvasResize();
}
let loadedLevelSeed = 0;
function loadLevelFromLevelDat(e, n, t, o) {
    if (
        (t || (t = (Math.random() + Date.now()).toString()),
        (loadedLevelSeed = t),
        (RandomGen = new RNG(loadedLevelSeed)),
        (forceRegenImages = !0),
        (ignoreNotJustPressedAction = !0),
        (titleScreen = !1),
        (titleMode = showContinueOptionOnTitleScreen() ? 1 : 0),
        (titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0),
        (titleSelected = !1),
        (againing = !1),
        void 0 === n)
    )
        return consolePrint("Trying to access a level that doesn't exist.", !0), void goToTitleScreen();
    void 0 === n.message
        ? ((titleMode = 0),
          (textMode = !1),
          tick_lazy_function_generation(!1),
          (level = n.clone()),
          RebuildLevelArrays(),
          void 0 !== e &&
              (void 0 !== e.metadata.flickscreen
                  ? (oldflickscreendat = [0, 0, Math.min(e.metadata.flickscreen[0], level.width), Math.min(e.metadata.flickscreen[1], level.height)])
                  : void 0 !== e.metadata.zoomscreen && (oldflickscreendat = [0, 0, Math.min(e.metadata.zoomscreen[0], level.width), Math.min(e.metadata.zoomscreen[1], level.height)])),
          (backups = []),
          (restartTarget = backupLevel()),
          (keybuffer = []),
          "run_rules_on_level_start" in e.metadata && ((runrulesonlevelstart_phase = !0), processInput(-1, !0), (runrulesonlevelstart_phase = !1)))
        : ((ignoreNotJustPressedAction = !0), tryPlayShowMessageSound(), drawMessageScreen(), canvasResize()),
        !0 === o && clearInputHistory();
}
function loadLevelFromStateTarget(e, n, t, o) {
    (curlevel = n), (curlevelTarget = t), void 0 === t.message && tryPlayStartLevelSound(), loadLevelFromLevelDat(e, e.levels[n], o), restoreLevel(t), (restartTarget = t);
}
function loadLevelFromState(e, n, t) {
    let o = e.levels[n];
    (curlevel = n), (curlevelTarget = null), void 0 !== o && void 0 === o.message && tryPlayStartLevelSound(), loadLevelFromLevelDat(e, o, t);
}
let sprites = [
    {
        color: "#423563",
        dat: [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
    },
    {
        color: "#252342",
        dat: [
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
        ],
    },
];
function tryPlaySimpleSound(e) {
    if (void 0 !== state.sfx_Events[e]) {
        playSound(state.sfx_Events[e], !0);
    }
}
function tryPlayTitleSound() {
    tryPlaySimpleSound("titlescreen");
}
function tryPlayStartGameSound() {
    tryPlaySimpleSound("startgame");
}
function tryPlayEndGameSound() {
    tryPlaySimpleSound("endgame");
}
function tryPlayCancelSound() {
    tryPlaySimpleSound("cancel");
}
function tryPlayStartLevelSound() {
    tryPlaySimpleSound("startlevel");
}
function tryPlayEndLevelSound() {
    tryPlaySimpleSound("endlevel");
}
function tryPlayUndoSound() {
    tryPlaySimpleSound("undo");
}
function tryPlayRestartSound() {
    tryPlaySimpleSound("restart");
}
function tryPlayShowMessageSound() {
    tryPlaySimpleSound("showmessage");
}
function tryPlayCloseMessageSound() {
    tryPlaySimpleSound("closemessage");
}
generateTitleScreen(), titleMode > 0 && (titleSelection = 1);
let restartTarget,
    backups = [];
function backupLevel() {
    return { dat: new Int32Array(level.objects), width: level.width, height: level.height, oldflickscreendat: oldflickscreendat.concat([]) };
}
function level4Serialization() {
    return { dat: Array.from(level.objects), width: level.width, height: level.height, oldflickscreendat: oldflickscreendat.concat([]) };
}
function setGameState(e, n, t) {
    if (void 0 === e) return void (e = introstate);
    (oldflickscreendat = []),
        (timer = 0),
        (autotick = 0),
        (winning = !1),
        (againing = !1),
        (messageselected = !1),
        (STRIDE_MOV = e.STRIDE_MOV),
        (STRIDE_OBJ = e.STRIDE_OBJ),
        (LAYER_COUNT = e.LAYER_COUNT),
        RebuildGameArrays(),
        (sfxCreateMask = new BitVec(STRIDE_OBJ)),
        (sfxDestroyMask = new BitVec(STRIDE_OBJ)),
        void 0 === n && (n = ["restart"]),
        (0 === state.levels.length || 0 === e.levels.length) && n.length > 0 && "rebuild" === n[0] && (n = ["restart"]),
        void 0 === t && (t = null),
        (RandomGen = new RNG(t)),
        (state = e),
        "rebuild" !== n[0] && (backups = []),
        (sprites = []);
    let o = Object.keys(state.objects),
        r = o.length;
    for (let e = 0; e < r; e++) {
        const n = o[e];
        let t = state.objects[n],
            r = { colors: t.colors, dat: t.spritematrix };
        sprites[t.id] = r;
    }
    switch (
        (void 0 !== state.metadata.realtime_interval ? ((autotick = 0), (autotickinterval = 1e3 * state.metadata.realtime_interval)) : ((autotick = 0), (autotickinterval = 0)),
        (repeatinterval = void 0 !== state.metadata.key_repeat_interval ? 1e3 * state.metadata.key_repeat_interval : 150),
        (againinterval = void 0 !== state.metadata.again_interval ? 1e3 * state.metadata.again_interval : 150),
        throttle_movement &&
            0 === autotickinterval &&
            logWarning("throttle_movement is designed for use in conjunction with realtime_interval. Using it in other situations makes games gross and unresponsive, broadly speaking.  Please don't."),
        (norepeat_action = void 0 !== state.metadata.norepeat_action),
        n[0])
    ) {
        case "restart":
            (winning = !1),
                (timer = 0),
                (titleScreen = !0),
                tryPlayTitleSound(),
                (textMode = !0),
                (titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0),
                (titleSelected = !1),
                (quittingMessageScreen = !1),
                (quittingTitleScreen = !1),
                (messageselected = !1),
                (titleMode = 0),
                showContinueOptionOnTitleScreen() && (titleMode = 1),
                generateTitleScreen();
            break;
        case "rebuild":
            titleScreen && generateTitleScreen();
            break;
        case "loadFirstNonMessageLevel":
            for (let e = 0; e < state.levels.length; e++) {
                if (state.levels[e].hasOwnProperty("message")) continue;
                let n = e;
                (curlevel = n),
                    (curlevelTarget = null),
                    (winning = !1),
                    (timer = 0),
                    (titleScreen = !1),
                    (textMode = !1),
                    tick_lazy_function_generation(!1),
                    (titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0),
                    (titleSelected = !1),
                    (quittingMessageScreen = !1),
                    (quittingTitleScreen = !1),
                    (messageselected = !1),
                    (titleMode = 0),
                    loadLevelFromState(state, n, t);
                break;
            }
            break;
        case "loadLevel": {
            let e = n[1];
            (curlevel = e),
                (curlevelTarget = null),
                (winning = !1),
                (timer = 0),
                (titleScreen = !1),
                (textMode = !1),
                tick_lazy_function_generation(!1),
                (titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0),
                (titleSelected = !1),
                (quittingMessageScreen = !1),
                (quittingTitleScreen = !1),
                (messageselected = !1),
                (titleMode = 0),
                loadLevelFromState(state, e, t);
            break;
        }
        case "levelline": {
            let e = n[1];
            for (let n = state.levels.length - 1; n >= 0; n--) {
                if (state.levels[n].lineNumber <= e + 1) {
                    (curlevel = n),
                        (curlevelTarget = null),
                        (winning = !1),
                        (timer = 0),
                        (titleScreen = !1),
                        (textMode = !1),
                        tick_lazy_function_generation(!1),
                        (titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0),
                        (titleSelected = !1),
                        (quittingMessageScreen = !1),
                        (quittingTitleScreen = !1),
                        (messageselected = !1),
                        (titleMode = 0),
                        loadLevelFromState(state, n);
                    break;
                }
            }
            break;
        }
    }
    "rebuild" !== n[0] && clearInputHistory(), canvasResize(), 0 === state.sounds.length ? killAudioButton() : showAudioButton();
}
function RebuildGameArrays() {
    (_o1 = new BitVec(STRIDE_OBJ)),
        (_o2 = new BitVec(STRIDE_OBJ)),
        (_o2_5 = new BitVec(STRIDE_OBJ)),
        (_o3 = new BitVec(STRIDE_OBJ)),
        (_o4 = new BitVec(STRIDE_OBJ)),
        (_o5 = new BitVec(STRIDE_OBJ)),
        (_o6 = new BitVec(STRIDE_OBJ)),
        (_o7 = new BitVec(STRIDE_OBJ)),
        (_o8 = new BitVec(STRIDE_OBJ)),
        (_o9 = new BitVec(STRIDE_OBJ)),
        (_o10 = new BitVec(STRIDE_OBJ)),
        (_o11 = new BitVec(STRIDE_OBJ)),
        (_o12 = new BitVec(STRIDE_OBJ)),
        (_m1 = new BitVec(STRIDE_MOV)),
        (_m2 = new BitVec(STRIDE_MOV)),
        (_m3 = new BitVec(STRIDE_MOV));
}
function RebuildLevelArrays() {
    (level.movements = new Int32Array(level.n_tiles * STRIDE_MOV)),
        (level.rigidMovementAppliedMask = []),
        (level.rigidGroupIndexMask = []),
        (level.rowCellContents = []),
        (level.rowCellContents_Movements = []),
        (level.colCellContents = []),
        (level.colCellContents_Movements = []),
        (level.mapCellContents = new BitVec(STRIDE_OBJ)),
        (level.mapCellContents_Movements = new BitVec(STRIDE_MOV)),
        (_movementVecs = [new BitVec(STRIDE_MOV), new BitVec(STRIDE_MOV), new BitVec(STRIDE_MOV)]);
    for (let e = 0; e < level.height; e++) level.rowCellContents[e] = new BitVec(STRIDE_OBJ);
    for (let e = 0; e < level.width; e++) level.colCellContents[e] = new BitVec(STRIDE_OBJ);
    for (let e = 0; e < level.height; e++) level.rowCellContents_Movements[e] = new BitVec(STRIDE_MOV);
    for (let e = 0; e < level.width; e++) level.colCellContents_Movements[e] = new BitVec(STRIDE_MOV);
    if (state.rigid) for (let e = 0; e < level.n_tiles; e++) (level.rigidMovementAppliedMask[e] = new BitVec(STRIDE_MOV)), (level.rigidGroupIndexMask[e] = new BitVec(STRIDE_MOV));
}
let messagetext = "";
function applyDiff(e, n) {
    let t = 0;
    for (; t < e.dat.length; ) {
        let o = e.dat[t],
            r = e.dat[t + 1];
        if (0 === r) break;
        for (let l = 0; l < r; l++) n[o + l] = e.dat[t + 2 + l];
        t += 2 + r;
    }
}
function unconsolidateDiff(e, n) {
    if (!e.hasOwnProperty("diff")) return e;
    let t = new Int32Array(n.dat);
    return applyDiff(e, t), { dat: t, width: e.width, height: e.height, oldflickscreendat: e.oldflickscreendat };
}
function restoreLevel(e) {
    let n = e.hasOwnProperty("diff");
    if (((oldflickscreendat = e.oldflickscreendat.concat([])), n ? applyDiff(e, level.objects) : (level.objects = new Int32Array(e.dat)), level.width !== e.width || level.height !== e.height))
        (level.width = e.width), (level.height = e.height), (level.n_tiles = e.width * e.height), RebuildLevelArrays();
    else {
        for (let e = 0; e < level.n_tiles; e++) level.movements[e] = 0;
        if (state.rigid) for (let e = 0; e < level.n_tiles; e++) level.rigidMovementAppliedMask[e].setZero(), level.rigidGroupIndexMask[e].setZero();
        for (let e = 0; e < level.height; e++) {
            level.rowCellContents[e].setZero();
        }
        for (let e = 0; e < level.width; e++) {
            level.colCellContents[e].setZero();
        }
    }
    (againing = !1), (level.commandQueue = []), (level.commandQueueSourceRules = []);
}
let zoomscreen = !1,
    flickscreen = !1,
    screenwidth = 0,
    screenheight = 0;
function consolidateDiff(e, n) {
    if (e.width !== n.width || e.height !== n.height || e.dat.length !== n.dat.length) return e;
    if (e.hasOwnProperty("diff") || n.hasOwnProperty("diff")) return e;
    if (e.dat.length < 1024) return e;
    let t = new Int32Array(128),
        o = 0,
        r = !1,
        l = -1,
        i = e.dat,
        s = n.dat;
    for (let e = 0; e < i.length; e++)
        if (!1 === r) {
            if (i[e] !== s[e]) {
                if (((r = !0), (l = o), t.length < o + 4)) {
                    let e = new Int32Array(2 * t.length);
                    e.set(t), (t = e);
                }
                (t[o + 0] = e), (t[o + 1] = 1), (t[o + 2] = i[e]), (o += 3);
            }
        } else if (i[e] !== s[e]) {
            if (o + 1 >= t.length && t.length < o + 4) {
                let e = new Int32Array(2 * t.length);
                e.set(t), (t = e);
            }
            t[l + 1]++, (t[o] = i[e]), o++;
        } else r = !1;
    return { diff: !0, dat: t, width: e.width, height: e.height, oldflickscreendat: e.oldflickscreendat };
}
function addUndoState(e, a) {
    if(a){
        solution.push(a);
    }
    backups.push(e), backups.length > 2 && !backups[backups.length - 1].hasOwnProperty("diff") && (backups[backups.length - 3] = consolidateDiff(backups[backups.length - 3], backups[backups.length - 2]));
}
function DoRestart(e) {
    !0 !== restarting &&
        ((!0 !== e && "norestart" in state.metadata) ||
            (againing && DoUndo(e, !0),
            (restarting = !0),
            !0 !== e && addUndoState(backupLevel()),
            verbose_logging && consolePrint("--- restarting ---", !0),
            restoreLevel(restartTarget),
            tryPlayRestartSound(),
            "run_rules_on_level_start" in state.metadata && processInput(-1, !0),
            (level.commandQueue = []),
            (level.commandQueueSourceRules = []),
            (restarting = !1)));
}
function backupDiffers() {
    if (0 === backups.length) return !0;
    let e = backups[backups.length - 1];
    if (e.hasOwnProperty("diff")) return 0 !== e.dat.length && 0 !== e.dat[1];
    for (let n = 0; n < level.objects.length; n++) if (level.objects[n] !== e.dat[n]) return !0;
    return !1;
}
function DoUndo(e, n) {
    if (levelEditorOpened || !("noundo" in state.metadata) || !0 === e) {
        if ((verbose_logging && consolePrint("--- undoing ---", !0), n)) for (; !1 === backupDiffers(); ) backups.pop();
        if (backups.length > 0) {
            if(solution.length > 0){
                solution.pop();
            }
            restoreLevel(backups[backups.length - 1]), (backups = backups.splice(0, backups.length - 1)), e || tryPlayUndoSound();
        }
    }
}
function getPlayerPositions() {
    let e = [],
        [n, t] = state.playerMask;
    if (n) for (let n = 0; n < level.n_tiles; n++) level.getCellInto(n, _o11), t.bitsSetInArray(_o11.data) && e.push(n);
    else for (let n = 0; n < level.n_tiles; n++) level.getCellInto(n, _o11), t.anyBitsInCommon(_o11) && e.push(n);
    return e;
}
function getLayersOfMask(e) {
    let n = [];
    for (let t = 0; t < state.objectCount; t++)
        if (e.get(t)) {
            let e = state.idDict[t],
                o = state.objects[e];
            n.push(o.layer);
        }
    return n;
}
let CACHE_MOVEENTITIESATINDEX = {};
function generate_moveEntitiesAtIndex(e, n) {
    const t = `'use strict';\n    let cellMask = level.getCell(positionIndex);\n\t${UNROLL(
        "cellMask &= entityMask",
        e
    )}\n    let layers = getLayersOfMask(cellMask);\n\n\tlet movementMask=_movementVecs[_movementVecIndex];\n\t_movementVecIndex=(_movementVecIndex+1)%_movementVecs.length;\n\t${LEVEL_GET_MOVEMENTS_INTO(
        "positionIndex",
        "movementMask",
        n
    )}\n\n    for (let i=0;i<layers.length;i++) {\n    \t${ISHIFTOR("movementMask", "dirMask", "(5 * layers[i])")}\n    }\n\t\t\n    ${LEVEL_SET_MOVEMENTS(
        "positionIndex",
        "movementMask",
        n
    )}\n\n\tconst colIndex=(positionIndex/level.height)|0;\n\tconst rowIndex=(positionIndex%level.height);\n\t${UNROLL("level.colCellContents_Movements[colIndex] |= movementMask", n)}\n\t${UNROLL(
        "level.rowCellContents_Movements[rowIndex] |= movementMask",
        n
    )}\n\t${UNROLL("level.mapCellContents_Movements |= movementMask", n)}\n\t`;
    return t in CACHE_MOVEENTITIESATINDEX ? CACHE_MOVEENTITIESATINDEX[t] : (CACHE_MOVEENTITIESATINDEX[t] = new Function("level", "positionIndex", "entityMask", "dirMask", t));
}
let CACHE_CALCULATEROWCOLMASKS = {};
function generate_calculateRowColMasks(e, n) {
    const t = `'use strict';\n\t\tlevel.mapCellContents.data.fill(0);\n\t\tlevel.mapCellContents_Movements.data.fill(0);\n\n\t\tfor (let i=0;i<level.width;i++) {\n\t\t\tlet ccc = level.colCellContents[i];\n\t\t\t${SET_ZERO(
        "ccc"
    )}\n\t\t\tlet ccc_Movements = level.colCellContents_Movements[i];\n\t\t\t${SET_ZERO("ccc_Movements")}\n\t\t}\n\n\t\tfor (let i=0;i<level.height;i++) {\n\t\t\tlet rcc = level.rowCellContents[i];\n\t\t\t${SET_ZERO(
        "rcc"
    )}\n\t\t\tlet rcc_Movements = level.rowCellContents_Movements[i];\n\t\t\t${SET_ZERO(
        "rcc_Movements"
    )}\n\t\t}\n\n\t\tfor (let i=0;i<level.width;i++) {\n\t\t\tfor (let j=0;j<level.height;j++) {\n\t\t\t\tlet index = j+i*level.height;\n\t\t\t\tlet cellContents=_o9;\n\t\t\t\t${LEVEL_GET_CELL_INTO(
        "level",
        "index",
        "cellContents",
        e
    )}\n\t\t\t\t${UNROLL("level.mapCellContents |= cellContents", e)}\n\t\t\t\t${UNROLL("level.rowCellContents[j] |= cellContents", e)}\n\t\t\t\t${UNROLL(
        "level.colCellContents[i] |= cellContents",
        e
    )}\n\t\t\t\t\n\t\t\t\tlet mapCellContents_Movements=level.getMovementsInto(index,_m1);\n\t\t\t\t${UNROLL("level.mapCellContents_Movements |= mapCellContents_Movements", n)}\n\t\t\t\t${UNROLL(
        "level.rowCellContents_Movements[j] |= mapCellContents_Movements",
        n
    )}\n\t\t\t\t${UNROLL("level.colCellContents_Movements[i] |= mapCellContents_Movements", n)}\n\t\t\t}\n\t\t}`;
    return t in CACHE_CALCULATEROWCOLMASKS ? CACHE_CALCULATEROWCOLMASKS[t] : (CACHE_CALCULATEROWCOLMASKS[t] = new Function("level", t));
}
function startMovement(e) {
    let n = getPlayerPositions();
    for (let t = 0; t < n.length; t++) {
        let o = n[t];
        state.moveEntitiesAtIndex(level, o, state.playerMask[1], e);
    }
    return n;
}
let dirMasksDelta = { 1: [0, -1], 2: [0, 1], 4: [-1, 0], 8: [1, 0], 15: [0, 0], 16: [0, 0], 3: [0, 0] },
    dirMaskName = { 1: "up", 2: "down", 4: "left", 8: "right", 15: "?", 16: "action", 3: "no" },
    seedsToPlay_CanMove = [],
    seedsToPlay_CantMove = [];
function repositionEntitiesOnLayer(e, n, t) {
    let o = dirMasksDelta[t],
        r = o[0],
        l = o[1],
        i = (e / level.height) | 0,
        s = e % level.height,
        a = level.width - 1,
        c = level.height - 1;
    if ((0 === i && r < 0) || (i === a && r > 0) || (0 === s && l < 0) || (s === c && l > 0)) return !1;
    let d = e + o[1] + o[0] * level.height,
        u = state.layerMasks[n],
        h = level.getCellInto(d, _o7),
        g = level.getCellInto(e, _o8);
    if (u.anyBitsInCommon(h) && 16 != t) return !1;
    for (let t = 0; t < state.sfx_MovementMasks[n].length; t++) {
        let o = state.sfx_MovementMasks[n][t];
        if (o.objectMask.anyBitsInCommon(g)) {
            let n = level.getMovements(e),
                t = o.directionMask;
            n.anyBitsInCommon(t) && -1 === seedsToPlay_CanMove.indexOf(o.seed) && seedsToPlay_CanMove.push(o.seed);
        }
    }
    let p = g.clone();
    g.iclear(u), p.iand(u), h.ior(p), level.setCell(e, g), level.setCell(d, h);
    let f = (d / level.height) | 0,
        m = d % level.height;
    return level.colCellContents[f].ior(p), level.rowCellContents[m].ior(p), !0;
}
let CACHE_REPOSITIONENTITIESATCELL = {};
function generate_repositionEntitiesAtCell(e, n) {
    let t = `'use strict';\n    const movementMask = level.getMovements(positionIndex);\n    if (${IS_ZERO("movementMask", n)}){\n        return false;\n\t}\n\n    let moved = false;\n    ${FOR(
        0,
        LAYER_COUNT,
        (e) =>
            `{\n        const layerMovement = ${GETSHIFTOR(
                "movementMask",
                31,
                5 * e
            )};\n        if (layerMovement !== 0) {\n            const thismoved = repositionEntitiesOnLayer(positionIndex, ${e}, layerMovement);\n            if (thismoved) {\n                ${ISHIFTCLEAR(
                "movementMask",
                "layerMovement",
                5 * e
            )}\n                moved = true;\n            }\n        }\n\t}`
    )}\n\n\t${FOR(
        0,
        n,
        (e) => `\n\t\tlevel.movements[positionIndex * STRIDE_MOV + ${e}] = movementMask.data[${e}];\n\t`
    )}\n\t\t\n\t//corresponding object stuff in repositionEntitiesOnLayer\n\tconst colIndex=(positionIndex/this.height)|0;\n\tconst rowIndex=(positionIndex%level.height);\n\t${UNROLL(
        "level.colCellContents_Movements[colIndex] |= movementMask",
        n
    )}\n\t${UNROLL("level.rowCellContents_Movements[rowIndex] |= movementMask", n)}\n\t${UNROLL("level.mapCellContents_Movements |= movementMask", n)}\n\t\n    return moved;`;
    return t in CACHE_REPOSITIONENTITIESATCELL ? CACHE_REPOSITIONENTITIESATCELL[t] : (CACHE_REPOSITIONENTITIESATCELL[t] = new Function("level", "positionIndex", t));
}
let ellipsisPattern = ["ellipsis"];
function Rule(e) {
    (this.direction = e[0]),
        (this.patterns = e[1]),
        (this.hasReplacements = e[2]),
        (this.lineNumber = e[3]),
        (this.ellipsisCount = e[4]),
        (this.groupNumber = e[5]),
        (this.rigid = e[6]),
        (this.commands = e[7]),
        (this.isRandom = e[8]),
        (this.cellRowMasks = e[9]),
        (this.cellRowMasks_Movements = e[10]),
        (this.ruleMask = new BitVec(STRIDE_OBJ)),
        (this.applyAt = this.generateApplyAt(this.patterns, this.ellipsisCount, STRIDE_OBJ, STRIDE_MOV));
    for (const e of this.cellRowMasks) this.ruleMask.ior(e);
    (this.cellRowMatches = []), lazyFunctionGeneration ? WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.push(this) : this.generate_all_MatchFunctions(), (this.findMatches = this.generateFindMatchesFunction());
}
Rule.prototype.generate_all_MatchFunctions = function () {
    for (let e = 0; e < this.patterns.length; e++) this.cellRowMatches.push(this.generateCellRowMatchesFunction(this.patterns[e], this.ellipsisCount[e]));
};
let CACHE_RULE_CELLROWMATCHESFUNCTION = {};
Rule.prototype.generateCellRowMatchesFunction = function (e, n) {
    if (0 === n) {
        let n = e.length,
            t = new Set(),
            o = new Set();
        for (let r = 0; r < n; r++) {
            const n = e[r];
            for (let e = 0; e < STRIDE_OBJ; e++) (n.objectsPresent.data[e] || n.objectsMissing.data[e]) && t.add(e);
            for (let e = 0; e < n.anyObjectsPresent.length; e++) for (let o = 0; o < STRIDE_OBJ; o++) n.anyObjectsPresent[e].data[o] && t.add(o);
            for (let e = 0; e < STRIDE_MOV; e++) (n.movementsPresent.data[e] || n.movementsMissing.data[e]) && o.add(e);
        }
        let r = "";
        const l = 1 === STRIDE_OBJ ? "" : "*" + STRIDE_OBJ,
            i = 1 === STRIDE_MOV ? "" : "*" + STRIDE_MOV;
        for (let e of t) r += "let cellObjects" + e + " = objects[i" + l + (e ? "+" + e : "") + "];\n";
        for (let e of o) r += "let cellMovements" + e + " = movements[i" + i + (e ? "+" + e : "") + "];\n";
        r += "return " + e[0].generateMatchString("0_");
        for (let e = 1; e < n; e++) r += "&&cellRow[" + e + "].matches(i+" + e + "*d, objects, movements)";
        return (r += ";"), r in CACHE_RULE_CELLROWMATCHESFUNCTION ? CACHE_RULE_CELLROWMATCHESFUNCTION[r] : (CACHE_RULE_CELLROWMATCHESFUNCTION[r] = new Function("cellRow", "i", "d", "objects", "movements", r));
    }
    if (1 === n) {
        let n = e.length,
            t = "let result = [];\nif(cellRow[0].matches(i, objects, movements)",
            o = 1;
        for (; e[o] !== ellipsisPattern; o++) t += "&&cellRow[" + o + "].matches(i+" + o + "*d, objects, movements)";
        for (o++, t += ") {\n\tfor (let k=kmin;k<kmax;k++) {\n\t\tif(cellRow[" + o + "].matches((i+d*(k+" + (o - 1) + ")), objects, movements)", o++; o < n; o++)
            t += "&&cellRow[" + o + "].matches((i+d*(k+" + (o - 1) + ")), objects, movements)";
        return (
            (t += "){\n\t\t\tresult.push([i,k]);\n\t\t}\n\t}\n}\n"),
            (t += "return result;"),
            t in CACHE_RULE_CELLROWMATCHESFUNCTION ? CACHE_RULE_CELLROWMATCHESFUNCTION[t] : (CACHE_RULE_CELLROWMATCHESFUNCTION[t] = new Function("cellRow", "i", "kmax", "kmin", "d", "objects", "movements", t))
        );
    }
    {
        let n = e.length,
            t = -1,
            o = -1;
        for (let r = 0; r < n; r++)
            if (e[r] === ellipsisPattern) {
                if (-1 !== t) {
                    o = r;
                    break;
                }
                t = r;
            }
        let r = "let result = [];\nif(cellRow[0].matches(i, objects, movements)";
        for (let e = 1; e < t; e++) r += "&&cellRow[" + e + "].matches(i+" + e + "*d, objects, movements)";
        (r += ") {\n"), (r += "\n\tfor (let k1=k1min;k1<k1max;k1++) {\n\t\tif(cellRow[" + (t + 1) + "].matches((i+d*(k1+" + (t + 1 - 1) + ")), objects, movements)");
        for (let e = t + 2; e < o; e++) r += "&&cellRow[" + e + "].matches((i+d*(k1+" + (e - 1) + ")), objects, movements)";
        (r += "\t\t){\n"), (r += "\n\t\t\tfor (let k2=k2min;k1+k2<kmax && k2<k2max;k2++) {\n\t\t\t\tif(cellRow[" + (o + 1) + "].matches((i+d*(k1+k2+" + (o + 1 - 2) + ")), objects, movements)");
        for (let e = o + 2; e < n; e++) r += "&&cellRow[" + e + "].matches((i+d*(k1+k2+" + (e - 2) + ")), objects, movements)";
        return (
            (r += "\n\t\t\t\t){\n\t\t\t\t\tresult.push([i,k1,k2]);\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\t\t\t\n}\t\nreturn result;"),
            r in CACHE_RULE_CELLROWMATCHESFUNCTION
                ? CACHE_RULE_CELLROWMATCHESFUNCTION[r]
                : (CACHE_RULE_CELLROWMATCHESFUNCTION[r] = new Function("cellRow", "i", "kmax", "kmin", "k1max", "k1min", "k2max", "k2min", "d", "objects", "movements", r))
        );
    }
};
let STRIDE_OBJ = 1,
    STRIDE_MOV = 1,
    LAYER_COUNT = 1;
const FALSE_FUNCTION = new Function("return false;");
function CellPattern(e) {
    (this.objectsPresent = e[0]),
        (this.objectsMissing = e[1]),
        (this.anyObjectsPresent = e[2]),
        (this.movementsPresent = e[3]),
        (this.movementsMissing = e[4]),
        lazyFunctionGeneration ? WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.push(this) : (this.matches = this.generateMatchFunction()),
        (this.replacement = e[5]);
}
function CellReplacement(e) {
    (this.objectsClear = e[0]), (this.objectsSet = e[1]), (this.movementsClear = e[2]), (this.movementsSet = e[3]), (this.movementsLayerMask = e[4]), (this.randomEntityMask = e[5]), (this.randomDirMask = e[6]), (this.replace = null);
}
(CellPattern.prototype.replace = function (e, n, t) {
    let o = this.generateReplaceFunction(STRIDE_OBJ, STRIDE_MOV, n);
    return (this.replace = o), this.replace(e, n, t);
}),
    (CellPattern.prototype.generateMatchString = function () {
        let e = "(true";
        for (let n = 0; n < Math.max(STRIDE_OBJ, STRIDE_MOV); ++n) {
            const t = "cellObjects" + n,
                o = "cellMovements" + n,
                r = this.objectsPresent.data[n],
                l = this.objectsMissing.data[n],
                i = this.movementsPresent.data[n],
                s = this.movementsMissing.data[n];
            r && (e += r & (r - 1) ? "\t\t&& ((" + t + "&" + r + ")===" + r + ")\n" : "\t\t&& (" + t + "&" + r + ")\n"),
                l && (e += "\t\t&& !(" + t + "&" + l + ")\n"),
                i && (e += i & (i - 1) ? "\t\t&& ((" + o + "&" + i + ")===" + i + ")\n" : "\t\t&& (" + o + "&" + i + ")\n"),
                s && (e += "\t\t&& !(" + o + "&" + s + ")\n");
        }
        for (let n = 0; n < this.anyObjectsPresent.length; n++) {
            e += "\t\t&& (0";
            for (let t = 0; t < STRIDE_OBJ; ++t) {
                const o = this.anyObjectsPresent[n].data[t];
                o && (e += "|(cellObjects" + t + "&" + o + ")");
            }
            e += ")";
        }
        return (e += "\t)"), e;
    });
let _o1,
    _o2,
    _o2_5,
    _o3,
    _o4,
    _o5,
    _o6,
    _o7,
    _o8,
    _o9,
    _o10,
    _o11,
    _o12,
    _m1,
    _m2,
    _m3,
    CACHE_CELLPATTERN_MATCHFUNCTION = new Map(),
    _generateMatchFunction_key_array = new Int32Array(0);
CellPattern.prototype.generateMatchFunction = function () {
    const e = 2 * STRIDE_OBJ + 2 * STRIDE_MOV + this.anyObjectsPresent.length * STRIDE_OBJ + 2;
    e !== _generateMatchFunction_key_array.length && (_generateMatchFunction_key_array = new Int32Array(e));
    const n = _generateMatchFunction_key_array;
    let t = 0;
    for (let e = 0; e < STRIDE_OBJ; e++) (n[t++] = this.objectsPresent.data[e] || 0), (n[t++] = this.objectsMissing.data[e] || 0);
    for (let e = 0; e < STRIDE_MOV; e++) (n[t++] = this.movementsPresent.data[e] || 0), (n[t++] = this.movementsMissing.data[e] || 0);
    for (let e = 0; e < this.anyObjectsPresent.length; e++) for (let o = 0; o < STRIDE_OBJ; o++) n[t++] = this.anyObjectsPresent[e].data[o] || 0;
    (n[t++] = STRIDE_OBJ), (n[t++] = STRIDE_MOV);
    let o = n.toString();
    if (CACHE_CELLPATTERN_MATCHFUNCTION.has(o)) return CACHE_CELLPATTERN_MATCHFUNCTION.get(o);
    const r = 1 === STRIDE_OBJ ? "" : `*${STRIDE_OBJ}`,
        l = 1 === STRIDE_MOV ? "" : `*${STRIDE_MOV}`;
    let i = "";
    for (let e = 0; e < STRIDE_OBJ; ++e) i += `const cellObjects${e} = objects[i${r}${e ? "+" + e : ""}];\n`;
    for (let e = 0; e < STRIDE_MOV; ++e) i += `const cellMovements${e} = movements[i${l}${e ? "+" + e : ""}];\n`;
    i += `return ${this.generateMatchString()};`;
    const s = new Function("i", "objects", "movements", i);
    return CACHE_CELLPATTERN_MATCHFUNCTION.set(o, s), s;
};
let CACHE_CELLPATTERN_REPLACEFUNCTION = {},
    CACHE_CHECK_COUNT = 0,
    CACHE_HIT_COUNT = 0,
    _replace_function_key_array = new Int32Array(0);
CellPattern.prototype.generateReplaceFunction = function (e, n, t) {
    if (null === this.replacement) return FALSE_FUNCTION;
    const o = 3 * e + 4 * n + 3;
    o !== _replace_function_key_array.length && (_replace_function_key_array = new Int32Array(o));
    const r = _replace_function_key_array;
    for (let t = 0; t < e; t++) (r[t] = this.replacement.objectsSet.data[t] || 0), (r[t + e] = this.replacement.objectsClear.data[t] || 0), (r[t + 2 * e + 3 * n] = this.replacement.randomEntityMask.data[t] || 0);
    for (let t = 0; t < n; t++)
        (r[t + 2 * e] = this.replacement.movementsSet.data[t] || 0),
            (r[t + 2 * e + n] = this.replacement.movementsClear.data[t] || 0),
            (r[t + 2 * e + 2 * n] = this.replacement.movementsLayerMask.data[t] || 0),
            (r[t + 3 * e + 3 * n] = this.replacement.randomDirMask.data[t] || 0);
    (r[3 * e + 4 * n] = e), (r[3 * e + 4 * n + 1] = n), (r[3 * e + 4 * n + 2] = t.rigid);
    const l = r.toString();
    if (l in CACHE_CELLPATTERN_REPLACEFUNCTION) return CACHE_CELLPATTERN_REPLACEFUNCTION[l];
    const i = this.replacement.randomEntityMask.iszero(),
        s = this.replacement.randomDirMask.iszero();
    let a = `\t\n\t\tlet replace = this.replacement;\n\n\t\tif (replace === null) {\n\t\t\treturn false;\n\t\t}\n\n\t\tconst replace_RandomEntityMask = replace.randomEntityMask;\n\t\tconst replace_RandomDirMask = replace.randomDirMask;\n\n\t\t// Using IMPORT_COMPILE_TIME_ARRAY should make the following three declarations faster,\n\t\t// but it really slows down the compiler.\n\t\tconst objectsSet = _o1;\t\n\t\t${UNROLL(
        "objectsSet = replace.objectsSet",
        e
    )}\n\t\n\t\tconst objectsClear = _o2;\n\t\t${UNROLL("objectsClear = replace.objectsClear", e)}\n\n\t\tconst movementsSet = _m1;\n\t\t${UNROLL(
        "movementsSet = replace.movementsSet",
        n
    )}\n\t\t\n\t\tconst movementsClear = _m2;\n\t\t\n\t\t${FOR(0, n, (e) => `movementsClear.data[${e}] = ${this.replacement.movementsClear.data[e] | this.replacement.movementsLayerMask.data[e]};\n`)}\n\n\t\t${IF_LAZY(
        !i,
        () =>
            `\n\t\t\tconst choices=[];\n\t\t\t${FOR(
                0,
                32 * e,
                (e) => `{\n\t\t\t\tif (${this.replacement.randomEntityMask.get(e)}) {\n\t\t\t\t\tchoices.push(${e});\n\t\t\t\t}\n\t\t\t}`
            )}\n\t\t\tconst rand = choices[Math.floor(RandomGen.uniform() * choices.length)];\n\t\t\tconst n = state.idDict[rand];\n\t\t\tconst o = state.objects[n];\n\t\t\t${IBITSET("objectsSet", "rand")}\n\t\t\t${UNROLL(
                "objectsClear |= state.layerMasks[o.layer]",
                e
            )}\n\t\t\t${ISHIFTOR("movementsClear", "0x1f", "(5 * o.layer)")}\n\t\t`
    )}\n\t\t${IF_LAZY(
        !s,
        () =>
            `\n\t\t\t${FOR(
                0,
                LAYER_COUNT,
                (e) => `{\n\t\t\t\tif (${GET("replace_RandomDirMask", 5 * e)}) {\n\t\t\t\t\tconst randomDir = Math.floor(RandomGen.uniform()*4);\n\t\t\t\t\t${IBITSET("movementsSet", `(randomDir + 5 * ${e})`)}\n\t\t\t\t}\n\t\t\t}`
            )}\n\t\t`
    )}\n\t\t\n\n\n\t\tconst oldCellMask = _o3;\n\t\t${LEVEL_GET_CELL_INTO("level", "currentIndex", "oldCellMask", e)}\n\n\t\tconst curCellMask = _o2_5;\n\t\t${FOR(
        0,
        e,
        (e) => `\n\t\t\tcurCellMask.data[${e}] = (oldCellMask.data[${e}] & (~objectsClear.data[${e}])) | objectsSet.data[${e}];\n\t\t`
    )}\n\n\t\tconst oldMovementMask = level.getMovements(currentIndex);\n\t\tconst curMovementMask = _m3;\n\t\t${FOR(
        0,
        n,
        (e) => `\n\t\t\tcurMovementMask.data[${e}] = (oldMovementMask.data[${e}] & (~movementsClear.data[${e}])) | movementsSet.data[${e}]\n\t\t`
    )}\n\n\n\t\tlet curRigidGroupIndexMask;\n\t\tlet curRigidMovementAppliedMask;\n\t\tlet rigidchange=false;\t\t\n\t\t${IF_LAZY(
        t.rigid,
        () =>
            `\n\t\t\tconst rigidGroupIndex = ${state.groupNumber_to_RigidGroupIndex[t.groupNumber] + 1};\n\t\t\tconst rigidMask = new BitVec(${STRIDE_MOV});\n\t\t\t${FOR(
                0,
                level.layerCount,
                (e) => `\n\t\t\t\t${ISHIFTOR("rigidMask", "rigidGroupIndex", 5 * e)}\n\t\t\t`
            )}\n\t\t\t${UNROLL(
                "rigidMask &= replace.movementsLayerMask",
                n
            )}\n\t\t\t\n\t\t\tcurRigidGroupIndexMask = level.rigidGroupIndexMask[currentIndex] || new BitVec(${STRIDE_MOV});\n\t\t\tcurRigidMovementAppliedMask = level.rigidMovementAppliedMask[currentIndex] || new BitVec(${STRIDE_MOV});\n\n\t\t\tif (${NOT_BITS_SET_IN_ARRAY(
                "rigidMask",
                "curRigidGroupIndexMask.data",
                n
            )} &&\n\t\t\t\t${NOT_BITS_SET_IN_ARRAY("replace.movementsLayerMask", "curRigidMovementAppliedMask.data", n)}) \n\t\t\t{\n\t\t\t\t${UNROLL("curRigidGroupIndexMask |= rigidMask", n)}\n\t\t\t\t${UNROLL(
                "curRigidMovementAppliedMask |= replace.movementsLayerMask",
                n
            )}\n\t\t\t\trigidchange=true;\n\t\t\t}\n\t\t`
    )}\n\t\t\n\t\tif (${EQUALS("oldCellMask", "curCellMask", e)} \n\t\t\t&& ${EQUALS(
        "oldMovementMask",
        "curMovementMask",
        n
    )} \n\t\t\t&& !rigidchange) { \n\t\t\t//nothing changed\n\t\t\treturn false;\n\t\t}\n\t\t\t\t\n\t\tif (rigidchange) {\n\t\t\tlevel.rigidGroupIndexMask[currentIndex] = curRigidGroupIndexMask;\n\t\t\tlevel.rigidMovementAppliedMask[currentIndex] = curRigidMovementAppliedMask;\n\t\t}\n\n\t\tconst created = _o4;\n\t\t${UNROLL(
        "created = curCellMask",
        e
    )}\n\t\t${UNROLL("created &= ~oldCellMask", e)}\n\t\t${UNROLL("sfxCreateMask |= created", e)}\n\t\t\n\t\tconst destroyed = _o5;\n\t\t${UNROLL("destroyed = oldCellMask", e)}\n\t\t${UNROLL("destroyed &= ~curCellMask", e)}\n\t\t${UNROLL(
        "sfxDestroyMask |= destroyed",
        e
    )}\n\n\t\t${LEVEL_SET_CELL("level", "currentIndex", "curCellMask", e)}\n\t\t${LEVEL_SET_MOVEMENTS(
        "currentIndex",
        "curMovementMask",
        n
    )}\n\n\t\tconst colIndex=(currentIndex/level.height)|0;\n\t\tconst rowIndex=(currentIndex%level.height);\n\n\t\t${UNROLL("level.colCellContents[colIndex] |= curCellMask", e)}\n\t\t${UNROLL(
        "level.rowCellContents[rowIndex] |= curCellMask",
        e
    )}\n\t\t${UNROLL("level.mapCellContents |= curCellMask", e)}\n\t\treturn true;\t\n\t`;
    return (CACHE_CELLPATTERN_REPLACEFUNCTION[l] = new Function("level", "rule", "currentIndex", a));
};
let CACHE_MATCHCELLROW = {};
function generateMatchCellRow(e, n) {
    const t = `'use strict';\n\tlet result=[];\n\t\n\tif ((${NOT_BITS_SET_IN_ARRAY("cellRowMask", "level.mapCellContents.data", e)})||\n\t(${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask_Movements",
        "level.mapCellContents_Movements.data",
        n
    )})) {\n\t\treturn result;\n\t}\n\n\tlet xmin=0;\n\tlet xmax=level.width;\n\tlet ymin=0;\n\tlet ymax=level.height;\n\n    let len=cellRow.length;\n\n    switch(direction) {\n    \tcase 1://up\n    \t{\n    \t\tymin+=(len-1);\n    \t\tbreak;\n    \t}\n    \tcase 2: //down \n    \t{\n\t\t\tymax-=(len-1);\n\t\t\tbreak;\n    \t}\n    \tcase 4: //left\n    \t{\n    \t\txmin+=(len-1);\n    \t\tbreak;\n    \t}\n    \tcase 8: //right\n\t\t{\n\t\t\txmax-=(len-1);\t\n\t\t\tbreak;\n\t\t}\n    \tdefault:\n    \t{\n    \t\twindow.console.log("EEEP "+direction);\n    \t}\n    }\n\n    const horizontal=direction>2;\n    if (horizontal) {\n\t\tfor (let y=ymin;y<ymax;y++) {\n\t\t\tif (${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask",
        "level.rowCellContents[y].data",
        e
    )} \n\t\t\t|| ${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask_Movements",
        "level.rowCellContents_Movements[y].data",
        n
    )}) {\n\t\t\t\tcontinue;\n\t\t\t}\n\n\t\t\tfor (let x=xmin;x<xmax;x++) {\n\t\t\t\tconst i = x*level.height+y;\n\t\t\t\tif (cellRowMatch(cellRow,i,d, level.objects, level.movements))\n\t\t\t\t{\n\t\t\t\t\tresult.push(i);\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t} else {\n\t\tfor (let x=xmin;x<xmax;x++) {\n\t\t\tif (${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask",
        "level.colCellContents[x].data",
        e
    )}\n\t\t\t|| ${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask_Movements",
        "level.colCellContents_Movements[x].data",
        n
    )}) {\n\t\t\t\tcontinue;\n\t\t\t}\n\n\t\t\tfor (let y=ymin;y<ymax;y++) {\n\t\t\t\tconst i = x*level.height+y;\n\t\t\t\tif (cellRowMatch(cellRow,i, d, level.objects, level.movements))\n\t\t\t\t{\n\t\t\t\t\tresult.push(i);\n\t\t\t\t}\n\t\t\t}\n\t\t}\t\t\n\t}\n\n\treturn result;`;
    return t in CACHE_MATCHCELLROW ? CACHE_MATCHCELLROW[t] : (CACHE_MATCHCELLROW[t] = new Function("level", "direction", "cellRowMatch", "cellRow", "cellRowMask", "cellRowMask_Movements", "d", t));
}
let CACHE_MATCHCELLROWWILDCARD = {};
function generateMatchCellRowWildCard(e, n) {
    const t = `'use strict';\n\tlet result=[];\n\tif ((${NOT_BITS_SET_IN_ARRAY("cellRowMask", "level.mapCellContents.data", e)})||\n\t(${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask_Movements",
        "level.mapCellContents_Movements.data",
        n
    )})) {\n\t\treturn result;\n\t}\n\t\n\tlet xmin=0;\n\tlet xmax=level.width;\n\tlet ymin=0;\n\tlet ymax=level.height;\n\n\tlet len=cellRow.length-wildcardCount;//remove one to deal with wildcard\n    switch(direction) {\n    \tcase 1://up\n    \t{\n    \t\tymin+=(len-1);\n    \t\tbreak;\n    \t}\n    \tcase 2: //down \n    \t{\n\t\t\tymax-=(len-1);\n\t\t\tbreak;\n    \t}\n    \tcase 4: //left\n    \t{\n    \t\txmin+=(len-1);\n    \t\tbreak;\n    \t}\n    \tcase 8: //right\n\t\t{\n\t\t\txmax-=(len-1);\t\n\t\t\tbreak;\n\t\t}\n    \tdefault:\n    \t{\n    \t\twindow.console.log("EEEP2 "+direction);\n    \t}\n    }\n\n    const horizontal=direction>2;\n    if (horizontal) {\n\t\tfor (let y=ymin;y<ymax;y++) {\n\t\t\tif (${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask",
        "level.rowCellContents[y].data",
        e
    )}\n\t\t\t|| ${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask_Movements",
        "level.rowCellContents_Movements[y].data",
        n
    )}) {\n\t\t\t\tcontinue;\n\t\t\t}\n\n\t\t\tfor (let x=xmin;x<xmax;x++) {\n\t\t\t\tconst i = x*level.height+y;\n\t\t\t\tlet kmax;\n\n\t\t\t\tif (direction === 4) { //left\n\t\t\t\t\tkmax=x-len+2;\n\t\t\t\t} else if (direction === 8) { //right\n\t\t\t\t\tkmax=level.width-(x+len)+1;\t\n\t\t\t\t} else {\n\t\t\t\t\twindow.console.log("EEEP2 "+direction);\t\t\t\t\t\n\t\t\t\t}\n\n\t\t\t\tif (wildcardCount===1) {\n\t\t\t\t\tresult.push.apply(result, cellRowMatch(cellRow,i,kmax,0, d, level.objects, level.movements));\n\t\t\t\t} else {\n\t\t\t\t\tresult.push.apply(result, cellRowMatch(cellRow,i,kmax,0,kmax,0,kmax,0, d, level.objects, level.movements));\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t} else {\n\t\tfor (let x=xmin;x<xmax;x++) {\n\t\t\tif (${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask",
        "level.colCellContents[x].data",
        e
    )}\n\t\t\t|| ${NOT_BITS_SET_IN_ARRAY(
        "cellRowMask_Movements",
        "level.colCellContents_Movements[x].data",
        n
    )}) {\n\t\t\t\tcontinue;\n\t\t\t}\n\n\t\t\tfor (let y=ymin;y<ymax;y++) {\n\t\t\t\tconst i = x*level.height+y;\n\t\t\t\tlet kmax;\n\n\t\t\t\tif (direction === 2) { // down\n\t\t\t\t\tkmax=level.height-(y+len)+1;\n\t\t\t\t} else if (direction === 1) { // up\n\t\t\t\t\tkmax=y-len+2;\t\t\t\t\t\n\t\t\t\t} else {\n\t\t\t\t\twindow.console.log("EEEP2 "+direction);\n\t\t\t\t}\n\t\t\t\tif (wildcardCount===1) {\n\t\t\t\t\tresult.push.apply(result, cellRowMatch(cellRow,i,kmax,0, d, level.objects, level.movements));\n\t\t\t\t} else {\n\t\t\t\t\tresult.push.apply(result, cellRowMatch(cellRow,i,kmax,0, kmax,0, kmax,0, d, level.objects, level.movements));\n\t\t\t\t}\n\t\t\t}\n\t\t}\t\t\n\t}\n\n\treturn result;`;
    return t in CACHE_MATCHCELLROWWILDCARD ? CACHE_MATCHCELLROWWILDCARD[t] : (CACHE_MATCHCELLROWWILDCARD[t] = new Function("direction", "cellRowMatch", "cellRow", "cellRowMask", "cellRowMask_Movements", "d", "wildcardCount", t));
}
function generateTuples(e) {
    let n = [[]];
    for (let t = 0; t < e.length; t++) {
        const o = e[t],
            r = [];
        for (let e = 0; e < o.length; e++) {
            let t = o[e];
            for (let e = 0; e < n.length; e++) {
                const o = n[e].concat([t]);
                r.push(o);
            }
        }
        n = r;
    }
    return n;
}
function IF(e) {
    return e ? "" : "/*";
}
function IF_LAZY(e, n) {
    return e ? n() : "";
}
function IF_ELSE_LAZY(e, n, t) {
    return e ? n() : t();
}
function ENDIF(e) {
    return e ? "" : "*/";
}
function ELSE(e) {
    return e ? "/*" : "*/";
}
function ENDELSE(e) {
    return e ? "*/" : "";
}
function FOR(e, n, t) {
    let o = "";
    for (let r = e; r < n; r++) o += t(r);
    return o;
}
(Rule.prototype.findMatches = function () {
    if (!this.ruleMask.bitsSetInArray(level.mapCellContents.data)) return [];
    const e = level.delta_index(this.direction);
    let n = [];
    const t = this.cellRowMasks,
        o = this.cellRowMasks_Movements;
    for (let r = 0; r < this.patterns.length; r++) {
        const l = this.patterns[r],
            i = this.cellRowMatches[r];
        let s;
        if (((s = 0 === this.ellipsisCount[r] ? state.matchCellRow(level, this.direction, i, l, t[r], o[r], e) : state.matchCellRowWildCard(this.direction, i, l, t[r], o[r], e, this.ellipsisCount[r])), 0 === s.length)) return [];
        n.push(s);
    }
    return n;
}),
    (Rule.prototype.directional = function () {
        for (let e = 0; e < state.rules.length; e++) {
            const n = state.rules[e];
            let t = 0;
            for (let e = 0; e < n.length; e++) if ((this.lineNumber === n[e].lineNumber && t++, t > 1)) return !0;
        }
        return !1;
    });
let CACHE_RULE_APPLYAT = {};
function showTempMessage() {
    (keybuffer = []), (textMode = !0), (titleScreen = !1), (quittingMessageScreen = !1), (messageselected = !1), (ignoreNotJustPressedAction = !0), tryPlayShowMessageSound(), drawMessageScreen(), canvasResize();
}
function processOutputCommands(e) {
    for (let n = 0; n < e.length; n++) {
        let t = e[n];
        "f" === t.charAt(1) && tryPlaySimpleSound(t), !1 === unitTesting && "message" === t && showTempMessage();
    }
}
function applyRandomRuleGroup(e, n) {
    let t = [];
    for (let o = 0; o < n.length; o++) {
        let r = n[o].findMatches(e);
        if (r.length > 0) {
            let e = generateTuples(r);
            for (let n = 0; n < e.length; n++) {
                let r = e[n];
                t.push([o, r]);
            }
        }
    }
    if (0 === t.length) return !1;
    let o = t[Math.floor(RandomGen.uniform() * t.length)],
        r = n[o[0]],
        l = o[1];
    const i = e.delta_index(r.direction);
    let s = r.applyAt(e, l, !1, i);
    return r.queueCommands(), s;
}
function applyRuleGroup(e) {
    if (e[0].isRandom) return applyRandomRuleGroup(level, e);
    const n = e.length,
        t = verbose_logging;
    let o = !1,
        r = !0,
        l = 0;
    for (; r && l++ < 200; ) {
        r = !1;
        let l = 0;
        for (let t = 0; t < n; t++) {
            if (e[t].tryApply(level)) (r = !0), (l = 0);
            else if ((l++, l === n)) break;
        }
        r && ((o = !0), t && (debugger_turnIndex++, addToDebugTimeline(level, -2)));
    }
    return l >= 200 && logErrorCacheable("Got caught looping lots in a rule group :O", e[0].lineNumber, !0), o;
}
function applyRules(e, n, t) {
    let o = !1,
        r = 0,
        l = 0;
    const i = e.length;
    for (; l < i; )
        if (((t && t[l]) || (o = applyRuleGroup(e[l]) || o), o && void 0 !== n[l])) {
            if (((l = n[l]), (o = !1), r++, r > 200)) {
                logErrorCacheable("got caught in an endless startloop...endloop vortex, escaping!", e[l][0].lineNumber, !0);
                break;
            }
            verbose_logging && (debugger_turnIndex++, addToDebugTimeline(level, -2));
        } else {
            if ((l++, l === e.length && o && void 0 !== n[l] && ((l = n[l]), (o = !1), r++, r > 200))) {
                logErrorCacheable("got caught in an endless startloop...endloop vortex, escaping!", e[l][0].lineNumber, !0);
                break;
            }
            verbose_logging && (debugger_turnIndex++, addToDebugTimeline(level, -2));
        }
}
(Rule.prototype.generateApplyAt = function (e, n, t, o) {
    const r = `'use strict';\n\t//have to double check they apply \n\t//(cf test ellipsis bug: rule matches two candidates, first replacement invalidates second)\n\tif (check)\n\t{\n\t${FOR(
        0,
        e.length,
        (e) =>
            `\n\t\t{\n\t\t\t${IF(
                0 === n[e]
            )}\n\t\t\t\tif ( ! this.cellRowMatches[${e}](\n\t\t\t\t\tthis.patterns[${e}], \n\t\t\t\t\t\ttuple[${e}], \n\t\t\t\t\t\tdelta, level.objects, level.movements\n\t\t\t\t\t\t) )\n\t\t\t\treturn false\n\t\t\t${ENDIF(
                0 === n[e]
            )}\n\t\t\t${IF(
                1 === n[e]
            )}\n\t\t\t\tif ( this.cellRowMatches[${e}](\n\t\t\t\t\t\tthis.patterns[${e}], \n\t\t\t\t\t\ttuple[${e}][0], \n\t\t\t\t\t\ttuple[${e}][1]+1, \n\t\t\t\t\t\t\ttuple[${e}][1], \n\t\t\t\t\t\tdelta, level.objects, level.movements\n\t\t\t\t\t).length === 0 )\n\t\t\t\t\treturn false\n\t\t\t${ENDIF(
                1 === n[e]
            )}\n\t\t\t${IF(
                2 === n[e]
            )}\n\t\t\t\tif ( this.cellRowMatches[${e}](\n\t\t\t\t\t\tthis.patterns[${e}], \n\t\t\t\t\t\ttuple[${e}][0],  \n\t\t\t\t\t\ttuple[${e}][1]+tuple[${e}][2]+1, \n\t\t\t\t\t\t\ttuple[${e}][1]+tuple[${e}][2], \n\t\t\t\t\t\ttuple[${e}][1]+1, \n\t\t\t\t\t\t\ttuple[${e}][1],  \n\t\t\t\t\t\ttuple[${e}][2]+1, \n\t\t\t\t\t\t\ttuple[${e}][2], \n\t\t\t\t\t\t\tdelta, level.objects, level.movements\n\t\t\t\t\t\t).length === 0 )\n\t\t\t\t\treturn false\n\t\t\t${ENDIF(
                2 === n[e]
            )}\n\t\t}`
    )}\n\t}\n\n    let result=false;\n\tlet anyellipses=false;\n\n    //APPLY THE RULE\n\t${FOR(0, e.length, (t) => {
        const o = e[t];
        return `\n\t\t\t{\n\t\t\t\tlet ellipse_index=0;\n\t\t\t\tlet currentIndex = ${n[t] > 0 ? `tuple[${t}][0]` : `tuple[${t}]`}\n\t\t\t\t${FOR(
            0,
            o.length,
            (e) =>
                `\n\t\t\t\t\t{\n\t\t\t\t\t\t${IF(
                    o[e] === ellipsisPattern
                )}\n\t\t\t\t\t\t\tconst k = tuple[${t}][1+ellipse_index];\n\t\t\t\t\t\t\tellipse_index++;\n\t\t\t\t\t\t\tanyellipses=true;\n\t\t\t\t\t\t\tcurrentIndex += delta*k;\n\t\t\t\t\t\t${ELSE(
                    o[e] === ellipsisPattern
                )}\n\t\t\t\t\t\t\tconst preCell = this.patterns[${t}][${e}];\n\t\t\t\t\t\t\tresult = preCell.replace(level,this, currentIndex) || result;\n\t\t\t\t\t\t\tcurrentIndex += delta;\n\t\t\t\t\t\t${ENDELSE(
                    o[e] === ellipsisPattern
                )}\n\t\t\t\t\t}\n\t\t\t\t`
        )}\n\t\t\t}`;
    })}\n\n\tif (verbose_logging && result){\n\t\tlet ruleDirection = dirMaskName[this.direction];\n\t\tif (!this.directional()){\n\t\t\truleDirection="";\n\t\t}\n\n\t\tlet inspect_ID =  addToDebugTimeline(level,this.lineNumber);\n\t\tlet gapMessage="";\n\t\t\n\t\tlet logString = '<font color="green">Rule <a onclick="jumpToLine(' + this.lineNumber + ');" href="javascript:void(0);">' + this.lineNumber + '</a> ' + ruleDirection + ' applied' + gapMessage + '.</font>';\n\t\tconsolePrint(logString,false,this.lineNumber,inspect_ID);\n\t\t\n\t}\n\n    return result;\n\t`;
    return r in CACHE_RULE_APPLYAT ? CACHE_RULE_APPLYAT[r] : (CACHE_RULE_APPLYAT[r] = new Function("level", "tuple", "check", "delta", r));
}),
    (Rule.prototype.tryApply = function (e) {
        const n = e.delta_index(this.direction);
        let t = this.findMatches(e);
        if (0 === t.length) return !1;
        let o = !1;
        if (this.hasReplacements) {
            let r = generateTuples(t);
            for (let t = 0; t < r.length; t++) {
                let l = r[t],
                    i = t > 0;
                o = this.applyAt(e, l, i, n) || o;
            }
        }
        return t.length > 0 && this.queueCommands(), o;
    }),
    (Rule.prototype.queueCommands = function () {
        if (0 === this.commands.length) return;
        let e = level.commandQueue.indexOf("cancel") >= 0,
            n = level.commandQueue.indexOf("restart") >= 0,
            t = !1,
            o = !1;
        for (let e = 0; e < this.commands.length; e++) {
            let n = this.commands[e][0];
            "cancel" === n ? (t = !0) : "restart" === n && (o = !0);
        }
        if (!e && (!n || t)) {
            (t || o) && ((level.commandQueue = []), (level.commandQueueSourceRules = []), (messagetext = ""));
            for (let e = 0; e < this.commands.length; e++) {
                const n = this.commands[e];
                if (!(level.commandQueue.indexOf(n[0]) >= 0)) {
                    if ((level.commandQueue.push(n[0]), level.commandQueueSourceRules.push(this), verbose_logging)) {
                        const e = this.lineNumber;
                        dirMaskName[this.direction];
                        consolePrint('<font color="green">Rule <a onclick="jumpToLine(' + e.toString() + ');"  href="javascript:void(0);">' + e.toString() + "</a> triggers command " + n[0] + ".</font>", !1, e, null);
                    }
                    "message" === n[0] && (messagetext = n[1]);
                }
            }
        }
    });
let CACHE_RESOLVEMOVEMENTS = {};
function generate_resolveMovements(e, n, t) {
    const o = `'use strict';\n\t\tlet moved=true;\n\t\twhile(moved){\n\t\t\tmoved=false;\n\t\t\tfor (let i=0;i<level.n_tiles;i++) {\n\t\t\t\tmoved = state.repositionEntitiesAtCell(level,i) || moved;\n\t\t\t}\n\t\t}\n\t\tlet doUndo=false;\n\t\n\t\t//Search for any rigidly-caused movements remaining\n\t\tfor (let i=0;i<level.n_tiles;i++) {\n\t\t\tlet cellMask = level.getCellInto(i,_o6);\n\t\t\tlet movementMask = level.getMovements(i);\n\t\t\tif (${IS_NONZERO(
        "movementMask",
        n
    )}) {\n\n\t\t\t\t${IF_LAZY(
        t.rigid,
        () =>
            `\n\t\t\t\t\tlet rigidMovementAppliedMask = level.rigidMovementAppliedMask[i];\n\t\t\t\tif (${IS_NONZERO("rigidMovementAppliedMask", n)}) {\n\t\t\t\t\t${UNROLL(
                "movementMask &= rigidMovementAppliedMask",
                n
            )}\n\t\t\t\t\tif (${IS_NONZERO("movementMask", n)}) \n\t\t\t\touter_area: {\n\t\t\t\t\t\t//find what layer was restricted\n\t\t\t\t\t\t${FOR(
                0,
                LAYER_COUNT,
                (e) =>
                    `{\n\t\t\t\t\t\t\tlet layerSection = ${GETSHIFTOR(
                        "movementMask",
                        31,
                        5 * e
                    )};\n\t\t\t\t\t\t\tif (layerSection!==0) {\n\t\t\t\t\t\t\t\t//this is our layer!\n\t\t\t\t\t\t\t\tlet rigidGroupIndexMask = level.rigidGroupIndexMask[i];\n\t\t\t\t\t\t\t\tlet rigidGroupIndex = ${GETSHIFTOR(
                        "rigidGroupIndexMask",
                        31,
                        5 * e
                    )};\n\t\t\t\t\t\t\t\trigidGroupIndex--;//group indices start at zero, but are incremented for storing in the bitfield\n\t\t\t\t\t\t\t\tlet groupIndex = state.rigidGroupIndex_to_GroupIndex[rigidGroupIndex];\n\t\t\t\t\t\t\t\tif (bannedGroup[groupIndex]!==true){\n\t\t\t\t\t\t\t\t\tbannedGroup[groupIndex]=true\n\t\t\t\t\t\t\t\t\tdoUndo=true;\n\t\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\t\tbreak outer_area;\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t}`
            )}\n\t\t\t\t\t}\n\t\t\t\t}`
    )}\n\n\n\t\t\t\tfor (let j=0;j<state.sfx_MovementFailureMasks.length;j++) {\n\t\t\t\t\tlet o = state.sfx_MovementFailureMasks[j];\n\t\t\t\t\tlet objectMask = o.objectMask;\n\t\t\n\t\t\t\t\tif (${ANY_BITS_IN_COMMON(
        "cellMask",
        "objectMask",
        e
    )} \n\t\t\t\t\t&& ${ANY_BITS_IN_COMMON(
        "o.directionMask",
        "movementMask",
        n
    )} \n\t\t\t\t\t&& seedsToPlay_CantMove.indexOf(o.seed)===-1) {\n\t\t\t\t\t\tseedsToPlay_CantMove.push(o.seed);\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\n\t\t\tfor (let j=0;j<STRIDE_MOV;j++) {\n\t\t\t\tlevel.movements[j+i*STRIDE_MOV]=0;\n\t\t\t}\n\n\t\t\t${IF(
        t.rigid
    )}\n\t\t\t\t${SET_ZERO("level.rigidGroupIndexMask[i]")}\n\t\t\t\t${SET_ZERO("level.rigidMovementAppliedMask[i]")}\n\t\t\t${ENDIF(t.rigid)}\n\n\t\t}\n\t\treturn doUndo;\n\t`;
    return o in CACHE_RESOLVEMOVEMENTS ? CACHE_RESOLVEMOVEMENTS[o] : (CACHE_RESOLVEMOVEMENTS[o] = new Function("level", "bannedGroup", o));
}
let sfxCreateMask = null,
    sfxDestroyMask = null;
function processInput(e, n, t) {
    againing = !1;
    let o = backupLevel(),
        r = e,
        l = [];
    if ((verbose_logging && (debugger_turnIndex++, addToDebugTimeline(level, -2)), e >= 0)) {
        switch (e) {
            case 0:
                e = parseInt("00001", 2);
                break;
            case 1:
                e = parseInt("00100", 2);
                break;
            case 2:
                e = parseInt("00010", 2);
                break;
            case 3:
                e = parseInt("01000", 2);
                break;
            case 4:
                e = parseInt("10000", 2);
        }
        l = startMovement(e);
    }
    if (verbose_logging) {
        consolePrint("Applying rules");
        let n = addToDebugTimeline(level, -1);
        consolePrint(-1 === e ? "Turn starts with no input." : `Turn starts with input of ${["up", "left", "down", "right", "action"][r]}.`, !1, null, n);
    }
    let i = [];
    (level.commandQueue = []), (level.commandQueueSourceRules = []);
    let s = !1;
    const a = {
        objects: new Int32Array(level.objects),
        movements: new Int32Array(level.movements),
        rigidGroupIndexMask: level.rigidGroupIndexMask.concat([]),
        rigidMovementAppliedMask: level.rigidMovementAppliedMask.concat([]),
        commandQueue: [],
        commandQueueSourceRules: [],
    };
    sfxCreateMask.setZero(), sfxDestroyMask.setZero(), (seedsToPlay_CanMove = []), (seedsToPlay_CantMove = []), state.calculateRowColMasks(level);
    let c = [],
        d = 0;
    do {
        if (((s = !1), d++, applyRules(state.rules, state.loopPoint, i), state.resolveMovements(level, i))) {
            if (((s = !0), IDE)) {
                let e = [];
                for (let n in i) c.includes(n) || (e.push(n), c.push(n));
                let n = e.map((e) => state.rules[e][0].lineNumber),
                    t = n.length > 1 ? "lines " : "line ";
                (t += n.map((e) => `<a onclick="jumpToLine(${e});" href="javascript:void(0);">${e}</a>`).join(", ")),
                    consolePrint(`Rigid movement application failed in rule-Group starting from ${t}, and will be disabled in resimulation. Rolling back...`);
            }
            (level.objects = new Int32Array(a.objects)),
                (level.movements = new Int32Array(a.movements)),
                (level.rigidGroupIndexMask = a.rigidGroupIndexMask.concat([])),
                (level.rigidMovementAppliedMask = a.rigidMovementAppliedMask.concat([])),
                (level.commandQueue = a.commandQueue.concat([])),
                (level.commandQueueSourceRules = a.commandQueueSourceRules.concat([])),
                sfxCreateMask.setZero(),
                sfxDestroyMask.setZero(),
                verbose_logging && s && d > 0 && (consolePrint("Relooping through rules because of rigid."), debugger_turnIndex++, addToDebugTimeline(level, -2));
        } else {
            if (verbose_logging) {
                let e = debug_visualisation_array[debugger_turnIndex].length + 1;
                consolePrint("Processed movements.", !1, null, addToDebugTimeline(level, e)), state.lateRules.length > 0 && (debugger_turnIndex++, addToDebugTimeline(level, -2), consolePrint("Applying late rules"));
            }
            applyRules(state.lateRules, state.lateLoopPoint);
        }
    } while (d < 50 && s);
    if ((d >= 50 && consolePrint("Looped through 50 times, gave up.  too many loops!"), l.length > 0 && void 0 !== state.metadata.require_player_movement)) {
        let e = !1;
        for (let n = 0; n < l.length; n++) {
            let t = l[n],
                o = level.getCell(t);
            if (state.playerMask[1].bitsClearInArray(o.data)) {
                e = !0;
                break;
            }
        }
        if (!1 === e)
            return (
                verbose_logging && (consolePrint("require_player_movement set, but no player movement detected, so cancelling turn."), consoleCacheDump()),
                addUndoState(o),
                DoUndo(!0, !1),
                (messagetext = ""),
                (textMode = !1),
                tick_lazy_function_generation(!1),
                !1
            );
    }
    let u = processCommandQueue(o, t, n, r, e);
    return verbose_logging && consoleCacheDump(), winning && (againing = !1), u;
}
function playSounds(e, n, t, o, r, l) {
    for (let n = 0; n < e.length; n++) playSound(e[n]);
    for (let e = 0; e < n.length; e++) playSound(n[e]);
    for (let e = 0; e < t.length; e++) {
        let n = t[e];
        r.anyBitsInCommon(n.objectMask) && playSound(n.seed);
    }
    for (let e = 0; e < o.length; e++) {
        let n = o[e];
        l.anyBitsInCommon(n.objectMask) && playSound(n.seed);
    }
}
function processCommandQueue(e, n, t, o, a) {
    const r = level.commandQueue.indexOf("cancel");
    if (r >= 0) {
        if (verbose_logging) {
            consoleCacheDump(), consolePrintFromRule("CANCEL command executed, cancelling turn.", level.commandQueueSourceRules[r], !0);
        }
        n || processOutputCommands(level.commandQueue);
        let t = level.commandQueue.length > 1;
        return addUndoState(e), DoUndo(!0, !1), tryPlayCancelSound(), t;
    }
    const l = level.commandQueue.indexOf("restart");
    if (l >= 0) {
        if (verbose_logging && runrulesonlevelstart_phase) {
            logWarning(
                'A "restart" command is being triggered in the "run_rules_on_level_start" section of level creation, which would cause an infinite loop if it was actually triggered, but it\'s being ignored.',
                level.commandQueueSourceRules[l].lineNumber,
                !0
            );
        }
        if (verbose_logging) {
            consolePrintFromRule("RESTART command executed, reverting to restart state.", level.commandQueueSourceRules[l]), consoleCacheDump();
        }
        n || processOutputCommands(level.commandQueue), addUndoState(e), n || DoRestart(!0);
    }
    let i = !1;
    for (let t = 0; t < level.objects.length; t++)
        if (level.objects[t] !== e.dat[t]) {
            if (n) return verbose_logging && consoleCacheDump(), addUndoState(e), DoUndo(!0, !1), !0;
            -1 !== o ? (a?addUndoState(e, a):addUndoState(e)) : backups.length > 0 && (backups[backups.length - 1] = unconsolidateDiff(backups[backups.length - 1], e)), (i = !0);
            break;
        }
    if (n && (level.commandQueue.includes("win") || level.commandQueue.includes("restart"))) return !0;
    if (
        (n || (playSounds(seedsToPlay_CantMove, seedsToPlay_CanMove, state.sfx_CreationMasks, state.sfx_DestructionMasks, sfxCreateMask, sfxDestroyMask), processOutputCommands(level.commandQueue)),
        !1 === textMode && (void 0 === t && (t = !1), checkWin(t)),
        !winning)
    ) {
        let e = level.commandQueue.indexOf("checkpoint");
        if (e >= 0) {
            if (verbose_logging) {
                consolePrintFromRule("CHECKPOINT command executed, saving current state to the restart state.", level.commandQueueSourceRules[e]);
            }
            (restartTarget = level4Serialization()), (hasUsedCheckpoint = !0);
            let n = JSON.stringify(restartTarget);
            storage_set(document.URL + "_checkpoint", n), storage_set(document.URL, curlevel);
        }
        let n = level.commandQueue.indexOf("again");
        if (n >= 0 && i) {
            let e = level.commandQueueSourceRules[n],
                t = verbose_logging,
                o = messagetext;
            (verbose_logging = !1),
                processInput(-1, !0, !0)
                    ? ((verbose_logging = t), verbose_logging && consolePrintFromRule("AGAIN command executed, with changes detected - will execute another turn.", e), (againing = !0), (timer = 0))
                    : ((verbose_logging = t), verbose_logging && consolePrintFromRule("AGAIN command not executed, it wouldn't make any changes.", e)),
                (verbose_logging = t),
                (messagetext = o);
        }
    }
    return verbose_logging && consolePrint("Turn complete"), (level.commandQueue = []), (level.commandQueueSourceRules = []), i;
}
function checkWin(e) {
    if ((levelEditorOpened && (e = !0), level.commandQueue.indexOf("win") >= 0))
        return (
            consolePrint(
                runrulesonlevelstart_phase
                    ? "Win Condition Satisfied (However this is in the run_rules_on_level_start rule pass, so I'm going to ignore it for you.  Why would you want to complete a level before it's already started?!)"
                    : "Win Condition Satisfied"
            ),
            void (e || DoWin())
        );
    if (0 === state.winconditions.length) return !1;
    let n = !1,
        t = !0;
    const o = state.winconditions.length;
    for (let e = 0; e < o; e++) {
        let n = state.winconditions[e],
            o = n[1],
            r = n[2],
            l = !0;
        const i = n[4] ? (e) => o.bitsSetInArray(e) : (e) => !o.bitsClearInArray(e),
            s = n[5] ? (e) => r.bitsSetInArray(e) : (e) => !r.bitsClearInArray(e);
        switch (n[0]) {
            case -1:
                for (let e = 0; e < level.n_tiles; e++) {
                    let n = level.getCellInto(e, _o10);
                    if (i(n.data) && s(n.data)) {
                        l = !1;
                        break;
                    }
                }
                break;
            case 0: {
                let e = !1;
                for (let n = 0; n < level.n_tiles; n++) {
                    let t = level.getCellInto(n, _o10);
                    if (i(t.data) && s(t.data)) {
                        e = !0;
                        break;
                    }
                }
                !1 === e && (l = !1);
                break;
            }
            case 1:
                for (let e = 0; e < level.n_tiles; e++) {
                    let n = level.getCellInto(e, _o10);
                    if (i(n.data) && !s(n.data)) {
                        l = !1;
                        break;
                    }
                }
        }
        !1 === l && (t = !1);
    }
    (n = t),
        n &&
            (consolePrint(
                runrulesonlevelstart_phase
                    ? "Win Condition Satisfied (However this is in the run_rules_on_level_start rule pass, so I'm going to ignore it for you.  Why would you want to complete a level before it's already started?!)"
                    : "Win Condition Satisfied"
            ),
            e || DoWin());
}
function DoWin() {
    submitForm(state.metadata.title, "" + curlevel, solution.join(","));
    solution = [];
    winning || ((againing = !1), tryPlayEndLevelSound(), unitTesting ? nextLevel() : ((winning = !0), (timer = 0)));
}
function nextLevel() {
    if (((againing = !1), (messagetext = ""), state && state.levels && curlevel > state.levels.length && (curlevel = state.levels.length - 1), (ignoreNotJustPressedAction = !0), titleScreen))
        0 === titleSelection && ((curlevel = 0), (curlevelTarget = null)), null !== curlevelTarget ? loadLevelFromStateTarget(state, curlevel, curlevelTarget) : loadLevelFromState(state, curlevel);
    else if ((hasUsedCheckpoint && ((curlevelTarget = null), (hasUsedCheckpoint = !1)), curlevel < state.levels.length - 1))
        curlevel++,
            (curlevelTarget = null),
            (textMode = !1),
            tick_lazy_function_generation(!1),
            tick_lazy_function_generation(!1),
            (titleScreen = !1),
            (quittingMessageScreen = !1),
            (messageselected = !1),
            loadLevelFromState(state, curlevel);
    else {
        try {
            storage_remove(document.URL), storage_remove(document.URL + "_checkpoint");
        } catch (e) {}
        (curlevel = 0), (curlevelTarget = null), goToTitleScreen(), tryPlayEndGameSound();
    }
    try {
        if ((storage_set(document.URL, curlevel), null !== curlevelTarget)) {
            restartTarget = level4Serialization();
            let e = JSON.stringify(restartTarget);
            storage_set(document.URL + "_checkpoint", e);
        } else storage_remove(document.URL + "_checkpoint");
    } catch (e) {}
    void 0 !== state && void 0 !== state.metadata.flickscreen && (oldflickscreendat = [0, 0, Math.min(state.metadata.flickscreen[0], level.width), Math.min(state.metadata.flickscreen[1], level.height)]), canvasResize();
}
function goToTitleScreen() {
    (againing = !1), (messagetext = ""), (titleScreen = !0), (textMode = !0), doSetupTitleScreenLevelContinue(), (titleSelection = showContinueOptionOnTitleScreen() ? 1 : 0), generateTitleScreen(), null !== canvas && regenSpriteImages();
}
let CACHE_RULE_FINDMATCHES = {};
Rule.prototype.generateFindMatchesFunction = function () {
    let e = "";
    (e += `if (${NOT_BITS_SET_IN_ARRAY("this.ruleMask", "level.mapCellContents.data", STRIDE_OBJ)}) return [];\n`), (e += "const d = level.delta_index(this.direction);\n"), (e += "const matches = [];\n");
    for (let n = 0; n < this.patterns.length; n++)
        (e += `let match${n};\n`),
            0 === this.ellipsisCount[n]
                ? (e += `match${n} = state.matchCellRow(level,this.direction, this.cellRowMatches[${n}], this.patterns[${n}], this.cellRowMasks[${n}], this.cellRowMasks_Movements[${n}], d);\n`)
                : 1 === this.ellipsisCount[n]
                ? (e += `match${n} = state.matchCellRowWildCard(this.direction, this.cellRowMatches[${n}], this.patterns[${n}], this.cellRowMasks[${n}], this.cellRowMasks_Movements[${n}], d, 1);\n`)
                : (e += `match${n} = state.matchCellRowWildCard(this.direction, this.cellRowMatches[${n}], this.patterns[${n}], this.cellRowMasks[${n}], this.cellRowMasks_Movements[${n}], d, 2);\n`),
            (e += `if (match${n}.length === 0) return [];\n`),
            (e += `matches.push(match${n});\n`);
    return (e += "return matches;"), e in CACHE_RULE_FINDMATCHES ? CACHE_RULE_FINDMATCHES[e] : (CACHE_RULE_FINDMATCHES[e] = new Function("level", e));
};
const MAX_ERRORS_FOR_REAL = 100;
let compiling = !1,
    errorStrings = [],
    errorCount = 0;
function TooManyErrors() {
    const e = compiling ? "Too many errors/warnings; aborting compilation." : "Too many errors/warnings; noping out.";
    throw (consolePrint(e, !0), new Error(e));
}
function logErrorCacheable(e, n, t) {
    if (compiling || t) {
        if (void 0 === n) return logErrorNoLine(e, t);
        let o = '<a onclick="jumpToLine(' + n.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + n.toString() + '</span></a> : <span class="errorText">' + e + "</span>";
        (errorStrings.indexOf(o) >= 0 && !t) || (consolePrint(o), errorStrings.push(o), errorCount++, errorStrings.length > MAX_ERRORS_FOR_REAL && TooManyErrors());
    }
}
function logError(e, n, t) {
    if (compiling || t) {
        if (void 0 === n) return logErrorNoLine(e, t);
        let o = '<a onclick="jumpToLine(' + n.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + n.toString() + '</span></a> : <span class="errorText">' + e + "</span>";
        (errorStrings.indexOf(o) >= 0 && !t) || (consolePrint(o, !0), errorStrings.push(o), errorCount++, errorStrings.length > MAX_ERRORS_FOR_REAL && TooManyErrors());
    }
}
function logWarning(e, n, t) {
    if (compiling || t) {
        if (void 0 === n) return logWarningNoLine(e, t);
        let o = '<a onclick="jumpToLine(' + n.toString() + ');"  href="javascript:void(0);"><span class="errorTextLineNumber"> line ' + n.toString() + '</span></a> : <span class="warningText">' + e + "</span>";
        (errorStrings.indexOf(o) >= 0 && !t) || (consolePrint(o, !0), errorStrings.push(o), errorStrings.length > MAX_ERRORS_FOR_REAL && TooManyErrors());
    }
}
function logWarningNoLine(e, n) {
    if (compiling || n) {
        let t = '<span class="warningText">' + e + "</span>";
        (errorStrings.indexOf(t) >= 0 && !n) || (consolePrint(t, !0), errorStrings.push(t), errorCount++, errorStrings.length > MAX_ERRORS_FOR_REAL && TooManyErrors());
    }
}
function logErrorNoLine(e, n) {
    if (compiling || n) {
        let t = '<span class="errorText">' + e + "</span>";
        (errorStrings.indexOf(t) >= 0 && !n) || (consolePrint(t, !0), errorStrings.push(t), errorCount++, errorStrings.length > MAX_ERRORS_FOR_REAL && TooManyErrors());
    }
}
function blankLineHandle(e) {
    "levels" === e.section ? e.levels[e.levels.length - 1].length > 0 && e.levels.push([]) : "objects" === e.section && (e.objects_section = 0);
}
function wordAlreadyDeclared(e, n) {
    if ((n = n.toLowerCase()) in e.objects) return e.objects[n];
    for (let t = 0; t < e.legend_aggregates.length; t++) {
        if (e.legend_aggregates[t][0] === n) return e.legend_aggregates[t];
    }
    for (let t = 0; t < e.legend_properties.length; t++) {
        if (e.legend_properties[t][0] === n) return e.legend_properties[t];
    }
    for (let t = 0; t < e.legend_synonyms.length; t++) {
        if (e.legend_synonyms[t][0] === n) return e.legend_synonyms[t];
    }
    return null;
}
"function" != typeof Object.assign &&
    (Object.assign = function (e) {
        if (null == e) throw new TypeError("Cannot convert undefined or null to object");
        let n = Object(e);
        for (let e = 1; e < arguments.length; e++) {
            let t = arguments[e];
            if (null != t) for (let e in t) t.hasOwnProperty(e) && (n[e] = t[e]);
        }
        return n;
    });
let codeMirrorFn = function () {
    function e(e, n) {
        if (void 0 === e.objects[n]) {
            for (let t = 0; t < e.legend_synonyms.length; t++) {
                if (e.legend_synonyms[t][0] == n) return;
            }
            for (let t = 0; t < e.legend_aggregates.length; t++) {
                if (e.legend_aggregates[t][0] == n) return;
            }
            for (let t = 0; t < e.legend_properties.length; t++) {
                if (e.legend_properties[t][0] == n) return;
            }
            logError(`You're talking about ${n.toUpperCase()} but it's not defined anywhere.`, e.lineNumber);
        }
    }
    function n(e, n, t, o) {
        let r = new RegExp("\\b" + (n.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&") + "\\b"), "i");
        let l = t.match(r);
        null != l && ((e.original_case_names[n] = l[0]), (e.original_line_numbers[n] = o));
    }
    function t(e) {
        let n = e.match(reg_match_until_commentstart_or_whitespace, !0);
        return null === n && (n = e.match(reg_notcommentstart, !0)), n;
    }
    function o(t, o) {
        let r = !0,
            l = t.current_line_wip_array;
        if (0 !== l.length)
            if (1 === l.length) logError('Incorrect format of legend - should be one of "A = B", "A = B or C [ or D ...]", "A = B and C [ and D ...]".', t.lineNumber), (r = !1);
            else if (l.length % 2 == 0)
                logError(
                    `Incorrect format of legend - should be one of "A = B", "A = B or C [ or D ...]", "A = B and C [ and D ...]", but it looks like you have a dangling "${t.current_line_wip_array[
                        t.current_line_wip_array.length - 1
                    ].toUpperCase()}"?`,
                    t.lineNumber
                ),
                    (r = !1);
            else {
                let i = l[0],
                    s = wordAlreadyDeclared(t, i);
                null !== s &&
                    (logError(`Name "${i.toUpperCase()}" already in use (on line <a onclick="jumpToLine(${s.lineNumber});" href="javascript:void(0);"><span class="errorTextLineNumber">line ${s.lineNumber}</span></a>).`, t.lineNumber),
                    (r = !1)),
                    keyword_array.indexOf(i) >= 0 && logWarning('You named an object "' + i.toUpperCase() + "\", but this is a keyword. Don't do that!", t.lineNumber);
                for (let e = 2; e < l.length; e += 2) {
                    let n = l[e];
                    if (n === i) {
                        logError("You can't define object " + i.toUpperCase() + " in terms of itself!", t.lineNumber), (r = !1);
                        let e = l.indexOf(i, 2);
                        for (; e >= 2; ) e >= 4 ? l.splice(e - 1, 2) : l.splice(e, 2), (e = l.indexOf(i, 2));
                    }
                    for (let o = 2; o < e; o += 2) {
                        let e = l[o];
                        e === n && logWarning("You're repeating the object " + e.toUpperCase() + " here multiple times on the RHS.  This makes no sense.  Don't do that.", t.lineNumber);
                    }
                }
                for (let n = 2; n < l.length; n += 2) {
                    let o = l[n];
                    o !== i && e(t, o);
                }
                if (3 === l.length) {
                    let e = [l[0], l[2]];
                    (e.lineNumber = t.lineNumber), n(t, l[0], o, t.lineNumber), t.legend_synonyms.push(e);
                } else if ("and" === l[3]) {
                    let e = function (n) {
                            if ((n = n.toLowerCase()) in t.objects) return [n];
                            for (let o = 0; o < t.legend_synonyms.length; o++) {
                                let r = t.legend_synonyms[o];
                                if (r[0] === n) return e(r[1]);
                            }
                            for (let o = 0; o < t.legend_aggregates.length; o++) {
                                let r = t.legend_aggregates[o];
                                if (r[0] === n) return [].concat.apply([], r.slice(1).map(e));
                            }
                            for (let e = 0; e < t.legend_properties.length; e++) {
                                if (t.legend_properties[e][0] === n) return logError("Cannot define an aggregate (using 'and') in terms of properties (something that uses 'or').", t.lineNumber), (r = !1), [n];
                            }
                            return [n];
                        },
                        i = [l[0]].concat(e(l[2])).concat(e(l[4]));
                    for (let n = 6; n < l.length; n += 2) i = i.concat(e(l[n]));
                    (i.lineNumber = t.lineNumber), n(t, i[0], o, t.lineNumber), t.legend_aggregates.push(i);
                } else if ("or" === l[3]) {
                    let e = !1,
                        r = function (n) {
                            if ((n = n.toLowerCase()) in t.objects) return [n];
                            for (let e = 0; e < t.legend_synonyms.length; e++) {
                                let o = t.legend_synonyms[e];
                                if (o[0] === n) return r(o[1]);
                            }
                            for (let o = 0; o < t.legend_aggregates.length; o++) {
                                if (t.legend_aggregates[o][0] === n)
                                    return (
                                        logError(
                                            `Cannot define a property (something defined in terms of 'or') in terms of an aggregate (something that uses 'and').  In this case, you can't define "${l[0]}" in terms of "${n}".`,
                                            t.lineNumber
                                        ),
                                        (e = !0),
                                        []
                                    );
                            }
                            for (let e = 0; e < t.legend_properties.length; e++) {
                                let o = t.legend_properties[e];
                                if (o[0] === n) {
                                    let e = [];
                                    for (let t = 1; t < o.length; t++) o[t] === n || (e = e.concat(r(o[t])));
                                    return e;
                                }
                            }
                            return [n];
                        };
                    for (let n = 5; n < l.length; n += 2)
                        if ("or" !== l[n].toLowerCase()) {
                            e = !0;
                            break;
                        }
                    if (!e) {
                        let e = [l[0]].concat(r(l[2])).concat(r(l[4]));
                        for (let n = 6; n < l.length; n += 2) e.push(l[n].toLowerCase());
                        (e.lineNumber = t.lineNumber), n(t, e[0], o, t.lineNumber), t.legend_properties.push(e);
                    }
                } else r && (logError("This legend-entry is incorrectly-formatted - it should be one of A = B, A = B or C ( or D ...), A = B and C (and D ...)", t.lineNumber), (r = !1));
            }
    }
    function r(e) {
        if (0 !== e.current_line_wip_array.length)
            if ("ERROR" === e.current_line_wip_array[e.current_line_wip_array.length - 1]);
            else {
                let n = e.current_line_wip_array;
                n.push(e.lineNumber), e.sounds.push(n);
            }
    }
    function l(e, n) {
        "legend" === e.section ? o(e, n) : "sounds" === e.section && r(e);
    }
    return {
        copyState: function (e) {
            let n = {};
            for (let t in e.objects)
                if (e.objects.hasOwnProperty(t)) {
                    let o = e.objects[t];
                    n[t] = { colors: o.colors.concat([]), lineNumber: o.lineNumber, spritematrix: o.spritematrix.concat([]) };
                }
            let t = [];
            for (let n = 0; n < e.collisionLayers.length; n++) t.push(e.collisionLayers[n].concat([]));
            let o = [],
                r = [],
                l = [],
                i = [],
                s = [],
                a = [],
                c = [];
            for (let n = 0; n < e.legend_synonyms.length; n++) o.push(e.legend_synonyms[n].concat([]));
            for (let n = 0; n < e.legend_aggregates.length; n++) r.push(e.legend_aggregates[n].concat([]));
            for (let n = 0; n < e.legend_properties.length; n++) l.push(e.legend_properties[n].concat([]));
            for (let n = 0; n < e.sounds.length; n++) i.push(e.sounds[n].concat([]));
            for (let n = 0; n < e.levels.length; n++) s.push(e.levels[n].concat([]));
            for (let n = 0; n < e.winconditions.length; n++) a.push(e.winconditions[n].concat([]));
            for (let n = 0; n < e.rules.length; n++) c.push(e.rules[n].concat([]));
            let d = Object.assign({}, e.original_case_names),
                u = Object.assign({}, e.original_line_numbers);
            return {
                lineNumber: e.lineNumber,
                objects: n,
                collisionLayers: t,
                commentLevel: e.commentLevel,
                section: e.section,
                visitedSections: e.visitedSections.concat([]),
                line_should_end: e.line_should_end,
                line_should_end_because: e.line_should_end_because,
                sol_after_comment: e.sol_after_comment,
                objects_candname: e.objects_candname,
                objects_section: e.objects_section,
                objects_spritematrix: e.objects_spritematrix.concat([]),
                tokenIndex: e.tokenIndex,
                current_line_wip_array: e.current_line_wip_array.concat([]),
                legend_synonyms: o,
                legend_aggregates: r,
                legend_properties: l,
                sounds: i,
                rules: c,
                names: e.names.concat([]),
                winconditions: a,
                original_case_names: d,
                original_line_numbers: u,
                abbrevNames: e.abbrevNames.concat([]),
                metadata: e.metadata.concat([]),
                metadata_lines: Object.assign({}, e.metadata_lines),
                levels: s,
                STRIDE_OBJ: e.STRIDE_OBJ,
                STRIDE_MOV: e.STRIDE_MOV,
            };
        },
        blankLine: function (e) {
            "levels" === e.section && e.levels[e.levels.length - 1].length > 0 && e.levels.push([]);
        },
        token: function (e, i) {
            let s = e.string,
                a = e.sol();
            a && (i.lineNumber++, (i.current_line_wip_array = []), (e.string = e.string.toLowerCase()), (i.tokenIndex = 0), (i.line_should_end = !1)), i.sol_after_comment && ((a = !0), (i.sol_after_comment = !1)), e.eatWhile(/[ \t]/);
            let c = e.peek();
            if ("(" === c && -4 !== i.tokenIndex) e.next(), i.commentLevel++;
            else if (")" === c) {
                if ((e.next(), !(i.commentLevel > 0)))
                    return logWarning("You're trying to close a comment here, but I can't find any opening bracket to match it? [This is highly suspicious; you probably want to fix it.]", i.lineNumber), "ERROR";
                if ((i.commentLevel--, 0 === i.commentLevel)) return (i.sol_after_comment = !0), "comment";
            }
            if (i.commentLevel > 0) {
                for (a && (i.sol_after_comment = !0); e.eatWhile(/[^\(\)]+/), !e.eol() && ((c = e.peek()), "(" === c ? i.commentLevel++ : ")" === c && i.commentLevel--, e.next(), 0 !== i.commentLevel); );
                return e.eol() && l(i, s), "comment";
            }
            if ((e.eatWhile(/[ \t]/), a && e.eol())) return l(i, s), blankLineHandle(i);
            if (i.line_should_end && !e.eol()) return logError("Only comments should go after " + i.line_should_end_because + " on a line.", i.lineNumber), e.skipToEnd(), "ERROR";
            let d = !0;
            if (a && "levels" === i.section) {
                let n = e.string.substring(e.pos);
                /^[\p{Z}\s=]*$/u.test(n) || (d = !1);
            }
            if (a && d && e.match(reg_equalsrow, !0)) return (i.line_should_end = !0), (i.line_should_end_because = "a bunch of equals signs ('===')"), "EQUALSBIT";
            if ("levels" !== i.section) {
                let n = e.match(reg_sectionNames, !0);
                if (a && n) {
                    (i.section = n[0].trim()),
                        i.visitedSections.indexOf(i.section) >= 0 && logError('cannot duplicate sections (you tried to duplicate "' + i.section.toUpperCase() + '").', i.lineNumber),
                        (i.line_should_end = !0),
                        (i.line_should_end_because = `a section name ("${i.section.toUpperCase()}")`),
                        i.visitedSections.push(i.section);
                    let e = sectionNames.indexOf(i.section);
                    if (
                        (0 == e
                            ? ((i.objects_section = 0), i.visitedSections.length > 1 && logError('section "' + i.section.toUpperCase() + '" must be the first section', i.lineNumber))
                            : -1 == i.visitedSections.indexOf(sectionNames[e - 1]) &&
                              logError(
                                  -1 === e
                                      ? 'no such section as "' + i.section.toUpperCase() + '".'
                                      : 'section "' +
                                            i.section.toUpperCase() +
                                            '" is out of order, must follow  "' +
                                            sectionNames[e - 1].toUpperCase() +
                                            '" (or it could be that the section "' +
                                            sectionNames[e - 1].toUpperCase() +
                                            '"is just missing totally.  You have to include all section headings, even if the section itself is empty).',
                                  i.lineNumber
                              ),
                        "sounds" === i.section)
                    ) {
                        for (let e in i.objects) i.objects.hasOwnProperty(e) && i.names.push(e);
                        for (let e = 0; e < i.legend_synonyms.length; e++) {
                            let n = i.legend_synonyms[e][0];
                            i.names.push(n);
                        }
                        for (let e = 0; e < i.legend_aggregates.length; e++) {
                            let n = i.legend_aggregates[e][0];
                            i.names.push(n);
                        }
                        for (let e = 0; e < i.legend_properties.length; e++) {
                            let n = i.legend_properties[e][0];
                            i.names.push(n);
                        }
                    } else if ("levels" === i.section) {
                        for (let e in i.objects) i.objects.hasOwnProperty(e) && 1 === e.length && i.abbrevNames.push(e);
                        for (let e = 0; e < i.legend_synonyms.length; e++) 1 === i.legend_synonyms[e][0].length && i.abbrevNames.push(i.legend_synonyms[e][0]);
                        for (let e = 0; e < i.legend_aggregates.length; e++) 1 === i.legend_aggregates[e][0].length && i.abbrevNames.push(i.legend_aggregates[e][0]);
                    }
                    return "HEADER";
                }
                void 0 === i.section && logError('must start with section "OBJECTS"', i.lineNumber);
            }
            if (e.eol()) return l(i, s), null;
            switch (i.section) {
                case "objects": {
                    let t = function () {
                        let t = a ? e.match(reg_name, !0) : e.match(/[^\p{Z}\s\()]+[\p{Z}\s]*/u, !0);
                        if (null == t)
                            return (
                                e.match(reg_notcommentstart, !0),
                                e.pos > 0 &&
                                    logWarning(
                                        'Unknown junk in object section (possibly: sprites have to be 5 pixels wide and 5 pixels high exactly. Or maybe: the main names for objects have to be words containing only the letters a-z0.9 - if you want to call them something like ",", do it in the legend section).',
                                        i.lineNumber
                                    ),
                                "ERROR"
                            );
                        {
                            let e = t[0].trim();
                            if (void 0 !== i.objects[e]) return logError('Object "' + e.toUpperCase() + '" defined multiple times.', i.lineNumber), "ERROR";
                            for (let n = 0; n < i.legend_synonyms.length; n++) {
                                i.legend_synonyms[n][0] == e && logError('Name "' + e.toUpperCase() + '" already in use.', i.lineNumber);
                            }
                            if ((keyword_array.indexOf(e) >= 0 && logWarning('You named an object "' + e.toUpperCase() + "\", but this is a keyword. Don't do that!", i.lineNumber), a))
                                (i.objects_candname = e), n(i, e, s, i.lineNumber), (i.objects[i.objects_candname] = { lineNumber: i.lineNumber, colors: [], spritematrix: [] });
                            else {
                                n(i, e, s, i.lineNumber);
                                let t = [e, i.objects_candname];
                                (t.lineNumber = i.lineNumber), i.legend_synonyms.push(t);
                            }
                            return (i.objects_section = 1), "NAME";
                        }
                    };
                    switch ((a && 2 == i.objects_section && (i.objects_section = 3), a && 1 == i.objects_section && (i.objects_section = 2), i.objects_section)) {
                        case 0:
                        case 1:
                            return (i.objects_spritematrix = []), t();
                        case 2: {
                            i.tokenIndex = 0;
                            let n = e.match(reg_color, !0);
                            if (null == n) {
                                let n = e.match(reg_name, !0) || e.match(reg_notcommentstart, !0);
                                return logError("Was looking for color for object " + i.objects_candname.toUpperCase() + ', got "' + n + '" instead.', i.lineNumber), null;
                            }
                            {
                                void 0 === i.objects[i.objects_candname].colors ? (i.objects[i.objects_candname].colors = [n[0].trim()]) : i.objects[i.objects_candname].colors.push(n[0].trim());
                                let e = n[0].trim().toLowerCase();
                                return e in colorPalettes.arnecolors ? "COLOR COLOR-" + e.toUpperCase() : "transparent" === e ? "COLOR FADECOLOR" : "MULTICOLOR" + n[0];
                            }
                        }
                        case 3: {
                            let n = e.eat(/[.\d]/),
                                o = i.objects_spritematrix;
                            if (void 0 === n) return 0 === o.length ? t() : (logError("Unknown junk in spritematrix for object " + i.objects_candname.toUpperCase() + ".", i.lineNumber), e.match(reg_notcommentstart, !0), null);
                            a && o.push("");
                            let r = i.objects[i.objects_candname];
                            if (((o[o.length - 1] += n), o[o.length - 1].length > 5)) return logWarning("Sprites must be 5 wide and 5 high.", i.lineNumber), e.match(reg_notcommentstart, !0), null;
                            if (((r.spritematrix = i.objects_spritematrix), 5 === o.length && 5 === o[o.length - 1].length && (i.objects_section = 0), "." !== n)) {
                                let e = parseInt(n);
                                return e >= r.colors.length
                                    ? (logError("Trying to access color number " + e + " from the color palette of sprite " + i.objects_candname.toUpperCase() + ", but there are only " + r.colors.length + " defined in it.", i.lineNumber),
                                      "ERROR")
                                    : "COLOR BOLDCOLOR COLOR-" + r.colors[e].toUpperCase();
                            }
                            return "COLOR FADECOLOR";
                        }
                        default:
                            window.console.logError("EEK shouldn't get here.");
                    }
                    break;
                }
                case "legend": {
                    let n = "",
                        t = null;
                    if (0 === i.tokenIndex) {
                        t = e.match(/[^=\p{Z}\s\(]*(\p{Z}\s)*/u, !0);
                        let o = t[0].trim();
                        (n = wordAlreadyDeclared(i, o) ? "ERROR" : "NAME"), i.tokenIndex++;
                    } else if (1 === i.tokenIndex)
                        (t = e.match(/=/u, !0)),
                            (null !== t && "=" === t[0].trim()) ||
                                (logError('In the legend, define new items using the equals symbol - declarations must look like "A = B", "A = B or C [ or D ...]", "A = B and C [ and D ...]".', i.lineNumber),
                                e.match(reg_notcommentstart, !0),
                                (n = "ERROR"),
                                (t = ["ERROR"])),
                            e.match(/[\p{Z}\s]*/u, !0),
                            i.tokenIndex++,
                            (n = "ASSIGNMENT");
                    else if (i.tokenIndex >= 3 && i.tokenIndex % 2 == 1) {
                        if (((t = e.match(reg_name, !0)), null === t)) {
                            logError("Something bad's happening in the LEGEND", i.lineNumber);
                            e.match(reg_notcommentstart, !0);
                            n = "ERROR";
                        } else {
                            let e = t[0].trim();
                            "and" === e || "or" === e
                                ? ((n = "LOGICWORD"), i.tokenIndex >= 5 && e !== i.current_line_wip_array[3] && (logError("Hey! You can't go mixing ANDs and ORs in a single legend entry.", i.lineNumber), (n = "ERROR")))
                                : (logError(
                                      `Expected and 'AND' or an 'OR' here, but got ${e.toUpperCase()} instead. In the legend, define new items using the equals symbol - declarations must look like 'A = B' or 'A = B and C' or 'A = B or C'.`,
                                      i.lineNumber
                                  ),
                                  (n = "ERROR"));
                        }
                        i.tokenIndex++;
                    } else if (((t = e.match(reg_name, !0)), null === t)) {
                        logError("Something bad's happening in the LEGEND", i.lineNumber);
                        e.match(reg_notcommentstart, !0);
                        n = "ERROR";
                    } else {
                        let e = t[0].trim();
                        (n = wordAlreadyDeclared(i, e) ? "NAME" : "ERROR"), i.tokenIndex++;
                    }
                    return null !== t && i.current_line_wip_array.push(t[0].trim()), e.eol() && o(i, s), n;
                }
                case "sounds": {
                    let n = "";
                    if (i.current_line_wip_array.length > 0 && "ERROR" === i.current_line_wip_array[i.current_line_wip_array.length - 1]) {
                        let o = null;
                        if (
                            (null === o && ((o = e.match(reg_soundevents, !0)), null !== o && (n = "SOUNDEVENT")),
                            null === o && ((o = e.match(reg_soundverbs, !0)), null !== o && (n = "SOUNDVERB")),
                            null === o && ((o = e.match(reg_sounddirectionindicators, !0)), null !== o && (n = "DIRECTION")),
                            null === o)
                        ) {
                            null !== e.match(reg_soundseed, !0) && (n = "SOUND");
                        }
                        null === o && ((o = e.match(reg_name, !0)), null !== o && (n = wordAlreadyDeclared(i, o[0].trim()) ? "NAME" : "ERROR")), null === o && ((o = t(e)), (n = "ERROR"));
                    } else if (0 === i.current_line_wip_array.length) {
                        let o = e.match(reg_soundevents, !0);
                        if (null == o)
                            if (((o = e.match(reg_name, !0)), null == o))
                                (n = "ERROR"), (o = t(e)), i.current_line_wip_array.push("ERROR"), logWarning("Was expecting a sound event (like SFX3, or ENDLEVEL) or an object name, but didn't find either.", i.lineNumber);
                            else {
                                let e = o[0].trim();
                                wordAlreadyDeclared(i, e)
                                    ? ((n = "NAME"), i.current_line_wip_array.push([e, n]), i.tokenIndex++)
                                    : ((n = "ERROR"), i.current_line_wip_array.push("ERROR"), logError(`unexpected sound token "${e}".`, i.lineNumber));
                            }
                        else (n = "SOUNDEVENT"), i.current_line_wip_array.push([o[0].trim(), n]), i.tokenIndex++;
                    } else if (1 === i.current_line_wip_array.length) {
                        if ("SOUNDEVENT" === i.current_line_wip_array[0][1]) {
                            let o = e.match(reg_soundseed, !0);
                            null !== o
                                ? ((n = "SOUND"), i.current_line_wip_array.push([o[0].trim(), n]), i.tokenIndex++)
                                : ((o = t(e)),
                                  logError("Was expecting a sound seed here (a number like 123123, like you generate by pressing the buttons above the console panel), but found something else.", i.lineNumber),
                                  (n = "ERROR"),
                                  i.current_line_wip_array.push("ERROR"));
                        } else {
                            let o = e.match(reg_soundverbs, !0);
                            null !== o
                                ? ((n = "SOUNDVERB"), i.current_line_wip_array.push([o[0].trim(), n]), i.tokenIndex++)
                                : ((o = t(e)), logError("Was expecting a soundverb here (MOVE, DESTROY, CANTMOVE, or the like), but found something else.", i.lineNumber), (n = "ERROR"), i.current_line_wip_array.push("ERROR"));
                        }
                    } else {
                        if ("SOUNDEVENT" === i.current_line_wip_array[0][1]) {
                            let o = t(e);
                            logError(
                                `I wasn't expecting anything after the sound declaration ${i.current_line_wip_array[
                                    i.current_line_wip_array.length - 1
                                ][0].toUpperCase()} on this line, so I don't know what to do with "${o[0].trim().toUpperCase()}" here.`,
                                i.lineNumber
                            ),
                                (n = "ERROR"),
                                i.current_line_wip_array.push("ERROR");
                        } else {
                            if ("SOUND" === i.current_line_wip_array[i.current_line_wip_array.length - 1][1]) {
                                let o = t(e);
                                logError(
                                    `I wasn't expecting anything after the sound declaration ${i.current_line_wip_array[
                                        i.current_line_wip_array.length - 1
                                    ][0].toUpperCase()} on this line, so I don't know what to do with "${o[0].trim().toUpperCase()}" here.`,
                                    i.lineNumber
                                ),
                                    (n = "ERROR"),
                                    i.current_line_wip_array.push("ERROR");
                            } else {
                                if (soundverbs_directional.indexOf(i.current_line_wip_array[1][0]) >= 0) {
                                    let o = e.match(reg_sounddirectionindicators, !0);
                                    if (null !== o) (n = "DIRECTION"), i.current_line_wip_array.push([o[0].trim(), n]), i.tokenIndex++;
                                    else {
                                        let o = e.match(reg_soundseed, !0);
                                        if (null !== o) (n = "SOUND"), i.current_line_wip_array.push([o[0].trim(), n]), i.tokenIndex++;
                                        else {
                                            let o = t(e);
                                            logError(
                                                `Ah I was expecting direction or a sound seed here after ${i.current_line_wip_array[
                                                    i.current_line_wip_array.length - 1
                                                ][0].toUpperCase()}, but I don't know what to make of "${o[0].trim().toUpperCase()}".`,
                                                i.lineNumber
                                            ),
                                                (n = "ERROR"),
                                                i.current_line_wip_array.push("ERROR");
                                        }
                                    }
                                } else {
                                    let o = e.match(reg_soundseed, !0);
                                    if (null !== o) (n = "SOUND"), i.current_line_wip_array.push([o[0].trim(), n]), i.tokenIndex++;
                                    else {
                                        let o = t(e);
                                        logError(
                                            `Ah I was expecting a sound seed here after ${i.current_line_wip_array[i.current_line_wip_array.length - 1][0].toUpperCase()}, but I don't know what to make of "${o[0].trim().toUpperCase()}".`,
                                            i.lineNumber
                                        ),
                                            (n = "ERROR"),
                                            i.current_line_wip_array.push("ERROR");
                                    }
                                }
                            }
                        }
                    }
                    return e.eol() && r(i), n;
                }
                case "collisionlayers": {
                    a && (i.collisionLayers.push([]), (i.current_line_wip_array = []), (i.tokenIndex = 0));
                    let n = e.match(reg_name, !0);
                    if (null === n) {
                        let n = e.pos;
                        return e.match(reg_csv_separators, !0), e.pos == n && (logError("error detected - unexpected character " + e.peek(), i.lineNumber), e.next()), null;
                    }
                    {
                        let e = n[0].trim(),
                            t = function (n) {
                                if ((n = n.toLowerCase()) in i.objects) return [n];
                                for (let e = 0; e < i.legend_synonyms.length; e++) {
                                    let o = i.legend_synonyms[e];
                                    if (o[0] === n) return t(o[1]);
                                }
                                for (let e = 0; e < i.legend_aggregates.length; e++) {
                                    if (i.legend_aggregates[e][0] === n)
                                        return logError('"' + n + '" is an aggregate (defined using "and"), and cannot be added to a single layer because its constituent objects must be able to coexist.', i.lineNumber), [];
                                }
                                for (let e = 0; e < i.legend_properties.length; e++) {
                                    let o = i.legend_properties[e];
                                    if (o[0] === n) {
                                        let e = [];
                                        for (let r = 1; r < o.length; r++) o[r] === n || (e = e.concat(t(o[r])));
                                        return e;
                                    }
                                }
                                return logError('Cannot add "' + e.toUpperCase() + '" to a collision layer; it has not been declared.', i.lineNumber), [];
                            };
                        "background" === e
                            ? (i.collisionLayers.length > 0 && i.collisionLayers[i.collisionLayers.length - 1].length > 0 && logError("Background must be in a layer by itself.", i.lineNumber), (i.tokenIndex = 1))
                            : 0 !== i.tokenIndex && logError("Background must be in a layer by itself.", i.lineNumber);
                        let o = t(e);
                        if (0 === i.collisionLayers.length) return logError("no layers found.", i.lineNumber), "ERROR";
                        let r = [],
                            l = [];
                        for (let e = 0; e < o.length; e++) {
                            let n = o[e];
                            for (let e = 0; e <= i.collisionLayers.length - 1; e++) {
                                i.collisionLayers[e].indexOf(n) >= 0 && (e !== i.collisionLayers.length - 1 ? r.push(e) : l.push(e));
                            }
                        }
                        if (r.length > 0) {
                            let n = 'Object "' + e.toUpperCase() + '" included in multiple collision layers ( layers ';
                            for (let e = 0; e < r.length; e++) n += "#" + (r[e] + 1) + ", ";
                            (n += "#" + i.collisionLayers.length), logWarning(n + " ). You should fix this!", i.lineNumber);
                        }
                        if (i.current_line_wip_array.indexOf(e) >= 0) {
                            logWarning('Object "' + e.toUpperCase() + "\" included explicitly multiple times in the same layer. Don't do that innit.", i.lineNumber);
                        }
                        return i.current_line_wip_array.push(e), (i.collisionLayers[i.collisionLayers.length - 1] = i.collisionLayers[i.collisionLayers.length - 1].concat(o)), o.length > 0 ? "NAME" : "ERROR";
                    }
                }
                case "rules":
                    if (a) {
                        let n = reg_notcommentstart.exec(e.string)[0];
                        i.rules.push([n, i.lineNumber, s]), (i.tokenIndex = 0);
                    }
                    if (-4 === i.tokenIndex) return e.skipToEnd(), "MESSAGE";
                    if (e.match(/[\p{Z}\s]*->[\p{Z}\s]*/u, !0)) return "ARROW";
                    if ("[" === c || "|" === c || "]" === c || "+" === c) return "+" !== c && (i.tokenIndex = 1), e.next(), e.match(/[\p{Z}\s]*/u, !0), "BRACKET";
                    {
                        let n = e.match(/[^\[\|\]\p{Z}\s]*/u, !0)[0].trim();
                        return 0 === i.tokenIndex && reg_loopmarker.exec(n)
                            ? "BRACKET"
                            : (0 === i.tokenIndex && reg_ruledirectionindicators.exec(n)) || (1 === i.tokenIndex && reg_directions.exec(n))
                            ? (e.match(/[\p{Z}\s]*/u, !0), "DIRECTION")
                            : i.names.indexOf(n) >= 0
                            ? a
                                ? (logError("Objects cannot appear outside of square brackets in rules, only directions can.", i.lineNumber), "ERROR")
                                : (e.match(/[\p{Z}\s]*/u, !0), "NAME")
                            : "..." === n || "rigid" === n || "random" === n
                            ? "DIRECTION"
                            : commandwords.indexOf(n) >= 0
                            ? ("message" === n && (i.tokenIndex = -4), "COMMAND")
                            : (logError('Name "' + n + '", referred to in a rule, does not exist.', i.lineNumber), "ERROR");
                    }
                case "winconditions": {
                    if (a) {
                        let n = reg_notcommentstart
                            .exec(e.string)[0]
                            .split(/[\p{Z}\s]/u)
                            .filter(function (e) {
                                return "" !== e;
                            });
                        n.push(i.lineNumber), i.winconditions.push(n), (i.tokenIndex = -1);
                    }
                    i.tokenIndex++;
                    let n = e.match(/[\p{Z}\s]*[\p{L}\p{N}_]+[\p{Z}\s]*/u);
                    if (null === n) return logError("incorrect format of win condition.", i.lineNumber), e.match(reg_notcommentstart, !0), "ERROR";
                    {
                        let e = n[0].trim();
                        return 0 === i.tokenIndex
                            ? reg_winconditionquantifiers.exec(e)
                                ? "LOGICWORD"
                                : (logError('Expecting the start of a win condition ("ALL","SOME","NO") but got "' + e.toUpperCase() + "'.", i.lineNumber), "ERROR")
                            : 2 === i.tokenIndex
                            ? "on" != e
                                ? (logError('Expecting the word "ON" but got "' + e.toUpperCase() + '".', i.lineNumber), "ERROR")
                                : "LOGICWORD"
                            : 1 === i.tokenIndex || 3 === i.tokenIndex
                            ? -1 === i.names.indexOf(e)
                                ? (logError('Error in win condition: "' + e.toUpperCase() + '" is not a valid object name.', i.lineNumber), "ERROR")
                                : "NAME"
                            : (logError("Error in win condition: I don't know what to do with " + e.toUpperCase() + ".", i.lineNumber), "ERROR");
                    }
                }
                case "levels":
                    if (a) {
                        if (e.match(/[\p{Z}\s]*message\b[\p{Z}\s]*/u, !0)) {
                            i.tokenIndex = -4;
                            let n = ["\n", s.slice(e.pos).trim(), i.lineNumber];
                            return 0 === i.levels[i.levels.length - 1].length ? i.levels.splice(i.levels.length - 1, 0, n) : i.levels.push(n), "MESSAGE_VERB";
                        }
                        if (e.match(/[\p{Z}\s]*message[\p{Z}\s]*/u, !0)) {
                            logWarning("You probably meant to put a space after 'message' innit.  That's ok, I'll still interpret it as a message, but you probably want to put a space there.", i.lineNumber), (i.tokenIndex = -4);
                            let n = ["\n", s.slice(e.pos).trim(), i.lineNumber];
                            return 0 === i.levels[i.levels.length - 1].length ? i.levels.splice(i.levels.length - 1, 0, n) : i.levels.push(n), "MESSAGE_VERB";
                        }
                        {
                            let n = e.match(reg_notcommentstart, !1);
                            if (null === n || 0 === n.length)
                                return (
                                    logError("Detected a comment where I was expecting a level. Oh gosh; if this is to do with you using '(' as a character in the legend, please don't do that ^^", i.lineNumber),
                                    i.commentLevel++,
                                    e.skipToEnd(),
                                    "comment"
                                );
                            {
                                let e = n[0].trim();
                                i.tokenIndex = 2;
                                let t = i.levels[i.levels.length - 1];
                                "\n" == t[0]
                                    ? i.levels.push([i.lineNumber, e])
                                    : (0 == t.length && t.push(i.lineNumber),
                                      t.push(e),
                                      t.length > 1 && e.length != t[1].length && logWarning("Maps must be rectangular, yo (In a level, the length of each row must be the same).", i.lineNumber));
                            }
                        }
                    } else if (-4 == i.tokenIndex) return e.skipToEnd(), "MESSAGE";
                    if (2 === i.tokenIndex && !e.eol()) {
                        let n = e.peek();
                        return e.next(), i.abbrevNames.indexOf(n) >= 0 ? "LEVEL" : (logError('Key "' + n.toUpperCase() + '" not found. Do you need to add it to the legend, or define a new object?', i.lineNumber), "ERROR");
                    }
                    break;
                default:
                    if (((a || i.sol_after_comment) && (i.tokenIndex = 0), 0 != i.tokenIndex)) {
                        e.match(reg_notcommentstart, !0), i.tokenIndex++;
                        let n = i.metadata[i.metadata.length - 2],
                            t = i.metadata[i.metadata.length - 1];
                        if (i.tokenIndex > 2) return logWarning("Error: you can't embed comments in metadata values. Anything after the comment will be ignored.", i.lineNumber), "ERROR";
                        if ("background_color" === n || "text_color" === n) {
                            let e = t.trim().toLowerCase();
                            if (e in colorPalettes.arnecolors) return "COLOR COLOR-" + e.toUpperCase();
                            if ("transparent" === e) return "COLOR FADECOLOR";
                            if (4 === e.length || 7 === e.length) {
                                let n = e.match(/#[0-9a-fA-F]+/);
                                if (null !== n) return "MULTICOLOR" + n[0];
                            }
                        }
                        return "METADATATEXT";
                    }
                    {
                        let n = e.match(/[\p{Z}\s]*[\p{L}\p{N}_]+[\p{Z}\s]*/u);
                        if (null !== n) {
                            let t = n[0].trim();
                            if (a) {
                                if (["title", "author", "homepage", "background_color", "text_color", "key_repeat_interval", "realtime_interval", "again_interval", "flickscreen", "zoomscreen", "color_palette", "youtube"].indexOf(t) >= 0) {
                                    ("author" !== t && "homepage" !== t && "title" !== t) || (e.string = s),
                                        "youtube" === t &&
                                            logWarning(
                                                "Unfortunately, YouTube support hasn't been working properly for a long time - it was always a hack and it hasn't gotten less hacky over time, so I can no longer pretend to support it.",
                                                i.lineNumber
                                            );
                                    let n = e.match(reg_notcommentstart, !1);
                                    if (null !== n) {
                                        if ((i.metadata.push(t), i.metadata.push(n[0].trim()), t in i.metadata_lines)) {
                                            let e = i.metadata_lines[t];
                                            logWarning(`You've already defined a ${t.toUpperCase()} in the prelude on line <a onclick="jumpToLine(${e})>${e}</a>.`, i.lineNumber);
                                        }
                                        i.metadata_lines[t] = i.lineNumber;
                                    } else logError('MetaData "' + t + '" needs a value.', i.lineNumber);
                                    return (i.tokenIndex = 1), "METADATA";
                                }
                                if (["run_rules_on_level_start", "norepeat_action", "require_player_movement", "debug", "verbose_logging", "throttle_movement", "noundo", "noaction", "norestart", "scanline"].indexOf(t) >= 0) {
                                    i.metadata.push(t), i.metadata.push("true"), (i.tokenIndex = -1);
                                    let n = e.match(reg_notcommentstart, !1);
                                    if (null !== n) {
                                        let e = n[0].trim();
                                        logWarning("MetaData " + t.toUpperCase() + " doesn't take any parameters, but you went and gave it \"" + e + '".', i.lineNumber);
                                    }
                                    return "METADATA";
                                }
                                return logError("Unrecognised stuff in the prelude.", i.lineNumber), "ERROR";
                            }
                            return -1 == i.tokenIndex ? (logError('MetaData "' + t + '" has no parameters.', i.lineNumber), "ERROR") : "METADATA";
                        }
                        logError(`Unrecognised stuff "${e.string}" in the prelude.`, i.lineNumber);
                    }
            }
            return e.eol() ? null : e.eol() ? void 0 : (e.next(), null);
        },
        startState: function () {
            return {
                objects: {},
                lineNumber: 0,
                commentLevel: 0,
                section: "",
                visitedSections: [],
                line_should_end: !1,
                line_should_end_because: "",
                sol_after_comment: !1,
                objects_candname: "",
                objects_section: 0,
                objects_spritematrix: [],
                collisionLayers: [],
                tokenIndex: 0,
                current_line_wip_array: [],
                legend_synonyms: [],
                legend_aggregates: [],
                legend_properties: [],
                sounds: [],
                rules: [],
                names: [],
                winconditions: [],
                metadata: [],
                metadata_lines: {},
                original_case_names: {},
                original_line_numbers: {},
                abbrevNames: [],
                levels: [[]],
                subsection: "",
            };
        },
    };
};
window.CodeMirror.defineMode("puzzle", codeMirrorFn);
const OAUTH_CLIENT_ID = "211570277eb588cddf44";
function github_authURL() {
    let e = window.btoa(
        Array.prototype.map
            .call(window.crypto.getRandomValues(new Uint8Array(24)), function (e) {
                return String.fromCharCode(e);
            })
            .join("")
    );
    return "https://github.com/login/oauth/authorize?client_id=" + OAUTH_CLIENT_ID + "&scope=gist&state=" + e + "&allow_signup=true";
}
function github_signOut() {
    storage_remove("oauth_access_token");
}
function github_isSignedIn() {
    return "string" == typeof storage_get("oauth_access_token");
}
function github_load(e, n) {
    let t = "https://api.github.com/gists/" + e,
        o = new XMLHttpRequest();
    if (
        (o.open("GET", t),
        (o.onreadystatechange = function () {
            if (4 != o.readyState) return;
            if ("" === o.responseText) return void n(null, "GitHub request returned nothing.  A connection fault, maybe?");
            let e = window.parseInt(o.getResponseHeader("x-ratelimit-limit")),
                t = window.parseInt(o.getResponseHeader("x-ratelimit-used")),
                r = new Date(1e3 * window.parseInt(o.getResponseHeader("x-ratelimit-reset")));
            console.log("Rate limit used " + t + "/" + e + " (resets " + r.toISOString() + ")");
            let l = JSON.parse(o.responseText);
            if (403 === o.status) github_isSignedIn() || -1 === l.message.indexOf("rate limit") ? n(null, l.message) : n(null, "Exceeded GitHub rate limits. Try signing in from the editor.");
            else if (401 === o.status) github_signOut(), n(null, "Authorization check failed.  Try reloading or signing back in from the editor.");
            else if (o.status >= 500) n(null, "HTTP Error " + o.status + " - " + o.statusText + ".");
            else if (200 !== o.status && 201 !== o.status) n(null, "HTTP Error " + o.status + " - " + o.statusText);
            else {
                let e = JSON.parse(o.responseText).files["script.txt"].content;
                n(e, null);
            }
        }),
        github_isSignedIn())
    ) {
        let e = storage_get("oauth_access_token");
        o.setRequestHeader("Authorization", "Token " + e);
    }
    o.setRequestHeader("Content-type", "application/x-www-form-urlencoded"), o.send();
}
function github_save(e, n, t) {
    let o = storage_get("oauth_access_token");
    if ("string" != typeof o) return void printUnauthorized();
    let r = { description: e, public: !0, files: { "readme.txt": { content: "Play this game by pasting the script in http://www.puzzlescript.net/editor.html" }, "script.txt": { content: n } } },
        l = new XMLHttpRequest();
    l.open("POST", "https://api.github.com/gists"),
        (l.onreadystatechange = function () {
            if (4 != l.readyState) return;
            let e = JSON.parse(l.responseText);
            403 === l.status
                ? t(null, e.message)
                : 401 === l.status
                ? (github_signOut(), t(null, "Authorization check failed.  You have to log back into GitHub (or give it permission again or something)."))
                : l.status >= 500
                ? t(null, "HTTP Error " + l.status + " - " + l.statusText + ".")
                : 200 !== l.status && 201 !== l.status
                ? (github_signOut(), t(null, "HTTP Error " + l.status + " - " + l.statusText + ".  Try giving puzzlescript permission again, that might fix things..."))
                : t(e.id, null);
        }),
        l.setRequestHeader("Content-type", "application/x-www-form-urlencoded"),
        l.setRequestHeader("Authorization", "Token " + o);
    let i = JSON.stringify(r);
    l.send(i);
}
function isColor(e) {
    return (e = e.trim()) in colorPalettes.arnecolors || !!REGEX_HEX.test(e) || "transparent" === e;
}
function colorToHex(e, n) {
    return (n = n.trim()) in e ? e[n] : n;
}
function generateSpriteMatrix(e) {
    let n = [];
    for (let t = 0; t < e.length; t++) {
        let o = [];
        for (let n = 0; n < e.length; n++) {
            let r = e[t].charAt(n);
            "." === r ? o.push(-1) : o.push(r);
        }
        n.push(o);
    }
    return n;
}
let debugMode, colorPalette;
function generateExtraMembers(e) {
    e.idDict = [];
    let n = 0;
    for (let t = 0; t < e.collisionLayers.length; t++)
        for (let o = 0; o < e.collisionLayers[t].length; o++) {
            let r = e.collisionLayers[t][o];
            if (r in e.objects) {
                let o = e.objects[r];
                (o.layer = t), (o.id = n), (e.idDict[n] = r), n++;
            }
        }
    e.objectCount = n;
    let t = e.collisionLayers.length,
        o = [];
    for (let e = 0; e < t; e++) o.push(-1);
    (STRIDE_OBJ = 0 | Math.ceil(e.objectCount / 32)),
        (STRIDE_MOV = 0 | Math.ceil(t / 5)),
        (LAYER_COUNT = t),
        (e.STRIDE_OBJ = STRIDE_OBJ),
        (e.STRIDE_MOV = STRIDE_MOV),
        (e.LAYER_COUNT = LAYER_COUNT),
        RebuildGameArrays(),
        (debugMode = !1),
        (verbose_logging = !1),
        (throttle_movement = !1),
        (colorPalette = colorPalettes.arnecolors);
    for (let n = 0; n < e.metadata.length; n += 2) {
        let t = e.metadata[n],
            o = e.metadata[n + 1];
        "color_palette" === t
            ? (o in colorPalettesAliases && (o = colorPalettesAliases[o]), void 0 === colorPalettes[o] ? logError('Palette "' + o + '" not found, defaulting to arnecolors.', 0) : (colorPalette = colorPalettes[o]))
            : "debug" === t
            ? IDE && !1 === unitTesting && ((debugMode = !0), (cache_console_messages = !0))
            : "verbose_logging" === t
            ? IDE && !1 === unitTesting && ((verbose_logging = !0), (cache_console_messages = !0))
            : "throttle_movement" === t && (throttle_movement = !0);
    }
    let r = [],
        l = {};
    const i = Object.keys(e.objects),
        s = i.length;
    for (let n = 0; n < s; n++) {
        const t = i[n];
        let s = e.objects[t];
        s.colors.length > 10 && logError("a sprite cannot have more than 10 colors.  Why you would want more than 10 is beyond me.", s.lineNumber + 1);
        for (let e = 0; e < s.colors.length; e++) {
            let n = s.colors[e];
            isColor(n) ? ((n = colorToHex(colorPalette, n)), (s.colors[e] = n)) : (logError('Invalid color specified for object "' + t + '", namely "' + s.colors[e] + '".', s.lineNumber + 1), (s.colors[e] = "#ff00ff"));
        }
        0 === s.colors.length && (logError('color not specified for object "' + t + '".', s.lineNumber), (s.colors = ["#ff00ff"])),
            0 === s.spritematrix.length
                ? (s.spritematrix = [
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                  ])
                : ((5 === s.spritematrix.length && 5 === s.spritematrix[0].length && 5 === s.spritematrix[1].length && 5 === s.spritematrix[2].length && 5 === s.spritematrix[3].length && 5 === s.spritematrix[4].length) ||
                      logWarning("Sprite graphics must be 5 wide and 5 high exactly.", s.lineNumber),
                  (s.spritematrix = generateSpriteMatrix(s.spritematrix)));
        let a = o.concat([]);
        (a[s.layer] = s.id), (l[t] = a), r.push([s.lineNumber, t]);
    }
    let a = !0;
    for (; a; ) {
        a = !1;
        for (let n = 0; n < e.legend_synonyms.length; n++) {
            let t = e.legend_synonyms[n],
                o = t[0],
                i = t[1];
            (o in l && void 0 !== l[o]) || void 0 === l[i] || ((a = !0), (l[o] = l[i]), r.push([t.lineNumber, o]));
        }
        for (let n = 0; n < e.legend_aggregates.length; n++) {
            let t = e.legend_aggregates[n],
                i = t[0],
                s = t.slice(1),
                c = !0;
            for (let e = 0; e < s.length; e++) {
                if (void 0 === l[s[e]]) {
                    c = !1;
                    break;
                }
            }
            if ((!(i in l) || void 0 === l[i]) && c) {
                let n = o.concat([]);
                for (let o = 1; o < t.length; o++) {
                    let r = t[o],
                        l = e.objects[r];
                    if ((void 0 === l && logError("Object not found with name " + r, e.lineNumber), -1 === n[l.layer])) n[l.layer] = l.id;
                    else if (void 0 === l.layer) logError('Object "' + r.toUpperCase() + '" has been defined, but not assigned to a layer.', t.lineNumber);
                    else {
                        logError(
                            'Trying to create an aggregate object (something defined in the LEGEND section using AND) with both "' +
                                r.toUpperCase() +
                                '" and "' +
                                e.idDict[n[l.layer]].toUpperCase() +
                                "\", which are on the same layer and therefore can't coexist.",
                            t.lineNumber
                        );
                    }
                }
                (a = !0), (l[t[0]] = n), r.push([t.lineNumber, i]);
            }
        }
    }
    r.sort((e, n) => e[0] - n[0]), (r = r.map((e) => e[1])), (e.glyphDict = l), (e.glyphOrder = r);
    let c = {};
    for (let n = 0; n < e.legend_aggregates.length; n++) {
        let t = e.legend_aggregates[n];
        c[t[0]] = t.slice(1);
    }
    e.aggregatesDict = c;
    let d = {};
    for (let n = 0; n < e.legend_properties.length; n++) {
        let t = e.legend_properties[n];
        d[t[0]] = t.slice(1);
    }
    e.propertiesDict = d;
    let u = {};
    for (let n = 0; n < e.legend_synonyms.length; n++) {
        let t = e.legend_synonyms[n],
            o = t[0],
            r = t[1];
        r in c ? (c[o] = c[r]) : r in d ? (d[o] = d[r]) : o !== r && (u[o] = r);
    }
    e.synonymsDict = u;
    let h = !0;
    for (; h; ) {
        h = !1;
        const e = Object.keys(u),
            n = e.length;
        for (let t = 0; t < n; t++) {
            const n = e[t];
            let o = u[n];
            o in d ? (delete u[n], (d[n] = d[o]), (h = !0)) : o in c ? (delete c[n], (c[n] = c[o]), (h = !0)) : o in u && (u[n] = u[o]);
        }
        const t = Object.keys(d),
            o = t.length;
        for (let e = 0; e < o; e++) {
            const n = t[e];
            let o = d[n];
            for (let e = 0; e < o.length; e++) {
                let t = o[e];
                if (t in u) (o[e] = u[t]), (h = !0);
                else if (t in d) {
                    o.splice(e, 1);
                    let n = d[t];
                    for (let e = 0; e < n.length; e++) {
                        let t = n[e];
                        -1 === o.indexOf(t) && o.push(t);
                    }
                    h = !0;
                }
                t in c && logError('Trying to define property "' + n.toUpperCase() + '" in terms of aggregate "' + t.toUpperCase() + '".');
            }
        }
        const r = Object.keys(c),
            l = r.length;
        for (let e = 0; e < l; e++) {
            const n = r[e];
            let t = c[n];
            for (let e = 0; e < t.length; e++) {
                let o = t[e];
                if (o in u) (t[e] = u[o]), (h = !0);
                else if (o in c) {
                    t.splice(e, 1);
                    let n = c[o];
                    for (let e = 0; e < n.length; e++) {
                        let o = n[e];
                        -1 === t.indexOf(o) && t.push(o);
                    }
                    h = !0;
                }
                o in d && logError('Trying to define aggregate "' + n.toUpperCase() + '" in terms of property "' + o.toUpperCase() + '".');
            }
        }
    }
    e.propertiesSingleLayer = {};
    const g = Object.keys(d),
        p = g.length;
    for (let n = 0; n < p; n++) {
        const t = g[n];
        let o = d[t],
            r = !0;
        for (let n = 1; n < o.length; n++)
            if (e.objects[o[n - 1]].layer !== e.objects[o[n]].layer) {
                r = !1;
                break;
            }
        r && (e.propertiesSingleLayer[t] = e.objects[o[0]].layer);
    }
    let f, m;
    if ((void 0 === e.idDict[0] && e.collisionLayers.length > 0 && logError("You need to have some objects defined"), void 0 === e.objects.background))
        if ("background" in e.synonymsDict) {
            let n = e.synonymsDict.background,
                t = e.objects[n];
            (f = t.id), (m = t.layer);
        } else if ("background" in e.propertiesDict) {
            let n = e.propertiesDict.background,
                t = n[0],
                o = e.objects[t];
            (f = o.id), (m = o.layer);
            for (let t = 1; t < n.length; t++) {
                let o = n[t];
                if (e.objects[o].layer !== m) {
                    logError("Background objects must be on the same layer", e.original_line_numbers.background);
                }
            }
        } else if ("background" in e.aggregatesDict) {
            let n = e.objects[e.idDict[0]];
            (f = n.id), (m = n.layer), logError("background cannot be an aggregate (declared with 'and'), it has to be a simple type, or property (declared in terms of others using 'or').", e.original_line_numbers.background);
        } else {
            let n = e.objects[e.idDict[0]];
            null != n && ((f = n.id), (m = n.layer)), logError("Seriously, you have to define something to be the background.");
        }
    else (f = e.objects.background.id), (m = e.objects.background.layer);
    (e.backgroundid = f), (e.backgroundlayer = m);
}
function levelFromString(e, n) {
    let t = e.backgroundlayer,
        o = (e.backgroundid, e.layerMasks[t]),
        r = new Level(n[0], n[1].length, n.length - 1, e.collisionLayers.length, null);
    r.objects = new Int32Array(r.width * r.height * STRIDE_OBJ);
    for (let t = 0; t < r.width; t++)
        for (let o = 0; o < r.height; o++) {
            let l = n[o + 1].charAt(t);
            0 === l.length && (l = n[o + 1].charAt(n[o + 1].length - 1));
            let i = e.glyphDict[l];
            if (void 0 === i)
                return (
                    void 0 === e.propertiesDict[l]
                        ? logError('Error, symbol "' + l + '", used in map, not found.', n[0] + o)
                        : logError('Error, symbol "' + l + '" is defined using OR, and therefore ambiguous - it cannot be used in a map. Did you mean to define it in terms of AND?', n[0] + o),
                    r
                );
            let s = new BitVec(STRIDE_OBJ);
            i = i.concat([]);
            for (let e = 0; e < r.layerCount; e++) i[e] >= 0 && s.ibitset(i[e]);
            for (let e = 0; e < STRIDE_OBJ; ++e) r.objects[STRIDE_OBJ * (t * r.height + o) + e] = s.data[e];
        }
    const l = r.calcBackgroundMask(e);
    for (let e = 0; e < r.n_tiles; e++) {
        let n = r.getCell(e);
        o.anyBitsInCommon(n) || (n.ior(l), r.setCell(e, n));
    }
    return r;
}
function levelsToArray(e) {
    let n = e.levels,
        t = [];
    for (let o = 0; o < n.length; o++) {
        let r = n[o];
        if (0 !== r.length)
            if ("\n" === r[0]) {
                let e = { message: r[1] };
                (splitMessage = wordwrap(e.message, intro_template[0].length)), splitMessage.length > 12 && logWarning("Message too long to fit on screen.", r[2]), t.push(e);
            } else {
                let n = levelFromString(e, r);
                t.push(n);
            }
    }
    e.levels = t;
}
function directionalRule(e) {
    for (let n = 0; n < e.lhs.length; n++) {
        let t = e.lhs[n];
        if (t.length > 1) return !0;
        for (let e = 0; e < t.length; e++) {
            let n = t[e];
            for (let e = 0; e < n.length; e += 2) if (relativeDirections.indexOf(n[e]) >= 0) return !0;
        }
    }
    for (let n = 0; n < e.rhs.length; n++) {
        let t = e.rhs[n];
        for (let e = 0; e < t.length; e++) {
            let n = t[e];
            for (let e = 0; e < n.length; e += 2) if (relativeDirections.indexOf(n[e]) >= 0) return !0;
        }
    }
    return !1;
}
function findIndexAfterToken(e, n, t) {
    e = e.toLowerCase();
    let o = 0;
    for (let r = 0; r <= t; r++) {
        let t = n[r];
        o = e.indexOf(t, o) + t.length;
    }
    return o;
}
function rightBracketToRightOf(e, n) {
    for (; n < e.length; n++) if ("]" === e[n]) return !0;
    return !1;
}
function processRuleString(e, n, t) {
    let o = e[0],
        r = e[1],
        l = e[2];
    (o = o.replace(/\[/g, " [ ").replace(/\]/g, " ] ").replace(/\|/g, " | ").replace(/\-\>/g, " -> ")), (o = o.trim()), "+" === o[0] && (o = o.substring(0, 1) + " " + o.substring(1, o.length));
    let i = o.split(/\s/).filter(function (e) {
        return "" !== e;
    });
    0 === i.length && logError("Spooky error!  Empty line passed to rule function.", r);
    let s = 0,
        a = [],
        c = null,
        d = [],
        u = !1,
        h = !1,
        g = [],
        p = [],
        f = !1,
        m = !1,
        _ = r,
        v = [],
        y = !1,
        b = !1;
    if (1 === i.length) {
        if ("startloop" === i[0]) {
            return { bracket: 1 };
        }
        if ("endloop" === i[0]) {
            return { bracket: -1 };
        }
    }
    -1 === i.indexOf("->") && logError("A rule has to have an arrow in it.  There's no arrow here! Consider reading up about rules - you're clearly doing something weird", r), (c = []);
    let w = 0;
    for (let e = 0; e < i.length; e++) {
        let o = i[e];
        switch (s) {
            case 0:
                "+" === o
                    ? ((b = !0),
                      _ === r
                          ? (0 === t.length && (logError('The "+" symbol, for joining a rule with the group of the previous rule, needs a previous rule to be applied to.', r), (b = !1)),
                            0 !== e && logError('The "+" symbol, for joining a rule with the group of the previous rule, must be the first symbol on the line ', r),
                            b && (_ = t[t.length - 1].groupNumber))
                          : logError('Two "+"s (the "append to previous rule group" symbol) applied to the same rule.', r))
                    : o in directionaggregates
                    ? (a = a.concat(directionaggregates[o]))
                    : "late" === o
                    ? (f = !0)
                    : "rigid" === o
                    ? (m = !0)
                    : "random" === o
                    ? ((y = !0),
                      b &&
                          logError(
                              "A rule-group can only be marked random by the opening rule in the group (aka, a '+' and 'random' can't appear as rule modifiers on the same line).  Why? Well, you see \"random\" isn't a property of individual rules, but of whole rule groups.  It indicates that a single possible application of some rule from the whole group should be applied at random.",
                              r
                          ))
                    : simpleAbsoluteDirections.indexOf(o) >= 0
                    ? a.push(o)
                    : simpleRelativeDirections.indexOf(o) >= 0
                    ? logError(
                          'You cannot use relative directions ("^v<>") to indicate in which direction(s) a rule applies.  Use absolute directions indicators (Up, Down, Left, Right, Horizontal, or Vertical, for instance), or, if you want the rule to apply in all four directions, do not specify directions',
                          r
                      )
                    : "[" === o
                    ? (0 === a.length && (a = a.concat(directionaggregates.orthogonal)), (s = 1), e--)
                    : logError(
                          "The start of a rule must consist of some number of directions (possibly 0), before the first bracket, specifying in what directions to look (with no direction specified, it applies in all four directions).  It seems you've just entered \"" +
                              o.toUpperCase() +
                              '".',
                          r
                      );
                break;
            case 1:
                if ("[" === o)
                    w++,
                        w > 1 && logWarning("Multiple opening brackets without closing brackets.  Something fishy here.  Every '[' has to be closed by a ']', and you can't nest them.", r),
                        c.length > 0 && logError('Error, malformed cell rule - encountered a "["" before previous bracket was closed', r),
                        (u = !0),
                        (c = []);
                else if (reg_directions_only.exec(o))
                    c.length % 2 == 1
                        ? logError("Error, an item can only have one direction/action at a time, but you're looking for several at once!", r)
                        : u
                        ? f && "no" !== o && "random" !== o && "randomdir" !== o
                            ? logError("Movements cannot appear in late rules.", r)
                            : c.push(o)
                        : logWarning("Invalid syntax. Directions should be placed at the start of a rule.", r);
                else if ("|" === o)
                    u
                        ? c.length % 2 == 1
                            ? logError("In a rule, if you specify a movement, it has to act on an object.", r)
                            : (d.push(c), (c = []))
                        : logWarning('Janky syntax.  "|" should only be used inside cell rows (the square brackety bits).', r);
                else if ("]" === o) {
                    if ((w--, w < 0)) return logWarning("Multiple closing brackets without corresponding opening brackets.  Something fishy here.  Every '[' has to be closed by a ']', and you can't nest them.", r), null;
                    c.length % 2 == 1 ? ("..." === c[0] ? logError("Cannot end a rule with ellipses.", r) : logError("In a rule, if you specify a movement, it has to act on an object.", r)) : (d.push(c), (c = [])),
                        h ? p.push(d) : g.push(d),
                        (d = []),
                        (u = !1);
                } else if ("->" === o) {
                    if (_ !== r) {
                        t[t.length - 1].late !== f && logWarning("Oh gosh you can mix late and non-late rules in a rule-group if you really want to, but gosh why would you want to do that?  What do you expect to accomplish?", r);
                    }
                    if (u) logWarning('Encountered an unexpected "->" inside square brackets.  It\'s used to separate states, it has no place inside them >:| .', r);
                    else {
                        if (h) return logError('Error, you can only use "->" once in a rule; it\'s used to separate before and after states.', r), null;
                        h = !0;
                    }
                } else if (n.names.indexOf(o) >= 0)
                    if (u) {
                        for (let e = 0; e < c.length; e += 2)
                            c[e + 1] === o &&
                                (logError(`You cannot specify the same object more than once in a single cell (in this case ${o} occurs mutliple times).`, r),
                                o in n.propertiesDict &&
                                    logWarningNoLine(
                                        "( However, noticing that you're committing this crime with <i>properties</i>, and not being able to help but acknowledge that you <i>may</i> be trying to do something esoteric and <i>clever</i> with the property inference system,  I might be brought to suggest that you consider this: you can have multiple equivalent properties with different names. )"
                                    ));
                        c.length % 2 == 0 ? (c.push(""), c.push(o)) : c.length % 2 == 1 && c.push(o);
                    } else logWarning("Invalid token " + o.toUpperCase() + ". Object names should only be used within cells (square brackets).", r);
                else if ("..." === o) u ? (c.push(o), c.push(o)) : logWarning("Invalid syntax, ellipses should only be used within cells (square brackets).", r);
                else if (commandwords.indexOf(o) >= 0)
                    if (
                        (!1 === h
                            ? logError("Commands should only appear at the end of rules, not in or before the pattern-detection/-replacement sections.", r)
                            : (u || rightBracketToRightOf(i, e)) && logWarning("Commands should only appear at the end of rules, not in or before the pattern-detection/-replacement sections.", r),
                        "message" === o)
                    ) {
                        let n = findIndexAfterToken(l, i, e),
                            t = l.substring(n).trim();
                        "" === t && (t = " "), v.push([o, t]), (e = i.length);
                    } else {
                        if (commandwords_sfx.indexOf(o) >= 0) {
                            let e = !1;
                            for (let t = 0; t < n.sounds.length; t++) {
                                n.sounds[t][0][0] === o && (e = !0);
                            }
                            e || logWarning('Sound effect "' + o + '" not defined.', r);
                        }
                        v.push([o]);
                    }
                else logError('Error, malformed cell rule - was looking for cell contents, but found "' + o + '".  What am I supposed to do with this, eh, please tell me that.', r);
        }
    }
    if ((f && m && logError("Late rules cannot be marked as rigid (rigid rules are all about dealing with the consequences of unresolvable movements, and late rules can't even have movements).", r), g.length != p.length))
        (v.length > 0 && 0 === p.length) || logWarning("Error, when specifying a rule, the number of matches (square bracketed bits) on the left hand side of the arrow must equal the number on the right", r);
    else
        for (let e = 0; e < g.length; e++)
            g[e].length != p[e].length && (logError("In a rule, each pattern to match on the left must have a corresponding pattern on the right of equal length (number of cells).", r), (n.invalid = !0)),
                0 === g[e].length && logError("You have an totally empty pattern on the left-hand side.  This will match *everything*.  You certainly don't want this.");
    0 === g.length && logError("This rule refers to nothing.  What the heck? :O", r);
    let E = { directions: a, lhs: g, rhs: p, lineNumber: r, late: f, rigid: m, groupNumber: _, commands: v, randomRule: y };
    return !1 === directionalRule(E) && E.directions.length > 1 && E.directions.splice(1), E;
}
function deepCloneHS(e) {
    return e.map(function (e) {
        return e.map(function (e) {
            return e.slice();
        });
    });
}
function deepCloneRule(e) {
    return { direction: e.direction, lhs: deepCloneHS(e.lhs), rhs: deepCloneHS(e.rhs), lineNumber: e.lineNumber, late: e.late, rigid: e.rigid, groupNumber: e.groupNumber, commands: e.commands, randomRule: e.randomRule };
}
function checkSuperfluousCoincidences(e, n) {
    let t = new BitVec(0 | Math.ceil(LAYER_COUNT / 32)),
        o = new BitVec(STRIDE_OBJ);
    const r = n.length;
    for (let s = 0; s < r; s++) {
        let r = n[s];
        const a = r.rhs.length;
        for (let n = 0; n < a; n++) {
            let t = r.rhs[n];
            const o = t.length;
            for (let n = 0; n < o; n++) {
                let o = t[n];
                var l = [],
                    i = new BitVec(STRIDE_OBJ);
                for (let n = 0; n < o.length; n += 2) {
                    if (!o[n].startsWith("no")) {
                        let t = o[n + 1];
                        e.objects.hasOwnProperty(t) ? (l.push(t), i.ibitset(e.objects[t].id)) : e.propertiesSingleLayer.hasOwnProperty(t) || e.propertiesDict.hasOwnProperty(t);
                    }
                }
                for (let n = 0; n < o.length; n += 2) {
                    if (o[n].startsWith("no")) {
                        let t = o[n + 1];
                        e.objectMasks[t].anyBitsInCommon(i) &&
                            logError(`You have specified that there should be NO ${t.toUpperCase()} but there is also a requirement that ${l.join(", ").toUpperCase()} be here.  This is a mistake right?`, r.lineNumber);
                    }
                }
            }
        }
        for (let e = 0; e < a; e++) {
            let n = r.rhs[e];
            const t = n.length;
            for (let o = 0; o < t; o++) {
                let t = n[o];
                for (let n = 0; n < t.length; n += 2) {
                    if (t[n].startsWith("no")) {
                        let l = t[n + 1];
                        const i = r.lhs[e][o];
                        for (let e = 0; e < i.length; e += 2)
                            if (i[e].startsWith("no") && i[e + 1] === l) {
                                t.splice(n, 2);
                                break;
                            }
                    }
                }
            }
        }
        const c = r.lhs.length;
        for (let n = 0; n < c; n++) {
            let l = r.lhs[n];
            const i = l.length;
            for (let n = 0; n < i; n++) {
                let i = l[n];
                t.setZero(), o.setZero();
                let s = {};
                for (let n = 0; n < i.length; n += 2) {
                    if ("no" === i[n]) continue;
                    let r = i[n + 1];
                    if (e.propertiesSingleLayer.hasOwnProperty(r)) {
                        let n = e.propertiesSingleLayer[r];
                        t.ibitset(n);
                        let l = e.propertiesDict[r];
                        const i = l.length;
                        for (let n = 0; n < i; n++) {
                            const t = l[n],
                                r = e.objects[t];
                            o.ibitset(r.id);
                        }
                        s[n] = r;
                    } else if (e.objects.hasOwnProperty(r)) {
                        let n = e.objects[r].layer,
                            l = e.objects[r].id;
                        t.ibitset(n), o.ibitset(l), (s[n] = r);
                    } else if (e.aggregatesDict.hasOwnProperty(r)) {
                        let n = e.aggregatesDict[r];
                        for (let l = 0; l < n.length; l++) {
                            let i = e.objects[n[l]],
                                a = i.layer,
                                c = i.id;
                            t.ibitset(a), o.ibitset(c), (s[a] = r);
                        }
                    }
                }
                for (let n = 0; n < i.length; n += 2) {
                    if (i[n].startsWith("no")) {
                        let l = i[n + 1];
                        if (!e.objectMasks.hasOwnProperty(l)) continue;
                        let a = e.objectMasks[l],
                            c = new BitVec(0 | Math.ceil(LAYER_COUNT / 32));
                        for (let n = 0; n < e.layerMasks.length; n++) e.layerMasks[n].anyBitsInCommon(a) && c.ibitset(n);
                        const d = !a.anyBitsInCommon(o);
                        if (c.bitsSetInArray(t.data) && d) {
                            const t = [],
                                o = [];
                            for (let n = 0; n < e.layerMasks.length; n++) s[n] && (t.push(s[n]), o.push(n));
                            logWarning(
                                `You have specified that there should be NO ${l.toUpperCase()} but there is also a requirement that ${t.join(", ").toUpperCase()} be there, which collectively occupies the same layers (Layers ${o
                                    .map((e) => e + 1)
                                    .join(", ")}), so you can leave this out.`,
                                r.lineNumber
                            ),
                                i.splice(n, 2),
                                (n -= 2);
                        }
                    }
                }
            }
        }
    }
}
function rulesToArray(e) {
    let n = e.rules,
        t = [],
        o = [];
    for (let r = 0; r < n.length; r++) {
        let l = n[r][1],
            i = processRuleString(n[r], e, t);
        null !== i && (void 0 === i.bracket ? t.push(i) : o.push([l, i.bracket]));
    }
    (e.loops = o), checkSuperfluousCoincidences(e, t);
    let r = [];
    for (let e = 0; e < t.length; e++) {
        let n = t[e],
            o = n.directions;
        for (let e = 0; e < o.length; e++) {
            let t = o[e];
            if (t in directionaggregates && directionalRule(n)) {
                let e = directionaggregates[t];
                for (let t = 0; t < e.length; t++) {
                    let o = deepCloneRule(n);
                    (o.direction = e[t]), r.push(o);
                }
            } else {
                let e = deepCloneRule(n);
                (e.direction = t), r.push(e);
            }
        }
    }
    for (let n = 0; n < r.length; n++) {
        let t = r[n];
        if ((convertRelativeDirsToAbsolute(t), rewriteUpLeftRules(t), atomizeAggregates(e, t), e.invalid)) return;
        rephraseSynonyms(e, t);
    }
    let l = [];
    for (let n = 0; n < r.length; n++) {
        let t = r[n];
        l = l.concat(concretizeMovingRule(e, t, t.lineNumber));
    }
    let i = [];
    for (let n = 0; n < l.length; n++) {
        let t = l[n];
        i = i.concat(concretizePropertyRule(e, t, t.lineNumber));
    }
    for (let n = 0; n < i.length; n++) {
        let t = i[n];
        makeSpawnedObjectsStationary(e, t, t.lineNumber);
    }
    e.rules = i;
}
function containsEllipsis(e) {
    for (let n = 0; n < e.lhs.length; n++) for (let t = 0; t < e.lhs[n].length; t++) if ("..." === e.lhs[n][t][1]) return !0;
    return !1;
}
function rewriteUpLeftRules(e) {
    if (!containsEllipsis(e)) {
        if ("up" === e.direction) e.direction = "down";
        else {
            if ("left" !== e.direction) return;
            e.direction = "right";
        }
        for (let n = 0; n < e.lhs.length; n++) e.lhs[n].reverse(), e.rhs.length > 0 && e.rhs[n].reverse();
    }
}
function getPossibleObjectsFromCell(e, n) {
    const t = [];
    for (let o = 0; o < n.length; o += 2) {
        n[o];
        const r = n[o + 1];
        if (r in e.objects) t.push(r);
        else if (r in e.propertiesDict) {
            const n = e.propertiesDict[r];
            for (let e = 0; e < n.length; e++) {
                const o = n[e];
                t.push(o);
            }
        }
    }
    return t;
}
function getPropertiesFromCell(e, n) {
    let t = [];
    for (let o = 0; o < n.length; o += 2) {
        let r = n[o],
            l = n[o + 1];
        "random" !== r && l in e.propertiesDict && t.push(l);
    }
    return t;
}
function getMovings(e, n) {
    let t = [];
    for (let e = 0; e < n.length; e += 2) {
        let o = n[e],
            r = n[e + 1];
        o in directionaggregates && t.push([r, o]);
    }
    return t;
}
function concretizePropertyInCell(e, n, t) {
    for (let o = 0; o < e.length; o += 2) e[o + 1] === n && "random" !== e[o] && (e[o + 1] = t);
}
function concretizeMovingInCell(e, n, t, o) {
    for (let r = 0; r < e.length; r += 2) e[r] === n && e[r + 1] === t && (e[r] = o);
}
function concretizeMovingInCellByAmbiguousMovementName(e, n, t) {
    for (let o = 0; o < e.length; o += 2) e[o] === n && (e[o] = t);
}
function expandNoPrefixedProperties(e, n) {
    let t = [];
    for (let o = 0; o < n.length; o += 2) {
        let r = n[o],
            l = n[o + 1];
        if ("no" === r && l in e.propertiesDict) {
            let n = e.propertiesDict[l];
            for (let e = 0; e < n.length; e++) {
                let o = n[e];
                t.push(r), t.push(o);
            }
        } else t.push(r), t.push(l);
    }
    return t;
}
function concretizePropertyRule(e, n, t) {
    for (let t = 0; t < n.lhs.length; t++) {
        let o = n.lhs[t];
        for (let r = 0; r < o.length; r++) (o[r] = expandNoPrefixedProperties(e, o[r])), n.rhs.length > 0 && (n.rhs[t][r] = expandNoPrefixedProperties(e, n.rhs[t][r]));
    }
    let o,
        r = {};
    for (let t = 0; t < n.rhs.length; t++) {
        let o = n.lhs[t],
            l = n.rhs[t];
        for (let n = 0; n < l.length; n++) {
            let t = getPropertiesFromCell(e, o[n]),
                i = getPropertiesFromCell(e, l[n]);
            for (let e = 0; e < i.length; e++) {
                let n = i[e];
                -1 === t.indexOf(n) && (r[n] = !0);
            }
        }
    }
    let l = [n],
        i = !0;
    for (; i; ) {
        i = !1;
        for (let n = 0; n < l.length; n++) {
            let t = l[n];
            o = !1;
            for (let n = 0; n < t.lhs.length && !o; n++) {
                let s = t.lhs[n];
                for (let a = 0; a < s.length && !o; a++) {
                    let c = getPropertiesFromCell(e, s[a]);
                    for (let s = 0; s < c.length; ++s) {
                        let d = c[s];
                        if (e.propertiesSingleLayer.hasOwnProperty(d) && !0 !== r[d]) continue;
                        let u = e.propertiesDict[d];
                        (o = !0), (i = !0);
                        for (let e = 0; e < u.length; e++) {
                            let o = u[e],
                                r = deepCloneRule(t);
                            if (((r.propertyReplacement = {}), t.propertyReplacement)) {
                                const e = Object.keys(t.propertyReplacement),
                                    n = e.length;
                                for (let o = 0; o < n; o++) {
                                    const n = e[o],
                                        l = t.propertyReplacement[n];
                                    r.propertyReplacement[n] = [l[0], l[1]];
                                }
                            }
                            concretizePropertyInCell(r.lhs[n][a], d, o),
                                r.rhs.length > 0 && concretizePropertyInCell(r.rhs[n][a], d, o),
                                void 0 === r.propertyReplacement[d] ? (r.propertyReplacement[d] = [o, 1]) : (r.propertyReplacement[d][1] = r.propertyReplacement[d][1] + 1),
                                l.push(r);
                        }
                        break;
                    }
                }
            }
            o && (l.splice(n, 1), n--);
        }
    }
    for (let e = 0; e < l.length; e++) {
        let n = l[e];
        if (void 0 !== n.propertyReplacement)
            for (let e of Object.keys(n.propertyReplacement)) {
                let t = n.propertyReplacement[e],
                    o = t[0];
                if (1 === t[1])
                    for (let t = 0; t < n.rhs.length; t++) {
                        let r = n.rhs[t];
                        for (let n = 0; n < r.length; n++) {
                            concretizePropertyInCell(r[n], e, o);
                        }
                    }
            }
    }
    let s = "";
    for (let n = 0; n < l.length; n++) {
        let t = l[n];
        delete t.propertyReplacement;
        for (let n = 0; n < t.rhs.length; n++) {
            let o = t.rhs[n];
            for (let n = 0; n < o.length; n++) {
                let t = getPropertiesFromCell(e, o[n]);
                for (let e = 0; e < t.length; e++) r.hasOwnProperty(t[e]) && (s = t[e]);
            }
        }
    }
    return s.length > 0
        ? (logError(
              'This rule has a property on the right-hand side, "' +
                  s.toUpperCase() +
                  "\", that can't be inferred from the left-hand side.  (either for every property on the right there has to be a corresponding one on the left in the same cell, OR, if there's a single occurrence of a particular property name on the left, all properties of the same name on the right are assumed to be the same).",
              t
          ),
          [])
        : l;
}
function makeSpawnedObjectsStationary(e, n, t) {
    if (!n.late)
        for (let t = 0; t < n.rhs.length; t++) {
            let o = n.lhs[t],
                r = n.rhs[t];
            for (let n = 0; n < r.length; n++) {
                let t = r[n],
                    l = getPossibleObjectsFromCell(e, o[n]),
                    i = l.map((n) => e.objects[n].layer);
                for (let n = 0; n < t.length; n += 2) {
                    if ("" !== t[n]) continue;
                    let o = t[n + 1];
                    if (o in e.propertiesDict || l.indexOf(o) >= 0) continue;
                    let r = e.objects[o].layer;
                    -1 === i.indexOf(r) && (t[n] = "stationary");
                }
            }
        }
}
function concretizeMovingRule(e, n, t) {
    let o,
        r = [n],
        l = !0;
    for (; l; ) {
        l = !1;
        for (let n = 0; n < r.length; n++) {
            let t = r[n];
            o = !1;
            for (let n = 0; n < t.lhs.length; n++) {
                let i = t.lhs[n];
                for (let s = 0; s < i.length; s++) {
                    let a = getMovings(e, i[s]);
                    if (a.length > 0) {
                        (o = !0), (l = !0);
                        let e = a[0][0],
                            i = a[0][1],
                            c = directionaggregates[i];
                        for (let o = 0; o < c.length; o++) {
                            let l = c[o],
                                a = deepCloneRule(t);
                            if (((a.movingReplacement = {}), t.movingReplacement)) {
                                const e = Object.keys(t.movingReplacement),
                                    n = e.length;
                                for (let o = 0; o < n; o++) {
                                    const n = e[o];
                                    let r = t.movingReplacement[n];
                                    a.movingReplacement[n] = [r[0], r[1], r[2], r[3], r[4], r[5]];
                                }
                            }
                            if (((a.aggregateDirReplacement = {}), t.aggregateDirReplacement)) {
                                const e = Object.keys(t.aggregateDirReplacement),
                                    n = e.length;
                                for (let o = 0; o < n; o++) {
                                    const n = e[o];
                                    let r = t.aggregateDirReplacement[n];
                                    a.aggregateDirReplacement[n] = [r[0], r[1], r[2]];
                                }
                            }
                            if ((concretizeMovingInCell(a.lhs[n][s], i, e, l), a.rhs.length > 0 && concretizeMovingInCell(a.rhs[n][s], i, e, l), void 0 === a.movingReplacement[e + i])) a.movingReplacement[e + i] = [l, 1, i, e, n, s];
                            else {
                                let t = a.movingReplacement[e + i];
                                (n === t[4] && s === t[5]) || (t[1] = t[1] + 1);
                            }
                            void 0 === a.aggregateDirReplacement[i] ? (a.aggregateDirReplacement[i] = [l, 1, i]) : (a.aggregateDirReplacement[i][1] = a.aggregateDirReplacement[i][1] + 1), r.push(a);
                        }
                    }
                }
            }
            o && (r.splice(n, 1), n--);
        }
    }
    for (let e = 0; e < r.length; e++) {
        let n = r[e];
        if (void 0 === n.movingReplacement) continue;
        let t = {};
        const o = Object.keys(n.movingReplacement);
        if (n.movingReplacement) {
            const e = o.length;
            for (let t = 0; t < e; t++) {
                const e = o[t];
                let r = n.movingReplacement[e],
                    l = r[0],
                    i = r[1],
                    s = r[2],
                    a = r[3];
                if (1 === i)
                    for (let e = 0; e < n.rhs.length; e++) {
                        let t = n.rhs[e];
                        for (let e = 0; e < t.length; e++) {
                            concretizeMovingInCell(t[e], s, a, l);
                        }
                    }
            }
        }
        let l = {};
        const i = Object.keys(n.aggregateDirReplacement),
            s = i.length;
        for (let e = 0; e < s; e++) {
            const t = i[e];
            let o = n.aggregateDirReplacement[t],
                r = o[0],
                s = o[1],
                a = o[2];
            l[a] = a in l || 1 !== s ? "INVALID" : r;
        }
        const a = Object.keys(t),
            c = a.length;
        for (let e = 0; e < c; e++) {
            const o = a[e];
            if ("INVALID" !== o) {
                if (((concreteMovement = t[o]), "INVALID" === concreteMovement)) continue;
                for (let e = 0; e < n.rhs.length; e++) {
                    let t = n.rhs[e];
                    for (let e = 0; e < t.length; e++) {
                        concretizeMovingInCellByAmbiguousMovementName(t[e], o, concreteMovement);
                    }
                }
            }
        }
        const d = Object.keys(l),
            u = d.length;
        for (let e = 0; e < u; e++) {
            const t = d[e];
            if ("INVALID" !== t) {
                let e = l[t];
                if ("INVALID" === e) continue;
                for (let o = 0; o < n.rhs.length; o++) {
                    let r = n.rhs[o];
                    for (let n = 0; n < r.length; n++) {
                        concretizeMovingInCellByAmbiguousMovementName(r[n], t, e);
                    }
                }
            }
        }
    }
    let i = "";
    e: for (const n of r) {
        delete n.movingReplacement;
        for (const t of n.rhs)
            for (const n of t) {
                const t = getMovings(e, n);
                if (t.length > 0) {
                    i = t[0][1];
                    break e;
                }
            }
    }
    return (
        i.length > 0 &&
            (logError(
                'This rule has an ambiguous movement on the right-hand side, "' +
                    i +
                    "\", that can't be inferred from the left-hand side.  (either for every ambiguous movement associated to an entity on the right there has to be a corresponding one on the left attached to the same entity, OR, if there's a single occurrence of a particular ambiguous movement on the left, all properties of the same movement attached to the same object on the right are assumed to be the same (or something like that)).",
                t
            ),
            (e.invalid = !0)),
        r
    );
}
function rephraseSynonyms(e, n) {
    const t = (n) => {
        for (let t = 1; t < n.length; t += 2) {
            const o = n[t];
            o in e.synonymsDict && (n[t] = e.synonymsDict[o]);
        }
    };
    for (let e = 0; e < n.lhs.length; e++) {
        const o = n.lhs[e],
            r = n.rhs[e];
        for (let e = 0; e < o.length; e++) t(o[e]), n.rhs.length > 0 && t(r[e]);
    }
}
function atomizeAggregates(e, n) {
    const t = (t) => {
        for (let o = 0; o < t.length; o++) atomizeCellAggregates(e, t[o], n.lineNumber);
    };
    for (let e = 0; e < n.lhs.length; e++) t(n.lhs[e]);
    for (let e = 0; e < n.rhs.length; e++) t(n.rhs[e]);
}
function atomizeCellAggregates(e, n, t) {
    for (let o = 0; o < n.length; o += 2) {
        const r = n[o],
            l = n[o + 1];
        if (l in e.aggregatesDict) {
            "no" === r &&
                logError(
                    "You cannot use 'no' to exclude the aggregate object " +
                        l.toUpperCase() +
                        " (defined using 'AND'), only regular objects, or properties (objects defined using 'OR').  If you want to do this, you'll have to write it out yourself the long way.",
                    t
                );
            const i = e.aggregatesDict[l];
            n[o + 1] = i[0];
            for (let e = 1; e < i.length; e++) n.push(r), n.push(i[e]);
        }
    }
}
function convertRelativeDirsToAbsolute(e) {
    const n = e.direction;
    for (let t = 0; t < e.lhs.length; t++) {
        const o = e.lhs[t];
        for (let e = 0; e < o.length; e++) {
            absolutifyRuleCell(n, o[e]);
        }
    }
    for (let t = 0; t < e.rhs.length; t++) {
        const o = e.rhs[t];
        for (let e = 0; e < o.length; e++) {
            absolutifyRuleCell(n, o[e]);
        }
    }
}
const relativeDirs = ["^", "v", "<", ">", "parallel", "perpendicular"],
    relativeDict = {
        right: ["up", "down", "left", "right", "horizontal_par", "vertical_perp"],
        up: ["left", "right", "down", "up", "vertical_par", "horizontal_perp"],
        down: ["right", "left", "up", "down", "vertical_par", "horizontal_perp"],
        left: ["down", "up", "right", "left", "horizontal_par", "vertical_perp"],
    };
function absolutifyRuleCell(e, n) {
    for (let t = 0; t < n.length; t += 2) {
        const o = n[t],
            r = relativeDirs.indexOf(o);
        r >= 0 && (n[t] = relativeDict[e][r]);
    }
}
const dirMasks = {
    up: parseInt("00001", 2),
    down: parseInt("00010", 2),
    left: parseInt("00100", 2),
    right: parseInt("01000", 2),
    moving: parseInt("01111", 2),
    no: parseInt("00011", 2),
    randomdir: parseInt("00101", 2),
    random: parseInt("10010", 2),
    action: parseInt("10000", 2),
    "": parseInt("00000", 2),
};
function getOverlapObjectNames(e, n, t) {
    let o = [];
    for (let r = 0; r < e.objectCount; r++) n.get(r) && t.get(r) && o.push(e.idDict[r]);
    return o;
}
function rulesToMask(e) {
    const n = e.collisionLayers.length,
        t = Array(n).fill(null);
    e: for (let r = 0; r < e.rules.length; r++) {
        const l = e.rules[r];
        for (let i = 0; i < l.lhs.length; i++) {
            const [s, a] = [l.lhs[i], l.rhs[i]];
            for (let i = 0; i < s.length; i++) {
                const c = s[i],
                    d = [...t],
                    u = { objectsPresent: new BitVec(STRIDE_OBJ), objectsMissing: new BitVec(STRIDE_OBJ), movementsPresent: new BitVec(STRIDE_MOV), movementsMissing: new BitVec(STRIDE_MOV), objectlayers_l: new BitVec(STRIDE_MOV) },
                    h = [];
                for (let n = 0; n < c.length; n += 2) {
                    const [t, o] = [c[n], c[n + 1]];
                    if ("..." === t) {
                        if (2 !== c.length) {
                            logError("You can't have anything in with an ellipsis. Sorry.", l.lineNumber), e.rules.splice(r, 1), r--;
                            continue e;
                        }
                        if (((0 !== i && i !== s.length - 1) || logError("There's no point in putting an ellipsis at the very start or the end of a rule", l.lineNumber), l.rhs.length > 0)) {
                            const e = a[i];
                            (2 === e.length && "..." === e[0]) || logError("An ellipsis on the left must be matched by one in the corresponding place on the right.", l.lineNumber);
                        }
                        u.objectsPresent = ellipsisPattern;
                        break;
                    }
                    if ("random" === t) {
                        logError("RANDOM cannot be matched on the left-hand side, it can only appear on the right", l.lineNumber);
                        continue;
                    }
                    const g = e.objects[o],
                        p = e.objectMasks[o],
                        f = g ? 0 | g.layer : e.propertiesSingleLayer[o];
                    if ((void 0 === f && logError(`Oops! ${o.toUpperCase()} not assigned to a layer.`, l.lineNumber), "no" === t)) u.objectsMissing.ior(p);
                    else {
                        const e = d[f];
                        null !== e && (l.discard = [o.toUpperCase(), e.toUpperCase()]), (d[f] = o), g ? (u.objectsPresent.ior(p), u.objectlayers_l.ishiftor(31, 5 * f)) : h.push(p);
                        ("stationary" === t ? u.movementsMissing : u.movementsPresent).ishiftor("stationary" === t ? 31 : dirMasks[t], 5 * f);
                    }
                }
                if (u.objectsPresent === ellipsisPattern) {
                    s[i] = ellipsisPattern;
                    continue;
                }
                if (((s[i] = new CellPattern([u.objectsPresent, u.objectsMissing, h, u.movementsPresent, u.movementsMissing, null])), !u.objectsPresent.iszero() && u.objectsPresent.anyBitsInCommon(u.objectsMissing))) {
                    l.lineNumber;
                    var o = getOverlapObjectNames(e, u.objectsPresent, u.objectsMissing)
                        .map((e) => e + " NO " + e)
                        .join(", ")
                        .toUpperCase();
                    logWarning(`This rule has something amounting to "${o}" on the left-hand-side,which can never match, so the rule is getting removed during compilation.`, l.lineNumber), e.rules.splice(r, 1), r--;
                    continue;
                }
                if (0 === l.rhs.length) continue;
                const g = a[i];
                "..." === g[0] && "..." !== c[0] && logError("An ellipsis on the right must be matched by one in the corresponding place on the left.", l.lineNumber);
                for (let e = 0; e < g.length; e += 2) "..." === g[e] && 2 !== g.length && logError("You can't have anything in with an ellipsis. Sorry.", l.lineNumber);
                const p = [...t],
                    f = [...t],
                    m = {
                        objectsClear: new BitVec(STRIDE_OBJ),
                        objectsSet: new BitVec(STRIDE_OBJ),
                        movementsClear: new BitVec(STRIDE_MOV),
                        movementsSet: new BitVec(STRIDE_MOV),
                        objectlayers_r: new BitVec(STRIDE_MOV),
                        randomMask_r: new BitVec(STRIDE_OBJ),
                        postMovementsLayerMask_r: new BitVec(STRIDE_MOV),
                        randomDirMask_r: new BitVec(STRIDE_MOV),
                    };
                for (let n = 0; n < g.length; n += 2) {
                    const [t, o] = [g[n], g[n + 1]];
                    if ("..." === t) break;
                    if ("random" === t) {
                        if (o in e.objectMasks) {
                            const n = e.objectMasks[o];
                            m.randomMask_r.ior(n);
                            const t = e.propertiesDict.hasOwnProperty(o) ? e.propertiesDict[o] : [o];
                            1 === t.length &&
                                logWarning(
                                    `In this rule you're asking me to spawn a random ${o.toUpperCase()} for you, but that's already a concrete single object. You wanna be using random with properties (things defined in terms of OR in the legend) so there's some things to select between.`,
                                    l.lineNumber
                                );
                            for (const n of t) {
                                const t = 0 | e.objects[n].layer,
                                    o = p[t];
                                if (null !== o) {
                                    const [e, t] = [n.toUpperCase(), o.toUpperCase()];
                                    e !== t && logWarning(`This rule may try to spawn a ${e} with random, but also requires a ${t} be here, which is on the same layer - they shouldn't be able to coexist!`, l.lineNumber);
                                }
                                f[t] = n;
                            }
                        } else logError(`You want to spawn a random "${o.toUpperCase()}", but I don't know how to do that`, l.lineNumber);
                        continue;
                    }
                    const r = e.objects[o],
                        i = e.objectMasks[o],
                        s = r ? 0 | r.layer : e.propertiesSingleLayer[o];
                    if ("no" === t) m.objectsClear.ior(i);
                    else {
                        const n = p[s] || f[s];
                        null === n || l.hasOwnProperty("discard") || logError(`Rule matches object types that can't overlap: "${o.toUpperCase()}" and "${n.toUpperCase()}".`, l.lineNumber),
                            (p[s] = o),
                            t.length > 0 && m.postMovementsLayerMask_r.ishiftor(31, 5 * s);
                        const i = e.layerMasks[s];
                        r && (m.objectsSet.ibitset(r.id), m.objectsClear.ior(i), m.objectlayers_r.ishiftor(31, 5 * s)),
                            "stationary" === t ? m.movementsClear.ishiftor(31, 5 * s) : "randomdir" === t ? m.randomDirMask_r.ishiftor(dirMasks[t], 5 * s) : m.movementsSet.ishiftor(dirMasks[t], 5 * s);
                    }
                }
                u.objectsPresent.bitsSetInArray(m.objectsSet.data) || m.objectsClear.ior(u.objectsPresent), u.movementsPresent.bitsSetInArray(m.movementsSet.data) || m.movementsClear.ior(u.movementsPresent);
                for (let t = 0; t < n; t++) null !== d[t] && null === p[t] && (m.objectsClear.ior(e.layerMasks[t]), m.postMovementsLayerMask_r.ishiftor(31, 5 * t));
                u.objectlayers_l.iclear(m.objectlayers_r), m.postMovementsLayerMask_r.ior(u.objectlayers_l);
                if (!(m.objectsClear.iszero() && m.objectsSet.iszero() && m.movementsClear.iszero() && m.movementsSet.iszero() && m.postMovementsLayerMask_r.iszero() && m.randomMask_r.iszero() && m.randomDirMask_r.iszero())) {
                    s[i].replacement = new CellReplacement([m.objectsClear, m.objectsSet, m.movementsClear, m.movementsSet, m.postMovementsLayerMask_r, m.randomMask_r, m.randomDirMask_r]);
                }
            }
        }
    }
}
function cellRowMasks(e) {
    const n = [],
        t = e[1];
    for (let e = 0; e < t.length; e++) {
        const o = t[e],
            r = new BitVec(STRIDE_OBJ);
        for (let e = 0; e < o.length; e++) o[e] !== ellipsisPattern && r.ior(o[e].objectsPresent);
        n.push(r);
    }
    return n;
}
function cellRowMasks_Movements(e) {
    const n = [],
        t = e[1];
    for (let e = 0; e < t.length; e++) {
        const o = t[e],
            r = new BitVec(STRIDE_MOV);
        for (let e = 0; e < o.length; e++) o[e] !== ellipsisPattern && r.ior(o[e].movementsPresent);
        n.push(r);
    }
    return n;
}
function collapseRules(e) {
    for (let n = 0; n < e.length; n++) {
        const t = e[n];
        for (let e = 0; e < t.length; e++) {
            const n = t[e],
                o = [0, [], n.rhs.length > 0, n.lineNumber],
                r = [];
            for (let e = 0; e < n.lhs.length; e++) r.push(0);
            o[0] = dirMasks[n.direction];
            for (let e = 0; e < n.lhs.length; e++) {
                const t = n.lhs[e];
                for (let o = 0; o < t.length; o++)
                    t[o] === ellipsisPattern &&
                        (r[e]++,
                        r[e] > 2
                            ? logError("You can't use more than two ellipses in a single cell match pattern.", n.lineNumber)
                            : o > 0 && t[o - 1] === ellipsisPattern && logWarning("Why would you go and have two ellipses in a row like that? It's exactly the same as just having a single ellipsis, right?", n.lineNumber));
                o[1][e] = t;
            }
            o.push(r), o.push(n.groupNumber), o.push(n.rigid), o.push(n.commands), o.push(n.randomRule), o.push(cellRowMasks(o)), o.push(cellRowMasks_Movements(o)), (t[e] = new Rule(o));
        }
    }
}
function ruleGroupDiscardOverlappingTest(e) {
    if (0 === e.length) return;
    let n = [];
    for (let t = 0; t < e.length; t++) {
        let o = e[t];
        if (o.hasOwnProperty("discard")) {
            let r = 0 !== t && e[t - 1].lineNumber === o.lineNumber,
                l = t !== e.length - 1 && e[t + 1].lineNumber === o.lineNumber;
            e.splice(t, 1);
            let i = !1;
            for (let e = 0; e < n.length; e++) {
                let t = n[e];
                if (t[0] === o.discard[0] && t[1] === o.discard[1]) {
                    i = !0;
                    break;
                }
            }
            if ((i || n.push(o.discard), (!r && !l) || 0 === e.length)) {
                const e = n[0];
                let t = "";
                if (n.length > 1) {
                    t = " (ditto for ";
                    for (let e = 1; e < n.length; e++) {
                        e > 1 && ((t += ", "), e === n.length - 1 && (t += "and "));
                        const o = n[e];
                        if (((t += `${o[0]}/${o[1]}`), 3 === e && n.length > 4)) {
                            t += " etc.";
                            break;
                        }
                    }
                    t += ")";
                }
                logError(`${e[0]} and ${e[1]} can never overlap${t}, but this rule requires that to happen, so it's being culled.`, o.lineNumber);
            }
            t--;
        }
    }
}
function arrangeRulesByGroupNumber(e) {
    let n = {},
        t = {};
    for (let o = 0; o < e.rules.length; o++) {
        let r = e.rules[o],
            l = n;
        r.late && (l = t), void 0 === l[r.groupNumber] && (l[r.groupNumber] = []), l[r.groupNumber].push(r);
    }
    let o = [];
    const r = Object.keys(n),
        l = r.length;
    for (let e = 0; e < l; e++) {
        let t = n[r[e]];
        ruleGroupDiscardOverlappingTest(t), t.length > 0 && o.push(t);
    }
    let i = [];
    const s = Object.keys(t),
        a = s.length;
    for (let e = 0; e < a; e++) {
        let n = t[s[e]];
        ruleGroupDiscardOverlappingTest(n), n.length > 0 && i.push(n);
    }
    (e.rules = o), (e.lateRules = i);
}
function generateRigidGroupList(e) {
    let n = [],
        t = [],
        o = [],
        r = [],
        l = [];
    for (let i = 0; i < e.rules.length; i++) {
        let s = e.rules[i],
            a = !1;
        for (let e = 0; e < s.length; e++) {
            s[e].rigid && (a = !0);
        }
        if (((l[i] = a), a)) {
            let e = s[0].groupNumber;
            o[e] = i;
            let l = n.length;
            (t[i] = l), (r[e] = l), n.push(i);
        }
    }
    if (n.length > 30) {
        let t = n[30];
        logError("There can't be more than 30 rigid groups (rule groups containing rigid members).", e.rules[t][0].lineNumber);
    }
    (e.rigid = t.length > 0), (e.rigidGroups = l), (e.rigidGroupIndex_to_GroupIndex = n), (e.groupNumber_to_RigidGroupIndex = r), (e.groupIndex_to_RigidGroupIndex = t);
}
function isObjectDefined(e, n) {
    let t = n in e.objects || (void 0 !== e.aggregatesDict && n in e.aggregatesDict) || (void 0 !== e.propertiesDict && n in e.propertiesDict) || (void 0 !== e.synonymsDict && n in e.synonymsDict);
    if (void 0 !== e.legend_aggregates)
        for (let o = 0; o < e.legend_aggregates.length; o++)
            if (e.legend_aggregates[o][0] === n) {
                t = !0;
                break;
            }
    if (void 0 !== e.legend_properties)
        for (let o = 0; o < e.legend_properties.length; o++)
            if (e.legend_properties[o][0] === n) {
                t = !0;
                break;
            }
    if (void 0 !== e.legend_synonyms)
        for (let o = 0; o < e.legend_synonyms.length; o++)
            if (e.legend_synonyms[o][0] === n) {
                t = !0;
                break;
            }
    return t;
}
function getMaskFromName(e, n) {
    const t = new BitVec(STRIDE_OBJ);
    let o = !1;
    if (n in e.objects) {
        const o = e.objects[n];
        t.ibitset(o.id);
    }
    if (n in e.aggregatesDict) {
        const r = e.aggregatesDict[n];
        o = !0;
        for (let n = 0; n < r.length; n++) {
            const o = r[n],
                l = e.objects[o];
            t.ibitset(l.id);
        }
    }
    if (n in e.propertiesDict) {
        const o = e.propertiesDict[n];
        for (let n = 0; n < o.length; n++) {
            const r = o[n],
                l = e.objects[r];
            t.ibitset(l.id);
        }
    }
    if (n in e.synonymsDict) {
        const o = e.synonymsDict[n],
            r = e.objects[o];
        t.ibitset(r.id);
    }
    return t.iszero() && logErrorNoLine(`Error, didn't find any object called ${n}, either in the objects section, or the legends section.`), [o, t];
}
function generateMasks(e) {
    e.playerMask = getMaskFromName(e, "player");
    let n = [],
        t = e.collisionLayers.length;
    for (let o = 0; o < t; o++) {
        let t = new BitVec(STRIDE_OBJ);
        for (let n = 0; n < e.objectCount; n++) {
            let r = e.idDict[n],
                l = e.objects[r];
            l.layer === o && t.ibitset(l.id);
        }
        n.push(t);
    }
    e.layerMasks = n;
    let o = {};
    const r = Object.keys(e.objects),
        l = r.length;
    for (let n = 0; n < l; n++) {
        const t = r[n];
        let l = e.objects[t];
        (o[t] = new BitVec(STRIDE_OBJ)), o[t].ibitset(l.id);
    }
    let i = e.legend_synonyms.concat(e.legend_properties);
    i.sort(function (e, n) {
        return e.lineNumber - n.lineNumber;
    });
    for (let e = 0; e < i.length; e++) {
        let n = i[e];
        if (2 === n.length) o[n[0]] = o[n[1]];
        else {
            let e = new BitVec(STRIDE_OBJ);
            for (let t = 1; t < n.length; t++) {
                let r = n[t];
                e.ior(o[r]);
            }
            o[n[0]] = e;
        }
    }
    let s = new BitVec(STRIDE_OBJ);
    s.inot(), (o["\nall\n"] = s), (e.objectMasks = o), (e.aggregateMasks = {});
    for (let n of Object.keys(e.aggregatesDict)) {
        let t = e.aggregatesDict[n],
            r = new BitVec(STRIDE_OBJ);
        for (let n = 0; n < t.length; n++) {
            let l = t[n];
            e.objects[l];
            r.ior(o[l]);
        }
        e.aggregateMasks[n] = r;
    }
}
function checkObjectsAreLayered(e) {
    const n = Object.keys(e.objects),
        t = n.length;
    e: for (let o = 0; o < t; o++) {
        const t = n[o];
        let r = !1;
        for (let n = 0; n < e.collisionLayers.length; n++) {
            const o = e.collisionLayers[n];
            for (let e = 0; e < o.length; e++)
                if (o[e] === t) {
                    r = !0;
                    continue e;
                }
        }
        if (!1 === r) {
            const n = e.objects[t];
            logError('Object "' + t.toUpperCase() + '" has been defined, but not assigned to a layer.', n.lineNumber);
        }
    }
}
function isInt(e) {
    return (
        !isNaN(e) &&
        (function (e) {
            return (0 | e) === e;
        })(parseFloat(e))
    );
}
function twiddleMetaData(e) {
    const n = {};
    for (let t = 0; t < e.metadata.length; t += 2) {
        const o = e.metadata[t],
            r = e.metadata[t + 1];
        n[o] = r;
    }
    const t = function (e, n) {
            if (!isFinite(e) || !isInt(e)) return logWarning(`Wasn't able to make sense of "${e}" as a (whole number) dimension.`, n), NaN;
            const t = parseInt(e);
            return isNaN(t) && logWarning(`Wasn't able to make sense of "${e}" as a dimension.`, n), t <= 0 && logWarning(`The dimension given to me (you gave "${e}") is baad - it should be greater than 0.`, n), t;
        },
        o = function (e, n) {
            const o = e.split("x");
            if (2 !== o.length) return logWarning("Dimensions must be of the form AxB.", n), null;
            {
                const r = [t(o[0], n), t(o[1], n)];
                return !isFinite(o[0]) || !isFinite(o[1]) || isNaN(r[0]) || isNaN(r[1])
                    ? (logWarning(`Couldn't understand the dimensions given to me (you gave "${e}") - should be of the form AxB.`, n), null)
                    : ((r[0] <= 0 || r[1] <= 0) && logWarning(`The dimensions given to me (you gave "${e}") are baad - they should be > 0.`, n), r);
            }
        };
    if (void 0 !== n.flickscreen) {
        const t = n.flickscreen;
        (n.flickscreen = o(t, e.metadata_lines.flickscreen)), null === n.flickscreen && delete n.flickscreen;
    }
    if (void 0 !== n.zoomscreen) {
        const t = n.zoomscreen;
        (n.zoomscreen = o(t, e.metadata_lines.zoomscreen)), null === n.zoomscreen && delete n.zoomscreen;
    }
    e.metadata = n;
}
function processWinConditions(e) {
    let n = [];
    for (let t = 0; t < e.winconditions.length; t++) {
        let o = e.winconditions[t];
        if (0 === o.length) return;
        let r = 0;
        switch (o[0]) {
            case "no":
                r = -1;
                break;
            case "all":
                r = 1;
        }
        let l,
            i = o[o.length - 1],
            s = o[1];
        l = 5 === o.length ? o[3] : "\nall\n";
        let a = 0,
            c = 0,
            d = !1,
            u = !1;
        o.length <= 2
            ? logError('Win conditions is badly formatted - needs to look something like "No Fruit", "All Target On Crate", "Some Fruit", "Some Gold on Chest", "No Gold on Chest", or the like.', i)
            : s in e.objectMasks
            ? ((d = !1), (a = e.objectMasks[s]))
            : s in e.aggregateMasks
            ? ((d = !0), (a = e.aggregateMasks[s]))
            : logError('Unwelcome term "' + s + "\" found in win condition. I don't know what I'm supposed to do with this. ", i),
            l in e.objectMasks
                ? ((u = !1), (c = e.objectMasks[l]))
                : l in e.aggregateMasks
                ? ((u = !0), (c = e.aggregateMasks[l]))
                : logError('Unwelcome term "' + s + "\" found in win condition. I don't know what I'm supposed to do with this. ", i);
        let h = [r, a, c, i, d, u];
        n.push(h);
    }
    e.winconditions = n;
}
function printCellRow(e) {
    let n = "[ ";
    for (let t = 0; t < e.length; t++) {
        t > 0 && (n += "| ");
        let o = e[t];
        for (let e = 0; e < o.length; e += 2) {
            let t = o[e],
                r = o[e + 1];
            n += "..." === t ? t + " " : t + " " + r + " ";
        }
    }
    return (n += "] "), n;
}
function cacheRuleStringRep(e) {
    let n = "(<a onclick=\"jumpToLine('" + e.lineNumber.toString() + '\');"  href="javascript:void(0);">' + e.lineNumber + "</a>) ",
        t = !1;
    for (let n = 0; n < e.lhs.length; n++) {
        if (e.lhs[n].length > 1) {
            t = !0;
            break;
        }
    }
    t && (n += e.direction.toString().toUpperCase() + " "), e.rigid && (n = "RIGID " + n + " "), e.randomRule && (n = "RANDOM " + n + " "), e.late && (n = "LATE " + n + " ");
    for (let t = 0; t < e.lhs.length; t++) {
        n += printCellRow(e.lhs[t]);
    }
    n += "-> ";
    for (let t = 0; t < e.rhs.length; t++) {
        n += printCellRow(e.rhs[t]);
    }
    for (let t = 0; t < e.commands.length; t++) {
        let o = e.commands[t];
        1 === o.length ? (n += o[0].toString()) : (n = n + "(" + o[0].toString() + ", " + o[1].toString() + ") ");
    }
    e.stringRep = n;
}
function cacheAllRuleNames(e) {
    for (let n = 0; n < e.rules.length; n++) {
        cacheRuleStringRep(e.rules[n]);
    }
}
function printRules(e) {
    let n = "",
        t = -1,
        o = !0,
        r = 0;
    for (let l = 0; l < e.rules.length; l++) {
        let i = e.rules[l];
        if (!o && t + 1 < e.loops.length) {
            e.loops[t + 1][0] < i.lineNumber && ((n += "ENDLOOP<br>"), (o = !0), t++);
        }
        if (o) {
            if (t + 1 < e.loops.length) {
                let n = e.loops[t + 1],
                    o = e.loops[t + 2];
                for (; t + 1 < e.loops.length && o[0] < i.lineNumber; ) (t += 2), (n = e.loops[t + 1]), (o = e.loops[t + 2]);
            }
            if (t + 1 < e.loops.length) {
                e.loops[t + 1][0] < i.lineNumber && ((n += "STARTLOOP<br>"), (o = !1), t++);
            }
        }
        if (i.hasOwnProperty("discard")) r++;
        else {
            (n += l > 0 && e.rules[l - 1].groupNumber === i.groupNumber ? "+ " : "&nbsp;&nbsp;"), (n += i.stringRep + "<br>");
        }
    }
    o || (n += "ENDLOOP<br>"), (n += "===========<br>"), (n = "<br>Rule Assembly : (" + (e.rules.length - r) + " rules)<br>===========<br>" + n), consolePrint(n);
}
function removeDuplicateRules(e) {
    let n = {},
        t = -1;
    for (let o = e.rules.length - 1; o >= 0; o--) {
        let r = e.rules[o],
            l = r.groupNumber;
        l !== t && (n = {});
        let i = r.stringRep;
        n.hasOwnProperty(i) ? e.rules.splice(o, 1) : (n[i] = !0), (t = l);
    }
}
function calculateLoopPoints(e, n) {
    let t = {};
    for (let o = 0; o < e.loops.length; o += 2) {
        let r = e.loops[o][0],
            l = e.loops[o + 1][0],
            i = -1;
        for (let e = 0; e < n.length; e++) {
            let o = n[e],
                s = o[0],
                a = o[o.length - 1],
                c = s.lineNumber,
                d = a.lineNumber;
            ((c <= r && r <= d) || (c <= l && l <= d)) && logError("Found a loop point in the middle of a rule. You probably don't want to do this, right?", r);
            let u = r <= c && c <= l;
            if (l < c) break;
            u && (-1 === i && (i = e), e > 0 && void 0 !== t[e - 1] && t[e - 1] === i && (t[e - 1] = void 0), (t[e] = i));
        }
    }
    return t;
}
function generateLoopPoints(e) {
    for (let n = 0; n < e.loops.length; n++) {
        let t = e.loops[n];
        n % 2 == 0 ? -1 === t[1] && logError("Found an ENDLOOP, but I'm not in a loop?", t[0]) : 1 === t[1] && logError("Found a STARTLOOP, but I'm already inside a loop? (Puzzlescript can't nest loops, FWIW).", t[0]);
    }
    e.loops.length % 2 != 0 && (logError("Yo I found a STARTLOOP without a corresponding ENDLOOP.", e.loops[e.loops.length - 1][0]), e.loops.push([e.loops[e.loops.length - 1][0], -1])),
        (e.loopPoint = calculateLoopPoints(e, e.rules)),
        (e.lateLoopPoint = calculateLoopPoints(e, e.lateRules));
}
let soundDirectionIndicatorMasks = {
        up: parseInt("00001", 2),
        down: parseInt("00010", 2),
        left: parseInt("00100", 2),
        right: parseInt("01000", 2),
        horizontal: parseInt("01100", 2),
        vertical: parseInt("00011", 2),
        orthogonal: parseInt("01111", 2),
        ___action____: parseInt("10000", 2),
    },
    soundDirectionIndicators = ["up", "down", "left", "right", "horizontal", "vertical", "orthogonal", "___action____"];
function generateSoundData(e) {
    let n = {},
        t = [],
        o = [],
        r = e.collisionLayers.map((e) => []),
        l = [];
    for (let i = 0; i < e.sounds.length; i++) {
        let s = e.sounds[i];
        if (s.length <= 1) continue;
        let a = s[s.length - 1];
        if (2 === s.length) {
            logWarning("incorrect sound declaration.", a);
            continue;
        }
        const c = s[0][0].trim(),
            d = s[0][1].trim(),
            u = s[1][0].trim();
        s[1][1].trim();
        let h = s[s.length - 2][0];
        if (("SOUND" !== s[s.length - 2][1] && logError('Expecting sfx data, instead found "' + h + '".', a), "SOUNDEVENT" === d))
            s.length > 4 ? logError("too much stuff to define a sound event.", a) : s.length > 3 && logWarning("too much stuff to define a sound event.", a),
                void 0 !== n[c] && logWarning(c.toUpperCase() + " already declared.", a),
                (n[c] = h);
        else {
            let n = c,
                i = u,
                d = [];
            for (let e = 2; e < s.length - 2; e++) "DIRECTION" === s[e][1] ? d.push(s[e][0]) : logError('Expected a direction here, but found instead "$(sound[j][0])".', a);
            d.length > 0 &&
                "move" !== i &&
                "cantmove" !== i &&
                logError("Incorrect sound declaration - cannot have directions (UP/DOWN/etc.) attached to non-directional sound verbs (CREATE is not directional, but MOVE is directional).", a),
                "action" === i && ((i = "move"), (d = ["___action____"])),
                0 === d.length && (d = ["orthogonal"]),
                n in e.aggregatesDict
                    ? logError('cannot assign sound events to aggregate objects (declared with "and"), only to regular objects, or properties, things defined in terms of "or" ("' + n + '").', a)
                    : n in e.objectMasks || logError('Object "' + n + '" not found.', a);
            let g = e.objectMasks[n],
                p = 0;
            for (let e = 0; e < d.length; e++) {
                d[e] = d[e].trim();
                let n = d[e];
                if (-1 === soundDirectionIndicators.indexOf(n)) logError('Was expecting a direction, instead found "' + n + '".', a);
                else {
                    p |= soundDirectionIndicatorMasks[n];
                }
            }
            let f = [n],
                m = !0;
            for (; m; ) {
                m = !1;
                for (let n = 0; n < f.length; n++) {
                    let t = f[n];
                    if (t in e.synonymsDict) (f[n] = e.synonymsDict[t]), (m = !0);
                    else if (t in e.propertiesDict) {
                        m = !0;
                        let o = e.propertiesDict[t];
                        f.splice(n, 1), n--;
                        for (let e = 0; e < o.length; e++) f.push(o[e]);
                    }
                }
            }
            if ("move" === i || "cantmove" === i)
                for (let n = 0; n < f.length; n++) {
                    let t = f[n],
                        o = e.objects[t],
                        s = o.layer,
                        a = new BitVec(STRIDE_OBJ);
                    if ((a.ibitset(o.id), void 0 === s)) continue;
                    let c = new BitVec(STRIDE_MOV);
                    c.ishiftor(p, 5 * s);
                    let d = { objectMask: a, directionMask: c, layer: s, seed: h };
                    "move" === i ? r[s].push(d) : l.push(d);
                }
            switch (i) {
                case "create": {
                    let e = { objectMask: g, seed: h };
                    t.push(e);
                    break;
                }
                case "destroy": {
                    let e = { objectMask: g, seed: h };
                    o.push(e);
                    break;
                }
            }
        }
    }
    (e.sfx_Events = n), (e.sfx_CreationMasks = t), (e.sfx_DestructionMasks = o), (e.sfx_MovementMasks = r), (e.sfx_MovementFailureMasks = l);
}
function formatHomePage(e) {
    if (
        ("background_color" in e.metadata ? (e.bgcolor = colorToHex(colorPalette, e.metadata.background_color)) : (e.bgcolor = "#000000"),
        "text_color" in e.metadata ? (e.fgcolor = colorToHex(colorPalette, e.metadata.text_color)) : (e.fgcolor = "#FFFFFF"),
        !1 === isColor(e.fgcolor) &&
            (logError("text_color in incorrect format - found " + e.fgcolor + ", but I expect a color name (like 'pink') or hex-formatted color (like '#1412FA').  Defaulting to white.", e.metadata_lines.text_color),
            (e.fgcolor = "#FFFFFF")),
        !1 === isColor(e.bgcolor) &&
            (logError("background_color in incorrect format - found " + e.bgcolor + ", but I expect a color name (like 'pink') or hex-formatted color (like '#1412FA').  Defaulting to black.", e.metadata_lines.background_color),
            (e.bgcolor = "#000000")),
        canSetHTMLColors && ("background_color" in e.metadata && (document.body.style.backgroundColor = e.bgcolor), "text_color" in e.metadata))
    ) {
        let n = document.getElementById("separator");
        null != n && (n.style.color = e.fgcolor);
        let t = document.getElementsByTagName("a");
        for (let n = 0; n < t.length; n++) t[n].style.color = e.fgcolor;
        let o = document.getElementsByTagName("h1");
        for (let n = 0; n < o.length; n++) o[n].style.color = e.fgcolor;
    }
    if ("homepage" in e.metadata) {
        let n = e.metadata.homepage;
        (n = n.replace("http://", "")), (n = n.replace("https://", "")), (e.metadata.homepage = n);
    }
}
let MAX_ERRORS = 5;
function loadFile(e) {
    let n = new codeMirrorFn(),
        t = n.startState(),
        o = e.split("\n");
    for (let e = 0; e < o.length; e++) {
        let r = o[e],
            l = new CodeMirror.StringStream(r, 4);
        do {
            if ((n.token(l, t), errorCount > MAX_ERRORS)) return void consolePrint("too many errors, aborting compilation");
        } while (!1 === l.eol());
    }
    return (
        isObjectDefined(t, "player") || logErrorNoLine("Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!"),
        isObjectDefined(t, "background") || logErrorNoLine("Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!"),
        0 === t.collisionLayers.length
            ? (logError("No collision layers defined.  All objects need to be in collision layers."), null)
            : (generateExtraMembers(t),
              generateMasks(t),
              levelsToArray(t),
              rulesToArray(t),
              t.invalid > 0
                  ? null
                  : (cacheAllRuleNames(t),
                    removeDuplicateRules(t),
                    rulesToMask(t),
                    debugMode && printRules(t),
                    arrangeRulesByGroupNumber(t),
                    collapseRules(t.rules),
                    collapseRules(t.lateRules),
                    generateRigidGroupList(t),
                    processWinConditions(t),
                    checkObjectsAreLayered(t),
                    twiddleMetaData(t),
                    generateLoopPoints(t),
                    generateSoundData(t),
                    formatHomePage(t),
                    addSpecializedFunctions(t),
                    delete t.commentLevel,
                    delete t.line_should_end,
                    delete t.line_should_end_because,
                    delete t.sol_after_comment,
                    delete t.names,
                    delete t.abbrevNames,
                    delete t.objects_candname,
                    delete t.objects_section,
                    delete t.objects_spritematrix,
                    delete t.section,
                    delete t.subsection,
                    delete t.tokenIndex,
                    delete t.current_line_wip_array,
                    delete t.visitedSections,
                    delete t.loops,
                    t))
    );
}
function addSpecializedFunctions(e) {
    const n = Math.ceil(e.objectCount / 32),
        t = Math.ceil(e.collisionLayers.length / 5);
    (e.moveEntitiesAtIndex = generate_moveEntitiesAtIndex(n, t)),
        (e.calculateRowColMasks = generate_calculateRowColMasks(n, t)),
        (e.resolveMovements = generate_resolveMovements(n, t, e)),
        (e.matchCellRow = generateMatchCellRow(n, t)),
        (e.matchCellRowWildCard = generateMatchCellRowWildCard(n, t)),
        (e.repositionEntitiesAtCell = generate_repositionEntitiesAtCell(n, t));
}
function compile(e, n, t) {
    if ((lazy_function_generation_clear_backlog(), (forceRegenImages = !0), void 0 === e && (e = ["restart"]), void 0 === t && (t = null), (lastDownTarget = canvas), "restart" === e[0] && levelEditorOpened && printLevel(), void 0 === n)) {
        n = window.form1.code.editorreference.getValue() + "\n";
    }
    let o;
    !0 === canDump && (compiledText = n), (errorCount = 0), (compiling = !0), (errorStrings = []), consolePrint("=================================");
    try {
        o = loadFile(n);
    } catch (e) {
        consoleError(e.message + "\n" + e.stack, !0), console.error(e), UnitTestingThrow(e);
    } finally {
        compiling = !1;
    }
    if ((o && o.levels && 0 === o.levels.length && logError("No levels found.  Add some levels!", void 0, !0), errorCount > 0)) {
        if (
            (consoleError(
                !1 === IDE
                    ? null === o
                        ? "<span class=\"systemMessage\">Errors detected during compilation; I can't salvage anything playable from it.  If this is an older game, and you think it just broke because of recent changes in the puzzlescript engine, please consider dropping an email to analytic@gmail.com with a link to the game and I'll try make sure it's back working ASAP.</span>"
                        : "<span class=\"systemMessage\">Errors detected during compilation; the game may not work correctly. If this is an older game, and you think it just broke because of recent changes in the puzzlescript engine, please consider dropping an email to analytic@gmail.com with a link to the game and I'll try make sure it's back working ASAP.</span>"
                    : null === o
                    ? '<span class="systemMessage">Errors detected during compilation; I can\'t salvage anything playable from it.</span>'
                    : '<span class="systemMessage">Errors detected during compilation; the game may not work correctly.</span>'
            ),
            errorCount > MAX_ERRORS)
        )
            return;
    } else {
        let n = 0;
        for (let e = 0; e < o.rules.length; e++) n += o.rules[e].length;
        for (let e = 0; e < o.lateRules.length; e++) n += o.lateRules[e].length;
        "restart" === e[0]
            ? consolePrint('<span class="systemMessage">Successful Compilation, generated ' + n + " instructions.</span>")
            : consolePrint('<span class="systemMessage">Successful live recompilation, generated ' + n + " instructions.</span>"),
            IDE && void 0 !== o.metadata.title && (document.title = "PuzzleScript - " + o.metadata.title);
    }
    null !== o && setGameState(o, e, t), clearInputHistory(), consoleCacheDump(), manage_compilation_caches();
}
const cache_CHECK_RATE = 20;
let cache_checkCount = 0;
function manage_compilation_caches() {
    (cache_checkCount = (cache_checkCount + 1) % cache_CHECK_RATE),
        0 === cache_checkCount &&
            (Object.keys(CACHE_CELLPATTERN_MATCHFUNCTION).length > 1e4 && (CACHE_CELLPATTERN_MATCHFUNCTION = {}),
            Object.keys(CACHE_MOVEENTITIESATINDEX).length > 200 && (CACHE_MOVEENTITIESATINDEX = {}),
            Object.keys(CACHE_CALCULATEROWCOLMASKS).length > 200 && (CACHE_CALCULATEROWCOLMASKS = {}),
            Object.keys(CACHE_RULE_CELLROWMATCHESFUNCTION).length > 1e3 && (CACHE_RULE_CELLROWMATCHESFUNCTION = {}),
            Object.keys(CACHE_CELLPATTERN_REPLACEFUNCTION).length > 200 && (CACHE_CELLPATTERN_REPLACEFUNCTION = {}),
            Object.keys(CACHE_MATCHCELLROW).length > 200 && (CACHE_MATCHCELLROW = {}),
            Object.keys(CACHE_MATCHCELLROWWILDCARD).length > 200 && (CACHE_MATCHCELLROWWILDCARD = {}),
            Object.keys(CACHE_RULE_APPLYAT).length > 200 && (CACHE_RULE_APPLYAT = {}),
            Object.keys(CACHE_RESOLVEMOVEMENTS).length > 200 && (CACHE_RESOLVEMOVEMENTS = {}),
            Object.keys(CACHE_RULE_FINDMATCHES).length > 200 && (CACHE_RULE_FINDMATCHES = {}));
}
function qualifyURL(e) {
    let n = document.createElement("a");
    return (n.href = e), n.href;
}
let keyRepeatTimer = 0,
    keyRepeatIndex = 0,
    input_throttle_timer = 0,
    lastinput = -100,
    dragging = !1,
    rightdragging = !1,
    columnAdded = !1;
function selectText(e, n) {
    n = n || window.event;
    let t = document.getElementById(e);
    if (n && (n.ctrlKey || n.metaKey)) {
        const e = ["console"].concat(t.innerText.split("\n")),
            n = levelFromString(state, e);
        loadLevelFromLevelDat(state, n, null), canvasResize();
    } else if (document.selection) {
        const e = document.body.createTextRange();
        e.moveToElementText(t), e.select();
    } else if (window.getSelection) {
        const e = document.createRange();
        e.selectNode(t);
        const n = window.getSelection();
        n.removeAllRanges(), n.addRange(e);
    }
}
function recalcLevelBounds() {}
function arrCopy(e, n, t, o, r) {
    for (; r--; ) t[o++] = e[n]++;
}
function adjustLevel(e, n, t) {
    backups.push(backupLevel());
    const o = e.clone();
    (e.width += n), (e.height += t), (e.n_tiles = e.width * e.height), (e.objects = new Int32Array(e.n_tiles * STRIDE_OBJ));
    const r = new BitVec(STRIDE_OBJ);
    r.ibitset(state.backgroundid);
    for (let n = 0; n < e.n_tiles; ++n) e.setCell(n, r);
    return (e.movements = new Int32Array(e.objects.length)), (columnAdded = !0), RebuildLevelArrays(), o;
}
function addLeftColumn() {
    const e = adjustLevel(level, 1, 0);
    for (let n = 1; n < level.width; ++n)
        for (let t = 0; t < level.height; ++t) {
            const o = n * level.height + t;
            level.setCell(o, e.getCell(o - level.height));
        }
}
function addRightColumn() {
    const e = adjustLevel(level, 1, 0);
    for (let n = 0; n < level.width - 1; ++n)
        for (let t = 0; t < level.height; ++t) {
            const o = n * level.height + t;
            level.setCell(o, e.getCell(o));
        }
}
function addTopRow() {
    const e = adjustLevel(level, 0, 1);
    for (let n = 0; n < level.width; ++n)
        for (let t = 1; t < level.height; ++t) {
            const o = n * level.height + t;
            level.setCell(o, e.getCell(o - n - 1));
        }
}
function addBottomRow() {
    const e = adjustLevel(level, 0, 1);
    for (let n = 0; n < level.width; ++n)
        for (let t = 0; t < level.height - 1; ++t) {
            const o = n * level.height + t;
            level.setCell(o, e.getCell(o - n));
        }
}
function removeLeftColumn() {
    if (level.width <= 1) return;
    const e = adjustLevel(level, -1, 0);
    for (let n = 0; n < level.width; ++n)
        for (let t = 0; t < level.height; ++t) {
            const o = n * level.height + t;
            level.setCell(o, e.getCell(o + level.height));
        }
}
function removeRightColumn() {
    if (level.width <= 1) return;
    const e = adjustLevel(level, -1, 0);
    for (let n = 0; n < level.width; ++n)
        for (let t = 0; t < level.height; ++t) {
            const o = n * level.height + t;
            level.setCell(o, e.getCell(o));
        }
}
function removeTopRow() {
    if (level.height <= 1) return;
    const e = adjustLevel(level, 0, -1);
    for (let n = 0; n < level.width; ++n)
        for (let t = 0; t < level.height; ++t) {
            const o = n * level.height + t;
            level.setCell(o, e.getCell(o + n + 1));
        }
}
function removeBottomRow() {
    if (level.height <= 1) return;
    const e = adjustLevel(level, 0, -1);
    for (let n = 0; n < level.width; ++n)
        for (let t = 0; t < level.height; ++t) {
            const o = n * level.height + t;
            level.setCell(o, e.getCell(o + n));
        }
}
function matchGlyph(e, n) {
    let t,
        o = -1;
    for (let r = 0; r < n.length; ++r) {
        const l = n[r][0],
            i = n[r][1],
            s = n[r][2];
        if (i.bitsSetInArray(e.data)) {
            let n = 0;
            for (let t = 0; t < 32 * STRIDE_OBJ; ++t) s.get(t) && e.get(t) && n++, i.get(t) && e.get(t) && n++;
            n > o && ((o = n), (t = l));
        }
    }
    return o > 0 ? t : ((compiling = !0), logWarningNoLine("Wasn't able to approximate glyph values for some tiles, using '.' as a placeholder.", !1), (compiling = !1), ".");
}
let htmlEntityMap = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;", "/": "&#x2F;" },
    selectableint = 0;
function printLevel() {
    try {
        (errorCount = 0), (errorStrings = []);
        const e = [];
        for (const n in state.glyphDict)
            if (state.glyphDict.hasOwnProperty(n) && 1 === n.length) {
                const t = state.glyphDict[n],
                    o = new BitVec(STRIDE_OBJ);
                for (let e = 0; e < t.length; e++) {
                    const n = t[e];
                    n >= 0 && o.ibitset(n);
                }
                const r = o.clone(),
                    l = state.layerMasks[state.backgroundlayer];
                o.iclear(l), e.push([n, o, r]);
            }
        selectableint++;
        const n = "selectable" + selectableint;
        let t = 'Printing level contents:<br><br><span id="' + n + '" onclick="selectText(\'' + n + "',event)\">";
        for (let n = 0; n < level.height; n++) {
            for (let o = 0; o < level.width; o++) {
                const r = n + o * level.height;
                let l = matchGlyph(level.getCell(r), e);
                l in htmlEntityMap && (l = htmlEntityMap[l]), (t += l);
            }
            n < level.height - 1 && (t += "<br>");
        }
        (t += "</span><br><br>"), consolePrint(t, !0);
    } catch (e) {
        console.error(e), consolePrint("unable to print level contents because of errors", !0);
    }
}
function levelEditorClick(e, n) {
    if (mouseCoordY <= -2) {
        const e = mouseCoordX + (screenwidth - 1) * (editorRowCount - (-mouseCoordY - 2) - 1);
        -1 === mouseCoordX ? printLevel() : mouseCoordX >= 0 && e < glyphImages.length && ((glyphSelectedIndex = e), redraw());
    } else if (mouseCoordX > -1 && mouseCoordY > -1 && mouseCoordX < screenwidth - 2 && mouseCoordY < screenheight - 2 - editorRowCount) {
        const e = glyphImagesCorrespondance[glyphSelectedIndex],
            n = state.glyphDict[e],
            t = new BitVec(STRIDE_OBJ);
        for (let e = 0; e < n.length; e++) {
            const o = n[e];
            o >= 0 && t.ibitset(o);
        }
        const o = state.layerMasks[state.backgroundlayer];
        t.bitsClearInArray(o.data) && t.ibitset(state.backgroundid);
        const r = mouseCoordY + mouseCoordX * level.height;
        if (level.getCell(r).equals(t)) return;
        !1 === anyEditsSinceMouseDown && ((anyEditsSinceMouseDown = !0), backups.push(backupLevel())), level.setCell(r, t), redraw();
    } else
        n &&
            (-1 === mouseCoordX ? (addLeftColumn(), canvasResize()) : mouseCoordX === screenwidth - 2 && (addRightColumn(), canvasResize()),
            -1 === mouseCoordY ? (addTopRow(), canvasResize()) : mouseCoordY === screenheight - 2 - editorRowCount && (addBottomRow(), canvasResize()));
}
function levelEditorRightClick(e, n) {
    if (-2 === mouseCoordY) mouseCoordX <= glyphImages.length && ((glyphSelectedIndex = mouseCoordX), redraw());
    else if (mouseCoordX > -1 && mouseCoordY > -1 && mouseCoordX < screenwidth - 2 && mouseCoordY < screenheight - 2 - editorRowCount) {
        const e = mouseCoordY + mouseCoordX * level.height,
            n = new BitVec(STRIDE_OBJ);
        n.ibitset(state.backgroundid), level.setCell(e, n), redraw();
    } else
        n &&
            (-1 === mouseCoordX ? (removeLeftColumn(), canvasResize()) : mouseCoordX === screenwidth - 2 && (removeRightColumn(), canvasResize()),
            -1 === mouseCoordY ? (removeTopRow(), canvasResize()) : mouseCoordY === screenheight - 2 - editorRowCount && (removeBottomRow(), canvasResize()));
}
let anyEditsSinceMouseDown = !1;
function onMouseDown(e) {
    if (e.handled) return;
    ULBS();
    let n = 0 === e.button,
        t = 2 === e.button;
    if (("touchstart" == e.type && (n = !0), n && (e.ctrlKey || e.metaKey) && ((n = !1), (t = !0)), n)) {
        if (((lastDownTarget = e.target), (keybuffer = []), (e.target === canvas || "tapFocusIndicator" === e.target.className) && (setMouseCoord(e), (dragging = !0), (rightdragging = !1), levelEditorOpened)))
            return (anyEditsSinceMouseDown = !1), levelEditorClick(e, !0);
        (dragging = !1), (rightdragging = !1), (e.handled = !0);
    } else if (t && (e.target === canvas || "tapFocusIndicator" === e.target.className) && (setMouseCoord(e), (dragging = !1), (rightdragging = !0), levelEditorOpened)) return (e.handled = !0), levelEditorRightClick(e, !0);
}
function rightClickCanvas(e) {
    return !levelEditorOpened || prevent(e);
}
function onMouseUp(e) {
    e.handled || ((dragging = !1), (rightdragging = !1), (e.handled = !0));
}
function onKeyDown(e) {
    ULBS(),
        (e = e || window.event),
        !IDE && [32, 37, 38, 39, 40].indexOf(e.keyCode) > -1 && ((e && (e.ctrlKey || e.metaKey)) || prevent(e)),
        IDE || 77 !== e.keyCode || toggleMute(),
        keybuffer.indexOf(e.keyCode) >= 0 ||
            ((lastDownTarget === canvas || (window.Mobile && lastDownTarget === window.Mobile.focusIndicator)) &&
                -1 === keybuffer.indexOf(e.keyCode) &&
                ((e && ((e.ctrlKey && 90 !== e.keyCode) || e.metaKey)) || (keybuffer.splice(keyRepeatIndex, 0, e.keyCode), (keyRepeatTimer = 0), checkKey(e, !e.repeat))),
            !0 === canDump &&
                (74 === e.keyCode && (e.ctrlKey || e.metaKey)
                    ? (dumpTestCase(), prevent(e))
                    : 75 === e.keyCode && (e.ctrlKey || e.metaKey)
                    ? (makeGIF(), prevent(e))
                    : 83 === e.keyCode && (e.ctrlKey || e.metaKey)
                    ? (saveClick(), prevent(e))
                    : 13 === e.keyCode && (e.ctrlKey || e.metaKey) && (canvas.focus(), editor.display.input.blur(), e.shiftKey ? runClick() : rebuildClick(), prevent(e))));
}
function relMouseCoords(e) {
    let n = 0,
        t = 0,
        o = 0,
        r = 0,
        l = this;
    do {
        (n += l.offsetLeft - l.scrollLeft), (t += l.offsetTop - l.scrollTop);
    } while ((l = l.offsetParent));
    return null == e.touches ? ((o = e.pageX - n), (r = e.pageY - t)) : ((o = e.touches[0].pageX - n), (r = e.touches[0].pageY - t)), { x: o, y: r };
}
function onKeyUp(e) {
    e = e || window.event;
    let n = keybuffer.indexOf(e.keyCode);
    n >= 0 && (keybuffer.splice(n, 1), keyRepeatIndex >= n && keyRepeatIndex--);
}
function onMyFocus(e) {
    (keybuffer = []), (keyRepeatIndex = 0), (keyRepeatTimer = 0);
}
function onMyBlur(e) {
    (keybuffer = []), (keyRepeatIndex = 0), (keyRepeatTimer = 0);
}
HTMLCanvasElement.prototype.relMouseCoords = relMouseCoords;
let mouseCoordX = 0,
    mouseCoordY = 0;
function setMouseCoord(e) {
    const n = canvas.relMouseCoords(e);
    (mouseCoordX = n.x - xoffset), (mouseCoordY = n.y - yoffset), (mouseCoordX = Math.floor(mouseCoordX / cellwidth)), (mouseCoordY = Math.floor(mouseCoordY / cellheight));
}
function mouseMove(e) {
    e.handled || (levelEditorOpened && (setMouseCoord(e), dragging ? levelEditorClick(e, !1) : rightdragging && levelEditorRightClick(e, !1), redraw()), (e.handled = !0));
}
function mouseOut() {}
function prevent(e) {
    return e.preventDefault && e.preventDefault(), e.stopImmediatePropagation && e.stopImmediatePropagation(), e.stopPropagation && e.stopPropagation(), (e.returnValue = !1), !1;
}
document.addEventListener("touchstart", onMouseDown, !1),
    document.addEventListener("touchmove", mouseMove, !1),
    document.addEventListener("touchend", onMouseUp, !1),
    document.addEventListener("mousedown", onMouseDown, !1),
    document.addEventListener("mouseup", onMouseUp, !1),
    document.addEventListener("keydown", onKeyDown, !1),
    document.addEventListener("keyup", onKeyUp, !1),
    window.addEventListener("focus", onMyFocus, !1),
    window.addEventListener("blur", onMyBlur, !1);
var gamepadKeys = [];
function pollGamepads() {
    function e(e, n) {
        return !(e.length <= n) && ("object" == typeof e[n] ? e[n].pressed : 1 == e[n]);
    }
    function n(e, n, t) {
        return !(e.length <= n) && e[n] * t > 0.5;
    }
    var t = [];
    function o(e) {
        -1 === keybuffer.indexOf(e) && (keybuffer.splice(keyRepeatIndex, 0, e), (keyRepeatTimer = 0), checkKey({ keyCode: e }, !0)), t.splice(0, 0, e);
    }
    function r() {
        for (var e = 0; e < gamepadKeys.length; e++)
            if (!(t.indexOf(gamepadKeys[e]) >= 0)) {
                var n = keybuffer.indexOf(gamepadKeys[e]);
                n >= 0 && (keybuffer.splice(n, 1), keyRepeatIndex >= n && keyRepeatIndex--);
            }
        gamepadKeys = t;
    }
    var l = navigator.getGamepads ? navigator.getGamepads() : navigator.webkitGetGamepads ? navigator.webkitGetGamepads() : [];
    if (null != l && 0 != l.length) {
        for (var i = 0; i < l.length; i++) {
            var s = l[i];
            s &&
                s.connected &&
                ((e(s.buttons, 3) || e(s.buttons, 4)) && o(82),
                (e(s.buttons, 1) || n(s.axes, 2, 1)) && o(90),
                (e(s.buttons, 2) || e(s.buttons, 0) || e(s.buttons, 5) || n(s.axes, 1, 1)) && o(88),
                (e(s.buttons, 7) || e(s.buttons, 8)) && o(27),
                e(s.buttons, 6) && o(69),
                (n(s.axes, 1, -1) || n(s.axes, 7, -1) || e(s.buttons, 12)) && o(38),
                (n(s.axes, 1, 1) || n(s.axes, 7, 1) || e(s.buttons, 13)) && o(40),
                (n(s.axes, 0, -1) || n(s.axes, 6, -1) || e(s.buttons, 14)) && o(37),
                (n(s.axes, 0, 1) || n(s.axes, 6, 1) || e(s.buttons, 15)) && o(39));
        }
        r();
    } else r();
}
function checkKey(e, n) {
    if ((ULBS(), winning)) return;
    if (e && ((e.ctrlKey && 90 !== e.keyCode) || e.metaKey || e.altKey)) return;
    let t = -1;
    switch (e.keyCode) {
        case 65:
        case 37:
            t = 1;
            break;
        case 38:
        case 87:
            t = 0;
            break;
        case 68:
        case 39:
            t = 3;
            break;
        case 83:
        case 40:
            t = 2;
            break;
        case 80:
            printLevel();
            break;
        case 13:
        case 32:
        case 67:
        case 88:
            if ((n && ignoreNotJustPressedAction && (ignoreNotJustPressedAction = !1), !1 === n && ignoreNotJustPressedAction)) return;
            if (!1 !== norepeat_action && !n) return;
            t = 4;
            break;
        case 85:
        case 90:
            if (!1 === textMode) return pushInput("undo"), DoUndo(!1, !0), canvasResize(), prevent(e);
            break;
        case 82:
            if (!1 === textMode && n) return pushInput("restart"), DoRestart(), canvasResize(), prevent(e);
            break;
        case 27:
            if (!1 === titleScreen) return goToTitleScreen(), tryPlayTitleSound(), canvasResize(), prevent(e);
            break;
        case 69:
            if (canOpenEditor)
                return (
                    n &&
                        (titleScreen && ("EMPTY GAME" === state.title ? compile(["loadFirstNonMessageLevel"]) : nextLevel()),
                        (levelEditorOpened = !levelEditorOpened),
                        !1 === levelEditorOpened && printLevel(),
                        (restartTarget = backupLevel()),
                        canvasResize()),
                    prevent(e)
                );
            break;
        case 48:
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 55:
        case 56:
        case 57:
            if (levelEditorOpened && n) {
                let n = 9;
                return e.keyCode >= 49 && (n = e.keyCode - 49), n < glyphImages.length ? (glyphSelectedIndex = n) : consolePrint("Trying to select tile outside of range in level editor.", !0), canvasResize(), prevent(e);
            }
            break;
        case 189:
        case 109:
            if (levelEditorOpened && n && glyphSelectedIndex > 0) return glyphSelectedIndex--, canvasResize(), prevent(e);
            break;
        case 187:
        case 107:
            if (levelEditorOpened && n && glyphSelectedIndex + 1 < glyphImages.length) return glyphSelectedIndex++, canvasResize(), prevent(e);
    }
    if (throttle_movement && t >= 0 && t <= 3) {
        if (lastinput == t && input_throttle_timer < repeatinterval) return;
        (lastinput = t), (input_throttle_timer = 0);
    }
    if (textMode) {
        if (0 === state.levels.length);
        else if (titleScreen)
            !1 === quittingTitleScreen &&
                (0 === titleMode
                    ? 4 === t && n && !1 === titleSelected && (tryPlayStartGameSound(), (titleSelected = !0), (messageselected = !1), (timer = 0), (quittingTitleScreen = !0), generateTitleScreen(), canvasResize(), clearInputHistory())
                    : 4 == t && n
                    ? !1 === titleSelected && (tryPlayStartGameSound(), (titleSelected = !0), (messageselected = !1), (timer = 0), (quittingTitleScreen = !0), generateTitleScreen(), redraw())
                    : (0 !== t && 2 !== t) || ((titleSelection = 0 === t ? 0 : 1), generateTitleScreen(), redraw()));
        else if (4 == t && n) {
            if (unitTesting) return void nextLevel();
            !1 === messageselected && ((messageselected = !0), (timer = 0), (quittingMessageScreen = !0), tryPlayCloseMessageSound(), (titleScreen = !1), drawMessageScreen());
        }
    } else !againing && t >= 0 && ((4 === t && "noaction" in state.metadata) || (pushInput(t), processInput(t) && redraw()));
    return prevent(e);
}
function update() {
    tick_lazy_function_generation(textMode && !unitTesting);
    const e = get_title_animation_frame();
    if (((timer += deltatime), (input_throttle_timer += deltatime), quittingTitleScreen)) {
        const n = get_title_animation_frame();
        timer / 1e3 > 0.3 && 0 === WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR.length ? ((quittingTitleScreen = !1), nextLevel()) : n > e && (generateTitleScreen(), redraw());
    }
    if (
        (againing && timer > againinterval && 0 == messagetext.length && processInput(-1) && (redraw(), (keyRepeatTimer = 0), (autotick = 0)),
        quittingMessageScreen &&
            timer / 1e3 > 0.15 &&
            ((quittingMessageScreen = !1),
            "" === messagetext
                ? nextLevel()
                : ((messagetext = ""),
                  (textMode = !1),
                  (titleScreen = !1),
                  (titleMode = curlevel > 0 || null !== curlevelTarget ? 1 : 0),
                  (titleSelected = !1),
                  (ignoreNotJustPressedAction = !0),
                  (titleSelection = 0),
                  canvasResize(),
                  checkWin())),
        winning && timer / 1e3 > 0.5 && ((winning = !1), nextLevel()),
        pollGamepads(),
        keybuffer.length > 0)
    ) {
        keyRepeatTimer += deltatime;
        const e = throttle_movement ? repeatinterval : repeatinterval / Math.sqrt(keybuffer.length);
        if (keyRepeatTimer > e) {
            (keyRepeatTimer = 0), (keyRepeatIndex = (keyRepeatIndex + 1) % keybuffer.length);
            checkKey({ keyCode: keybuffer[keyRepeatIndex] }, !1);
        }
    }
    !(autotickinterval > 0) || textMode || levelEditorOpened || againing || winning || ((autotick += deltatime), autotick > autotickinterval && ((autotick = 0), pushInput("tick"), processInput(-1) && redraw()));
}
let prevTimestamp;
function loop(e) {
    void 0 !== prevTimestamp && (deltatime = e - prevTimestamp), (prevTimestamp = e), update(), window.requestAnimationFrame(loop);
}
function Animatable(e, n, t) {
    var o;
    function r() {
        var e;
        return (o += n) >= 1 && ((e = !0), (o = 1)), t(o), e;
    }
    function l() {
        var e;
        return (o -= n) <= 0 && ((e = !0), (o = 0)), t(o), e;
    }
    return (
        (o = 0),
        {
            animateUp: function () {
                Animator.getInstance().animate(e, r);
            },
            animateDown: function () {
                Animator.getInstance().animate(e, l);
            },
        }
    );
}
window.requestAnimationFrame(loop),
    (window.Mobile = {}),
    (Mobile.hasTouch = function () {
        var e;
        return ("ontouchstart" in window || (window.DocumentTouch && document instanceof DocumentTouch)) && (e = !0), e;
    }),
    (Mobile.enable = function (e) {
        return (e || (Mobile.hasTouch() && !Mobile._instance)) && ((Mobile._instance = new Mobile.GestureHandler()), Mobile._instance.bindEvents(), Mobile._instance.bootstrap()), Mobile._instance;
    }),
    (window.Mobile.GestureHandler = function () {
        this.initialize.apply(this, arguments);
    }),
    (Mobile.log = function (e) {
        document.getElementsByTagName("h1")[0].innerHTML = Math.random().toString().substring(4, 1) + "-" + e;
    }),
    (Mobile.debugDot = function (e) {
        var n, t;
        (t = "border-radius: 50px;width: 5px;height: 5px;background: red;position: absolute;left: " + e.touches[0].clientX + "px;top: " + e.touches[0].clientY + "px;"),
            (n = document.createElement("div")).setAttribute("style", t),
            document.getElementsByTagName("body")[0].appendChild(n);
    }),
    (function (e) {
        var n = { action: 88, left: 37, right: 39, up: 38, down: 40, undo: 85, restart: 82, quit: 27 },
            t = ['<div class="tab">', '  <div class="tab-affordance"></div>', '  <div class="tab-icon">', '    <div class="slice"></div>', '    <div class="slice"></div>', "  </div>", "</div>"].join("\n");
        (e.initialize = function () {
            (this.firstPos = { x: 0, y: 0 }),
                (this.setTabAnimationRatio = this.setTabAnimationRatio.bind(this)),
                (this.setMenuAnimationRatio = this.setMenuAnimationRatio.bind(this)),
                (this.repeatTick = this.repeatTick.bind(this)),
                (this.isFocused = !0);
        }),
            (e.setFocusElement = function (e) {
                (this.focusElement = e), (this.isFocused = !1), this.buildFocusIndicator();
            }),
            (e.bindEvents = function () {
                window.addEventListener("touchstart", this.onTouchStart.bind(this)), window.addEventListener("touchend", this.onTouchEnd.bind(this)), window.addEventListener("touchmove", this.onTouchMove.bind(this));
            }),
            (e.bootstrap = function () {
                this.showTab(), this.disableScrolling(), this.isAudioSupported() || this.disableAudio(), this.disableSelection();
            }),
            (e.onTouchStart = function (e) {
                this.isTouching ||
                    (this.handleFocusChange(e),
                    this.isFocused &&
                        "A" !== e.target.tagName.toUpperCase() &&
                        ((this.isTouching = !0),
                        (this.mayBeSwiping = !0),
                        (this.gestured = !1),
                        (this.swipeDirection = void 0),
                        (this.swipeDistance = 0),
                        (this.startTime = new Date().getTime()),
                        (this.firstPos.x = e.touches[0].clientX),
                        (this.firstPos.y = e.touches[0].clientY)));
            }),
            (e.onTouchEnd = function (e) {
                this.isFocused &&
                    this.isTouching &&
                    (this.gestured || (0 === e.touches.length && "unMuteButton" !== e.target.id && "muteButton" !== e.target.id && this.handleTap()), 0 === e.touches.length && ((this.isTouching = !1), this.endRepeatWatcher()));
            }),
            (e.onTouchMove = function (e) {
                if (this.isFocused && !levelEditorOpened)
                    return (
                        this.isSuccessfulSwipe()
                            ? (this.handleSwipe(this.swipeDirection, this.touchCount), (this.gestured = !0), (this.mayBeSwiping = !1), this.beginRepeatWatcher(e))
                            : this.mayBeSwiping
                            ? this.swipeStep(e)
                            : this.isRepeating && this.repeatStep(e),
                        prevent(e),
                        !1
                    );
            }),
            (e.handleFocusChange = function (e) {
                this.focusElement && ((this.isFocused = this.isTouchInsideFocusElement(e)), this.setFocusIndicatorVisibility(this.isFocused), canvas.focus(), editor.display.input.blur());
            }),
            (e.isTouchInsideFocusElement = function (e) {
                var n;
                return (
                    !(!e.touches || !e.touches[0]) &&
                    ((n = this.absoluteElementPosition(this.focusElement)),
                    !(e.touches[0].clientX < n.left || e.touches[0].clientY < n.top) && !(e.touches[0].clientX > n.left + this.focusElement.clientWidth || e.touches[0].clientY > n.top + this.focusElement.clientHeight))
                );
            }),
            (e.setFocusIndicatorVisibility = function (e) {}),
            (e.absoluteElementPosition = function (e) {
                var n, t;
                for (n = { top: e.offsetTop || 0, left: e.offsetLeft || 0 }, t = document.getElementsByTagName("body")[0], n.top -= t.scrollTop || 0; (e = e.offsetParent); ) (n.top += e.offsetTop || 0), (n.left += e.offsetLeft || 0);
                return n;
            }),
            (e.beginRepeatWatcher = function (e) {
                var n;
                this.repeatInterval || ((this.isRepeating = !0), (n = 1e3 * state.metadata.key_repeat_interval), (!isNaN(n) && n) || (n = 150), (this.repeatInterval = setInterval(this.repeatTick, n)), this.recenter(e));
            }),
            (e.endRepeatWatcher = function () {
                this.repeatInterval && (clearInterval(this.repeatInterval), delete this.repeatInterval, (this.isRepeating = !1));
            }),
            (e.repeatTick = function () {
                this.isTouching && this.handleSwipe(this.direction, this.touchCount);
            }),
            (e.recenter = function (e) {
                (this.firstPos.x = e.touches[0].clientX), (this.firstPos.y = e.touches[0].clientY);
            }),
            (e.isSuccessfulSwipe = function () {
                var e;
                return this.mayBeSwiping && void 0 !== this.swipeDirection && this.swipeDistance >= 50 && (e = !0), e;
            }),
            (e.swipeStep = function (e) {
                var n, t, o;
                this.mayBeSwiping &&
                    ((n = { x: e.touches[0].clientX, y: e.touches[0].clientY }),
                    (t = new Date().getTime()),
                    (o = e.touches.length),
                    (this.swipeDistance = this.cardinalDistance(this.firstPos, n)),
                    this.swipeDirection ? t - this.startTime > 1e3 && (this.mayBeSwiping = !1) : this.swipeDistance > 10 && ((this.swipeDirection = this.dominantDirection(this.firstPos, n)), (this.touchCount = o)));
            }),
            (e.repeatStep = function (e) {
                var n;
                (n = { x: e.touches[0].clientX, y: e.touches[0].clientY }), this.cardinalDistance(this.firstPos, n) >= 50 && ((this.swipeDirection = this.dominantDirection(this.firstPos, n)), this.recenter(e));
            }),
            (e.cardinalDistance = function (e, n) {
                var t, o;
                return (t = Math.abs(e.x - n.x)), (o = Math.abs(e.y - n.y)), Math.max(t, o);
            }),
            (e.dominantDirection = function (e, n) {
                var t, o, r;
                return (t = n.x - e.x), (o = n.y - e.y), (r = "x"), Math.abs(o) > Math.abs(t) && (r = "y"), "x" === r ? (t > 0 ? "right" : "left") : o > 0 ? "down" : "up";
            }),
            (e.handleSwipe = function (e, n) {
                1 === n ? this.emitKeydown(this.swipeDirection) : n > 1 && this.toggleMenu();
            }),
            (e.handleTap = function () {
                this.emitKeydown("action");
            }),
            (e.emitKeydown = function (e) {
                var t;
                (t = { keyCode: n[e] }), this.fakeCanvasFocus(), onKeyDown(t), onKeyUp(t);
            }),
            (e.fakeCanvasFocus = function () {
                onMouseDown({ button: 0, target: document.getElementById("gameCanvas") });
            }),
            (e.toggleMenu = function () {
                this.isMenuVisible ? this.hideMenu() : this.showMenu();
            }),
            (e.showMenu = function () {
                this.menuElem || this.buildMenu(), this.getAnimatables().menu.animateUp(), (this.isMenuVisible = !0), this.hideTab();
            }),
            (e.hideMenu = function () {
                this.menuElem && this.getAnimatables().menu.animateDown(), (this.isMenuVisible = !1), this.showTab();
            }),
            (e.getAnimatables = function () {
                return this._animatables || (this._animatables = { tab: Animatable("tab", 0.1, this.setTabAnimationRatio), menu: Animatable("menu", 0.1, this.setMenuAnimationRatio) }), this._animatables;
            }),
            (e.showTab = function () {
                this.tabElem || this.buildTab(), this.getAnimatables().tab.animateDown();
            }),
            (e.hideTab = function () {
                this.tabElem && this.tabElem.setAttribute("style", "display: none;"), this.getAnimatables().tab.animateUp();
            }),
            (e.buildTab = function () {
                var e,
                    n,
                    o,
                    r = this;
                ((e = document.createElement("div")).innerHTML = t),
                    (o = e.children[0]),
                    (n = function (e) {
                        e.stopPropagation(), r.showMenu();
                    }),
                    (this.tabAffordance = o.getElementsByClassName("tab-affordance")[0]),
                    (this.tabElem = o.getElementsByClassName("tab-icon")[0]),
                    this.tabAffordance.addEventListener("touchstart", n),
                    this.tabAffordance.addEventListener("click", (e) => {}),
                    this.tabElem.addEventListener("touchstart", n),
                    this.tabElem.addEventListener("click", (e) => {}),
                    document.getElementsByTagName("body")[0].appendChild(o);
            }),
            (e.buildMenu = function () {
                var e,
                    n,
                    t,
                    o,
                    r,
                    l,
                    i = this;
                ((e = document.createElement("div")).innerHTML = this.buildMenuString(state)),
                    (this.menuElem = e.children[0]),
                    (this.closeElem = this.menuElem.getElementsByClassName("close")[0]),
                    (l = function (e) {
                        e.stopPropagation(), i.hideMenu();
                    }),
                    (this.closeAffordance = this.menuElem.getElementsByClassName("close-affordance")[0]),
                    (r = this.menuElem.getElementsByClassName("close")[0]),
                    this.closeAffordance.addEventListener("touchstart", l),
                    this.closeAffordance.addEventListener("click", (e) => {}),
                    r.addEventListener("touchstart", l),
                    r.addEventListener("click", (e) => {}),
                    (n = this.menuElem.getElementsByClassName("undo")[0]) &&
                        (n.addEventListener("touchstart", function (e) {
                            e.stopPropagation(), i.emitKeydown("undo");
                        }),
                        n.addEventListener("click", (e) => {})),
                    (t = this.menuElem.getElementsByClassName("restart")[0]) &&
                        (t.addEventListener("touchstart", function (e) {
                            e.stopPropagation(), i.emitKeydown("restart");
                        }),
                        t.addEventListener("click", (e) => {})),
                    (o = this.menuElem.getElementsByClassName("quit")[0]).addEventListener("touchstart", function (e) {
                        e.stopPropagation(), i.emitKeydown("quit");
                    }),
                    o.addEventListener("click", (e) => {}),
                    document.getElementsByTagName("body")[0].appendChild(this.menuElem);
            }),
            (e.buildMenuString = function (e) {
                var n, t, o, r;
                return (
                    (n = 3),
                    (o = e.metadata.noundo) && (n -= 1),
                    (r = e.metadata.norestart) && (n -= 1),
                    (t = ['<div class="mobile-menu item-count-' + n + '">', '  <div class="close-affordance"></div>', '  <div class="close">', '    <div class="slice"></div>', '    <div class="slice"></div>', "  </div>"]),
                    o || t.push('  <div class="undo button">Undo</div>'),
                    r || t.push('  <div class="restart button">Restart</div>'),
                    (t = t.concat(['  <div class="quit button">Quit to Menu</div>', '  <div class="clear"></div>', "</div>"])).join("\n")
                );
            }),
            (e.buildFocusIndicator = function () {
                (this.focusIndicator = document.createElement("DIV")),
                    this.focusIndicator.setAttribute("class", "tapFocusIndicator"),
                    this.focusIndicator.setAttribute("style", "visibility: hidden;"),
                    this.focusElement.parentNode.appendChild(this.focusIndicator);
            }),
            (e.setTabAnimationRatio = function (e) {
                var n;
                (e = Math.round(1e3 * e) / 1e3) >= 0.999 ? this.tabAffordance.setAttribute("style", "display: none;") : this.tabAffordance.setAttribute("style", "display: block;"),
                    (n = "opacity: " + (1 - e) + ";" + " width: " + (66 * e + 18 * (1 - e)) + "px;"),
                    this.tabElem.setAttribute("style", n);
            }),
            (e.setMenuAnimationRatio = function (e) {
                var n, t, o;
                (o = "left: " + ((n = -18 * (e = Math.round(1e3 * e) / 1e3) + -66 * (1 - e)) - 4) + "px; " + (t = "opacity: " + e + ";") + " width: " + -n + "px;"),
                    (e = Math.round(1e3 * e) / 1e3) <= 0.001 ? (this.closeAffordance.setAttribute("style", "display: none;"), (t = "display:none;")) : this.closeAffordance.setAttribute("style", "display: block;"),
                    this.closeElem.setAttribute("style", o),
                    this.menuElem.setAttribute("style", t);
            }),
            (e.disableScrolling = function () {
                var e = { height: "100%", overflow: "hidden", position: "fixed", width: "100%", margin: 0 },
                    n = "";
                for (var t in e) n += t + ": " + e[t] + "; ";
                document.body.setAttribute("style", n);
            }),
            (e.disableAudio = function () {
                window.playSeed = function () {};
            }),
            (e.isAudioSupported = function () {
                var e = !0;
                return "undefined" != typeof webkitAudioContext && (e = !1), e;
            }),
            (e.disableSelection = function () {
                var e;
                (e = document.getElementsByTagName("body")[0]).setAttribute("class", e.getAttribute("class") + " disable-select");
            });
    })(window.Mobile.GestureHandler.prototype),
    (window.Animator = function () {
        this.initialize.apply(this, arguments);
    }),
    (function (e) {
        (e.initialize = function () {
            (this._animations = {}), (this.tick = this.tick.bind(this));
        }),
            (e.animate = function (e, n) {
                (this._animations[e] = n), this.wakeup();
            }),
            (e.wakeup = function () {
                this._isAnimating || ((this._isAnimating = !0), this.tick());
            }),
            (e.tick = function () {
                var e, n, t;
                for (e in ((t = []), (n = !0), this._animations)) {
                    if (!this._animations.hasOwnProperty(e)) return;
                    this._animations[e]() ? t.push(e) : (n = !1);
                }
                if (n) {
                    for (0; 0 < t.length; t++) delete this._isAnimating[t[0]];
                    this._isAnimating = !1;
                } else requestAnimationFrame(this.tick);
            });
    })(window.Animator.prototype),
    (window.Animator.getInstance = function () {
        return window.Animator._instance || (window.Animator._instance = new window.Animator()), window.Animator._instance;
    }),
    (function () {
        var e,
            n,
            t = ["ms", "moz", "webkit", "o"];
        for (e = 0; e < t.length && !window.requestAnimationFrame; e++)
            (window.requestAnimationFrame = window[t[e] + "RequestAnimationFrame"]),
                (window.cancelAnimationFrame = window[t[e] + "CancelAnimationFrame"]),
                window.cancelAnimationFrame || (window.cancelAnimationFrame = window[t[e] + "CancelRequestAnimationFrame"]);
        window.requestAnimationFrame ||
            ((n = 0),
            (window.requestAnimationFrame = function (e, t) {
                var o, r, l;
                return (
                    (o = new Date().getTime()),
                    (r = Math.max(0, 16 - (o - n))),
                    (l = window.setTimeout(function () {
                        e(o + r);
                    }, r)),
                    (n = o + r),
                    l
                );
            })),
            window.cancelAnimationFrame ||
                (window.cancelAnimationFrame = function (e) {
                    clearTimeout(e);
                }),
            Mobile.enable();
    })();
//# sourceMappingURL=scripts_play_compiled.js.map
