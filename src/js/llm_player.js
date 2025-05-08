// LLM Player for PuzzleScript games
// This script allows an LLM to play PuzzleScript games by sending game state and rules to the LLM

// Global variables
let isLLMPlaying = false;
let llmPlayInterval = null;
let currentGameState = null;
let gameRules = null;

// Function to extract game rules from the editor
function extractGameRules() {
    // Get the game code from the editor
    const gameCode = editor.getValue();
    
    // Extract the RULES section
    const rulesMatch = gameCode.match(/======\s*RULES\s*======\s*([\s\S]*?)(?:======\s*(?:WINCONDITIONS|LEVELS)\s*======)/i);
    
    // Extract the LEGEND section
    const legendMatch = gameCode.match(/======\s*LEGEND\s*======\s*([\s\S]*?)(?:======\s*(?:SOUNDS|COLLISIONLAYERS)\s*======)/i);
    
    // Extract the OBJECTS section
    const objectsMatch = gameCode.match(/======\s*OBJECTS\s*======\s*([\s\S]*?)(?:======\s*(?:LEGEND)\s*======)/i);
    
    // Extract the COLLISIONLAYERS section
    const collisionLayersMatch = gameCode.match(/======\s*COLLISIONLAYERS\s*======\s*([\s\S]*?)(?:======\s*(?:RULES)\s*======)/i);
    
    // Extract the WINCONDITIONS section
    const winConditionsMatch = gameCode.match(/======\s*WINCONDITIONS\s*======\s*([\s\S]*?)(?:======\s*(?:LEVELS)\s*======|$)/i);
    
    // Combine all sections into a rules object
    return {
        objects: objectsMatch ? objectsMatch[1].trim() : "",
        legend: legendMatch ? legendMatch[1].trim() : "",
        collisionLayers: collisionLayersMatch ? collisionLayersMatch[1].trim() : "",
        rules: rulesMatch ? rulesMatch[1].trim() : "",
        winConditions: winConditionsMatch ? winConditionsMatch[1].trim() : ""
    };
}

// Function to get the current game name
function getCurrentGameName() {
    // 尝试从下拉菜单获取当前游戏名称
    try {
        const dropdown = document.getElementById('exampleDropdown');
        if (dropdown && dropdown.value && dropdown.value !== 'Load Example') {
            return dropdown.value;
        }
    } catch (error) {
        consolePrint(`Error getting game name from dropdown: ${error.message}`);
    }
    
    // 如果无法从下拉菜单获取，则返回默认值
    return "sokoban_basic";
}

// Function to get the current game state
function getCurrentGameState() {
    // Get the current level state
    if (!state || !state.levels || !state.levels[curlevel]) {
        consolePrint("No game state available");
        return null;
    }
    
    // Convert the level to a string representation
    const level = state.levels[curlevel];
    let stateRepresentation = "";
    
    for (let y = 0; y < level.height; y++) {
        let row = "";
        for (let x = 0; x < level.width; x++) {
            const cell = level.getCell(x, y);
            // Use the first character of each cell or a dot if empty
            row += cell.length > 0 ? cell[0].charAt(0) : ".";
        }
        stateRepresentation += row + "\n";
    }
    
    return stateRepresentation.trim();
}

// Function to send the game state to the LLM and get a decision
async function getLLMDecision(gameState) {
    try {
        // 获取当前游戏名称
        const gameName = getCurrentGameName();
        
        const response = await fetch('/llm_action', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                state: gameState,
                game_name: gameName,
                goal: "Solve the puzzle by following the game rules and win conditions."
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data.action;
    } catch (error) {
        consolePrint(`Error getting LLM decision: ${error.message}`);
        return null;
    }
}

// 跟踪动作执行次数
let actionCounter = 0;

// Function to execute the LLM's decision
function executeAction(action) {
    // Map action to key code
    let keyCode;
    switch (action) {
        case 'up':
            keyCode = 38; // Up arrow
            break;
        case 'down':
            keyCode = 40; // Down arrow
            break;
        case 'left':
            keyCode = 37; // Left arrow
            break;
        case 'right':
            keyCode = 39; // Right arrow
            break;
        case 'use':
        case 'action':
            keyCode = 88; // X key
            break;
        default:
            consolePrint(`Unknown action: ${action}`);
            return;
    }
    
    // 确保游戏画布获得焦点
    const gameCanvas = document.getElementById('gameCanvas');
    if (gameCanvas) {
        gameCanvas.focus();
    }
    
    // 模拟按键的更可靠方法
    try {
        // 方法1：使用KeyboardEvent
        const keyDownEvent = new KeyboardEvent('keydown', {
            keyCode: keyCode,
            which: keyCode,
            code: getKeyCodeString(keyCode),
            key: getKeyString(keyCode),
            bubbles: true,
            cancelable: true
        });
        
        // 方法2：直接调用游戏的按键处理函数（如果存在）
        if (typeof onKeyDown === 'function') {
            // 直接调用游戏的按键处理函数
            onKeyDown(keyDownEvent);
        } else {
            // 如果没有直接的处理函数，分发事件到文档和游戏画布
            document.dispatchEvent(keyDownEvent);
            if (gameCanvas) {
                gameCanvas.dispatchEvent(keyDownEvent);
            }
        }
        
        // 短暂延迟后发送keyup事件
        setTimeout(() => {
            const keyUpEvent = new KeyboardEvent('keyup', {
                keyCode: keyCode,
                which: keyCode,
                code: getKeyCodeString(keyCode),
                key: getKeyString(keyCode),
                bubbles: true,
                cancelable: true
            });
            
            if (typeof onKeyUp === 'function') {
                onKeyUp(keyUpEvent);
            } else {
                document.dispatchEvent(keyUpEvent);
                if (gameCanvas) {
                    gameCanvas.dispatchEvent(keyUpEvent);
                }
            }
        }, 100);
        
        consolePrint(`LLM executed action: ${action}`);
        
        // 增加动作计数器
        actionCounter++;
        
        // 只在前两次动作执行后点击游戏画布
        if (actionCounter <= 2) {
            consolePrint(`Action ${actionCounter}: Clicking on game canvas`);
            setTimeout(() => {
                const gameCanvas = document.getElementById('gameCanvas');
                if (gameCanvas) {
                    // 获取游戏画布的位置和尺寸
                    const rect = gameCanvas.getBoundingClientRect();
                    
                    // 定义要点击的位置（只点击上部和中心）
                    const clickPositions = [
                        { x: rect.left + rect.width / 2, y: rect.top + rect.height / 4 },    // 上部
                        { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }     // 中心
                    ];
                    
                    // 依次点击每个位置
                    for (let i = 0; i < clickPositions.length; i++) {
                        setTimeout((pos) => {
                            // 创建并分发mousedown事件
                            const mouseDownEvent = new MouseEvent('mousedown', {
                                bubbles: true,
                                cancelable: true,
                                view: window,
                                clientX: pos.x,
                                clientY: pos.y
                            });
                            gameCanvas.dispatchEvent(mouseDownEvent);
                            
                            // 短暂延迟后发送mouseup事件
                            setTimeout(() => {
                                const mouseUpEvent = new MouseEvent('mouseup', {
                                    bubbles: true,
                                    cancelable: true,
                                    view: window,
                                    clientX: pos.x,
                                    clientY: pos.y
                                });
                                gameCanvas.dispatchEvent(mouseUpEvent);
                                
                                // 最后发送click事件
                                const clickEvent = new MouseEvent('click', {
                                    bubbles: true,
                                    cancelable: true,
                                    view: window,
                                    clientX: pos.x,
                                    clientY: pos.y
                                });
                                gameCanvas.dispatchEvent(clickEvent);
                                
                                consolePrint(`Clicked on game canvas at position ${i+1} after action: ${action}`);
                            }, 50);
                        }, i * 100, clickPositions[i]); // 每个点击间隔100毫秒
                    }
                }
            }, 200); // 稍微延迟点击，确保动作已执行
        }
    } catch (error) {
        consolePrint(`Error executing action: ${error.message}`);
    }
}

// 辅助函数：获取键码对应的code字符串
function getKeyCodeString(keyCode) {
    switch (keyCode) {
        case 37: return 'ArrowLeft';
        case 38: return 'ArrowUp';
        case 39: return 'ArrowRight';
        case 40: return 'ArrowDown';
        case 88: return 'KeyX';
        default: return '';
    }
}

// 辅助函数：获取键码对应的key字符串
function getKeyString(keyCode) {
    switch (keyCode) {
        case 37: return 'ArrowLeft';
        case 38: return 'ArrowUp';
        case 39: return 'ArrowRight';
        case 40: return 'ArrowDown';
        case 88: return 'x';
        default: return '';
    }
}

// Function to start the LLM playing
function startLLMPlaying() {
    if (isLLMPlaying) return;
    
    consolePrint("LLM is now playing the game...");
    
    // 尝试确保游戏已经启动
    ensureGameIsRunning();
    
    isLLMPlaying = true;
    
    // Start the play loop
    llmPlayInterval = setInterval(async () => {
        // Get current game state
        currentGameState = getCurrentGameState();
        if (!currentGameState) {
            consolePrint("No game state available, trying to ensure game is running...");
            ensureGameIsRunning();
            
            // 再次尝试获取游戏状态
            currentGameState = getCurrentGameState();
            if (!currentGameState) {
                consolePrint("Still no game state, stopping LLM player");
                stopLLMPlaying();
                return;
            }
        }
        
        // Get LLM decision
        const action = await getLLMDecision(currentGameState);
        if (!action) {
            consolePrint("Failed to get LLM decision");
            return;
        }
        
        // Execute the action
        executeAction(action);
        
        // Check if the game is won
        if (typeof winning !== 'undefined' && winning) {
            consolePrint("LLM has won the game!");
            stopLLMPlaying();
        }
    }, 1000); // Make a move every second
}

// 确保游戏已经启动
function ensureGameIsRunning() {
    consolePrint("Ensuring game is running...");
    
    // 尝试多种方法启动游戏
    
    // 1. 尝试调用游戏的内部函数
    if (typeof canvasResize === 'function') {
        consolePrint("Calling canvasResize()");
        canvasResize();
    }
    
    if (typeof redraw === 'function') {
        consolePrint("Calling redraw()");
        redraw();
    }
    
    // 2. 尝试模拟按键事件
    const gameCanvas = document.getElementById('gameCanvas');
    if (gameCanvas) {
        // 确保游戏画布获得焦点
        gameCanvas.focus();
        
        // 模拟按各种可能的按键
        const keyCodes = [32, 13, 38, 40, 37, 39, 88, 90]; // 空格, 回车, 上下左右, X, Z
        for (let i = 0; i < keyCodes.length; i++) {
            const keyCode = keyCodes[i];
            
            // 创建并分发keydown事件
            const keyDownEvent = new KeyboardEvent('keydown', {
                keyCode: keyCode,
                which: keyCode,
                code: getKeyCodeString(keyCode),
                key: getKeyString(keyCode),
                bubbles: true,
                cancelable: true
            });
            
            // 尝试直接调用游戏的按键处理函数
            if (typeof onKeyDown === 'function') {
                consolePrint("Calling onKeyDown() with keyCode " + keyCode);
                onKeyDown(keyDownEvent);
            } else {
                consolePrint("Dispatching keydown event with keyCode " + keyCode);
                document.dispatchEvent(keyDownEvent);
                gameCanvas.dispatchEvent(keyDownEvent);
            }
            
            // 短暂延迟后发送keyup事件
            setTimeout(function(kc) {
                return function() {
                    const keyUpEvent = new KeyboardEvent('keyup', {
                        keyCode: kc,
                        which: kc,
                        code: getKeyCodeString(kc),
                        key: getKeyString(kc),
                        bubbles: true,
                        cancelable: true
                    });
                    
                    if (typeof onKeyUp === 'function') {
                        onKeyUp(keyUpEvent);
                    } else {
                        document.dispatchEvent(keyUpEvent);
                        gameCanvas.dispatchEvent(keyUpEvent);
                    }
                };
            }(keyCode), 100);
        }
        
        // 模拟点击游戏画布中心
        const rect = gameCanvas.getBoundingClientRect();
        const clickEvent = new MouseEvent('click', {
            bubbles: true,
            cancelable: true,
            view: window,
            clientX: rect.left + rect.width / 2,
            clientY: rect.top + rect.height / 2
        });
        gameCanvas.dispatchEvent(clickEvent);
    }
    
// 3. 尝试直接设置游戏状态变量和初始化游戏
    try {
        const initScript = `
            // 确保游戏状态变量正确初始化
            if (typeof gameRunning !== 'undefined') {
                gameRunning = true;
                consolePrint("Setting gameRunning = true");
            }
            
            if (typeof textMode !== 'undefined') {
                textMode = false;
                consolePrint("Setting textMode = false");
            }
            
            if (typeof titleScreen !== 'undefined') {
                titleScreen = false;
                consolePrint("Setting titleScreen = false");
            }
            
            // 确保游戏级别正确加载
            if (typeof state !== 'undefined' && state.levels && state.levels.length > 0) {
                // 确保当前级别索引有效
                if (typeof curlevel === 'undefined' || curlevel < 0 || curlevel >= state.levels.length) {
                    curlevel = 0;
                    consolePrint("Resetting curlevel to 0");
                }
                
                // 确保级别数据已正确初始化
                var level = state.levels[curlevel];
                if (level) {
                    // 重新编译级别以确保所有数据结构都正确初始化
                    if (typeof recompileLevel === 'function') {
                        recompileLevel();
                        consolePrint("Recompiled level");
                    }
                    
                    // 确保背景图层已初始化
                    if (level.layerData === undefined || level.layerData.length === 0) {
                        if (typeof regenerateCellLayer === 'function') {
                            regenerateCellLayer();
                            consolePrint("Regenerated cell layer");
                        }
                    }
                    
                    // 确保背景颜色已设置
                    if (typeof state.bgcolor !== 'undefined') {
                        state.bgcolor = state.bgcolor || [0, 0, 0];
                        consolePrint("Ensured bgcolor is set");
                    }
                    
                    // 确保游戏对象已正确初始化
                    if (typeof state.objects !== 'undefined') {
                        consolePrint("Game objects exist");
                    }
                }
            }
            
            // 尝试强制重绘游戏
            if (typeof canvasResize === 'function') {
                canvasResize();
                consolePrint("Forced canvas resize");
            }
            
            if (typeof redraw === 'function') {
                redraw();
                consolePrint("Forced redraw");
            }
            
            return "Game state initialization attempted";
        `;
        
        const result = eval(initScript);
        consolePrint(result);
    } catch (e) {
        consolePrint(`Error in game state initialization: ${e.message}`);
    }
    
    // 4. 尝试调用其他可能的游戏函数
    if (typeof loadLevelFromStateString === 'function' && typeof state !== 'undefined' && state.levels && state.levels[curlevel]) {
        consolePrint("Calling loadLevelFromStateString()");
        const levelString = state.levels[curlevel].toString();
        loadLevelFromStateString(levelString);
    }
    
    // 只有在必要的对象都已定义时才调用applyRandomRuleGroup
    if (typeof applyRandomRuleGroup === 'function' && 
        typeof state !== 'undefined' && 
        state.rules !== undefined && 
        state.rules.random_rules !== undefined && 
        Array.isArray(state.rules.random_rules)) {
        consolePrint("Calling applyRandomRuleGroup()");
        try {
            applyRandomRuleGroup();
        } catch (e) {
            consolePrint(`Error in applyRandomRuleGroup: ${e.message}`);
        }
    }
    
    consolePrint("Game start attempts completed");
}

// 辅助函数：获取键码对应的code字符串
function getKeyCodeString(keyCode) {
    switch (keyCode) {
        case 32: return 'Space';
        case 13: return 'Enter';
        case 37: return 'ArrowLeft';
        case 38: return 'ArrowUp';
        case 39: return 'ArrowRight';
        case 40: return 'ArrowDown';
        case 88: return 'KeyX';
        case 90: return 'KeyZ';
        default: return '';
    }
}

// 辅助函数：获取键码对应的key字符串
function getKeyString(keyCode) {
    switch (keyCode) {
        case 32: return ' ';
        case 13: return 'Enter';
        case 37: return 'ArrowLeft';
        case 38: return 'ArrowUp';
        case 39: return 'ArrowRight';
        case 40: return 'ArrowDown';
        case 88: return 'x';
        case 90: return 'z';
        default: return '';
    }
}

// Function to stop the LLM playing
function stopLLMPlaying() {
    if (!isLLMPlaying) return;
    
    clearInterval(llmPlayInterval);
    llmPlayInterval = null;
    isLLMPlaying = false;
    consolePrint("LLM has stopped playing");
}

// Function to reset the current level
function resetGame() {
    try {
        // 停止LLM玩家（如果正在运行）
        if (isLLMPlaying) {
            stopLLMPlaying();
        }
        
        // 确保游戏画布获得焦点
        const gameCanvas = document.getElementById('gameCanvas');
        if (gameCanvas) {
            gameCanvas.focus();
        }
        
        // 直接使用R键重置游戏
        consolePrint("Using R key to reset game");
        
        // 模拟按R键重置游戏
        const resetKeyCode = 82; // R键的keyCode
        
        const keyDownEvent = new KeyboardEvent('keydown', {
            keyCode: resetKeyCode,
            which: resetKeyCode,
            code: 'KeyR',
            key: 'r',
            bubbles: true,
            cancelable: true
        });
        
        // 尝试直接调用游戏的按键处理函数
        if (typeof onKeyDown === 'function') {
            onKeyDown(keyDownEvent);
        } else {
            document.dispatchEvent(keyDownEvent);
            if (gameCanvas) {
                gameCanvas.dispatchEvent(keyDownEvent);
            }
        }
        
        // 发送keyup事件
        setTimeout(() => {
            const keyUpEvent = new KeyboardEvent('keyup', {
                keyCode: resetKeyCode,
                which: resetKeyCode,
                code: 'KeyR',
                key: 'r',
                bubbles: true,
                cancelable: true
            });
            
            if (typeof onKeyUp === 'function') {
                onKeyUp(keyUpEvent);
            } else {
                document.dispatchEvent(keyUpEvent);
                if (gameCanvas) {
                    gameCanvas.dispatchEvent(keyUpEvent);
                }
            }
        }, 100);
        
        // 确保游戏状态被重置
        setTimeout(() => {
            // 尝试重置一些可能的游戏状态变量
            try {
                // 重置游戏状态的JavaScript代码
                const resetScript = `
                    // 重置一些常见的游戏状态变量
                    if (typeof level !== 'undefined' && typeof curlevel !== 'undefined') {
                        level = curlevel;
                    }
                    if (typeof winning !== 'undefined') {
                        winning = false;
                    }
                    if (typeof againing !== 'undefined') {
                        againing = false;
                    }
                    if (typeof messagetext !== 'undefined') {
                        messagetext = "";
                    }
                    
                    // 重新聚焦游戏画布
                    var gameCanvas = document.getElementById('gameCanvas');
                    if (gameCanvas) {
                        gameCanvas.focus();
                        
                        // 模拟点击游戏画布中心
                        var rect = gameCanvas.getBoundingClientRect();
                        var clickEvent = new MouseEvent('click', {
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            clientX: rect.left + rect.width / 2,
                            clientY: rect.top + rect.height / 2
                        });
                        gameCanvas.dispatchEvent(clickEvent);
                    }
                    
                    consolePrint("Game state reset attempted");
                `;
                
                const result = eval(resetScript);
                consolePrint(result);
            } catch (e) {
                consolePrint(`Error in reset script: ${e.message}`);
            }
            
            consolePrint("Game reset completed");
            
            // 如果之前LLM正在玩游戏，重新开始
            if (isLLMPlaying) {
                setTimeout(() => {
                    startLLMPlaying();
                }, 500);
            }
        }, 300);
        
        return true;
    } catch (error) {
        consolePrint(`Error resetting game: ${error.message}`);
        return false;
    }
}

// Function to toggle LLM playing
function toggleLLMPlaying() {
    if (isLLMPlaying) {
        stopLLMPlaying();
    } else {
        startLLMPlaying();
    }
}

// Function to provide feedback to the LLM about the game state
function feedStateToLLM() {
    if (!currentGameState) {
        consolePrint("No game state available to feed back to LLM");
        return false;
    }
    
    // 获取当前游戏状态
    const gameState = getCurrentGameState();
    
    // 获取当前游戏名称
    const gameName = getCurrentGameName();
    
    // 发送到服务器，提供反馈
    fetch('/llm_feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            state: gameState,
            game_name: gameName,
            result: winning ? 'success' : 'in_progress',
            reward: winning ? 1.0 : 0.0
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        consolePrint(`Feedback sent to LLM: ${data.status}`);
    })
    .catch(error => {
        consolePrint(`Error sending feedback to LLM: ${error.message}`);
    });
    
    return true;
}

// Add event listeners for the buttons
document.addEventListener('DOMContentLoaded', function() {
    // LLM Play按钮
    const llmPlayLink = document.getElementById('llmPlayClickLink');
    if (llmPlayLink) {
        llmPlayLink.addEventListener('click', toggleLLMPlaying);
    }
    
    // 创建Reset按钮
    const toolbar = document.getElementById('uppertoolbar');
    if (toolbar) {
        // 查找LLM PLAY链接元素
        const llmPlayElement = document.getElementById('llmPlayClickLink');
        
        if (llmPlayElement) {
            // 创建Reset按钮
            const resetButton = document.createElement('a');
            resetButton.id = 'resetGameLink';
            resetButton.href = 'javascript:void(0);';
            resetButton.textContent = 'RESET';
            resetButton.style.marginLeft = '5px';
            resetButton.addEventListener('click', resetGame);
            
            // 创建分隔符
            const separator = document.createTextNode(' - ');
            
            // 在LLM PLAY后面插入
            llmPlayElement.parentNode.insertBefore(separator, llmPlayElement.nextSibling);
            llmPlayElement.parentNode.insertBefore(resetButton, separator.nextSibling);
            
            // 创建Feedback按钮
            const feedbackButton = document.createElement('a');
            feedbackButton.id = 'feedStateLink';
            feedbackButton.href = 'javascript:void(0);';
            feedbackButton.textContent = 'FEEDBACK';
            feedbackButton.style.marginLeft = '5px';
            feedbackButton.addEventListener('click', feedStateToLLM);
            
            // 创建另一个分隔符
            const separator2 = document.createTextNode(' - ');
            
            // 在Reset后面插入
            resetButton.parentNode.insertBefore(separator2, resetButton.nextSibling);
            resetButton.parentNode.insertBefore(feedbackButton, separator2.nextSibling);
        }
    }
});
