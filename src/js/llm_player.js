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
    // Try to get current game name from dropdown menu
    try {
        const dropdown = document.getElementById('exampleDropdown');
        if (dropdown && dropdown.value && dropdown.value !== 'Load Example') {
            return dropdown.value;
        }
    } catch (error) {
        consolePrint(`Error getting game name from dropdown: ${error.message}`);
    }
    
    // If unable to get from dropdown menu, return default value
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
        // Get current game name
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

// Track number of action executions
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
    
    // Ensure game canvas has focus
    const gameCanvas = document.getElementById('gameCanvas');
    if (gameCanvas) {
        gameCanvas.focus();
    }
    
    // More reliable method for simulating key press
    try {
        // Method 1: Use KeyboardEvent
        const keyDownEvent = new KeyboardEvent('keydown', {
            keyCode: keyCode,
            which: keyCode,
            code: getKeyCodeString(keyCode),
            key: getKeyString(keyCode),
            bubbles: true,
            cancelable: true
        });
        
        // Method 2: Directly call the game's key handling function (if it exists)
        if (typeof onKeyDown === 'function') {
            // Directly call the game's key handling function
            onKeyDown(keyDownEvent);
        } else {
            // If there's no direct handling function, dispatch events to document and game canvas
            document.dispatchEvent(keyDownEvent);
            if (gameCanvas) {
                gameCanvas.dispatchEvent(keyDownEvent);
            }
        }
        
        // Send keyup event after a short delay
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
        
        // Increment action counter
        actionCounter++;
        
        // Only click on game canvas after the first two actions
        if (actionCounter <= 2) {
            consolePrint(`Action ${actionCounter}: Clicking on game canvas`);
            setTimeout(() => {
                const gameCanvas = document.getElementById('gameCanvas');
                if (gameCanvas) {
                    // Get position and size of game canvas
                    const rect = gameCanvas.getBoundingClientRect();
                    
                    // Define positions to click (only top and center)
                    const clickPositions = [
                        { x: rect.left + rect.width / 2, y: rect.top + rect.height / 4 },    // Top
                        { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }     // Center
                    ];
                    
                    // Click each position in sequence
                    for (let i = 0; i < clickPositions.length; i++) {
                        setTimeout((pos) => {
                            // Create and dispatch mousedown event
                            const mouseDownEvent = new MouseEvent('mousedown', {
                                bubbles: true,
                                cancelable: true,
                                view: window,
                                clientX: pos.x,
                                clientY: pos.y
                            });
                            gameCanvas.dispatchEvent(mouseDownEvent);
                            
                            // Send mouseup event after a short delay
                            setTimeout(() => {
                                const mouseUpEvent = new MouseEvent('mouseup', {
                                    bubbles: true,
                                    cancelable: true,
                                    view: window,
                                    clientX: pos.x,
                                    clientY: pos.y
                                });
                                gameCanvas.dispatchEvent(mouseUpEvent);
                                
                                // Finally send click event
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
                        }, i * 100, clickPositions[i]); // 100ms interval between each click
                    }
                }
            }, 200); // Slightly delay clicks to ensure action has been executed
        }
    } catch (error) {
        consolePrint(`Error executing action: ${error.message}`);
    }
}

// Helper function: Get code string corresponding to key code
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

// Helper function: Get key string corresponding to key code
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
    
    // Try to ensure game is running
    ensureGameIsRunning();
    
    isLLMPlaying = true;
    
    // Start the play loop
    llmPlayInterval = setInterval(async () => {
        // Get current game state
        currentGameState = getCurrentGameState();
        if (!currentGameState) {
            consolePrint("No game state available, trying to ensure game is running...");
            ensureGameIsRunning();
            
            // Try to get game state again
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

// Ensure game is running
function ensureGameIsRunning() {
    consolePrint("Ensuring game is running...");
    
    // Try multiple methods to start game
    
    // 1. Try to call game's internal functions
    if (typeof canvasResize === 'function') {
        consolePrint("Calling canvasResize()");
        canvasResize();
    }
    
    if (typeof redraw === 'function') {
        consolePrint("Calling redraw()");
        redraw();
    }
    
    // 2. Try to simulate key events
    const gameCanvas = document.getElementById('gameCanvas');
    if (gameCanvas) {
        // Ensure game canvas has focus
        gameCanvas.focus();
        
        // Simulate pressing various possible keys
        const keyCodes = [32, 13, 38, 40, 37, 39, 88, 90]; // Space, Enter, Up, Down, Left, Right, X, Z
        for (let i = 0; i < keyCodes.length; i++) {
            const keyCode = keyCodes[i];
            
            // Create and dispatch keydown event
            const keyDownEvent = new KeyboardEvent('keydown', {
                keyCode: keyCode,
                which: keyCode,
                code: getKeyCodeString(keyCode),
                key: getKeyString(keyCode),
                bubbles: true,
                cancelable: true
            });
            
            // Try to directly call game's key handling function
            if (typeof onKeyDown === 'function') {
                consolePrint("Calling onKeyDown() with keyCode " + keyCode);
                onKeyDown(keyDownEvent);
            } else {
                consolePrint("Dispatching keydown event with keyCode " + keyCode);
                document.dispatchEvent(keyDownEvent);
                gameCanvas.dispatchEvent(keyDownEvent);
            }
            
            // Send keyup event after a short delay
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
        
        // Simulate clicking center of game canvas
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
    
// 3. Try to directly set game state variables and initialize game
    try {
        const initScript = `
            // Ensure game state variables are correctly initialized
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
            
            // Ensure game levels are correctly loaded
            if (typeof state !== 'undefined' && state.levels && state.levels.length > 0) {
                // Ensure current level index is valid
                if (typeof curlevel === 'undefined' || curlevel < 0 || curlevel >= state.levels.length) {
                    curlevel = 0;
                    consolePrint("Resetting curlevel to 0");
                }
                
                // Ensure level data is correctly initialized
                var level = state.levels[curlevel];
                if (level) {
                    // Recompile level to ensure all data structures are correctly initialized
                    if (typeof recompileLevel === 'function') {
                        recompileLevel();
                        consolePrint("Recompiled level");
                    }
                    
                    // Ensure background layer is initialized
                    if (level.layerData === undefined || level.layerData.length === 0) {
                        if (typeof regenerateCellLayer === 'function') {
                            regenerateCellLayer();
                            consolePrint("Regenerated cell layer");
                        }
                    }
                    
                    // Ensure background color is set
                    if (typeof state.bgcolor !== 'undefined') {
                        state.bgcolor = state.bgcolor || [0, 0, 0];
                        consolePrint("Ensured bgcolor is set");
                    }
                    
                    // Ensure game objects are correctly initialized
                    if (typeof state.objects !== 'undefined') {
                        consolePrint("Game objects exist");
                    }
                }
            }
            
            // Try to force redraw of game
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
    
    // 4. Try to call other possible game functions
    if (typeof loadLevelFromStateString === 'function' && typeof state !== 'undefined' && state.levels && state.levels[curlevel]) {
        consolePrint("Calling loadLevelFromStateString()");
        const levelString = state.levels[curlevel].toString();
        loadLevelFromStateString(levelString);
    }
    
    // Only call applyRandomRuleGroup when necessary objects are defined
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

// Helper function: Get code string corresponding to key code
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

// Helper function: Get key string corresponding to key code
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
        // Stop LLM player (if running)
        if (isLLMPlaying) {
            stopLLMPlaying();
        }
        
        // Ensure game canvas has focus
        const gameCanvas = document.getElementById('gameCanvas');
        if (gameCanvas) {
            gameCanvas.focus();
        }
        
        // Directly use R key to reset game
        consolePrint("Using R key to reset game");
        
        // Simulate pressing R key to reset game
        const resetKeyCode = 82; // R键的keyCode
        
        const keyDownEvent = new KeyboardEvent('keydown', {
            keyCode: resetKeyCode,
            which: resetKeyCode,
            code: 'KeyR',
            key: 'r',
            bubbles: true,
            cancelable: true
        });
        
        // Try to directly call game's key handling function
        if (typeof onKeyDown === 'function') {
            onKeyDown(keyDownEvent);
        } else {
            document.dispatchEvent(keyDownEvent);
            if (gameCanvas) {
                gameCanvas.dispatchEvent(keyDownEvent);
            }
        }
        
        // Send keyup event
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
        
        // Ensure game state is reset
        setTimeout(() => {
            // Try to reset some possible game state variables
            try {
                // JavaScript code to reset game state
                const resetScript = `
                    // Reset some common game state variables
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
                    
                    // Refocus game canvas
                    var gameCanvas = document.getElementById('gameCanvas');
                    if (gameCanvas) {
                        gameCanvas.focus();
                        
                        // Simulate clicking center of game canvas
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
            
            // If LLM was playing before, restart
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
    
    // Get current game state
    const gameState = getCurrentGameState();
    
    // Get current game name
    const gameName = getCurrentGameName();
    
    // Send to server, provide feedback
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
    // LLM Play button
    const llmPlayLink = document.getElementById('llmPlayClickLink');
    if (llmPlayLink) {
        llmPlayLink.addEventListener('click', toggleLLMPlaying);
    }
    
    // Create Reset button
    const toolbar = document.getElementById('uppertoolbar');
    if (toolbar) {
        // Find LLM PLAY link element
        const llmPlayElement = document.getElementById('llmPlayClickLink');
        
        if (llmPlayElement) {
            // Create Reset button
            const resetButton = document.createElement('a');
            resetButton.id = 'resetGameLink';
            resetButton.href = 'javascript:void(0);';
            resetButton.textContent = 'RESET';
            resetButton.style.marginLeft = '5px';
            resetButton.addEventListener('click', resetGame);
            
            // Create separator
            const separator = document.createTextNode(' - ');
            
            // Insert after LLM PLAY
            llmPlayElement.parentNode.insertBefore(separator, llmPlayElement.nextSibling);
            llmPlayElement.parentNode.insertBefore(resetButton, separator.nextSibling);
            
            // Create Feedback button
            const feedbackButton = document.createElement('a');
            feedbackButton.id = 'feedStateLink';
            feedbackButton.href = 'javascript:void(0);';
            feedbackButton.textContent = 'FEEDBACK';
            feedbackButton.style.marginLeft = '5px';
            feedbackButton.addEventListener('click', feedStateToLLM);
            
            // Create another separator
            const separator2 = document.createTextNode(' - ');
            
            // Insert after Reset
            resetButton.parentNode.insertBefore(separator2, resetButton.nextSibling);
            resetButton.parentNode.insertBefore(feedbackButton, separator2.nextSibling);
        }
    }
});
