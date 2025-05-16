import atexit
import threading
from functools import partial
import time
import os
import base64
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from flask import Flask, jsonify, request, send_from_directory
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

# Import LLM agent
from LLM_agent import LLMAgent
@dataclass
class Config:
    port: int = 8000
    headless: bool = False
    auto_open_devtools: bool = True
    maximize_browser: bool = True
    save_gifs: bool = True
    gif_output_dir: str = "gifs"
    game: str = "sokoban_basic"  # Default game
    play_all: bool = False  # Whether to play all games

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# Create Flask application
app = Flask(__name__)

# Global variables
driver = None
llm_agent = None
rl_wrapper = None
is_recording = False
current_game_state = None
game_rules = None
cfg = None  # Global configuration object

# Route: Serve static files
@app.route('/')
def serve_doctor():
    return send_from_directory('src', 'doctor.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('src', filename)

# Load game rules from data/min_games directory
def load_game_rules(game_name):
    try:
        # Build file path
        file_path = os.path.join('data', 'min_games', f"{game_name}.txt")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Game file not found: {file_path}")
            return {}
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            game_code = f.read()
        
        # Extract rules section
        rules_match = re.search(r'======\s*RULES\s*======\s*([\s\S]*?)(?:======\s*(?:WINCONDITIONS|LEVELS)\s*======)', game_code)
        legend_match = re.search(r'======\s*LEGEND\s*======\s*([\s\S]*?)(?:======\s*(?:SOUNDS|COLLISIONLAYERS)\s*======)', game_code)
        objects_match = re.search(r'======\s*OBJECTS\s*======\s*([\s\S]*?)(?:======\s*(?:LEGEND)\s*======)', game_code)
        collision_layers_match = re.search(r'======\s*COLLISIONLAYERS\s*======\s*([\s\S]*?)(?:======\s*(?:RULES)\s*======)', game_code)
        win_conditions_match = re.search(r'======\s*WINCONDITIONS\s*======\s*([\s\S]*?)(?:======\s*(?:LEVELS)\s*======|$)', game_code)
        
        # Combine all parts into rules object
        return {
            'objects': objects_match.group(1).strip() if objects_match else "",
            'legend': legend_match.group(1).strip() if legend_match else "",
            'collisionLayers': collision_layers_match.group(1).strip() if collision_layers_match else "",
            'rules': rules_match.group(1).strip() if rules_match else "",
            'winConditions': win_conditions_match.group(1).strip() if win_conditions_match else ""
        }
    except Exception as e:
        print(f"Error loading game rules: {e}")
        return {}

# LLM action API
@app.route('/llm_action', methods=['POST'])
def llm_action():
    try:
        data = request.json
        state_repr = data['state']
        game_name = data.get('game_name', 'sokoban_basic')  # Default to sokoban_basic
        goal = data.get('goal', 'Solve the puzzle by following the game rules and win conditions.')
        
        # Load rules from data/min_games directory
        rules = load_game_rules(game_name)
        
        # Process game state
        processed_state = {
            'raw_state': state_repr,
            'entities': llm_agent._extract_entities(state_repr),
            'metrics': {'complexity': len(state_repr)}
        }
        
        # Generate decision
        action = llm_agent.choose_action(processed_state=processed_state, goal=goal)
        
        # Record history
        llm_agent.update_history(action, "pending")
        
        return jsonify({
            'action': action,
            'state_hash': hash(state_repr)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# LLM feedback API
@app.route('/llm_feedback', methods=['POST'])
def llm_feedback():
    try:
        data = request.json
        state_repr = data['state']
        game_name = data.get('game_name', 'sokoban_basic')  # Default to sokoban_basic
        result = data.get('result', 'in_progress')
        reward = data.get('reward', 0.0)
        
        # Load rules from data/min_games directory
        rules = load_game_rules(game_name)
        
        # Process game state
        processed_state = {
            'raw_state': state_repr,
            'entities': llm_agent._extract_entities(state_repr),
            'metrics': {'complexity': len(state_repr)}
        }
        
        # Update LLM agent's history record
        if result == 'success':
            llm_agent.update_history(llm_agent.action_history[-1]['action'] if llm_agent.action_history else 'none', "success")
        else:
            llm_agent.update_history(llm_agent.action_history[-1]['action'] if llm_agent.action_history else 'none', "in_progress")
        
        # Apply reinforcement learning reward
        if rl_wrapper and reward != 0.0:
            rl_wrapper.reinforce(reward)
        
        return jsonify({
            'status': 'feedback_received',
            'state_hash': hash(state_repr)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API for saving GIF
@app.route('/save_gif', methods=['POST'])
def save_gif():
    try:
        data = request.json
        gif_data = data['gif_data']
        level = data.get('level', 0)
        
        # Decode Base64 data
        gif_binary = base64.b64decode(gif_data.split(',')[1])
        
        # Ensure output directory exists
        os.makedirs(cfg.gif_output_dir, exist_ok=True)
        
        # Save GIF file
        filename = f"level_{level}_{int(time.time())}.gif"
        filepath = os.path.join(cfg.gif_output_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(gif_binary)
            
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to open browser
def open_browser(url, cfg):
    global driver
    
    # Set Selenium WebDriver options
    options = Options()
    if cfg.auto_open_devtools:
        options.add_argument("--auto-open-devtools-for-tabs")
    if cfg.maximize_browser:
        options.add_argument("--start-maximized")
    if cfg.headless:
        options.add_argument("--headless")
    
    options.add_experimental_option("detach", True)
    
    # Create WebDriver
    driver = webdriver.Chrome(options=options)
    
    # Open URL
    driver.get(url)
    
    # Wait for page to load completely
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, "gameCanvas"))
    )
    
    # Inject custom JavaScript to enhance GIF recording functionality
    inject_gif_recorder(driver)
    
    print(f"Browser opened at {url}")
    return driver

# Check if makegif.js exists and inject GIF recording enhancement script
def inject_gif_recorder(driver):
    # First wait for the page to fully load
    time.sleep(3)
    
    # Check if makeGIF function is loaded
    check_script = """
    console.log('Checking if makegif.js is loaded...');
    console.log('makeGIF exists:', typeof makeGIF !== 'undefined');
    return typeof makeGIF !== 'undefined';
    """
    
    # Try to execute check script
    try:
        is_loaded = driver.execute_script(check_script)
        print(f"makegif.js loaded: {is_loaded}")
    except Exception as e:
        print(f"Error checking makegif.js: {e}")
        is_loaded = False
    
    if not is_loaded:
        print("Warning: makegif.js not loaded or makeGIF function not found")
        return False
    
    # This script enhances makegif.js functionality to automatically send GIF data to our server
    script = """
    console.log('Enhancing GIF recorder...');
    
    // Create a simple recording function that directly calls the makeGIF function
    window.startGifRecording = function() {
        console.log('Starting GIF recording manually');
        if (typeof makeGIF === 'function') {
            console.log('makeGIF function found, calling...');
            
            // Save the original consolePrint function
            var originalConsolePrint = window.consolePrint;
            
            // Override consolePrint function to capture GIF data
            window.consolePrint = function(text) {
                // Call original function
                originalConsolePrint.apply(this, arguments);
                
                // Check if it contains GIF data
                if (typeof text === 'string' && text.includes('data:image/gif;base64,')) {
                    console.log('GIF data found in console output');
                    
                    // Extract GIF data
                    var gifDataMatch = text.match(/src="(data:image\/gif;base64,[^"]+)"/);
                    if (gifDataMatch && gifDataMatch[1]) {
                        var gifData = gifDataMatch[1];
                        
                        console.log('Sending GIF data to server');
                        // Send to server
                        fetch('/save_gif', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                gif_data: gifData,
                                level: typeof curlevel !== 'undefined' ? curlevel : 0
                            })
                        })
                        .then(response => response.json())
                        .then(data => {
                            console.log('GIF saved:', data.filepath);
                        })
                        .catch(error => {
                            console.error('Error saving GIF:', error);
                        });
                        
                        // Restore original consolePrint function
                        window.consolePrint = originalConsolePrint;
                    }
                }
            };
            
            // Call makeGIF function
            makeGIF();
            return true;
        } else {
            console.log('makeGIF function not found');
            return false;
        }
    };
    
    console.log('GIF recorder enhanced successfully');
    return true;
    """
    
    try:
        result = driver.execute_script(script)
        print(f"GIF recorder enhancement result: {result}")
        return result
    except Exception as e:
        print(f"Error enhancing GIF recorder: {e}")
        return False

# Start LLM player
def start_llm_player(driver):
    # Click LLM PLAY button
    try:
        llm_play_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "llmPlayClickLink"))
        )
        llm_play_button.click()
        print("LLM player started")
        return True
    except Exception as e:
        print(f"Error starting LLM player: {e}")
        return False

# Run game
def run_game(driver):
    # Click RUN button
    try:
        run_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "runClickLink"))
        )
        run_button.click()
        print("Game running")
        return True
    except Exception as e:
        print(f"Error running game: {e}")
        return False

# Start recording GIF
def start_recording_gif(driver):
    try:
        # Call our injected recording function
        result = driver.execute_script("return window.startGifRecording();")
        if result:
            print("GIF recording started successfully")
        else:
            print("Failed to start GIF recording")
        return result
    except Exception as e:
        print(f"Error starting GIF recording: {e}")
        return False

# Check if the game has been won
def check_game_won(driver):
    try:
        # Check if there is a victory message
        script = """
        return typeof winning !== 'undefined' && winning;
        """
        return driver.execute_script(script)
    except:
        return False

# Select game example
def select_example_game(driver, game_name="sokoban_basic"):
    try:
        # Open example dropdown menu
        script = """
        console.log('Selecting example game:', arguments[0]);
        var dropdown = document.getElementById('exampleDropdown');
        if (!dropdown) {
            console.log('Example dropdown not found');
            return false;
        }
        
        // Set dropdown menu value
        dropdown.value = arguments[0];
        
        // Trigger change event
        var event = new Event('change', { bubbles: true });
        dropdown.dispatchEvent(event);
        
        console.log('Example game selected');
        return true;
        """
        
        result = driver.execute_script(script, game_name)
        if result:
            print(f"Selected example game: {game_name}")
        else:
            print(f"Failed to select example game: {game_name}")
        return result
    except Exception as e:
        print(f"Error selecting example game: {e}")
        return False

# Simulate pressing space key to start game and click game canvas
def press_space_key_and_click_canvas(driver):
    try:
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.common.action_chains import ActionChains
        
        # First ensure game canvas has focus
        script = """
        var gameCanvas = document.getElementById('gameCanvas');
        if (gameCanvas) {
            gameCanvas.focus();
            return true;
        }
        return false;
        """
        focus_result = driver.execute_script(script)
        
        if not focus_result:
            print("Failed to focus on game canvas")
            return False
        
        # Get position and size of game canvas
        script = """
        var gameCanvas = document.getElementById('gameCanvas');
        if (gameCanvas) {
            var rect = gameCanvas.getBoundingClientRect();
            return {
                x: rect.left + rect.width / 2,
                y: rect.top + rect.height / 2,
                width: rect.width,
                height: rect.height
            };
        }
        return null;
        """
        canvas_info = driver.execute_script(script)
        
        if not canvas_info:
            print("Failed to get canvas position")
            return False
        
        # Click center of game canvas multiple times
        for i in range(3):
            actions = ActionChains(driver)
            actions.move_to_element_with_offset(
                driver.find_element(By.ID, "gameCanvas"),
                canvas_info['width'] / 2,
                canvas_info['height'] / 2
            )
            actions.click()
            actions.perform()
            print(f"Canvas clicked (attempt {i+1})")
            time.sleep(0.5)
        
        # Press space key multiple times
        for i in range(3):
            actions = ActionChains(driver)
            actions.send_keys(Keys.SPACE)
            actions.perform()
            print(f"Space key pressed (attempt {i+1})")
            time.sleep(0.5)
        
        # Directly use JavaScript to simulate key press and click
        script = """
        var gameCanvas = document.getElementById('gameCanvas');
        if (gameCanvas) {
            // Simulate click
            var rect = gameCanvas.getBoundingClientRect();
            var clickEvent = new MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                view: window,
                clientX: rect.left + rect.width / 2,
                clientY: rect.top + rect.height / 2
            });
            gameCanvas.dispatchEvent(clickEvent);
            
            // Simulate pressing space key
            var spaceKeyEvent = new KeyboardEvent('keydown', {
                key: ' ',
                code: 'Space',
                keyCode: 32,
                which: 32,
                bubbles: true,
                cancelable: true
            });
            gameCanvas.dispatchEvent(spaceKeyEvent);
            
            // Send keyup event after a short delay
            setTimeout(function() {
                var keyUpEvent = new KeyboardEvent('keyup', {
                    key: ' ',
                    code: 'Space',
                    keyCode: 32,
                    which: 32,
                    bubbles: true,
                    cancelable: true
                });
                gameCanvas.dispatchEvent(keyUpEvent);
            }, 100);
            
            return true;
        }
        return false;
        """
        js_result = driver.execute_script(script)
        print(f"JavaScript key and click simulation: {js_result}")
        
        return True
    except Exception as e:
        print(f"Error in press_space_key_and_click_canvas: {e}")
        return False

# Get list of available games
def get_available_games(driver):
    try:
        script = """
        var dropdown = document.getElementById('exampleDropdown');
        if (!dropdown) {
            return [];
        }
        
        var games = [];
        for (var i = 0; i < dropdown.options.length; i++) {
            var option = dropdown.options[i];
            if (option.value && option.value !== 'Load Example') {
                games.push({
                    name: option.text,
                    value: option.value
                });
            }
        }
        return games;
        """
        
        games = driver.execute_script(script)
        print(f"Found {len(games)} available games")
        return games
    except Exception as e:
        print(f"Error getting available games: {e}")
        return []

# Automatically play and record game
def auto_play_and_record(driver, cfg, game_name=None):
    # Wait for page to fully load
    time.sleep(3)
    print("Page loaded, starting game automation")
    
    # If no game specified, get all available games
    if not game_name:
        games = get_available_games(driver)
        if not games:
            print("No games available")
            return False
        
        # Iterate through all games
        for game in games:
            print(f"Playing game: {game['name']} ({game['value']})")
            play_single_game(driver, cfg, game['value'])
            time.sleep(2)  # Wait for game to end before starting the next one
        
        return True
    else:
        # Play a single specified game
        return play_single_game(driver, cfg, game_name)

# Play a single game
def play_single_game(driver, cfg, game_name):
    # Select example game
    if not select_example_game(driver, game_name):
        print(f"Failed to select example game: {game_name}")
        return False
    
    # Wait for example game to load
    time.sleep(2)
    
    # Run game
    if not run_game(driver):
        print("Failed to run game")
        return False
    
    # Wait for game to load
    time.sleep(3)
    print("Game should be running now")
    
    # Check if game is running
    try:
        is_running = driver.execute_script("return typeof gameRunning !== 'undefined' && gameRunning;")
        print(f"Game running status: {is_running}")
    except Exception as e:
        print(f"Error checking game status: {e}")
    
    # Try to start game directly using JavaScript
    try:
        script = """
        // Try multiple methods to start game
        console.log('Attempting to start game using JavaScript...');
        
        // Method 1: Try to call game's internal functions
        if (typeof canvasResize === 'function') {
            console.log('Calling canvasResize()');
            canvasResize();
        }
        
        if (typeof redraw === 'function') {
            console.log('Calling redraw()');
            redraw();
        }
        
        // Method 2: Try to simulate key events
        var gameCanvas = document.getElementById('gameCanvas');
        if (gameCanvas) {
            // Ensure game canvas has focus
            gameCanvas.focus();
            
            // Simulate pressing various possible keys
            var keyCodes = [32, 13, 38, 40, 37, 39, 88, 90]; // Space, Enter, Up, Down, Left, Right, X, Z
            for (var i = 0; i < keyCodes.length; i++) {
                var keyCode = keyCodes[i];
                
                // Create and dispatch keydown event
                var keyDownEvent = new KeyboardEvent('keydown', {
                    keyCode: keyCode,
                    which: keyCode,
                    code: getKeyCodeString(keyCode),
                    key: getKeyString(keyCode),
                    bubbles: true,
                    cancelable: true
                });
                
                // Try to directly call game's key handling function
                if (typeof onKeyDown === 'function') {
                    console.log('Calling onKeyDown() with keyCode ' + keyCode);
                    onKeyDown(keyDownEvent);
                } else {
                    console.log('Dispatching keydown event with keyCode ' + keyCode);
                    document.dispatchEvent(keyDownEvent);
                    gameCanvas.dispatchEvent(keyDownEvent);
                }
                
                // Send keyup event after a short delay
                setTimeout(function(kc) {
                    return function() {
                        var keyUpEvent = new KeyboardEvent('keyup', {
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
        
        // Method 3: Try to directly set game state variables
        if (typeof gameRunning !== 'undefined') {
            console.log('Setting gameRunning = true');
            gameRunning = true;
        }
        
        if (typeof textMode !== 'undefined') {
            console.log('Setting textMode = false');
            textMode = false;
        }
        
        if (typeof titleScreen !== 'undefined') {
            console.log('Setting titleScreen = false');
            titleScreen = false;
        }
        
        // Helper functions
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
        
        return 'Game start attempts completed';
        """
        
        result = driver.execute_script(script)
        print(f"JavaScript game start result: {result}")
    except Exception as e:
        print(f"Error in JavaScript game start attempt: {e}")
    
    # Ensure game canvas has focus and click it
    for attempt in range(3):  # Try multiple times to ensure success
        if press_space_key_and_click_canvas(driver):
            print(f"Successfully focused and clicked game canvas on attempt {attempt+1}")
            break
        else:
            print(f"Failed to focus and click game canvas on attempt {attempt+1}, retrying...")
            time.sleep(0.5)
    
    # Wait for game to actually start
    time.sleep(1)
    
    # Ensure game canvas has focus again
    try:
        script = """
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
            
            return true;
        }
        return false;
        """
        driver.execute_script(script)
        print("Additional focus and click on game canvas")
    except Exception as e:
        print(f"Error in additional focus attempt: {e}")
    
    # Start recording GIF
    if cfg.save_gifs:
        if not start_recording_gif(driver):
            print("Failed to start GIF recording, but continuing...")
    
    # Start LLM player
    if not start_llm_player(driver):
        print("Failed to start LLM player")
        return False
    
    # Wait for game to end or timeout
    timeout = 300  # 5 minute timeout
    start_time = time.time()
    
    print("Waiting for game to complete...")
    while time.time() - start_time < timeout:
        if check_game_won(driver):
            print("Game won!")
            # Wait for GIF generation to complete
            time.sleep(2)
            return True
        time.sleep(1)  # Check once per second
    
    print("Game timed out")
    return False

# Main function
@hydra.main(config_name="config", version_base="1.3")
def main(cfg_param: Config):
    global llm_agent, rl_wrapper, driver, cfg
    
    # Set global configuration object
    cfg = cfg_param
    
    # Initialize LLM agent
    llm_agent = LLMAgent(model_name="gpt-4o")

    
    # Create browser thread
    url = f"http://127.0.0.1:{cfg.port}"
    browser_thread = threading.Thread(
        target=partial(open_browser, url=url, cfg=cfg)
    )
    browser_thread.daemon = True
    browser_thread.start()
    
    # Wait for browser to start
    browser_ready = threading.Event()
    
    def wait_for_browser():
        global driver
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 second timeout
            if driver is not None:
                browser_ready.set()
                break
            time.sleep(0.5)
    
    wait_thread = threading.Thread(target=wait_for_browser)
    wait_thread.daemon = True
    wait_thread.start()
    
    # Register function to close browser
    def close_browser():
        global driver
        if driver:
            try:
                driver.quit()
                print("Browser closed.")
            except:
                pass
    
    atexit.register(close_browser)
    
    # Create thread for automatically playing game
    def auto_play_thread():
        # Wait for browser to be ready
        if browser_ready.wait(30):  # Wait up to 30 seconds
            # Wait for the page to fully load
            time.sleep(3)
            # Automatically play and record the game
            if cfg.play_all:
                print("Playing all available games")
                auto_play_and_record(driver, cfg)  # No specific game name, will play all games
            else:
                print(f"Playing game: {cfg.game}")
                auto_play_and_record(driver, cfg, cfg.game)  # Play the specified game
    
    play_thread = threading.Thread(target=auto_play_thread)
    play_thread.daemon = True
    play_thread.start()
    
    # Run the Flask application
    print(f"Starting server on port {cfg.port}")
    app.run(port=cfg.port)

if __name__ == "__main__":
    main()
