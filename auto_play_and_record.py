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

# 导入LLM代理
from LLM_agent import LLMAgent, ReinforcementWrapper, StateVisualizer

@dataclass
class Config:
    port: int = 8000
    headless: bool = False
    auto_open_devtools: bool = True
    maximize_browser: bool = True
    save_gifs: bool = True
    gif_output_dir: str = "gifs"
    game: str = "sokoban_basic"  # 默认游戏
    play_all: bool = False  # 是否玩所有游戏

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# 创建Flask应用
app = Flask(__name__)

# 全局变量
driver = None
llm_agent = None
rl_wrapper = None
is_recording = False
current_game_state = None
game_rules = None
cfg = None  # 全局配置对象

# 路由：提供静态文件
@app.route('/')
def serve_doctor():
    return send_from_directory('src', 'doctor.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('src', filename)

# 从data/min_games目录加载游戏规则
def load_game_rules(game_name):
    try:
        # 构建文件路径
        file_path = os.path.join('data', 'min_games', f"{game_name}.txt")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"Game file not found: {file_path}")
            return {}
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            game_code = f.read()
        
        # 提取规则部分
        rules_match = re.search(r'======\s*RULES\s*======\s*([\s\S]*?)(?:======\s*(?:WINCONDITIONS|LEVELS)\s*======)', game_code)
        legend_match = re.search(r'======\s*LEGEND\s*======\s*([\s\S]*?)(?:======\s*(?:SOUNDS|COLLISIONLAYERS)\s*======)', game_code)
        objects_match = re.search(r'======\s*OBJECTS\s*======\s*([\s\S]*?)(?:======\s*(?:LEGEND)\s*======)', game_code)
        collision_layers_match = re.search(r'======\s*COLLISIONLAYERS\s*======\s*([\s\S]*?)(?:======\s*(?:RULES)\s*======)', game_code)
        win_conditions_match = re.search(r'======\s*WINCONDITIONS\s*======\s*([\s\S]*?)(?:======\s*(?:LEVELS)\s*======|$)', game_code)
        
        # 组合所有部分到规则对象
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

# LLM动作API
@app.route('/llm_action', methods=['POST'])
def llm_action():
    try:
        data = request.json
        state_repr = data['state']
        game_name = data.get('game_name', 'sokoban_basic')  # 默认使用sokoban_basic
        goal = data.get('goal', 'Solve the puzzle by following the game rules and win conditions.')
        
        # 从data/min_games目录加载规则
        rules = load_game_rules(game_name)
        
        # 处理游戏状态
        processed_state = {
            'raw_state': state_repr,
            'entities': llm_agent._extract_entities(state_repr),
            'metrics': {'complexity': len(state_repr)}
        }
        
        # 生成决策
        action = llm_agent.choose_action(processed_state=processed_state, goal=goal)
        
        # 记录历史
        llm_agent.update_history(action, "pending")
        
        return jsonify({
            'action': action,
            'state_hash': hash(state_repr)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# LLM反馈API
@app.route('/llm_feedback', methods=['POST'])
def llm_feedback():
    try:
        data = request.json
        state_repr = data['state']
        game_name = data.get('game_name', 'sokoban_basic')  # 默认使用sokoban_basic
        result = data.get('result', 'in_progress')
        reward = data.get('reward', 0.0)
        
        # 从data/min_games目录加载规则
        rules = load_game_rules(game_name)
        
        # 处理游戏状态
        processed_state = {
            'raw_state': state_repr,
            'entities': llm_agent._extract_entities(state_repr),
            'metrics': {'complexity': len(state_repr)}
        }
        
        # 更新LLM代理的历史记录
        if result == 'success':
            llm_agent.update_history(llm_agent.action_history[-1]['action'] if llm_agent.action_history else 'none', "success")
        else:
            llm_agent.update_history(llm_agent.action_history[-1]['action'] if llm_agent.action_history else 'none', "in_progress")
        
        # 应用强化学习奖励
        if rl_wrapper and reward != 0.0:
            rl_wrapper.reinforce(reward)
        
        return jsonify({
            'status': 'feedback_received',
            'state_hash': hash(state_repr)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 保存GIF的API
@app.route('/save_gif', methods=['POST'])
def save_gif():
    try:
        data = request.json
        gif_data = data['gif_data']
        level = data.get('level', 0)
        
        # 解码Base64数据
        gif_binary = base64.b64decode(gif_data.split(',')[1])
        
        # 确保输出目录存在
        os.makedirs(cfg.gif_output_dir, exist_ok=True)
        
        # 保存GIF文件
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

# 打开浏览器函数
def open_browser(url, cfg):
    global driver
    
    # 设置Selenium WebDriver选项
    options = Options()
    if cfg.auto_open_devtools:
        options.add_argument("--auto-open-devtools-for-tabs")
    if cfg.maximize_browser:
        options.add_argument("--start-maximized")
    if cfg.headless:
        options.add_argument("--headless")
    
    options.add_experimental_option("detach", True)
    
    # 创建WebDriver
    driver = webdriver.Chrome(options=options)
    
    # 打开URL
    driver.get(url)
    
    # 等待页面加载完成
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, "gameCanvas"))
    )
    
    # 注入自定义JavaScript以增强GIF录制功能
    inject_gif_recorder(driver)
    
    print(f"Browser opened at {url}")
    return driver

# 检查makegif.js是否存在并注入GIF录制增强脚本
def inject_gif_recorder(driver):
    # 首先等待页面完全加载
    time.sleep(3)
    
    # 检查makeGIF函数是否已加载
    check_script = """
    console.log('Checking if makegif.js is loaded...');
    console.log('makeGIF exists:', typeof makeGIF !== 'undefined');
    return typeof makeGIF !== 'undefined';
    """
    
    # 尝试执行检查脚本
    try:
        is_loaded = driver.execute_script(check_script)
        print(f"makegif.js loaded: {is_loaded}")
    except Exception as e:
        print(f"Error checking makegif.js: {e}")
        is_loaded = False
    
    if not is_loaded:
        print("Warning: makegif.js not loaded or makeGIF function not found")
        return False
    
    # 这个脚本会增强makegif.js的功能，使其能够自动将GIF数据发送到我们的服务器
    script = """
    console.log('Enhancing GIF recorder...');
    
    // 创建一个简单的录制函数，直接调用makeGIF函数
    window.startGifRecording = function() {
        console.log('Starting GIF recording manually');
        if (typeof makeGIF === 'function') {
            console.log('makeGIF function found, calling...');
            
            // 保存原始的consolePrint函数
            var originalConsolePrint = window.consolePrint;
            
            // 重写consolePrint函数以捕获GIF数据
            window.consolePrint = function(text) {
                // 调用原始函数
                originalConsolePrint.apply(this, arguments);
                
                // 检查是否包含GIF数据
                if (typeof text === 'string' && text.includes('data:image/gif;base64,')) {
                    console.log('GIF data found in console output');
                    
                    // 提取GIF数据
                    var gifDataMatch = text.match(/src="(data:image\/gif;base64,[^"]+)"/);
                    if (gifDataMatch && gifDataMatch[1]) {
                        var gifData = gifDataMatch[1];
                        
                        console.log('Sending GIF data to server');
                        // 发送到服务器
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
                        
                        // 恢复原始consolePrint函数
                        window.consolePrint = originalConsolePrint;
                    }
                }
            };
            
            // 调用makeGIF函数
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

# 启动LLM玩家
def start_llm_player(driver):
    # 点击LLM PLAY按钮
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

# 运行游戏
def run_game(driver):
    # 点击RUN按钮
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

# 开始录制GIF
def start_recording_gif(driver):
    try:
        # 调用我们注入的录制函数
        result = driver.execute_script("return window.startGifRecording();")
        if result:
            print("GIF recording started successfully")
        else:
            print("Failed to start GIF recording")
        return result
    except Exception as e:
        print(f"Error starting GIF recording: {e}")
        return False

# 检查游戏是否已经赢了
def check_game_won(driver):
    try:
        # 检查是否有胜利消息
        script = """
        return typeof winning !== 'undefined' && winning;
        """
        return driver.execute_script(script)
    except:
        return False

# 选择游戏示例
def select_example_game(driver, game_name="sokoban_basic"):
    try:
        # 打开示例下拉菜单
        script = """
        console.log('Selecting example game:', arguments[0]);
        var dropdown = document.getElementById('exampleDropdown');
        if (!dropdown) {
            console.log('Example dropdown not found');
            return false;
        }
        
        // 设置下拉菜单值
        dropdown.value = arguments[0];
        
        // 触发change事件
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

# 模拟按空格键开始游戏并点击游戏画布
def press_space_key_and_click_canvas(driver):
    try:
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.common.action_chains import ActionChains
        
        # 首先确保游戏画布获得焦点
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
        
        # 获取游戏画布的位置和大小
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
        
        # 多次点击游戏画布中心
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
        
        # 多次按空格键
        for i in range(3):
            actions = ActionChains(driver)
            actions.send_keys(Keys.SPACE)
            actions.perform()
            print(f"Space key pressed (attempt {i+1})")
            time.sleep(0.5)
        
        # 直接使用JavaScript模拟按键和点击
        script = """
        var gameCanvas = document.getElementById('gameCanvas');
        if (gameCanvas) {
            // 模拟点击
            var rect = gameCanvas.getBoundingClientRect();
            var clickEvent = new MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                view: window,
                clientX: rect.left + rect.width / 2,
                clientY: rect.top + rect.height / 2
            });
            gameCanvas.dispatchEvent(clickEvent);
            
            // 模拟按空格键
            var spaceKeyEvent = new KeyboardEvent('keydown', {
                key: ' ',
                code: 'Space',
                keyCode: 32,
                which: 32,
                bubbles: true,
                cancelable: true
            });
            gameCanvas.dispatchEvent(spaceKeyEvent);
            
            // 短暂延迟后发送keyup事件
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

# 获取可用游戏列表
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

# 自动玩游戏并录制
def auto_play_and_record(driver, cfg, game_name=None):
    # 等待页面完全加载
    time.sleep(3)
    print("Page loaded, starting game automation")
    
    # 如果没有指定游戏，则获取所有可用游戏
    if not game_name:
        games = get_available_games(driver)
        if not games:
            print("No games available")
            return False
        
        # 遍历所有游戏
        for game in games:
            print(f"Playing game: {game['name']} ({game['value']})")
            play_single_game(driver, cfg, game['value'])
            time.sleep(2)  # 等待游戏结束后再开始下一个游戏
        
        return True
    else:
        # 玩单个指定的游戏
        return play_single_game(driver, cfg, game_name)

# 玩单个游戏
def play_single_game(driver, cfg, game_name):
    # 选择示例游戏
    if not select_example_game(driver, game_name):
        print(f"Failed to select example game: {game_name}")
        return False
    
    # 等待示例游戏加载
    time.sleep(2)
    
    # 运行游戏
    if not run_game(driver):
        print("Failed to run game")
        return False
    
    # 等待游戏加载
    time.sleep(3)
    print("Game should be running now")
    
    # 检查游戏是否正在运行
    try:
        is_running = driver.execute_script("return typeof gameRunning !== 'undefined' && gameRunning;")
        print(f"Game running status: {is_running}")
    except Exception as e:
        print(f"Error checking game status: {e}")
    
    # 尝试直接使用JavaScript启动游戏
    try:
        script = """
        // 尝试多种方法启动游戏
        console.log('Attempting to start game using JavaScript...');
        
        // 方法1: 尝试调用游戏的内部函数
        if (typeof canvasResize === 'function') {
            console.log('Calling canvasResize()');
            canvasResize();
        }
        
        if (typeof redraw === 'function') {
            console.log('Calling redraw()');
            redraw();
        }
        
        // 方法2: 尝试模拟按键事件
        var gameCanvas = document.getElementById('gameCanvas');
        if (gameCanvas) {
            // 确保游戏画布获得焦点
            gameCanvas.focus();
            
            // 模拟按各种可能的按键
            var keyCodes = [32, 13, 38, 40, 37, 39, 88, 90]; // 空格, 回车, 上下左右, X, Z
            for (var i = 0; i < keyCodes.length; i++) {
                var keyCode = keyCodes[i];
                
                // 创建并分发keydown事件
                var keyDownEvent = new KeyboardEvent('keydown', {
                    keyCode: keyCode,
                    which: keyCode,
                    code: getKeyCodeString(keyCode),
                    key: getKeyString(keyCode),
                    bubbles: true,
                    cancelable: true
                });
                
                // 尝试直接调用游戏的按键处理函数
                if (typeof onKeyDown === 'function') {
                    console.log('Calling onKeyDown() with keyCode ' + keyCode);
                    onKeyDown(keyDownEvent);
                } else {
                    console.log('Dispatching keydown event with keyCode ' + keyCode);
                    document.dispatchEvent(keyDownEvent);
                    gameCanvas.dispatchEvent(keyDownEvent);
                }
                
                // 短暂延迟后发送keyup事件
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
        
        // 方法3: 尝试直接设置游戏状态变量
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
        
        // 辅助函数
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
    
    # 确保游戏画布获得焦点并点击
    for attempt in range(3):  # 尝试多次以确保成功
        if press_space_key_and_click_canvas(driver):
            print(f"Successfully focused and clicked game canvas on attempt {attempt+1}")
            break
        else:
            print(f"Failed to focus and click game canvas on attempt {attempt+1}, retrying...")
            time.sleep(0.5)
    
    # 等待游戏真正开始
    time.sleep(1)
    
    # 再次确保游戏画布获得焦点
    try:
        script = """
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
            
            return true;
        }
        return false;
        """
        driver.execute_script(script)
        print("Additional focus and click on game canvas")
    except Exception as e:
        print(f"Error in additional focus attempt: {e}")
    
    # 开始录制GIF
    if cfg.save_gifs:
        if not start_recording_gif(driver):
            print("Failed to start GIF recording, but continuing...")
    
    # 启动LLM玩家
    if not start_llm_player(driver):
        print("Failed to start LLM player")
        return False
    
    # 等待游戏结束或超时
    timeout = 300  # 5分钟超时
    start_time = time.time()
    
    print("Waiting for game to complete...")
    while time.time() - start_time < timeout:
        if check_game_won(driver):
            print("Game won!")
            # 等待GIF生成完成
            time.sleep(2)
            return True
        time.sleep(1)  # 每秒检查一次
    
    print("Game timed out")
    return False

# 主函数
@hydra.main(config_name="config", version_base="1.3")
def main(cfg_param: Config):
    global llm_agent, rl_wrapper, driver, cfg
    
    # 设置全局配置对象
    cfg = cfg_param
    
    # 初始化LLM代理
    llm_agent = LLMAgent(model_name="gpt-4o")
    rl_wrapper = ReinforcementWrapper(llm_agent)
    
    # 创建浏览器线程
    url = f"http://127.0.0.1:{cfg.port}"
    browser_thread = threading.Thread(
        target=partial(open_browser, url=url, cfg=cfg)
    )
    browser_thread.daemon = True
    browser_thread.start()
    
    # 等待浏览器启动
    browser_ready = threading.Event()
    
    def wait_for_browser():
        global driver
        start_time = time.time()
        while time.time() - start_time < 30:  # 30秒超时
            if driver is not None:
                browser_ready.set()
                break
            time.sleep(0.5)
    
    wait_thread = threading.Thread(target=wait_for_browser)
    wait_thread.daemon = True
    wait_thread.start()
    
    # 注册关闭浏览器的函数
    def close_browser():
        global driver
        if driver:
            try:
                driver.quit()
                print("Browser closed.")
            except:
                pass
    
    atexit.register(close_browser)
    
    # 创建自动玩游戏的线程
    def auto_play_thread():
        # 等待浏览器准备好
        if browser_ready.wait(30):  # 等待最多30秒
            # 等待页面完全加载
            time.sleep(3)
            # 自动玩游戏并录制
            if cfg.play_all:
                print("Playing all available games")
                auto_play_and_record(driver, cfg)  # 不指定游戏名称，将玩所有游戏
            else:
                print(f"Playing game: {cfg.game}")
                auto_play_and_record(driver, cfg, cfg.game)  # 玩指定的游戏
    
    play_thread = threading.Thread(target=auto_play_thread)
    play_thread.daemon = True
    play_thread.start()
    
    # 运行Flask应用
    print(f"Starting server on port {cfg.port}")
    app.run(port=cfg.port)

if __name__ == "__main__":
    main()
