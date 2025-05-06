# 自动运行PuzzleScript游戏并保存GIF

这个脚本允许LLM自动运行PuzzleScript游戏并保存游戏过程的GIF。

## 功能特点

- 自动打开浏览器并访问PuzzleScript游戏编辑器
- 自动运行游戏
- 使用LLM代理自动玩游戏
- 自动录制游戏过程并保存为GIF
- 不使用sleep等待，而是使用事件驱动的方式检测游戏状态

## 依赖项

- Python 3.7+
- Flask
- Selenium
- Hydra
- 其他依赖项（见requirements.txt）

## 安装

1. 确保已安装Python 3.7+
2. 安装依赖项：

```bash
pip install -r requirements.txt
```

3. 确保已安装Chrome浏览器和ChromeDriver

## 使用方法

1. 运行脚本：

```bash
python auto_play_and_record.py
```

2. 默认配置：

- 端口：8000
- 浏览器：非无头模式，最大化，自动打开开发者工具
- GIF保存：启用，保存到gifs目录

3. 自定义配置：

```bash
python auto_play_and_record.py port=8080 headless=true save_gifs=false
```

## 配置选项

- `port`：服务器端口，默认为8000
- `headless`：是否使用无头模式运行浏览器，默认为false
- `auto_open_devtools`：是否自动打开开发者工具，默认为true
- `maximize_browser`：是否最大化浏览器窗口，默认为true
- `save_gifs`：是否保存GIF，默认为true
- `gif_output_dir`：GIF保存目录，默认为"gifs"

## 工作原理

1. 脚本启动一个Flask服务器，提供PuzzleScript游戏编辑器的静态文件。
2. 在一个单独的线程中，脚本使用Selenium打开浏览器并访问游戏编辑器。
3. 脚本注入自定义JavaScript代码，增强GIF录制功能。
4. 脚本自动点击"RUN"按钮运行游戏。
5. 脚本自动开始录制GIF。
6. 脚本自动点击"LLM PLAY"按钮，启动LLM代理玩游戏。
7. LLM代理通过/llm_action API与服务器通信，获取游戏状态并返回动作。
8. 脚本每秒检查一次游戏是否已经赢了，如果赢了，就停止录制并保存GIF。
9. GIF数据通过/save_gif API发送到服务器，服务器将其保存到指定目录。

## 注意事项

- 确保PuzzleScript游戏编辑器的静态文件位于src目录下。
- 确保LLM_agent.py文件可用，它定义了LLM代理类。
- 脚本使用事件驱动的方式检测游戏状态，而不是使用sleep等待，这样可以更准确地捕捉游戏结束的时刻。
- 如果游戏在5分钟内没有赢，脚本会超时并停止录制。
