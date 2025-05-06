# Automatically Run PuzzleScript Game and Save GIF

This script allows an LLM to automatically run a PuzzleScript game and save a GIF of the gameplay.

## Features

* Automatically opens the browser and accesses the PuzzleScript game editor
* Automatically runs the game
* Uses an LLM agent to play the game
* Automatically records the gameplay and saves it as a GIF
* Uses event-driven detection of game state instead of relying on sleep delays

## Dependencies

* Python 3.7+
* Flask
* Selenium
* Hydra
* Other dependencies (see `requirements.txt`)

## Installation

1. Make sure Python 3.7+ is installed.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure that Google Chrome and ChromeDriver are installed.

## Usage

1. Run the script:

```bash
python auto_play_and_record.py
```

2. Default configuration:

* Port: 8000
* Browser: Non-headless mode, maximized, with developer tools automatically opened
* GIF Saving: Enabled, saved to the `gifs` directory

3. Custom configuration:

```bash
python auto_play_and_record.py port=8080 headless=true save_gifs=false
```

## Configuration Options

* `port`: Server port (default: 8000)
* `headless`: Whether to run the browser in headless mode (default: false)
* `auto_open_devtools`: Whether to automatically open developer tools (default: true)
* `maximize_browser`: Whether to maximize the browser window (default: true)
* `save_gifs`: Whether to save the GIF recording (default: true)
* `gif_output_dir`: Directory to save GIFs (default: "gifs")

## How It Works

1. The script starts a Flask server that serves the static files of the PuzzleScript game editor.
2. In a separate thread, the script uses Selenium to open the browser and access the editor.
3. The script injects custom JavaScript code to enhance GIF recording functionality.
4. The script automatically clicks the "RUN" button to launch the game.
5. It then automatically starts recording a GIF.
6. The script clicks the "LLM PLAY" button to let the LLM agent begin playing.
7. The LLM agent communicates with the server via the `/llm_action` API to get the current game state and return an action.
8. The script checks once per second to see if the game is won. If so, it stops recording and saves the GIF.
9. GIF data is sent to the server via the `/save_gif` API and saved to the specified directory.

## Notes

* Ensure the static files for the PuzzleScript editor are located in the `src` directory.
* Make sure the `LLM_agent.py` file is available—it defines the LLM agent class.
* The script uses an event-driven approach to detect game completion, allowing for more accurate capture of the win moment.
* If the game is not won within 5 minutes, the script will timeout and stop recording.


