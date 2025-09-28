
# PuzzleJAX

This repository contains the code for *PuzzleJAX*, a GPU-accelerated implementation of PuzzleScript (https://www.puzzlescript.net)

PuzzleScript is a concise and expressive game description language that has been used by designers to create a plethora of grid-based puzzle games.
At its core are *local pattern rewrite rules*. We take advantage of the convolutional nature of these rewrite rules to implement the engine in JAX, allowing AI practitioners to e.g. efficiently train Reinforcement Learning agents to play arbitrary PuzzleScript games.

![A GIF showing tree-search-generated (partial) solutions to a diverse series of PuzzleScript games.](gifs/header.gif)

## Setup

Create a conda environment (or other virtual environment) running python 3.13, then run:
```
pip install -r requirements.txt
```
But first! Make sure the lines requiring `jax` or `jax[cuda]` are (un)commented appropriately, depending on whether or not you have CUDA available on your system.


## Collecting and parsing data

First, collect games, both from the original PuzzleScript website/editor/javascript-engine repository, (which is checkpointed here under `src`) and an online archive with the following command:
```
python collect_games.py
```
This will also attempt to scrape a dataset of ~900 games from an online database. For this, you will need a Github REST API key saved in `.env`.

To preprocess these files, so that we can validate, profile and benchmark them in the jax, nodejs, and javascript versions of PuzzleScript, run:
```
python preprocess_games.py
```

## Interactive playtesting 

To play a game interactively on a local machine, using the jax environment to run the engine, run, e.g.:
```
python human_env.py game=sokoban_basic jit=True debug=False
```
You can add new/custom games to the `custom_games` folder and refer to them in the above command line argument to playtest them yourself, or similarly supply the game as a command line argument to the RL training script below.

Note that when playing a new level, the first 2 timesteps will have JAX trace and compile the engine's step function, which can be slow (especially for games with more rules, objects, and larger levels).

You can toggle the `jit` and `debug` command line arguments to replace jitted functions with traditional python control loops, and print out verbose logging about rule applications, etc., respectively. (When `jit=False`, we're also able to print out text representations of intermediary level states, which is useful for fine-grained engine debugging, and understanding the rule execution order.)

Similarly, you can launch the javascript PuzzleScript editor with:
```
python server.py mode=None headless=False auto_launch_client=True port=8002
```
, then copy games into the editor and compile and playtest them there as well.

## Validating the jax engine

To generate solutions PuzzleScript games by applying tree search to the original engine, run:
```
python profile_nodejs.py for_validation=True
```
This will run a standalone NodeJS version of the original PuzzleScript engine and save solutions (and terminal states) to disk.

We can then validate that these solutions lead to the same win conditions and level states in PuzzleJAX with:
```
python validate_sols.py overwrite=True
```
This will run the solutions generated above in PuzzleJAX, and ensure that they lead to the same results.

## Profiling the speed of random actions
```
python profile_rand_jax.py
```
```
python profile_rand_nodejs.py
```


## Reinforcement learning

To train an agent using reinforcement learning to play a particular game level, run, e.g.:
```
python train.py game=sokoban_basic level=0 n_envs=600 model=conv2 render_freq=5 hidden_dims=[128,128] seed=0
```
This will log plots and gifs to wandb.

## LLM player agent

To run the LLM agent, use the `llm_agent_loop.py` script. This script iterates through a predefined list of games, running each one for a specified number of trials (`--num_runs`) across all its levels.

### Basic Usage
To run the agent for all priority games with a specific model for a certain number of runs:
```bash
python llm_agent_loop.py --model gemini --num_runs 10
```
Supported models are `4o-mini`, `o3-mini`, `gemini`, `deepseek`, `qwen`, and `deepseek-r1`.

### Command-line options
The script offers several options to customize its execution:
-   `--model`: (Required) The LLM model to use.
-   `--num_runs`: The number of times to run each game level (default: 10).
-   `--max_steps`: The maximum number of steps allowed per episode (default: 100).
-   `--resume_game_name`: The name of the game to start from in the priority list.
-   `--level`: The level number to start from for each game (default: 0).
-   `--reverse`: Process the games in reverse order.
-   `--force`: Rerun all games, even if result files already exist.

### Execution flow
The script performs the following steps:
1.  It loads a predefined list of priority games.
2.  It iterates from `run_id` 1 to `--num_runs`.
3.  In each run, it processes every game in the list.
4.  For each game, it iterates through all available levels, starting from the specified `--level`.
5.  The results for each run and level are saved as individual JSON files in the `llm_agent_results/<model_name>/` directory.

### Generating Analysis
After running the agent, you can generate heatmaps and other analysis by running:
```bash
python llm_agent_results/analysis/result_analysis.py
```
This will save the generated images in the `llm_agent_results/analysis/` directory.


### Custom Environment Integration

To integrate this system with other environments, you need to implement the following functions:

1. Convert environment observations to ASCII representation
2. Convert agent actions to environment-acceptable format
3. Process environment feedback and update the agent

Refer to the implementation in `jax_sokoban_agent.py`, especially the following methods:
- `_observation_to_ascii`
- `_action_to_env_action`
- `run_episode`

### Notes

- Ensure environment variables are correctly configured, especially API keys
- For large game states, you may need to adjust the LLM token limit

## Citing this work

*omitted for anonymity*

## License

MIT