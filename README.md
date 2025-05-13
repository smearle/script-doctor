ScriptDoctor & PuzzleJax
============

## Setup

Create a conda environment or whatnot running python 3.13, then:
```
pip install -r requirements.txt
```
But first! Make sure the lines requiring `jax` or `jax[cuda]` are (un)commented appropriately, depending on whether or not you have CUDA available on your system.


## Usage

First, collect games, both from the original PuzzleScript website/editor/javascript-engine repository, (which is checkpointed here under `src`) with the following command:
```
python collect_games.py
```
This will also attempt to scrape a dataset of ~900 games from an online database. For this, you will need a Github REST API key saved in `.env`.

To play a game interactively on a local machine, using the jax environment to run the engine, run, e.g.:
```
python human_env.py game=sokoban_basic jit=True debug=False
```
Note that the first 2 timesteps will have jax trace and compile the engine's step function, which can be painfully slow (especially for games with more rules, objects, and larger levels).
(TODO: Figure out why the first *2* timesteps, and not just the first, require compilation.)

You can toggle the `jit` and `debug` command line arguments to replace jitted functions with traditional python control loops, and print out verbose logging about rule applications, etc., respectively. (When `jit=False`, we're also able to print out text representations of intermediary level states, which is useful for fine-grained engine debugging, and understanding the rule execution order.)

### Tree search

To generate solutions for games using the javascript version of puzzlescript, run:
```
python server.py mode=gen_solutions auto_launch_client=True headless=True port=8001
```
This will launch a (headless) browser that runs PS in JS.

Note that this requires first sorting games according to the number of rules in each. This can be achieved with:
```
python sort_games_by_n_rules.py
```

### Reinforcement learning

To train an agent using reinforcement learning to play a particular game level, run, e.g.:
```
python train.py game=sokoban_basic level=0 n_envs=600 model=conv2 render_freq=5 hidden_dims=[128,128] seed=0
```
This will attempt to log plots and gifs to wandb.


## Generate data for training a world model

```
python server.py
```
Open up the local web-page, then press `GEN DATA` in the top-left. This will run `processAllGames` in `src/js/ScriptDoctor.js`, which loads games (in random order) and applies A*, saving images of unique level states, as well as a record of which actions led to which transitions between states. This data is stored in `transitions`.

TODO:
- save symbolic representations of the map instead
- write a script that converts these symbolic representations to 2D multi-hot arrays
- train an MLP to predict the next state given a previous state and an action.

## Collecting/scraping and parsing games

```
python collect_games.py
python parse_lark.py
```

## Fine-tuning a model

```
python finetune.py
```

## Evolving games with OpenAI queries

Put your OpenAI API key in a file called `.env`, which will have a single line that reads `OPENAI_API_KEY=yourkeyhere`.

To install requirements: `pip install -r requirements.txt`. To run a local server: `python server.py`. Then, open the local IP address. You should see the PuzzleScript editor page displayed, and some GPT slop appearing in the PuzzleScript terminal and level code editor. 

The main function is run client-side, from inside `ScriptDoctor.js`, which is included in the `editor.html` (which is served by the server). This JS process asks for a game from the server (which makes an OpenAI API call), then throws it in the editor.

Next: playtesting. Making generated games not suck.

Notes:
- We made a single edit to `compile.js` to fix an issue importing gzipper, but we don't actually use the compressed version of the engine at the moment (the one in `bin`---instead just using the one in `src`).

TODO (feel free to claim a task---they're relatively standalone):
- Submodule focused solely on adding new levels to functioning games
- Save gifs of solutions being played out (there is some existing functionality for saving gifs in the js codebase---use it)
- Feed screenshots of generated levels to GPT-4o to iterate on sprites
- Some kind of evolutionary loop that will generate a bunch of games for us, diverse/novel along some axis (implemented, need to debug)

## Running experiments

To sweep over fewshot and chain of thought prompting, uncomment `sweep()` in `src/js/ScriptDoctor.js`, launch the server with `python server.py` and open the webpage at `127.0.0.1:8000` (or whatever pops up in the terminal where you've launched the server). Then the javascript code, and the `sweep()` function, will be run. Once this is done, run `python compute_edit_distances.py` then `python eval_fewshot_cot_sweep.py` to generate tables of results.

To generate game ideas, run `python brainstorm.py`, then uncomment `fromIdeaSweep()` in `src/js/ScriptDoctor.js`, launch the server and open the webpage, then run `python compute_from_idea_edit_distances.py` and `python eval_from_idea_sweep.py`.

PuzzleScript
============

Open Source HTML5 Puzzle Game Engine

Try it out at https://www.puzzlescript.net.

-----

If you're interested in recompiling/modifing/hacking the engine, there is [development setup info here](DEVELOPMENT.md).  If you're just interested in learning how to use the engine/make games in it, [the documentation is here](https://www.puzzlescript.net/Documentation/documentation.html).
