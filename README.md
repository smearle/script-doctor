
ScriptDoctor & PuzzleJax



This project provides a unified LLM client and agent system for using Large Language Models (LLMs) as intelligent agents in Sokoban puzzle game environments. The system supports multiple LLM models and providers, and integrates reinforcement learning capabilities.
## Setup

Create a conda environment or whatnot running python 3.13, then:
```
pip install -r requirements.txt
```
But first! Make sure the lines requiring `jax` or `jax[cuda]` are (un)commented appropriately, depending on whether or not you have CUDA available on your system.


## Collecting and parsing data

First, collect games, both from the original PuzzleScript website/editor/javascript-engine repository, (which is checkpointed here under `src`) with the following command:
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
Note that the first 2 timesteps will have jax trace and compile the engine's step function, which can be painfully slow (especially for games with more rules, objects, and larger levels).
(TODO: Figure out why the first *2* timesteps, and not just the first, require compilation.)

You can toggle the `jit` and `debug` command line arguments to replace jitted functions with traditional python control loops, and print out verbose logging about rule applications, etc., respectively. (When `jit=False`, we're also able to print out text representations of intermediary level states, which is useful for fine-grained engine debugging, and understanding the rule execution order.)

Similarly, you can launch the javascript PuzzleScript editor with:
```
python server.py mode=None headless=False auto_launch_client=True port=8002
```
, then copy games into the editor and compile and playtest them there manually. (TODO: could be nice to automatically load a game from a command-line argument.)

## Validating the jax engine

To generate solutions for games using the javascript version of puzzlescript, run:
```
python server.py mode=gen_solutions auto_launch_client=True headless=True port=8001
```
This will launch a (headless) browser that runs PS in JS, and save the resultant solutions.

We can then validate that these solutions lead to the same win conditions

## Profiling the speed of random actions
```
python profile_rand_jax.py
```
```
python profile_rand_nodejs.py
```

## Tree search



## Reinforcement learning

To train an agent using reinforcement learning to play a particular game level, run, e.g.:
```
python train.py game=sokoban_basic level=0 n_envs=600 model=conv2 render_freq=5 hidden_dims=[128,128] seed=0
```
This will attempt to log plots and gifs to wandb.



## Custom Environment Integration

To integrate this system with other environments, you need to implement the following functions:

1. Convert environment observations to ASCII representation
2. Convert agent actions to environment-acceptable format
3. Process environment feedback and update the agent

Refer to the implementation in `jax_sokoban_agent.py`, especially the following methods:
- `_observation_to_ascii`
- `_action_to_env_action`
- `run_episode`

## Notes

- Ensure environment variables are correctly configured, especially API keys
- JAX environments require additional dependencies
- For large game states, you may need to adjust the LLM token limit
- Reinforcement learning features require sufficient training episodes to significantly improve performance

## License

MIT
