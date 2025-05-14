
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
python sort_games_by_n_rules.py
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

## 


## Generate data for training a world model

## Main Components

The project includes the following main components:

1. **LLMClient** - A unified LLM query client supporting multiple models and providers
2. **EnhancedLLMAgent** - An enhanced LLM agent for game decision-making
3. **ReinforcementWrapper** - A reinforcement learning wrapper to improve agent performance
4. **StateVisualizer** - A state visualization tool for displaying game states
5. **JaxSokobanAgent** - A Sokoban agent integrated with JAX environments

## File Description

- `client.py` - Core client and agent implementation
- `example_sokoban_agent.py` - Simple example script demonstrating basic usage
- `jax_sokoban_agent.py` - Complete agent implementation integrated with JAX environments

## Installation Dependencies

```bash
# Basic dependencies
pip install numpy requests tiktoken openai python-dotenv

# JAX environment dependencies (if using JAX environments)
pip install jax jaxlib dm-haiku
```

## Usage

### Basic Example

## Collecting/scraping and parsing games

```bash
python example_sokoban_agent.py --model gpt-4o --provider azure --visualize
```

Parameter description:
- `--model`: LLM model name (gpt-4o, o1, o3-mini, gemini-2.0-flash-exp)
- `--provider`: LLM provider (azure, portkey)
- `--game`: Game name
- `--max_steps`: Maximum number of steps
- `--save_dir`: Directory to save agent data
- `--visualize`: Whether to visualize game state

### JAX Environment Integration

Run the agent integrated with JAX environment:

```bash
python jax_sokoban_agent.py --model gpt-4o --provider azure --episodes 5 --render
python collect_games.py
python parse_lark.py
```

Parameter description:
- `--model`: LLM model name
- `--provider`: LLM provider
- `--episodes`: Number of episodes to run
- `--max_steps`: Maximum steps per episode
- `--save_dir`: Directory to save agent data
- `--level`: Sokoban level name
- `--render`: Whether to render game state
- `--no_rl`: Disable reinforcement learning
- `--quiet`: Reduce output verbosity

## Environment Variable Configuration

Create a `.env` file in the project root directory and configure the following environment variables:

```
# Azure OpenAI configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
ENDPOINT_URL=your_azure_endpoint_url
O3_MINI_KEY=your_o3_mini_key

# Portkey API configuration
PORTKEY_API_KEY=your_portkey_api_key
```

## Code Examples

### Using LLMClient

```python
from client import LLMClient

# Initialize client
client = LLMClient()

# Query LLM
response = client.query(
    system_prompt="You are a helpful assistant.",
    prompt="Explain what a Sokoban game is.",
    model="gpt-4o",
    provider="azure"
)

print(response)
```

### Using EnhancedLLMAgent

```python
from client import EnhancedLLMAgent

# Initialize agent
agent = EnhancedLLMAgent(model_name="gpt-4o", provider="azure")

# Process game state
game_state = """
#####
#@$.#
#####
"""
processed_state = agent.process_state(game_state)

# Choose action
action = agent.choose_action(processed_state, goal="Push the box to the target position")
print(f"Chosen action: {action}")
```

### Using ReinforcementWrapper

```python
from client import EnhancedLLMAgent, ReinforcementWrapper

# Initialize agent and RL wrapper
agent = EnhancedLLMAgent(model_name="gpt-4o")
rl_wrapper = ReinforcementWrapper(agent)

# Use RL to choose action
state_hash = "some_state_hash"
action = rl_wrapper.choose_action(state_hash)

# Update Q-value
next_state_hash = "next_state_hash"
reward = 1.0
rl_wrapper.reinforce(state_hash, action, reward, next_state_hash)
```

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
