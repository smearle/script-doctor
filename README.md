# LLM Agent for Sokoban

This project provides a unified LLM client and agent system for using Large Language Models (LLMs) as intelligent agents in Sokoban puzzle game environments. The system supports multiple LLM models and providers, and integrates reinforcement learning capabilities.
## Setup

Requires python 3.12
```
pip install -r requirements.txt
```
(If you don't have cuda available, install `jax` instead of `jax[cuda]`.)

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
python gen_trees.py
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
