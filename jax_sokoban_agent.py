import os
import time
import argparse
import json
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# Import our LLM client and agent
from client import LLMClient, EnhancedLLMAgent, ReinforcementWrapper, StateVisualizer

# Note: This script assumes you have installed jax and related sokoban environment
# If not, please install first: pip install jax jaxlib dm-haiku

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("Warning: JAX not found. Please install JAX: pip install jax jaxlib")

# Try to import sokoban environment
try:
    # Assuming the sokoban environment is in a package, adjust import path as needed
    from env import SokobanEnv
except ImportError:
    print("Warning: Sokoban environment not found. Please ensure related packages are installed.")
    # Define a placeholder class so the script can continue running
    class SokobanEnv:
        def __init__(self, *args, **kwargs):
            print("Using placeholder SokobanEnv class. Please provide a real environment when running.")
        
        def reset(self):
            return {"observation": np.zeros((10, 10, 3)), "info": {}}
        
        def step(self, action):
            return {"observation": np.zeros((10, 10, 3)), "reward": 0, "done": False, "info": {}}


class JaxSokobanAgent:
    """
    LLM agent for interacting with JAX Sokoban environment
    """
    def __init__(self, 
                 model_name: str = "gpt-4o", 
                 provider: str = "azure",
                 env_config: Optional[Dict[str, Any]] = None,
                 save_dir: str = "agent_data",
                 use_rl: bool = True,
                 verbose: bool = True):
        """
        Initialize JAX Sokoban agent
        
        Parameters:
            model_name: LLM model name
            provider: LLM provider (azure or portkey)
            env_config: Sokoban environment configuration
            save_dir: Directory to save agent data
            use_rl: Whether to use reinforcement learning
            verbose: Whether to print detailed information
        """
        self.model_name = model_name
        self.provider = provider
        self.env_config = env_config or {}
        self.save_dir = save_dir
        self.use_rl = use_rl
        self.verbose = verbose
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize LLM agent
        self.agent = EnhancedLLMAgent(model_name=model_name, provider=provider)
        
        # If using reinforcement learning, initialize RL wrapper
        self.rl_wrapper = ReinforcementWrapper(self.agent) if use_rl else None
        
        # Initialize environment
        self.env = None
        self.reset_env()
        
        # Load previous agent state and Q-table
        self._load_agent_state()
    
    def reset_env(self):
        """Reset Sokoban environment"""
        try:
            self.env = SokobanEnv(**self.env_config)
            self.state = self.env.reset()
            if self.verbose:
                print("Environment reset")
            return self.state
        except Exception as e:
            print(f"Error resetting environment: {e}")
            return None
    
    def _load_agent_state(self):
        """Load agent state and Q-table"""
        env_name = self.env_config.get('level_name', 'sokoban')
        agent_state_path = os.path.join(self.save_dir, f'{env_name}_agent_state.json')
        q_table_path = os.path.join(self.save_dir, f'{env_name}_q_table.json')
        
        # Load agent state
        if os.path.exists(agent_state_path):
            if self.verbose:
                print(f"Loading agent state: {agent_state_path}")
            self.agent.load_agent_state(agent_state_path)
        
        # Load Q-table
        if self.use_rl and os.path.exists(q_table_path):
            if self.verbose:
                print(f"Loading Q-table: {q_table_path}")
            self.rl_wrapper.load_q_table(q_table_path)
    
    def _save_agent_state(self):
        """Save agent state and Q-table"""
        env_name = self.env_config.get('level_name', 'sokoban')
        agent_state_path = os.path.join(self.save_dir, f'{env_name}_agent_state.json')
        q_table_path = os.path.join(self.save_dir, f'{env_name}_q_table.json')
        
        # Save agent state
        self.agent.save_agent_state(agent_state_path)
        if self.verbose:
            print(f"Agent state saved: {agent_state_path}")
        
        # Save Q-table
        if self.use_rl:
            self.rl_wrapper.save_q_table(q_table_path)
            if self.verbose:
                print(f"Q-table saved: {q_table_path}")
    
    def _observation_to_ascii(self, observation) -> str:
        """
        Convert JAX environment observation to ASCII representation
        
        Parameters:
            observation: JAX environment observation (typically an array)
            
        Returns:
            ASCII representation of the game state
        """
        # This function needs to be adjusted based on the actual observation format
        # Here we assume the observation is a 3D array where different values represent different entities
        
        if isinstance(observation, dict) and "observation" in observation:
            observation = observation["observation"]
        
        # Convert JAX array to NumPy array
        if hasattr(observation, "device_buffer"):
            observation = np.array(observation)
        
        # Assume observation is an array with shape (height, width, channels)
        height, width = observation.shape[:2]
        
        # Define entity mapping (adjust based on actual environment)
        # Assume:
        # 0 = empty space
        # 1 = wall
        # 2 = box
        # 3 = target
        # 4 = box on target
        # 5 = player
        # 6 = player on target
        entity_map = {
            0: ' ',  # empty space
            1: '#',  # wall
            2: '$',  # box
            3: '.',  # target
            4: '*',  # box on target
            5: '@',  # player
            6: '+'   # player on target
        }
        
        # Create ASCII representation
        ascii_state = []
        for y in range(height):
            row = []
            for x in range(width):
                # Get entity type at this position
                # Assume entity type is the maximum value in the observation
                if len(observation.shape) == 3:
                    entity_type = np.argmax(observation[y, x])
                else:
                    entity_type = observation[y, x]
                
                # Map to ASCII character
                char = entity_map.get(entity_type, '?')
                row.append(char)
            
            ascii_state.append(''.join(row))
        
        return '\n'.join(ascii_state)
    
    def _action_to_env_action(self, action: str) -> int:
        """
        Convert agent's action string to environment-acceptable action ID
        
        Parameters:
            action: Action chosen by the agent (up, down, left, right)
            
        Returns:
            Environment-acceptable action ID
        """
        # Adjust action mapping based on actual environment
        action_map = {
            'up': 0,
            'down': 1,
            'left': 2,
            'right': 3
        }
        
        return action_map.get(action, 0)  # Default to up (0)
    
    def run_episode(self, max_steps: int = 100, render: bool = False) -> Dict[str, Any]:
        """
        Run a complete game episode
        
        Parameters:
            max_steps: Maximum number of steps
            render: Whether to render game state
            
        Returns:
            Dictionary containing episode statistics
        """
        # Reset environment
        state = self.reset_env()
        if state is None:
            return {"success": False, "error": "Unable to reset environment"}
        
        # Initialize statistics
        stats = {
            "total_reward": 0,
            "steps": 0,
            "success": False,
            "actions": [],
            "rewards": []
        }
        
        # Convert initial observation to ASCII
        ascii_state = self._observation_to_ascii(state)
        prev_state_hash = hash(ascii_state)
        
        if self.verbose:
            print("Initial state:")
            print(ascii_state)
        
        # Game loop
        for step in range(max_steps):
            if self.verbose:
                print(f"\nStep {step+1}/{max_steps}")
            
            # Process current state
            processed_state = self.agent.process_state(ascii_state)
            current_state_hash = hash(ascii_state)
            
            # If rendering is needed, display ASCII state
            if render and self.verbose:
                print(StateVisualizer.render_ascii(processed_state))
            
            # Choose action
            if self.use_rl and step > 10:  # First 10 steps use LLM, then RL
                action = self.rl_wrapper.choose_action(str(current_state_hash))
                if self.verbose:
                    print(f"RL chosen action: {action}")
            else:
                action = self.agent.choose_action(processed_state, goal="Push all boxes to target positions")
                if self.verbose:
                    print(f"LLM chosen action: {action}")
            
            # Record action
            stats["actions"].append(action)
            
            # Convert action to environment-acceptable format
            env_action = self._action_to_env_action(action)
            
            # Execute action in environment
            try:
                next_state = self.env.step(env_action)
                reward = next_state.get("reward", 0)
                done = next_state.get("done", False)
                
                # Record reward
                stats["rewards"].append(reward)
                stats["total_reward"] += reward
                
                if self.verbose:
                    print(f"Executed action: {action} (environment action ID: {env_action})")
                    print(f"Reward: {reward}")
                
                # Convert new state to ASCII
                new_ascii_state = self._observation_to_ascii(next_state)
                
                if self.verbose:
                    print("New state:")
                    print(new_ascii_state)
                
                # Update agent history
                self.agent.update_history(action, "success" if reward > 0 else "in_progress")
                
                # If using RL and have previous state, update Q-values
                if self.use_rl and prev_state_hash is not None:
                    new_state_hash = hash(new_ascii_state)
                    self.rl_wrapper.reinforce(str(prev_state_hash), action, reward, str(new_state_hash))
                
                # Update state
                ascii_state = new_ascii_state
                prev_state_hash = current_state_hash
                
                # Check if game is complete
                if done:
                    stats["success"] = True
                    if self.verbose:
                        print(f"\nGame completed in {step+1} steps!")
                    break
                
            except Exception as e:
                print(f"Error executing action: {e}")
                break
            
            # Update step count
            stats["steps"] = step + 1
        
        # Save agent state and Q-table
        self._save_agent_state()
        
        return stats
    
    def run_multiple_episodes(self, num_episodes: int = 10, max_steps: int = 100, render: bool = False) -> List[Dict[str, Any]]:
        """
        Run multiple game episodes
        
        Parameters:
            num_episodes: Number of episodes
            max_steps: Maximum steps per episode
            render: Whether to render game state
            
        Returns:
            List containing statistics for each episode
        """
        all_stats = []
        
        for episode in range(num_episodes):
            if self.verbose:
                print(f"\n=== Episode {episode+1}/{num_episodes} ===\n")
            
            # Run one episode
            stats = self.run_episode(max_steps=max_steps, render=render)
            
            # Add episode number
            stats["episode"] = episode + 1
            
            # Add to statistics list
            all_stats.append(stats)
            
            # Print episode summary
            if self.verbose:
                print(f"\nEpisode {episode+1} summary:")
                print(f"  Steps: {stats['steps']}")
                print(f"  Total reward: {stats['total_reward']}")
                print(f"  Success: {stats['success']}")
        
        # Save all statistics
        stats_path = os.path.join(self.save_dir, "episode_stats.json")
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        
        if self.verbose:
            print(f"\nAll episode statistics saved to: {stats_path}")
        
        return all_stats
    
    def get_llm_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        return self.agent.llm_client.get_usage_statistics()


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='JAX Sokoban LLM Agent')
    parser.add_argument('--model', type=str, default='gpt-4o', 
                        help='LLM model name (gpt-4o, o1, o3-mini, gemini-2.0-flash-exp)')
    parser.add_argument('--provider', type=str, default='azure', 
                        help='LLM provider (azure, portkey)')
    parser.add_argument('--episodes', type=int, default=5, 
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=100, 
                        help='Maximum steps per episode')
    parser.add_argument('--save_dir', type=str, default='jax_agent_data', 
                        help='Directory to save agent data')
    parser.add_argument('--level', type=str, default='unfiltered/train/5',
                        help='Sokoban level name')
    parser.add_argument('--render', action='store_true', 
                        help='Whether to render game state')
    parser.add_argument('--no_rl', action='store_true',
                        help='Disable reinforcement learning')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    args = parser.parse_args()
    
    # Environment configuration
    env_config = {
        'level_name': args.level,
        'max_steps': args.max_steps,
        # Add other configurations based on actual environment
    }
    
    # Create agent
    agent = JaxSokobanAgent(
        model_name=args.model,
        provider=args.provider,
        env_config=env_config,
        save_dir=args.save_dir,
        use_rl=not args.no_rl,
        verbose=not args.quiet
    )
    
    # Run multiple episodes
    all_stats = agent.run_multiple_episodes(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render
    )
    
    # Print LLM usage statistics
    if not args.quiet:
        print("\nLLM Usage Statistics:")
        stats = agent.get_llm_usage_stats()
        for model, model_stats in stats.items():
            print(f"Model: {model}")
            print(f"  Total queries: {model_stats['total_queries']}")
            print(f"  Successful queries: {model_stats['successful_queries']}")
            print(f"  Success rate: {model_stats['success_rate']:.2%}")
            
            for provider, provider_stats in model_stats['by_provider'].items():
                print(f"  Provider {provider}:")
                print(f"    Total queries: {provider_stats['total']}")
                print(f"    Successful queries: {provider_stats['successful']}")
                success_rate = provider_stats['successful'] / provider_stats['total'] if provider_stats['total'] > 0 else 0
                print(f"    Success rate: {success_rate:.2%}")
    
    # Print overall statistics
    success_count = sum(1 for stats in all_stats if stats["success"])
    success_rate = success_count / len(all_stats) if all_stats else 0
    avg_steps = sum(stats["steps"] for stats in all_stats) / len(all_stats) if all_stats else 0
    avg_reward = sum(stats["total_reward"] for stats in all_stats) / len(all_stats) if all_stats else 0
    
    print("\nOverall Statistics:")
    print(f"  Episodes: {len(all_stats)}")
    print(f"  Successful episodes: {success_count}")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Average steps: {avg_steps:.2f}")
    print(f"  Average reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
