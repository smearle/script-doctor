import os
import time
import argparse
from client import LLMClient, EnhancedLLMAgent, ReinforcementWrapper, StateVisualizer

def main():
    """
    Example script demonstrating how to use the LLM client and agent to play Sokoban
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM agent playing Sokoban example')
    parser.add_argument('--model', type=str, default='gpt-4o', 
                        help='LLM model name (gpt-4o, o1, o3-mini, gemini-2.0-flash-exp)')
    parser.add_argument('--provider', type=str, default='azure', 
                        help='LLM provider (azure, portkey)')
    parser.add_argument('--game', type=str, default='sokoban_basic', 
                        help='Game name')
    parser.add_argument('--max_steps', type=int, default=50, 
                        help='Maximum number of steps')
    parser.add_argument('--save_dir', type=str, default='agent_data', 
                        help='Directory to save agent data')
    parser.add_argument('--visualize', action='store_true', 
                        help='Whether to visualize game state')
    args = parser.parse_args()
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize LLM agent
    agent = EnhancedLLMAgent(model_name=args.model, provider=args.provider)
    rl_wrapper = ReinforcementWrapper(agent)
    
    # Try to load previous agent state
    agent_state_path = os.path.join(args.save_dir, f'{args.game}_agent_state.json')
    q_table_path = os.path.join(args.save_dir, f'{args.game}_q_table.json')
    
    if os.path.exists(agent_state_path):
        print(f"Loading agent state: {agent_state_path}")
        agent.load_agent_state(agent_state_path)
    
    if os.path.exists(q_table_path):
        print(f"Loading Q-table: {q_table_path}")
        rl_wrapper.load_q_table(q_table_path)
    
    # Example Sokoban game state
    # This is just an example, in a real application you would get the state from the game environment
    example_state = """
#####
#@$.#
#####
""".strip()
    
    print("Initial game state:")
    print(example_state)
    
    # Game loop
    current_state = example_state
    prev_state_hash = None
    
    for step in range(args.max_steps):
        print(f"\nStep {step+1}/{args.max_steps}")
        
        # Process current state
        processed_state = agent.process_state(current_state)
        current_state_hash = hash(current_state)
        
        # Visualize current state
        if args.visualize:
            print(StateVisualizer.render_ascii(processed_state))
        
        # Choose action
        # Can use either LLM agent or reinforcement learning wrapper
        use_rl = step > 10  # First 10 steps use LLM, then RL (just as an example)
        
        if use_rl:
            action = rl_wrapper.choose_action(str(current_state_hash))
            print(f"RL chosen action: {action}")
        else:
            action = agent.choose_action(processed_state, goal="Push the box to the target position")
            print(f"LLM chosen action: {action}")
        
        # Execute action in the environment and get new state
        # Here we use a simple simulation function
        new_state, reward, done = simulate_action(current_state, action)
        
        print(f"Executed action: {action}")
        print(f"Reward: {reward}")
        print("New state:")
        print(new_state)
        
        # Update agent history
        agent.update_history(action, "success" if reward > 0 else "in_progress")
        
        # If we have a previous state, update Q-values
        if prev_state_hash is not None:
            new_state_hash = hash(new_state)
            rl_wrapper.reinforce(str(prev_state_hash), action, reward, str(new_state_hash))
        
        # Update state
        current_state = new_state
        prev_state_hash = current_state_hash
        
        # Check if game is complete
        if done:
            print(f"\nGame completed in {step+1} steps!")
            break
        
        # Short pause for observation
        time.sleep(0.5)
    
    # Save agent state and Q-table
    agent.save_agent_state(agent_state_path)
    rl_wrapper.save_q_table(q_table_path)
    
    # Print usage statistics
    print("\nLLM Usage Statistics:")
    stats = agent.llm_client.get_usage_statistics()
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


def simulate_action(state: str, action: str):
    """
    Simulate executing an action in the Sokoban game
    
    Parameters:
        state: String representation of the current game state
        action: Action to execute (up, down, left, right)
        
    Returns:
        (new_state, reward, done): New state, reward, and whether the game is complete
    """
    # Convert state to 2D grid
    grid = [list(line) for line in state.split('\n')]
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    # Find player position
    player_x, player_y = -1, -1
    for y in range(height):
        for x in range(width):
            if grid[y][x] in ['@', '+']: 
                player_x, player_y = x, y
                break
    
    if player_x == -1 or player_y == -1:
        return state, 0, False  # Player not found, return original state
    
    # Determine movement direction
    dx, dy = 0, 0
    if action == 'up':
        dy = -1
    elif action == 'down':
        dy = 1
    elif action == 'left':
        dx = -1
    elif action == 'right':
        dx = 1
    
    # Calculate new position
    new_x, new_y = player_x + dx, player_y + dy
    
    # Check if out of bounds
    if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
        return state, -0.1, False  # Invalid move, return original state
    
    # Check new position
    new_pos = grid[new_y][new_x]
    
    # If wall, cannot move
    if new_pos == '#':
        return state, -0.1, False
    
    # If empty space or target
    if new_pos in [' ', '.']:
        # Move player
        is_on_target = grid[player_y][player_x] == '+'
        grid[player_y][player_x] = '.' if is_on_target else ' '
        grid[new_y][new_x] = '+' if new_pos == '.' else '@'
        
        # Small reward
        reward = 0.1
    
    # If box
    elif new_pos in ['$', '*']:
        # Calculate box's new position
        box_x, box_y = new_x + dx, new_y + dy
        
        # Check if box's new position is valid
        if box_x < 0 or box_x >= width or box_y < 0 or box_y >= height:
            return state, -0.1, False
        
        box_new_pos = grid[box_y][box_x]
        
        # If box's new position is empty space or target
        if box_new_pos in [' ', '.']:
            # Move box
            is_box_on_target = new_pos == '*'
            grid[new_y][new_x] = '.' if is_box_on_target else ' '
            grid[box_y][box_x] = '*' if box_new_pos == '.' else '$'
            
            # Move player
            is_on_target = grid[player_y][player_x] == '+'
            grid[player_y][player_x] = '.' if is_on_target else ' '
            grid[new_y][new_x] = '+' if is_box_on_target else '@'
            
            # Larger reward if box is pushed onto target
            reward = 1.0 if box_new_pos == '.' else 0.2
        else:
            # Box cannot move
            return state, -0.1, False
    else:
        # Unknown case
        return state, 0, False
    
    # Convert back to string
    new_state = '\n'.join(''.join(row) for row in grid)
    
    # Check if game is complete (all boxes are on targets)
    done = all(row.count('$') == 0 for row in grid)  # No boxes not on targets
    
    # Large reward if game is complete
    if done:
        reward = 10.0
    
    return new_state, reward, done


if __name__ == "__main__":
    main()
