import json
import os
from typing import List, Dict
import numpy as np

class LLMAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.action_history = []
        self.state_memory = []
        
    def process_state(self, state_repr: str) -> dict:
        """Parse game state representation"""
        return {
            'raw_state': state_repr,
            'entities': self._extract_entities(state_repr),
            'metrics': self._calculate_metrics(state_repr)
        }
    
    def _extract_entities(self, state: str) -> List[Dict]:
        """Identify game objects from state string"""
        # Basic entity extraction logic
        entities = []
        for y, line in enumerate(state.split('\n')):
            for x, char in enumerate(line):
                if char != ' ':
                    entities.append({
                        'type': 'object',
                        'symbol': char,
                        'position': (x, y)
                    })
        return entities
    
    def _calculate_metrics(self, state: str) -> Dict:
        """Calculate basic state metrics"""
        return {
            'complexity': len(state.replace(' ', '')),
            'diversity': len(set(state)) - 1  # Exclude space
        }
    
    def choose_action(self, processed_state: dict, goal: str = "") -> str:
        """Core decision-making logic"""
        # Placeholder for actual LLM integration
        return np.random.choice(['up', 'down', 'left', 'right', 'use'])
    
    def update_history(self, action: str, result: str):
        """Maintain action-result history"""
        self.action_history.append({
            'action': action,
            'result': result,
            'timestamp': len(self.action_history)
        })
    
    def save_agent_state(self, path: str):
        """Persist agent state to disk"""
        with open(path, 'w') as f:
            json.dump({
                'history': self.action_history,
                'model': self.model_name
            }, f)

class ReinforcementWrapper:
    def __init__(self, base_agent: LLMAgent):
        self.base_agent = base_agent
        self.q_table = {}
        
    def reinforce(self, reward: float):
        """Update Q-values based on recent actions"""
        if len(self.base_agent.action_history) > 0:
            last_action = self.base_agent.action_history[-1]['action']
            self.q_table[last_action] = self.q_table.get(last_action, 0) + reward

class StateVisualizer:
    @staticmethod
    def render_ascii(state: dict) -> str:
        """Convert processed state back to ASCII"""
        max_x = max(e['position'][0] for e in state['entities'])
        max_y = max(e['position'][1] for e in state['entities'])
        
        grid = [[' ' for _ in range(max_x+1)] for _ in range(max_y+1)]
        for e in state['entities']:
            x, y = e['position']
            grid[y][x] = e['symbol']
        return '\n'.join(''.join(row) for row in grid)
