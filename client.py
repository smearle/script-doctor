
import atexit
import time
from selenium import webdriver


url = 'http://127.0.0.1:8000'
import json
import os
import random
import re
import requests
import tiktoken
from typing import List, Dict, Optional, Union, Any
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Azure OpenAI configuration
GPT4V_ENDPOINT = "https://aoai-physics.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview"
GPT4V_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_HEADERS = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}

def open_browser(url=url, headless=False):

    # Set up Selenium WebDriver
    options = webdriver.ChromeOptions()
    if headless:

        # Enable browser console logging
        options.set_capability("goog:loggingPrefs", {"browser": "ALL"})

        options.add_argument("--headless")  # Run in headless mode
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
    else:
        options.add_argument("--auto-open-devtools-for-tabs")  # Open developer tools
        options.add_argument("--start-maximized")  # Open browser maximized

    driver = webdriver.Chrome(options=options)
    driver.get(url)  # Open the URL

    # Switch to the "console" tab of the developer tools
    # driver.execute_script("window.open('');")
    # driver.switch_to.window(driver.window_handles[1])
    # driver.get('chrome://devtools/console')
    atexit.register(_safe_quit, driver)

    seen = set()
    print("Streaming JS console output...\n")

    try:
        while True:
            logs = driver.get_log("browser")
            for entry in logs:
                print(f"[{entry['level']}] {entry['message']}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        driver.quit()  

    return driver
# Azure OpenAI other model configurations
O_ENDPOINT = os.getenv("ENDPOINT_URL", "https://sc-pn-m898m3wl-eastus2.openai.azure.com/")
O_KEY = os.getenv("O3_MINI_KEY")

# Portkey API configuration
PORTKEY_API_KEY = os.getenv('PORTKEY_API_KEY')
PORTKEY_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1"

# Global client instance
azure_client = None

class LLMClient:
    """
    Unified LLM client supporting multiple models and providers
    """
    def __init__(self):
        self.azure_client = None
        self.portkey_client = None
        self.model_history = {}  # Record usage history for each model
    
    def query(self, 
              system_prompt: str, 
              prompt: str, 
              seed: Optional[int] = None, 
              model: str = "gpt-4o", 
              provider: str = "azure",
              max_retries: int = 3,
              retry_delay: int = 5,
              temperature: float = 0.7,
              top_p: float = 0.95,
              max_tokens: int = 1000000) -> str:
        """
        Unified query interface supporting multiple models and providers
        
        Parameters:
            system_prompt: System prompt
            prompt: User prompt
            seed: Random seed
            model: Model name, such as "gpt-4o", "o1", "o3-mini", "gemini-2.0-flash-exp", etc.
            provider: Provider, can be "azure" or "portkey"
            max_retries: Maximum number of retry attempts
            retry_delay: Retry delay in seconds
            temperature: Temperature parameter
            top_p: Top-p parameter
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            LLM response text
        """
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        # Record query history
        if model not in self.model_history:
            self.model_history[model] = []
        
        self.model_history[model].append({
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "success": False  # Default to failure, update on success
        })
        
        # Choose query method based on provider
        if provider == 'azure':
            response = self._query_azure(messages, model, max_retries, retry_delay, temperature, top_p, max_tokens)
        elif provider == 'portkey':
            response = self._query_portkey(messages, model, max_retries, retry_delay, max_tokens)
        else:
            raise ValueError("Invalid provider. Please choose 'azure' or 'portkey'.")
        
        # Update query history to success
        self.model_history[model][-1]["success"] = True
        
        return response
    
    def _query_azure(self, 
                    messages: List[Dict[str, str]], 
                    model: str, 
                    max_retries: int, 
                    retry_delay: int,
                    temperature: float,
                    top_p: float,
                    max_tokens: int) -> str:
        """
        Query using Azure OpenAI API
        """
        if model == 'gpt-4o':
            payload = {
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
            }
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    print('Querying Azure OpenAI...')
                    response = requests.post(GPT4V_ENDPOINT, headers=AZURE_HEADERS, json=payload)
                    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
                    print('Query completed.')
                    return response.json()['choices'][0]['message']['content']
                except requests.RequestException as e:
                    print(f"Request failed. RequestException: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Will retry in {retry_delay} seconds... ({retry_count}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        raise Exception(f"Query failed after {max_retries} attempts")
                except requests.HTTPError as e:
                    print(f"HTTPError: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Will retry in {retry_delay} seconds... ({retry_count}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        raise Exception(f"Query failed after {max_retries} attempts")
        else:
            # Use other Azure models
            global azure_client
            if azure_client is None:
                azure_client = AzureOpenAI(  
                    azure_endpoint=O_ENDPOINT,  
                    api_key=O_KEY,  
                    api_version="2024-12-01-preview",
                )
            
            if model not in ['o1', 'o3-mini']:
                raise ValueError(f"Unsupported Azure model: {model}. Please choose 'gpt-4o', 'o1', or 'o3-mini'.")
            
            deployment = os.getenv('DEPLOYMENT_NAME', model)
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    print(f'Querying Azure OpenAI ({model})...')
                    completion = azure_client.chat.completions.create(  
                        model=deployment,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        stop=None,  
                        stream=False
                    )
                    print('Query completed.')
                    return completion.choices[0].message.content
                except Exception as e:
                    print(f"Query failed: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Will retry in {retry_delay} seconds... ({retry_count}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        raise Exception(f"Query failed after {max_retries} attempts")
    
    def _query_portkey(self, 
                      messages: List[Dict[str, str]], 
                      model: str, 
                      max_retries: int, 
                      retry_delay: int,
                      max_tokens: int) -> str:
        """
        Query using Portkey API
        """
        try:
            # Select appropriate virtual key
            virtual_key = "vertex-ai-3e806d"  # Default to vertex-ai
            
            if model == "o3-mini":
                virtual_key = "o3-mini-5791cb"
            elif model == "gpt-4o-mini":
                virtual_key = "gpt-4o-mini-efbb71"
            
            print(f'Querying Portkey API with model {model}, virtual key {virtual_key}...')
            
            # Prepare request
            url = f"{PORTKEY_BASE_URL}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {PORTKEY_API_KEY}",
                "x-portkey-virtual-key": virtual_key
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens
            }
            
            # Send request with timeout and retry settings
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                    
                    # Check response status
                    if response.status_code == 200:
                        response_data = response.json()
                        print('Query completed successfully.')
                        return response_data['choices'][0]['message']['content']
                    elif response.status_code == 504:  # Gateway timeout
                        print(f"Gateway timeout (504), retrying... ({retry_count+1}/{max_retries})")
                        retry_count += 1
                        time.sleep(retry_delay)
                    else:
                        print(f"Request failed with status code: {response.status_code}")
                        print(f"Response text: {response.text}")
                        # If it's a 404 error and the model is vertex-ai, try falling back to o3-mini
                        if response.status_code == 404 and model == "vertex-ai":
                            print("Vertex AI model not found, falling back to o3-mini...")
                            return self._query_portkey(messages, "o3-mini", max_retries, retry_delay, max_tokens)
                        # For other errors, throw an exception
                        raise Exception(f"API request failed with status code: {response.status_code}")
                except requests.exceptions.Timeout:
                    print(f"Request timed out, retrying... ({retry_count+1}/{max_retries})")
                    retry_count += 1
                    time.sleep(retry_delay)
                except requests.exceptions.RequestException as e:
                    print(f"Request exception: {e}")
                    retry_count += 1
                    time.sleep(retry_delay)
            
            # If still failing after maximum retries, throw an exception
            raise Exception(f"Query failed after {max_retries} attempts")
            
        except ImportError:
            # If portkey is not installed, fall back to Azure implementation
            print("Portkey not installed, falling back to Azure implementation")
            return self._query_azure(messages, "gpt-4o" if model == "gpt-4o-mini" else "o3-mini", max_retries, retry_delay, 0.7, 0.95, max_tokens)

    def get_token_count(self, text: str, model: str = "gpt-4o") -> int:
        """
        Calculate the number of tokens in a text
        
        Parameters:
            text: The text to calculate tokens for
            model: Model name
            
        Returns:
            Number of tokens
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Error calculating token count: {e}")
            # Fallback to simple estimation: approximately 1.3 tokens per word
            return int(len(text.split()) * 1.3)
    
    def truncate_text_to_token_limit(self, text: str, model: str, max_tokens: int) -> str:
        """
        Truncate text to a specified token limit
        
        Parameters:
            text: Text to truncate
            model: Model name
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)
        except Exception as e:
            print(f"Error truncating text: {e}")
            # Fallback to simple estimation
            words = text.split()
            estimated_tokens_per_word = 1.3
            estimated_words = int(max_tokens / estimated_tokens_per_word)
            return " ".join(words[:estimated_words])
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics
        
        Returns:
            Dictionary containing usage statistics
        """
        stats = {}
        
        for model, history in self.model_history.items():
            total_queries = len(history)
            successful_queries = sum(1 for entry in history if entry["success"])
            
            # Group by provider
            by_provider = {}
            for entry in history:
                provider = entry["provider"]
                if provider not in by_provider:
                    by_provider[provider] = {
                        "total": 0,
                        "successful": 0
                    }
                
                by_provider[provider]["total"] += 1
                if entry["success"]:
                    by_provider[provider]["successful"] += 1
            
            stats[model] = {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
                "by_provider": by_provider
            }
        
        return stats


class EnhancedLLMAgent:
    """
    Enhanced LLM agent with integrated LLMClient
    """
    def __init__(self, model_name: str = "gpt-4o", provider: str = "azure"):
        self.model_name = model_name
        self.provider = provider
        self.action_history = []
        self.state_memory = []
        self.llm_client = LLMClient()
        self.system_prompt = """
        You are an intelligent agent that understands and solves Sokoban puzzles.
        Your goal is to push all boxes to their target positions.
        You can move in these directions: up, down, left, right.
        You cannot move through walls or push multiple boxes at once.
        Analyze the current game state and choose the best movement direction.
        """
    
    def process_state(self, state_repr: str) -> dict:
        """Parse game state representation"""
        return {
            'raw_state': state_repr,
            'entities': self._extract_entities(state_repr),
            'metrics': self._calculate_metrics(state_repr)
        }
    
    def _extract_entities(self, state: str) -> List[Dict]:
        """Identify game objects from state string"""
        entities = []
        for y, line in enumerate(state.split('\n')):
            for x, char in enumerate(line):
                if char != ' ':
                    entities.append({
                        'type': self._get_entity_type(char),
                        'symbol': char,
                        'position': (x, y)
                    })
        return entities
    
    def _get_entity_type(self, symbol: str) -> str:
        """Determine entity type based on symbol"""
        entity_types = {
            '#': 'wall',
            '@': 'player',
            '$': 'box',
            '.': 'target',
            '*': 'box_on_target',
            '+': 'player_on_target',
            ' ': 'empty'
        }
        return entity_types.get(symbol, 'unknown')
    
    def _calculate_metrics(self, state: str) -> Dict:
        """Calculate basic state metrics"""
        entities = self._extract_entities(state)
        
        # Calculate number of boxes and targets
        boxes = sum(1 for e in entities if e['type'] in ['box', 'box_on_target'])
        targets = sum(1 for e in entities if e['type'] in ['target', 'box_on_target', 'player_on_target'])
        boxes_on_target = sum(1 for e in entities if e['type'] == 'box_on_target')
        
        return {
            'complexity': len(state.replace(' ', '')),
            'diversity': len(set(state)) - 1,  # Exclude space
            'boxes': boxes,
            'targets': targets,
            'boxes_on_target': boxes_on_target,
            'completion_percentage': (boxes_on_target / boxes) * 100 if boxes > 0 else 0
        }
    
    def choose_action(self, processed_state: dict, goal: str = "") -> str:
        """Core decision-making logic"""
        # Use LLM for decision making
        prompt = self._generate_decision_prompt(processed_state, goal)
        
        try:
            response = self.llm_client.query(
                system_prompt=self.system_prompt,
                prompt=prompt,
                model=self.model_name,
                provider=self.provider
            )
            
            # Parse response to extract action
            action = self._parse_action_from_response(response)
            return action
        except Exception as e:
            print(f"LLM decision failed: {e}")
            # Fallback to random action
            return np.random.choice(['up', 'down', 'left', 'right'])
    
    def _generate_decision_prompt(self, processed_state: dict, goal: str) -> str:
        """Generate decision prompt"""
        state_repr = processed_state['raw_state']
        metrics = processed_state['metrics']
        
        prompt = f"""
        Current game state:
        ```
        {state_repr}
        ```
        
        Legend:
        - # represents a wall
        - @ represents the player
        - $ represents a box
        - . represents a target position
        - * represents a box on a target
        - + represents the player on a target
        
        Game statistics:
        - Total boxes: {metrics.get('boxes', 'unknown')}
        - Total targets: {metrics.get('targets', 'unknown')}
        - Boxes on targets: {metrics.get('boxes_on_target', 'unknown')}
        - Completion percentage: {metrics.get('completion_percentage', 'unknown'):.1f}%
        
        Goal: {goal if goal else "Push all boxes to target positions"}
        
        Please analyze the current state and choose the best movement direction (up, down, left, right).
        Your answer should include:
        1. A brief analysis of the current state
        2. Your chosen action and reasoning
        3. On the last line, clearly state your action choice: ACTION: [up/down/left/right]
        """
        
        # Add action history as context
        if self.action_history:
            recent_actions = self.action_history[-5:]  # Last 5 actions
            action_history_text = "\n".join([
                f"- Action {i+1}: {action['action']} (result: {action['result']})"
                for i, action in enumerate(recent_actions)
            ])
            
            prompt += f"""
            
            Recent action history:
            {action_history_text}
            """
        
        return prompt
    
    def _parse_action_from_response(self, response: str) -> str:
        """Parse action from LLM response"""
        # Try to find explicit ACTION marker
        action_match = re.search(r'ACTION:\s*(up|down|left|right)', response, re.IGNORECASE)
        if action_match:
            return action_match.group(1).lower()
        
        # If no explicit marker, try to extract direction words from text
        direction_words = {
            'up': ['up', 'upward', 'upwards', 'north'],
            'down': ['down', 'downward', 'downwards', 'south'],
            'left': ['left', 'leftward', 'leftwards', 'west'],
            'right': ['right', 'rightward', 'rightwards', 'east']
        }
        
        # Count occurrences of each direction in the response
        direction_counts = {direction: 0 for direction in direction_words}
        
        for direction, words in direction_words.items():
            for word in words:
                direction_counts[direction] += len(re.findall(r'\b' + word + r'\b', response, re.IGNORECASE))
        
        # Choose the direction with the most occurrences
        if any(direction_counts.values()):
            return max(direction_counts.items(), key=lambda x: x[1])[0]
        
        # If unable to determine, return random direction
        return np.random.choice(['up', 'down', 'left', 'right'])
    
    def update_history(self, action: str, result: str):
        """Maintain action-result history"""
        self.action_history.append({
            'action': action,
            'result': result,
            'timestamp': len(self.action_history)
        })
    
    def save_agent_state(self, path: str):
        """Save agent state to disk"""
        with open(path, 'w') as f:
            json.dump({
                'history': self.action_history,
                'model': self.model_name,
                'provider': self.provider
            }, f)
    
    def load_agent_state(self, path: str):
        """Load agent state from disk"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.action_history = data.get('history', [])
                self.model_name = data.get('model', self.model_name)
                self.provider = data.get('provider', self.provider)
            return True
        except Exception as e:
            print(f"Failed to load agent state: {e}")
            return False


class ReinforcementWrapper:
    """
    Reinforcement learning wrapper
    """
    def __init__(self, base_agent: EnhancedLLMAgent):
        self.base_agent = base_agent
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.state_action_history = []
    
    def choose_action(self, state_hash: str, available_actions: List[str] = None) -> str:
        """
        Choose action using Q-learning
        """
        if available_actions is None:
            available_actions = ['up', 'down', 'left', 'right']
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Exploration: choose random action
            return random.choice(available_actions)
        else:
            # Exploitation: choose action with highest Q-value
            if state_hash not in self.q_table:
                self.q_table[state_hash] = {action: 0.0 for action in available_actions}
            
            # Find action with highest Q-value
            best_action = max(self.q_table[state_hash].items(), key=lambda x: x[1])[0]
            return best_action
    
    def update_q_value(self, state_hash: str, action: str, reward: float, next_state_hash: str):
        """
        Update Q-value
        """
        # Initialize state in Q-table if not present
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0}
        
        if next_state_hash not in self.q_table:
            self.q_table[next_state_hash] = {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0}
        
        # Calculate current Q-value
        current_q = self.q_table[state_hash][action]
        
        # Calculate maximum Q-value for next state
        max_next_q = max(self.q_table[next_state_hash].values())
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_hash][action] = new_q
    
    def reinforce(self, state_hash: str, action: str, reward: float, next_state_hash: str):
        """
        Update Q-value based on recent action and reward
        """
        # Record state-action pair
        self.state_action_history.append((state_hash, action, reward, next_state_hash))
        
        # Update Q-value
        self.update_q_value(state_hash, action, reward, next_state_hash)
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
    
    def save_q_table(self, path: str):
        """
        Save Q-table to file
        """
        with open(path, 'w') as f:
            json.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate,
                'state_action_history': self.state_action_history[-100:]  # Only save the most recent 100 records
            }, f)
    
    def load_q_table(self, path: str):
        """
        Load Q-table from file
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.q_table = data.get('q_table', {})
                self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
                self.state_action_history = data.get('state_action_history', [])
            return True
        except Exception as e:
            print(f"Failed to load Q-table: {e}")
            return False


class StateVisualizer:
    """
    State visualization tool
    """
    @staticmethod
    def render_ascii(state: dict) -> str:
        """Convert processed state back to ASCII representation"""
        if 'raw_state' in state:
            return state['raw_state']
        
        entities = state.get('entities', [])
        if not entities:
            return "Empty state"
        
        # Find maximum coordinates
        max_x = max(e['position'][0] for e in entities)
        max_y = max(e['position'][1] for e in entities)
        
        # Create empty grid
        grid = [[' ' for _ in range(max_x+1)] for _ in range(max_y+1)]
        
        # Fill in entities
        for e in entities:
            x, y = e['position']
            grid[y][x] = e['symbol']
        
        # Convert to string
        return '\n'.join(''.join(row) for row in grid)
    
    @staticmethod
    def render_html(state: dict) -> str:
        """Render state as HTML"""
        if 'raw_state' not in state:
            state['raw_state'] = StateVisualizer.render_ascii(state)
        
        raw_state = state['raw_state']
        
        # Create HTML table
        html = "<table style='border-collapse: collapse; font-family: monospace;'>"
        
        for y, line in enumerate(raw_state.split('\n')):
            html += "<tr>"
            for x, char in enumerate(line):
                color = StateVisualizer._get_color_for_symbol(char)
                html += f"<td style='width: 20px; height: 20px; text-align: center; background-color: {color};'>{char}</td>"
            html += "</tr>"
        
        html += "</table>"
        return html
    
    @staticmethod
    def _get_color_for_symbol(symbol: str) -> str:
        """Return color corresponding to symbol"""
        color_map = {
            '#': '#333333',  # Wall - dark gray
            '@': '#3498db',  # Player - blue
            '$': '#e74c3c',  # Box - red
            '.': '#2ecc71',  # Target - green
            '*': '#f39c12',  # Box on target - orange
            '+': '#9b59b6',  # Player on target - purple
            ' ': '#ffffff',  # Empty space - white
        }
        return color_map.get(symbol, '#ffffff')
