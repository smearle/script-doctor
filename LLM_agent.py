from typing import List, Dict, Optional
from utils import llm_text_query

class LLMGameAgent:
    """
    LLM agent for games: given ascii_map, mapping, rules, and action_space, returns an action id (1-5).
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    def choose_action(self, ascii_map: str, rules: str, action_space: List[int], action_meanings: Dict[int, str]) -> int:
        """
        Query the LLM to select an action id from action_space, given ascii_map, mapping, rules, and action_meanings.
        Returns an integer action id.
        """
        system_prompt = (
            "You are an expert game-playing agent. "
            "Given the current game state, the ASCII map, the legend mapping, the game rules, and the action meanings, "
            "your task is to select the best action. "
            "Only respond with the action id (an integer from the provided action_space)."
        )

        action_space_str = ", ".join(str(a) for a in action_space)
        # Provide action mapping (number to meaning) dynamically
        action_map_str = ", ".join([f"{k}={v}" for k, v in action_meanings.items()])
        prompt = (
            f"Game state (ASCII map) and Legend:\n{ascii_map}\n\n"
  
            f"Game rules:\n{rules}\n\n"
            f"Available actions (action_space): {action_space_str}\n"
            f"Action mapping: {action_map_str}\n"
            f"Please select the best action and ONLY return the action id (an integer from action_space)."
        )
        # Use a fixed seed for reproducibility if needed
        response = llm_text_query(
            system_prompt,
            prompt,
            model=self.model_name,
        )

        # Handle cases where llm_text_query returns None (e.g., after max retries)
        if response is None:
            print("LLM query failed after multiple retries. Falling back to the first action.")
            return action_space[0]

        # Extract the first integer in the response as the action id
        import re
        # Accept any valid action id from action_space
        action_pattern = r"\b(" + "|".join(str(a) for a in action_space) + r")\b"
        match = re.search(action_pattern, response)
        if match:
            return int(match.group(1))
        
        # Fallback: pick the first action if LLM output is not as expected
        print(f"LLM response did not contain a valid action. Response: '{response}'. Falling back to the first action.")
        return action_space[0]
