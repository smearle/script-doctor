from typing import List, Dict, Optional, Tuple
import re
from ascii_prompting import build_game_action_prompt, build_human_like_prompt
from puzzlescript_jax.utils import llm_text_query

class LLMGameAgent:
    """
    LLM agent for games: given ascii_map, mapping, rules, and action_space, returns an action id (1-5).
    """

    def __init__(self, model_name: str = "gpt-4o", enable_thinking: Optional[bool] = None):
        self.model_name = model_name
        # For vLLM Qwen3 models: True enables thinking, False disables, None lets server decide.
        self.enable_thinking = enable_thinking

    def choose_action(self, ascii_map: str, rules: str, action_space: List[int], action_meanings: Dict[int, str],
                      think_aloud: bool, memory: int, state_history: List, log_file: Optional[str] = None) -> Tuple[int, Optional[str]]:
        """
        Query the LLM to select an action id from action_space, given ascii_map, mapping, rules, and action_meanings.
        Returns an integer action id.
        """
        system_prompt = (
            "You are an expert game-playing agent. "
            "Given the current game state, the ASCII map, the legend mapping, the game rules, and the action meanings, "
            "your task is to select the best action. "
        )
        if not think_aloud:
            system_prompt += "Only respond with the action id (an integer from the provided action_space)."
        else:
            system_prompt += "You may first reason about the problem, then select an action by outputting `ACTION: ID`, where `ID` is an integer from the provided action_space."

        prompt = build_game_action_prompt(
            ascii_map=ascii_map,
            rules=rules,
            action_space=action_space,
            action_meanings=action_meanings,
            think_aloud=think_aloud,
            memory=memory,
            state_history=state_history,
        )
        response = llm_text_query(
            system_prompt,
            prompt,
            model=self.model_name,
            enable_thinking=self.enable_thinking,
        )

        if log_file is not None:
            with open(log_file, "w") as f:
                f.write("System Prompt:\n")
                f.write(system_prompt + "\n\n")
                f.write("User Prompt:\n")
                f.write(prompt + "\n\n")
                f.write("LLM Response:\n")
                f.write(response if response is not None else "No response (LLM query failed)\n")

        if response is None:
            print("LLM query failed after multiple retries. Falling back to the first action.")
            return action_space[0]

        action_pattern = r"\b(" + "|".join(str(a) for a in action_space) + r")\b"
        if not think_aloud:
            match = re.search(action_pattern, response)
        else:
            match = re.search(r"ACTION:\s*(" + "|".join(str(a) for a in action_space) + r")\b", response, re.IGNORECASE)
        if match:
            return int(match.group(1))

        print(f"LLM response did not contain a valid action. Response: '{response}'. Falling back to the first action.")
        return action_space[0]

    def choose_action_human(
        self,
        *,
        title: str,
        author: str,
        legend_text: str,
        ascii_map: str,
        action_space: List[int],
        action_meanings: Dict[int, str],
        state_history: List,
        history_limit: int,
        scratchpad: str,
        action_only: bool = False,
        messages: Optional[List[str]] = None,
        level_number: Optional[int] = None,
        log_file: Optional[str] = None,
    ) -> Tuple[int, str]:
        """Human-like agent: no rules shown, uses scratchpad for reasoning.

        Returns (action_id, updated_scratchpad).
        """
        if action_only:
            system_prompt = (
                "You are playing a puzzle game. "
                "Respond only with `ACTION: <id>` where <id> is one of the available action ids."
            )
        else:
            system_prompt = (
                "You are playing a puzzle game. "
                "Use your scratchpad to keep notes about what you've learned.\n"
                "You must end your response with `ACTION: <id>` where <id> is one of "
                "the available action ids."
            )

        prompt = build_human_like_prompt(
            title=title,
            author=author,
            legend_text=legend_text,
            ascii_map=ascii_map,
            action_space=action_space,
            action_meanings=action_meanings,
            state_history=state_history,
            history_limit=history_limit,
            scratchpad=scratchpad,
            include_scratchpad=not action_only,
            messages=messages,
            level_number=level_number,
        )

        response = llm_text_query(
            system_prompt,
            prompt,
            model=self.model_name,
            enable_thinking=self.enable_thinking,
        )

        if log_file is not None:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("System Prompt:\n")
                f.write(system_prompt + "\n\n")
                f.write("User Prompt:\n")
                f.write(prompt + "\n\n")
                f.write("LLM Response:\n")
                f.write(response if response is not None else "No response (LLM query failed)\n")

        if response is None:
            print("LLM query failed after multiple retries. Falling back to the first action.")
            return action_space[0], scratchpad

        new_scratchpad = scratchpad
        if not action_only:
            sp_match = re.search(r"SCRATCHPAD:\s*(.*?)(?=\nACTION:|\Z)", response, re.DOTALL | re.IGNORECASE)
            if sp_match:
                new_scratchpad = sp_match.group(1).strip()

        action_pattern = "|".join(str(a) for a in action_space)
        match = re.search(r"ACTION:\s*(" + action_pattern + r")\b", response, re.IGNORECASE)
        if match:
            return int(match.group(1)), new_scratchpad

        print(f"LLM response did not contain a valid action. Falling back to first action.")
        return action_space[0], new_scratchpad
