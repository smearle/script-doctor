
import os
import dotenv
from puzzlejax.utils import llm_text_query

dotenv.load_dotenv()

# Test using Portkey to call LLM to generate PuzzleScript games
system_prompt = "You are a creative and resourceful indie puzzle game designer, familiar with the PuzzleScript game description language."
user_prompt = "Create a simple Sokoban-like puzzle game with a player, walls, boxes, and targets. The player should push boxes onto targets to win. Include at least one simple level."
seed = 42
model = "gpt-4o-mini"  # Can be "o3-mini", "gpt-4o-mini", "vertex-ai"

print(f"Testing LLM query with model: {model}")
response = llm_text_query(system_prompt, user_prompt, seed, model)

print("\n--- Response from LLM ---")
print(response)
print("\n--- End of response ---")
print("\nTest completed successfully!")
