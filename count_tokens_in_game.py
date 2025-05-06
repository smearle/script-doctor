from utils import num_tokens_from_string

game_path = 'data/scraped_games/atlas shrank.txt'

with open(game_path, 'r', encoding='utf-8') as f:
    game = f.read()
    
n_tokens = num_tokens_from_string(game, 'gpt-4o')

print(f"The game at {game_path} has {n_tokens} tokens.")