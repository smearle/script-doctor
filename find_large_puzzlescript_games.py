import os
import glob
from lark import Lark
from puzzlejax.preprocess_games import preprocess_ps, StripPuzzleScript
from globals import GAMES_DIR, LARK_SYNTAX_PATH

def main():
    print(f"Loading grammar from {LARK_SYNTAX_PATH}...")
    with open(LARK_SYNTAX_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    
    print("Initializing parser...")
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    transformer = StripPuzzleScript()
    
    game_files = glob.glob(os.path.join(GAMES_DIR, "*.txt"))
    game_files.sort()
    
    large_games = []
    
    print(f"Found {len(game_files)} games in {GAMES_DIR}. Scanning...")

    for i, game_path in enumerate(game_files):
        game_name = os.path.basename(game_path)
        # Optional: print progress every 100 games
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(game_files)} games...")

        try:
            with open(game_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Preprocess
            try:
                content = preprocess_ps(content)
            except Exception as e:
                # print(f"Skipping {game_name}: Preprocessing failed: {e}")
                continue

            # Parse
            try:
                tree = parser.parse(content)
            except Exception as e:
                # print(f"Skipping {game_name}: Parsing failed")
                continue

            # Transform
            try:
                stripped_tree = transformer.transform(tree)
            except Exception as e:
                # print(f"Skipping {game_name}: Transform failed: {e}")
                continue
            
            # Find objects_section
            objects_section = None
            if hasattr(stripped_tree, 'children'):
                for child in stripped_tree.children:
                    if hasattr(child, 'data') and child.data == 'objects_section':
                        objects_section = child
                        break
            
            if objects_section:
                num_objects = len(objects_section.children)
                if num_objects > 32:
                    print(f"FOUND: {game_name} has {num_objects} objects")
                    large_games.append((game_name, num_objects))
            
        except Exception as e:
            print(f"Error checking {game_name}: {e}")

    print("-" * 40)
    print(f"Total games with > 32 objects: {len(large_games)}")
    print("-" * 40)
    for name, count in large_games:
        print(f"{name}: {count}")

if __name__ == "__main__":
    main()
