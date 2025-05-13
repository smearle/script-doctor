with open('data/scraped_games/Depth-First_Maze.txt', 'r', encoding='utf-8') as f:
    s = f.read()

for i, c in enumerate(s):
    print(f'{i}: {repr(c)} {ord(c)}')
