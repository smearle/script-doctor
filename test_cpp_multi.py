"""Test C++ engine against JS on a sample of games with solutions.
Uses subprocess to isolate each game test (prevents JS bridge hangs).
"""
import json
import os
import sys
import glob
import random
import subprocess

JS_SOLS_DIR = 'data/js_sols'
SIMPLIFIED_GAMES_DIR = 'data/simplified_games'
TIMEOUT = 30  # seconds per game

def find_simplified_game(game_name):
    path = os.path.join(SIMPLIFIED_GAMES_DIR, f'{game_name}_simplified.txt')
    return path if os.path.isfile(path) else None

def main():
    all_games = []
    for game_dir in os.listdir(JS_SOLS_DIR):
        sol_dir = os.path.join(JS_SOLS_DIR, game_dir)
        if not os.path.isdir(sol_dir):
            continue
        sols = glob.glob(os.path.join(sol_dir, 'bfs_*level-*.json'))
        if sols and find_simplified_game(game_dir):
            all_games.append(game_dir)
    
    print(f"Found {len(all_games)} games with both simplified files and solutions")
    
    sample_size = min(50, len(all_games))
    random.seed(42)
    sample = random.sample(all_games, sample_size)
    sample.sort()
    
    total_success = 0
    total_fail = 0
    total_error = 0
    total_timeout = 0
    total_levels_ok = 0
    total_levels_fail = 0
    failed_games = []
    error_games = []
    timeout_games = []
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    for i, game in enumerate(sample):
        sys.stdout.write(f"[{i+1}/{sample_size}] {game[:55]:55s} ")
        sys.stdout.flush()
        
        try:
            result = subprocess.run(
                [sys.executable, 'test_cpp_single.py', game],
                capture_output=True, text=True, timeout=TIMEOUT, cwd=cwd
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                total_success += 1
                total_levels_ok += data.get("successes", 0)
                print(f"OK ({data.get('successes', 0)} levels)")
            elif result.returncode == 1:
                data = json.loads(result.stdout.strip())
                total_fail += 1
                total_levels_ok += data.get("successes", 0)
                total_levels_fail += data.get("failures", 0)
                failed_games.append((game, data))
                print(f"FAIL ({data.get('successes', 0)} ok, {data.get('failures', 0)} fail)")
            else:
                total_error += 1
                try:
                    data = json.loads(result.stdout.strip())
                    error_games.append((game, data.get("error", "unknown")))
                    print(f"ERROR: {data.get('error', 'unknown')[:50]}")
                except Exception:
                    msg = result.stderr[:100] if result.stderr else "unknown"
                    error_games.append((game, msg))
                    print(f"ERROR: {msg[:50]}")
        except subprocess.TimeoutExpired:
            total_timeout += 1
            timeout_games.append(game)
            print(f"TIMEOUT ({TIMEOUT}s)")
    
    print(f"\n{'='*70}")
    print(f"Results: {total_success} OK / {total_fail} FAIL / {total_error} ERROR / {total_timeout} TIMEOUT")
    print(f"Levels:  {total_levels_ok} OK / {total_levels_fail} FAIL")
    
    if failed_games:
        print(f"\nFailed games ({len(failed_games)}):")
        for game, data in failed_games:
            print(f"  {game}")
            for d in data.get("details", [])[:3]:
                print(f"    {d}")
    
    if timeout_games:
        print(f"\nTimeout games ({len(timeout_games)}):")
        for g in timeout_games:
            print(f"  {g}")
    
    if error_games:
        print(f"\nError games ({len(error_games)}):")
        for g, e in error_games[:10]:
            print(f"  {g}: {str(e)[:80]}")

if __name__ == '__main__':
    main()
