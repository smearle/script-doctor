import glob
import json
import os

import hydra
from matplotlib import pyplot as plt
import pandas as pd

from conf.config import PlotStandaloneBFS
from globals import GAMES_N_RULES_SORTED_PATH, PLOTS_DIR, STANDALONE_NODEJS_RESULTS_PATH
from profile_nodejs import get_standalone_run_params_from_name
from utils import get_list_of_games_for_testing
from validate_sols import JS_SOLS_DIR


BFS_RESULTS_PATH = os.path.join('data', 'bfs_results.json')


@hydra.main(version_base="1.3", config_path="conf", config_name="plot_standalone_bfs_config")
def main(cfg: PlotStandaloneBFS):
    if cfg.aggregate:
        aggregate_results(cfg)
    else:
        plot(cfg)
    

def aggregate_results(cfg: PlotStandaloneBFS):
    games = os.listdir(JS_SOLS_DIR)
    results = {}
    for game in games:
        if game.startswith('test_'):
            continue
        game_dir = os.path.join(JS_SOLS_DIR, game)
        sol_jsons = glob.glob(f"{game_dir}/*.json")
        n_levels = 0
        n_solved = 0
        max_iters = 0
        for sol_json in sol_jsons:
            level_i = sol_json.split("/")[-1].split(".")[0].split('-')[1]
            with open(sol_json, 'r') as f:
                sol_dict = json.load(f)
            if 'iterations' not in sol_dict or 'won' not in sol_dict:
                print(f"Skipping {sol_json} because it doesn't have 'iterations' or 'won'")
                continue
            solved = sol_dict['won']
            if solved:
                n_solved += 1
            n_levels += 1
            max_iters = max(max_iters, sol_dict['iterations'])
        if n_levels > 0:
            pct_solved = n_solved / n_levels
            results[game] = {
                'pct_solved': pct_solved,
                'n_levels': n_levels,
                'max_iters': max_iters,
            }
    with open(BFS_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)

    plot(cfg, results)


def plot(cfg: PlotStandaloneBFS, results=None):
    M = 40  # max number of games per table

    if results is None:
        with open(BFS_RESULTS_PATH, 'r') as f:
            results = json.load(f)
    
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.sort_values(by=['pct_solved', 'max_iters'], ascending=[False, True])

    csv_file_path = os.path.join(PLOTS_DIR, 'standalone_bfs_results.csv')
    df.to_csv(csv_file_path, index=True, float_format="%.2f")
    print(f'Saved results to {csv_file_path}')

    col_renames = {
        'pct_solved': 'Solved Levels \\%',
        'n_levels': '\\# Total Levels',
        'max_iters': 'Max Search Iterations',
    }
    df.rename(columns=col_renames, inplace=True)

    # Clean and prepare index for LaTeX
    df.index = df.index.str.replace('_', ' ')
    df.index = df.index.str.replace('&', r'\&', regex=False)
    df.index = df.index.str.replace('^', r'\^', regex=False)
    df.index = df.index.to_series().apply(
        lambda name: f"\\parbox{{3.5cm}}{{\\strut {name[:50]}{'...' if len(name) > 50 else ''}}}"
    )
    df.index.name = 'Game'

    # Modify % column to show, e.g. "0.75" as "75\%"
    df['Solved Levels \\%'] = df['Solved Levels \\%'].apply(lambda x: f"{x * 100:.0f}\\%")

    latex_file_path = os.path.join(PLOTS_DIR, 'bfs_results.tex')
    with open(latex_file_path, 'w') as f:
        f.write(df.to_latex(index=True, float_format="%.2f", escape=False, caption="Results of BFS on full dataset of games, with max $100,000$ max search iterations and a timeout of 1 minute.",
                            longtable=True, label="tab:bfs_results"))
    print(f'Saved latex table to {latex_file_path}')


@hydra.main(version_base="1.3", config_path="conf", config_name="plot_standalone_bfs_config")
def old_plot(cfg: PlotStandaloneBFS):
    with open(STANDALONE_NODEJS_RESULTS_PATH, 'r') as f:
        results = json.load(f)

    # Create a directory for the plots if necessary
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sorted_games = get_list_of_games_for_testing(cfg.all_games) 
    # Build a sorting key for each index based on the sorted_games list
    game_order = {game: i for i, game in enumerate(sorted_games)}


    for run_name in results.keys():
        run_results = results[run_name]
        if len(run_results) == 0:
            print(f'No results for {run_name}. Skipping.')
            continue

        print(f'Plotting results for {run_name}.')

        df_col_keys = [
            'solved',
            # 'FPS',
            'iterations',
            # 'score'
        ]
        df_row_headers = ['game', 'level']
        df_row_indices = []
        df_rows = []

        for game in run_results.keys():
            game_results = run_results[game]
            level_keys = list(game_results.keys())

            if len(game_results) == 0:
                print(f'No results for {game}. Skipping.')
                continue
                
            if game not in sorted_games:
                print(f'Game {game} not in sorted games. Skipping.')
                continue

            for level in level_keys:
                level_results = game_results[level]

                if len(level_results) == 0:
                    print(f'No results for {level}. Skipping.')
                    continue

                # Add this stuff to the dataframe
                df_row_indices.append((game, level))
                # df_rows.append([level_results[key] for key in df_col_keys])
                row_data = []
                for key in df_col_keys:
                    val = level_results[key]
                    if key == "solved":
                        val = r"\colorbox{green}{~}" if val else r"\colorbox{red}{~}"
                    row_data.append(val)
                df_rows.append(row_data)

        # Create a dataframe from the results
        df = pd.DataFrame(df_rows, columns=df_col_keys, index=pd.MultiIndex.from_tuples(df_row_indices, names=df_row_headers))

        # Sort games according to their ordering in `sorted_games`
        # Sort levels according to their integer value. Convert them to integers here!
        df = df.sort_values(
            by=['game', 'level'],
            key=lambda col: (
                col.map(game_order) if col.name == 'game'
                else col.astype(int)  # assumes levels like '0', '1', ...
            )
        )

        algo_name, n_steps, device_name = get_standalone_run_params_from_name(run_name)
        concise_run_name = f'{algo_name}_{n_steps}-steps'

        # Save the dataframe to a CSV file
        csv_file_name = f'{run_name}.csv'
        csv_file_path = os.path.join(PLOTS_DIR, csv_file_name)

        # Remove underscores from game names
        df.index = df.index.set_levels([df.index.levels[0].str.replace('_', ' '), df.index.levels[1]])
        
        # Save to a latex table
        latex_file_name = f'{concise_run_name}.tex'
        latex_file_path = os.path.join(PLOTS_DIR, latex_file_name)
        # df.to_latex(latex_file_path, index=True, float_format="%.2f", escape=False)
        # print(f'Saved latex table to {latex_file_path}')

        # Split the dataframe roughly in half
        split_index = len(df) // 2
        df1 = df.iloc[:split_index]
        df2 = df.iloc[split_index:]

        # Build LaTeX minipage output
        latex_output = r"""
\centering
\begin{subtable}[t]{0.48\linewidth}
\centering
""" + df1.to_latex(index=True, float_format="%.2f", escape=False, caption=False) + r"""
\end{subtable}%
\hfill
\begin{subtable}[t]{0.48\linewidth}
\centering
""" + df2.to_latex(index=True, float_format="%.2f", escape=False, caption=False) + r"""
\end{subtable}
        """

        latex_file_name = f'{concise_run_name}_split.tex'
        latex_file_path = os.path.join(PLOTS_DIR, latex_file_name)
        with open(latex_file_path, 'w') as f:
            f.write(latex_output)

        print(f'Saved split latex table to {latex_file_path}')

    
if __name__ == "__main__":
    main()