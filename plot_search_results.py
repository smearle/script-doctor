import glob
import json
import os

import hydra
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from conf.config import PlotSearch
from globals import PLOTS_DIR, STANDALONE_NODEJS_RESULTS_PATH, JS_SOLS_DIR
from profile_nodejs import get_algo_name, get_standalone_run_params_from_name
from puzzlejax.utils import get_list_of_games_for_testing, game_names_remap


BFS_RESULTS_PATH = os.path.join('data', 'bfs_results.json')


@hydra.main(version_base="1.3", config_path="conf", config_name="plot_standalone_bfs_config")
def main(cfg: PlotSearch):
    if cfg.aggregate:
        aggregate_results(cfg)
    else:
        plot(cfg)
    

def aggregate_results(cfg: PlotSearch):
    # games = os.listdir(JS_SOLS_DIR)
    games = get_list_of_games_for_testing(cfg.all_games)
    results = {}
    max_iters = cfg.n_steps
    if cfg.algo == 'bfs':
        algo_name = 'solveBFS'
    elif cfg.algo == 'astar':
        algo_name = 'solveAStar'
    elif cfg.algo == 'mcts':
        algo_name = 'solveMCTS'
    else:
        raise ValueError(f'Unknown algo: {cfg.algo}')
    for game in games:
        if game.startswith('test_'):
            continue
        game_dir = os.path.join(JS_SOLS_DIR, game)
        sol_jsons = glob.glob(f"{game_dir}/*.json")
        n_levels = 0
        n_solved = 0
        n_stepss = []
        solution_lengths = []
        for sol_json in sol_jsons:
            run_name, level_i = os.path.basename(sol_json).rsplit('_level-', 1)
            sol_algo_name = run_name.split('_')[0]
            n_steps = run_name.split('_')[1].split('-steps')[0]
            level_i = level_i.split('.json')[0]
            if sol_algo_name != algo_name or int(n_steps) != cfg.n_steps:
                continue
            with open(sol_json, 'r') as f:
                sol_dict = json.load(f)
            if 'iterations' not in sol_dict or 'won' not in sol_dict:
                print(f"Skipping {sol_json} because it doesn't have 'iterations' or 'won'")
                continue
            solved = sol_dict['won']
            actions = sol_dict.get('actions')
            if solved:
                n_solved += 1
                solution_lengths.append(len(actions))
            n_levels += 1
            n_steps = min(cfg.n_steps, sol_dict['iterations'])
            n_stepss.append(n_steps)
        if n_levels > 0:
            n_steps_mean = np.mean(n_stepss)
            pct_solved = n_solved / n_levels
            mean_solution_length = float(np.mean(solution_lengths))
            results[game] = {
                'pct_solved': pct_solved,
                'n_levels': n_levels,
                'n_iters': n_steps_mean,
                'mean_sol_len': mean_solution_length
            }
    print(results)
    with open(BFS_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)

    plot(cfg, results)


def plot(cfg: PlotSearch, results=None):
    M = 40  # max number of games per table

    if results is None:
        with open(BFS_RESULTS_PATH, 'r') as f:
            results = json.load(f)
    
    df = pd.DataFrame.from_dict(results, orient='index')
    if 'sol_len' in df.columns and 'mean_sol_len' not in df.columns:
        df.rename(columns={'sol_len': 'mean_sol_len'}, inplace=True)
    # df = df.sort_values(by=['pct_solved', 'n_iters'], ascending=[False, True])

    os.makedirs(PLOTS_DIR, exist_ok=True)

    csv_file_path = os.path.join(PLOTS_DIR, 'standalone_bfs_results.csv')
    df.to_csv(csv_file_path, index=True, float_format="%.2f")
    print(f'Saved results to {csv_file_path}')

    # Remap game names
    df.index = df.index.to_series().replace(game_names_remap)
    # Replace underscores and capitalize game names
    df.index = df.index.str.replace('_', ' ')
    df.index = df.index.str.title()

    heatmap_source_df = df.copy()

    generate_heatmaps(heatmap_source_df)

    latex_df = df.copy()

    col_renames = {
        'pct_solved': 'Solved Levels \\%',
        'n_levels': '\\# Total Levels',
        'n_iters': 'Mean Search Iterations',
        'mean_sol_len': 'Mean Solution Length',
    }
    latex_df.rename(columns=col_renames, inplace=True)

    # Clean and prepare index for LaTeX
    latex_df.index = latex_df.index.str.replace('_', ' ')
    latex_df.index = latex_df.index.str.replace('&', r'\&', regex=False)
    latex_df.index = latex_df.index.str.replace('^', r'\^', regex=False)
    latex_df.index = latex_df.index.to_series().apply(
        lambda name: f"\\parbox{{3.5cm}}{{\\strut {name[:50]}{'...' if len(name) > 50 else ''}}}"
    )
    latex_df.index.name = 'Game'

    # Modify % column to show, e.g. "0.75" as "75\%"
    latex_df['Solved Levels \\%'] = latex_df['Solved Levels \\%'].apply(lambda x: f"{x * 100:.0f}\\%")

    latex_file_path = os.path.join(PLOTS_DIR, 'bfs_results.tex')
    with open(latex_file_path, 'w') as f:
        f.write(latex_df.to_latex(index=True, float_format="%.2f", escape=False, caption="Results of BFS on full dataset of games, with max $100,000$ max search iterations and a timeout of 1 minute.",
                            longtable=True, label="tab:bfs_results"))
    print(f'Saved latex table to {latex_file_path}')

def generate_heatmaps(df: pd.DataFrame) -> None:
    if df.empty:
        print('No data available for heatmap generation.')
        return

    heatmap_configs = [
        {
            'column': 'pct_solved',
            'title': 'BFS Percent of Levels Solved per Game',
            'cmap': 'RdYlGn',
            'vmin': 0.0,
            'vmax': 1.0,
            'colorbar_label': 'Average Win Rate',
            'formatter': lambda v: f"{v:.0%}",
            'output': 'bfs_pct_solved_heatmap.png',
        },
        {
            'column': 'n_iters',
            'title': 'BFS Mean Search Iterations per Game',
            'cmap': 'Blues',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Search Iterations',
            'formatter': lambda v: f"{v:.0f}",
            'output': 'bfs_mean_iterations_heatmap.png',
        },
        {
            'column': 'mean_sol_len',
            'title': 'BFS Mean Solution Length per Game',
            'cmap': 'Purples',
            'vmin': 0.0,
            'vmax': None,
            'colorbar_label': 'Mean Solution Length',
            'formatter': lambda v: f"{v:.0f}",
            'output': 'bfs_mean_solution_length_heatmap.png',
        },
    ]

    target_cell_height = 1.0
    target_cell_width = 1.0
    min_total_figure_width = 8.0
    min_total_figure_height = 3.0
    h_padding = 3.0
    v_padding = 1.5

    for config in heatmap_configs:
        column = config['column']
        if column not in df.columns:
            continue

        series = df[column]
        if series.dropna().empty:
            continue

        valid_series = series.dropna()

        vmin = config.get('vmin')
        vmax = config.get('vmax')
        if vmin is None:
            vmin = float(valid_series.min())
        if vmax is None:
            vmax = float(valid_series.max())
        if np.isfinite(vmin) and np.isfinite(vmax) and np.isclose(vmin, vmax):
            vmax = vmin + (abs(vmin) * 0.05 + 1)

        display_names = [name.replace('_', ' ') for name in series.index]
        data = pd.DataFrame([series.values], columns=display_names, index=[''])

        annot_row = [
            config['formatter'](val) if pd.notnull(val) else ''
            for val in series.values
        ]
        annot_data = [annot_row]

        num_cols = len(series)
        fig_h = max(target_cell_height + v_padding, min_total_figure_height)
        fig_w = max(num_cols * target_cell_width + h_padding, min_total_figure_width)

        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            data,
            annot=annot_data,
            fmt="",
            cmap=config['cmap'],
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': config['colorbar_label'], 'shrink': 0.8, 'pad': 0.01},
            annot_kws={"size": 9},
            linewidths=0.5,
            linecolor='white'
        )
        plt.title(config['title'])
        plt.xlabel('Game', labelpad=10)
        plt.ylabel('')
        plt.yticks([])
        plt.xticks(rotation=45, ha='right')
        # plt.tight_layout(pad=1.2, rect=[0, 0, 0.92, 1])
        plt.tight_layout()

        output_path = os.path.join(PLOTS_DIR, config['output'])
        try:
            plt.savefig(output_path, dpi=300)
            print(f"Saved heatmap to {output_path}")
        except Exception as e:
            print(f"Error saving heatmap {config['output']}: {e}")
        finally:
            plt.close()


@hydra.main(version_base="1.3", config_path="conf", config_name="plot_standalone_bfs_config")
def old_plot(cfg: PlotSearch):
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
