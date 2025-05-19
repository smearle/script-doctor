import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_filename(filename):
    """
    Parses the filename to extract LLM, game, run, and level.
    Example: 4o-mini_limerick_run_1_level_1.json
    Example: 4o-mini_atlas shrank_run_1.json (level is 0 by default)
    """
    match = re.match(r"^(.*?)_(.*?)_run_(\d+)(?:_level_(\d+))?\.json$", filename)
    if match:
        llm_model, game_name, run_number, level_number = match.groups()
        if level_number is None: # Handles cases like '4o-mini_atlas shrank_run_1.json'
            # Check if game_name itself contains 'level_X'
            game_match = re.match(r"(.*?)_level_(\d+)$", game_name)
            if game_match:
                game_name = game_match.group(1)
                level_number = game_match.group(2)
            else: # Default level to 0 if not specified and not in game_name
                 # Further split game_name if it's a multi-part name before '_run_'
                if ' shrank' in llm_model: # Special case for "atlas shrank"
                    parts = llm_model.split(' shrank')
                    llm_model = parts[0]
                    game_name = f"atlas shrank_{game_name}" # Reconstruct game name
                level_number = '0'

        # Normalize game name here - convert "atlas shrank" to "atlas_shrank"
        if game_name == "atlas shrank" or game_name == "atlas_shrank":
            game_name = "atlas_shrank"

        return llm_model, game_name, int(run_number), int(level_number)

    # Fallback for filenames that might not perfectly match the primary pattern
    print(f"Warning: Could not parse filename: {filename}")
    return None, None, None, None


def collect_results(results_dir):
    """
    Collects results from all .json files in the specified results_dir **and all its subdirectories**.
    """
    data = []

    if not os.path.isdir(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return pd.DataFrame()

    # 遍历所有子目录和文件
    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                llm, game, run, level = parse_filename(filename)

                if not llm or not game:
                    print(f"Skipping unparsable file: {filename}")
                    continue
                    
                # Skip Gemini results
                # if "gemini" in llm.lower():
                #     print(f"Skipping Gemini result file: {filename}")
                #     continue

                # Normalize game name for atlas shrank cases
                if "atlas shrank" in game or "atlas_shrank" in game:
                    game = "atlas_shrank"

                try:
                    with open(filepath, 'r') as f:
                        content = json.load(f)
                    
                    win_status = content.get("win", content.get("state_data", {}).get("win", False))
                    steps = content.get("state_data", {}).get("step", float('nan')) # Get steps, use NaN if not found

                    data.append({
                        "llm": llm,
                        "game": game,
                        "run": run,
                        "level": level,
                        "win": win_status,
                        "steps": steps,
                        "filename": filename
                    })
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filepath}")
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
    
    return pd.DataFrame(data)


def main():
    # The script is in llm_agent_results/analysis/, so results_dir is ../
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_script_path, "..") # Points to llm_agent_results/

    df = collect_results(results_dir)

    if df.empty:
        print("No data collected. Exiting.")
        return
        
    # Standardize game names - ensure all atlas shrank variations are normalized
    df['game'] = df['game'].apply(lambda x: "atlas_shrank" if "atlas" in x and "shrank" in x else x)

    # Map LLM names (remove Gemini from mapping)
    llm_name_mapping = {
        "4o-mini": "ChatGPT 4o-mini",
        "deepseek": "Deepseek-chat",
        "qwen": "Qwen-plus" 
    }
    df['llm'] = df['llm'].replace(llm_name_mapping)

    # Calculate average win rate and average steps per LLM
    llm_agg_data = df.groupby("llm").agg(
        average_win_rate=('win', 'mean'),
        average_steps=('steps', 'mean')
    ).reset_index()

    print("\nAverage Win Rates and Steps per LLM:")
    print(llm_agg_data)

    if llm_agg_data.empty:
        print("No win rates to plot. Exiting.")
        return

    target_cell_height_global = 0.7
    target_cell_width_global = 1.0
    min_total_figure_width_global = 8.0

    # Calculate steps min and max for consistent coloring
    valid_steps = df['steps'].dropna()
    min_steps = valid_steps.min() if not valid_steps.empty else 0
    max_steps = valid_steps.max() if not valid_steps.empty else 10
    
    # --- Heatmap 1: LLM Average Win Rates ---
    heatmap_plot_data = llm_agg_data.set_index('llm')[['average_win_rate']]
    
    h_padding1 = 3.0
    v_padding1 = 2.0
    num_rows1 = len(heatmap_plot_data.index)
    num_cols1 = len(heatmap_plot_data.columns)

    fig_h1 = (num_rows1 * target_cell_height_global) + v_padding1
    ideal_data_w1 = num_cols1 * target_cell_width_global
    ideal_fig_w1 = ideal_data_w1 + h_padding1
    final_fig_w1 = max(ideal_fig_w1, min_total_figure_width_global)
    
    plt.figure(figsize=(final_fig_w1, fig_h1))
    
    # Create annotation strings (win rates as percentages)
    annot_data_llm = llm_agg_data.set_index('llm').apply(
        lambda x: f"{x['average_win_rate']:.0%}" if pd.notnull(x['average_win_rate']) else "N/A",
        axis=1
    ).values.reshape(heatmap_plot_data.shape)

    # Customize the heatmap with tighter color bar
    sns.heatmap(heatmap_plot_data, annot=annot_data_llm, fmt="", cmap="RdYlGn", vmin=0, vmax=1, 
                cbar_kws={'label': 'Average Win Rate', 'shrink': 0.8, 'pad': 0.01}, 
                annot_kws={"size": 10})
    plt.title("Average Win Rate per LLM")
    plt.ylabel("LLM Model")
    plt.xticks([])
    # 调整布局，使颜色条更靠近图表
    plt.tight_layout(pad=1.2, rect=[0, 0, 0.9, 1])  # 右侧留出更少空间让颜色条贴近
    
    output_path = os.path.join(current_script_path, "llm_win_rate_heatmap.png")
    try:
        plt.savefig(output_path)
        print(f"\nWin rate heatmap saved to {output_path}")
    except Exception as e:
        print(f"Error saving win rate heatmap: {e}")
    plt.close()

    # --- Heatmap 2: LLM Average Steps ---
    heatmap_steps_data = llm_agg_data.set_index('llm')[['average_steps']]
    
    plt.figure(figsize=(final_fig_w1, fig_h1))
    
    # Create annotation strings (steps as integers)
    annot_data_llm_steps = llm_agg_data.set_index('llm').apply(
        lambda x: f"{x['average_steps']:.0f}" if pd.notnull(x['average_steps']) else "N/A",
        axis=1
    ).values.reshape(heatmap_steps_data.shape)

    # Use Blues colormap for steps (more steps = darker blue) with tighter color bar
    sns.heatmap(heatmap_steps_data, annot=annot_data_llm_steps, fmt="", cmap="Blues", 
                vmin=max(0, min_steps), vmax=max_steps, 
                cbar_kws={'label': 'Average Steps', 'shrink': 0.8, 'pad': 0.01}, 
                annot_kws={"size": 10})
    plt.title("Average Steps per LLM")
    plt.ylabel("LLM Model")
    plt.xticks([])
    # 调整布局，使颜色条更靠近图表
    plt.tight_layout(pad=1.2, rect=[0, 0, 0.9, 1])  # 右侧留出更少空间让颜色条贴近
    
    output_path_steps = os.path.join(current_script_path, "llm_steps_heatmap.png")
    try:
        plt.savefig(output_path_steps)
        print(f"Steps heatmap saved to {output_path_steps}")
    except Exception as e:
        print(f"Error saving steps heatmap: {e}")
    plt.close()

    # Heatmaps for LLM vs Game (win rates and steps)
    llm_game_agg = df.groupby(["llm", "game"]).agg(
        average_win_rate=('win', 'mean'),
        average_steps=('steps', 'mean')
    )
    
    if not llm_game_agg.empty:
        llm_game_win_rates_plot = llm_game_agg['average_win_rate'].unstack()
        llm_game_steps_plot = llm_game_agg['average_steps'].unstack()
        
        if not llm_game_win_rates_plot.empty:
            # --- Heatmap 3: LLM vs Game (Win Rates) ---
            # Create annotation strings (win rates as percentages)
            annot_data_game_win = []
            for r_idx, row_name in enumerate(llm_game_win_rates_plot.index):
                annot_row = []
                for c_idx, col_name in enumerate(llm_game_win_rates_plot.columns):
                    win_rate_val = llm_game_win_rates_plot.iat[r_idx, c_idx]
                    if pd.notnull(win_rate_val):
                        annot_str = f"{win_rate_val:.0%}"
                        annot_row.append(annot_str)
                    else:
                        annot_row.append("")
                annot_data_game_win.append(annot_row)

            h_padding2 = 3.0
            v_padding2 = 2.5
            num_rows2 = len(llm_game_win_rates_plot.index)
            num_cols2 = len(llm_game_win_rates_plot.columns)

            fig_h2 = (num_rows2 * target_cell_height_global) + v_padding2
            ideal_data_w2 = num_cols2 * target_cell_width_global
            ideal_fig_w2 = ideal_data_w2 + h_padding2
            final_fig_w2 = max(ideal_fig_w2, min_total_figure_width_global)

            plt.figure(figsize=(final_fig_w2, fig_h2))
            sns.heatmap(llm_game_win_rates_plot, annot=pd.DataFrame(annot_data_game_win, index=llm_game_win_rates_plot.index, columns=llm_game_win_rates_plot.columns), 
                        fmt="", cmap="RdYlGn", vmin=0, vmax=1, 
                        cbar_kws={'label': 'Average Win Rate', 'shrink': 0.8, 'pad': 0.01}, 
                        annot_kws={"size": 9})
            plt.title("Average Win Rate: LLM vs. Game")
            plt.xlabel("Game", labelpad=10)  # 增加labelpad让标签往上移
            plt.ylabel("LLM Model")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            # 调整布局，使颜色条更靠近图表
            plt.tight_layout(pad=1.2, rect=[0, 0, 0.9, 1])  # 右侧留出更少空间让颜色条贴近
            
            output_path_game_win = os.path.join(current_script_path, "llm_vs_game_win_rate_heatmap.png")
            try:
                plt.savefig(output_path_game_win)
                print(f"LLM vs Game win rate heatmap saved to {output_path_game_win}")
            except Exception as e:
                print(f"Error saving LLM vs Game win rate heatmap: {e}")
            plt.close()
            
            # --- Heatmap 4: LLM vs Game (Steps) ---
            # Create annotation strings (steps as integers)
            annot_data_game_steps = []
            for r_idx, row_name in enumerate(llm_game_steps_plot.index):
                annot_row = []
                for c_idx, col_name in enumerate(llm_game_steps_plot.columns):
                    steps_val = llm_game_steps_plot.iat[r_idx, c_idx]
                    if pd.notnull(steps_val):
                        annot_str = f"{steps_val:.0f}"
                        annot_row.append(annot_str)
                    else:
                        annot_row.append("")
                annot_data_game_steps.append(annot_row)

            plt.figure(figsize=(final_fig_w2, fig_h2))
            sns.heatmap(llm_game_steps_plot, annot=pd.DataFrame(annot_data_game_steps, index=llm_game_steps_plot.index, columns=llm_game_steps_plot.columns), 
                        fmt="", cmap="Blues", vmin=max(0, min_steps), vmax=max_steps, cbar_kws={'label': 'Average Steps'}, annot_kws={"size": 9})
            plt.title("Average Steps: LLM vs. Game")
            plt.xlabel("Game", labelpad=10)  # 增加labelpad让标签往上移
            plt.ylabel("LLM Model")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            # 调整布局，使颜色条更靠近图表
            plt.tight_layout(pad=1.2, rect=[0, 0, 0.9, 1])  # 右侧留出更少空间让颜色条贴近
            
            output_path_game_steps = os.path.join(current_script_path, "llm_vs_game_steps_heatmap.png")
            try:
                plt.savefig(output_path_game_steps)
                print(f"LLM vs Game steps heatmap saved to {output_path_game_steps}")
            except Exception as e:
                print(f"Error saving LLM vs Game steps heatmap: {e}")
            plt.close()
        else:
            print("No win rate data for LLM vs Game heatmap.")
    else:
        print("No aggregated data for LLM vs Game heatmap.")

    # Generate heatmaps per game: LLM vs Level (win rates and steps)
    if not df.empty:
        games = df["game"].unique()
        for game_name in games:
            game_df = df[df["game"] == game_name]
            if game_df.empty:
                print(f"No data for game: {game_name}")
                continue

            llm_level_agg = game_df.groupby(["llm", "level"]).agg(
                average_win_rate=('win', 'mean'),
                average_steps=('steps', 'mean')
            )
            
            if llm_level_agg.empty:
                print(f"No aggregated data to plot for game: {game_name}")
                continue

            llm_level_win_rates_plot = llm_level_agg['average_win_rate'].unstack()
            llm_level_steps_plot = llm_level_agg['average_steps'].unstack()

            if llm_level_win_rates_plot.empty:
                print(f"No win rate data to plot for game: {game_name}")
                continue
            
            # Check if there's only one level for this game after unstacking
            if len(llm_level_win_rates_plot.columns) <= 1:
                print(f"Skipping LLM vs Level heatmap for game '{game_name}' as it has only one distinct level or no level variation for comparison.")
                continue
                
            # Create annotation strings for win rate heatmap (win rates as percentages)
            annot_data_level_win = []
            for r_idx, row_name in enumerate(llm_level_win_rates_plot.index):
                annot_row = []
                for c_idx, col_name in enumerate(llm_level_win_rates_plot.columns):
                    win_rate_val = llm_level_win_rates_plot.iat[r_idx, c_idx]
                    if pd.notnull(win_rate_val):
                        annot_str = f"{win_rate_val:.0%}"
                        annot_row.append(annot_str)
                    else:
                        annot_row.append("")
                annot_data_level_win.append(annot_row)
            
            # --- Heatmap: LLM vs Level (per game) for Win Rate ---
            h_padding3 = 3.0
            v_padding3 = 2.5
            num_rows3 = len(llm_level_win_rates_plot.index)
            num_cols3 = len(llm_level_win_rates_plot.columns)

            fig_h3 = (num_rows3 * target_cell_height_global) + v_padding3
            ideal_data_w3 = num_cols3 * target_cell_width_global
            ideal_fig_w3 = ideal_data_w3 + h_padding3
            final_fig_w3 = max(ideal_fig_w3, min_total_figure_width_global)
            
            plt.figure(figsize=(final_fig_w3, fig_h3))
            sns.heatmap(llm_level_win_rates_plot, annot=pd.DataFrame(annot_data_level_win, index=llm_level_win_rates_plot.index, columns=llm_level_win_rates_plot.columns), 
                        fmt="", cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={'label': 'Average Win Rate'}, annot_kws={"size": 9})
            
            safe_game_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in game_name).rstrip()
            safe_game_name = safe_game_name.replace(' ', '_')

            plt.title(f"Average Win Rate: LLM vs Level for Game: {game_name}")
            plt.xlabel("Level", labelpad=10)  # 增加labelpad让标签往上移
            plt.ylabel("LLM Model")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            # 调整布局，使颜色条更靠近图表
            plt.tight_layout(pad=1.2, rect=[0, 0, 0.9, 1])  # 右侧留出更少空间让颜色条贴近
            
            output_path_game_level_win = os.path.join(current_script_path, f"game_{safe_game_name}_llm_vs_level_win_rate_heatmap.png")
            try:
                plt.savefig(output_path_game_level_win)
                print(f"LLM vs Level win rate heatmap for game '{game_name}' saved to {output_path_game_level_win}")
            except Exception as e:
                print(f"Error saving LLM vs Level win rate heatmap for game '{game_name}': {e}")
            plt.close()
            
            # Create annotation strings for steps heatmap (steps as integers)
            annot_data_level_steps = []
            for r_idx, row_name in enumerate(llm_level_steps_plot.index):
                annot_row = []
                for c_idx, col_name in enumerate(llm_level_steps_plot.columns):
                    steps_val = llm_level_steps_plot.iat[r_idx, c_idx]
                    if pd.notnull(steps_val):
                        annot_str = f"{steps_val:.0f}"
                        annot_row.append(annot_str)
                    else:
                        annot_row.append("")
                annot_data_level_steps.append(annot_row)
            
            # --- Heatmap: LLM vs Level (per game) for Steps ---
            plt.figure(figsize=(final_fig_w3, fig_h3))
            sns.heatmap(llm_level_steps_plot, annot=pd.DataFrame(annot_data_level_steps, index=llm_level_steps_plot.index, columns=llm_level_steps_plot.columns), 
                        fmt="", cmap="Blues", vmin=max(0, min_steps), vmax=max_steps, cbar_kws={'label': 'Average Steps'}, annot_kws={"size": 9})
                        
            plt.title(f"Average Steps: LLM vs Level for Game: {game_name}")
            plt.xlabel("Level", labelpad=10)  # 增加labelpad让标签往上移
            plt.ylabel("LLM Model")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            # 调整布局，使颜色条更靠近图表
            plt.tight_layout(pad=1.2, rect=[0, 0, 0.9, 1])  # 右侧留出更少空间让颜色条贴近
            
            output_path_game_level_steps = os.path.join(current_script_path, f"game_{safe_game_name}_llm_vs_level_steps_heatmap.png")
            try:
                plt.savefig(output_path_game_level_steps)
                print(f"LLM vs Level steps heatmap for game '{game_name}' saved to {output_path_game_level_steps}")
            except Exception as e:
                print(f"Error saving LLM vs Level steps heatmap for game '{game_name}': {e}")
            plt.close()
    else:
        print("No data to generate per-game heatmaps.")


if __name__ == "__main__":
    main()