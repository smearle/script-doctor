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


        # Handle cases where game name might be part of the llm_model string initially
        # e.g., "4o-mini_atlas shrank" where "atlas shrank" is the game.
        # The initial regex might capture "4o-mini" as llm and "atlas shrank" as game.
        # Or "4o-mini" as llm and "limerick" as game.
        # We need to be careful if the game name itself has underscores.
        # The current regex tries to capture the longest possible llm_model name first.

        # If llm_model still contains game-like parts separated by underscore
        # and game_name is simple (no underscores), it might be a parsing issue.
        # However, the prompt examples "4o-mini_atlas shrank_run_1" and "4o-mini_limerick_run_1_level_1"
        # suggest the first part before the game is the model.

        # Let's refine the game name extraction for multi-word games
        # The regex (.*?)_(.*?)_run_ might be too greedy for the first part.
        # A more robust way might be to identify known game names or assume the part after llm is the game.

        # Based on "4o-mini_atlas shrank_run_1", llm="4o-mini", game="atlas shrank"
        # Based on "4o-mini_limerick_run_1_level_1", llm="4o-mini", game="limerick"

        # The provided regex: r"^(.*?)_(.*?)_run_(\d+)(?:_level_(\d+))?\.json$"
        # For "4o-mini_atlas shrank_run_1.json":
        #   llm_model = "4o-mini"
        #   game_name = "atlas shrank"
        #   run_number = "1"
        #   level_number = None -> will be set to '0'

        # For "4o-mini_limerick_run_1_level_1.json":
        #   llm_model = "4o-mini"
        #   game_name = "limerick"
        #   run_number = "1"
        #   level_number = "1"
        return llm_model, game_name, int(run_number), int(level_number)

    # Fallback for filenames that might not perfectly match the primary pattern
    # e.g. if "level" is missing but implied, or if game name has "run" in it.
    # This part needs careful consideration based on actual filename variations.
    # For now, we rely on the primary regex.
    print(f"Warning: Could not parse filename: {filename}")
    return None, None, None, None


def collect_results(results_dir):
    """
    Collects results from all .json files in the specified results_dir.
    """
    data = []
    # parent_dir = os.path.dirname(results_dir) # Get the parent directory (llm_agent_results)

    if not os.path.isdir(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return pd.DataFrame()

    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            llm, game, run, level = parse_filename(filename)

            if not llm or not game:
                print(f"Skipping unparsable file: {filename}")
                continue

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

    # Map LLM names
    llm_name_mapping = {
        "4o-mini": "ChatGPT 4o-mini",
        "deepseek": "Deepseek-chat",
        "gemini": "Gemini-flash 2.0 exp",
        "qwen": "Qwen-plus" 
    }
    df['llm'] = df['llm'].replace(llm_name_mapping)

    # Calculate average win rate per LLM
    # We group by LLM and game, then calculate win rate for that, then average for the LLM.
    # Or, if we want overall win rate per LLM across all games and levels:
    # Calculate average win rate and average steps per LLM
    llm_agg_data = df.groupby("llm").agg(
        average_win_rate=('win', 'mean'),
        average_steps=('steps', 'mean')
    ).reset_index()

    print("\nAverage Win Rates and Steps per LLM:")
    print(llm_agg_data)

    # Create heatmap
    # For the heatmap, we need a matrix: LLMs vs Games, with win rate as values.
    # If you want a single bar for each LLM showing its overall average win rate,
    # a heatmap might not be the best viz unless you have another dimension.
    # The request was "heatmap showing each llm, average win rate".
    # This implies LLMs on one axis, and their win rate represented by color.
    # A simple bar chart might be more direct for LLM vs. average_win_rate.
    # If the intention is LLM vs Game heatmap, the pivot is different.

    # Let's assume a heatmap where each LLM is a row, and the color represents its average win rate.
    # This is essentially a colored table.

    if llm_agg_data.empty: # Changed llm_win_rates to llm_agg_data
        print("No win rates to plot. Exiting.")
        return

    # For a heatmap of LLM average win rates (LLMs on y-axis, single column for win rate)
    # Prepare data for heatmap (win rates) and annotations (win rates + steps)
    heatmap_plot_data = llm_agg_data.set_index('llm')[['average_win_rate']]

    target_cell_height_global = 0.7
    target_cell_width_global = 1.0
    min_total_figure_width_global = 8.0

    # --- Heatmap 1: LLM Average Win Rates ---
    h_padding1 = 3.0
    v_padding1 = 2.0
    num_rows1 = len(heatmap_plot_data.index)
    num_cols1 = len(heatmap_plot_data.columns)

    fig_h1 = (num_rows1 * target_cell_height_global) + v_padding1
    ideal_data_w1 = num_cols1 * target_cell_width_global
    ideal_fig_w1 = ideal_data_w1 + h_padding1
    final_fig_w1 = max(ideal_fig_w1, min_total_figure_width_global)
    
    plt.figure(figsize=(final_fig_w1, fig_h1))
    
    # Create annotation strings (only steps)
    annot_data_llm = llm_agg_data.set_index('llm').apply(
        lambda x: f"{x['average_steps']:.0f}" if pd.notnull(x['average_steps']) else "N/A",
        axis=1
    ).values.reshape(heatmap_plot_data.shape)

    sns.heatmap(heatmap_plot_data, annot=annot_data_llm, fmt="", cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={'label': 'Average Win Rate'}, annot_kws={"size": 10}) # Increased font size
    plt.title("Average Win Rate (Color) and Avg Steps (Text) per LLM") # Clarified title
    plt.ylabel("LLM Model")
    plt.xticks([]) # No x-axis labels needed for a single column heatmap
    plt.tight_layout()
    
    output_path = os.path.join(current_script_path, "llm_win_rate_steps_heatmap.png") # Updated filename
    try:
        plt.savefig(output_path)
        print(f"\nHeatmap saved to {output_path}")
    except Exception as e:
        print(f"Error saving heatmap: {e}")

    # Optional: Heatmap of LLM vs Game win rates and steps
    llm_game_agg = df.groupby(["llm", "game"]).agg(
        average_win_rate=('win', 'mean'),
        average_steps=('steps', 'mean')
    )
    if not llm_game_agg.empty:
        llm_game_win_rates_plot = llm_game_agg['average_win_rate'].unstack()
        
        if not llm_game_win_rates_plot.empty:
            llm_game_steps_plot = llm_game_agg['average_steps'].unstack()
            
            annot_data_game = []
            for r_idx, row_name in enumerate(llm_game_win_rates_plot.index):
                annot_row = []
                for c_idx, col_name in enumerate(llm_game_win_rates_plot.columns):
                    # win_val = llm_game_win_rates_plot.iat[r_idx, c_idx] # Color is based on this
                    step_val = llm_game_steps_plot.iat[r_idx, c_idx]
                    if pd.notnull(step_val):
                        annot_str = f"{step_val:.0f}"
                        annot_row.append(annot_str)
                    elif pd.notnull(llm_game_win_rates_plot.iat[r_idx, c_idx]): # If there's a win rate but no steps
                        annot_row.append("N/A steps")
                    else: # If no win rate either (cell should be blank or NaN color)
                        annot_row.append("") # Empty string for annotation if no data
                annot_data_game.append(annot_row)

            # --- Heatmap 2: LLM vs Game ---
            h_padding2 = 3.0
            v_padding2 = 2.5
            num_rows2 = len(llm_game_win_rates_plot.index)
            num_cols2 = len(llm_game_win_rates_plot.columns)

            fig_h2 = (num_rows2 * target_cell_height_global) + v_padding2
            ideal_data_w2 = num_cols2 * target_cell_width_global
            ideal_fig_w2 = ideal_data_w2 + h_padding2
            final_fig_w2 = max(ideal_fig_w2, min_total_figure_width_global)

            plt.figure(figsize=(final_fig_w2, fig_h2))
            sns.heatmap(llm_game_win_rates_plot, annot=pd.DataFrame(annot_data_game, index=llm_game_win_rates_plot.index, columns=llm_game_win_rates_plot.columns), 
                        fmt="", cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={'label': 'Average Win Rate'}, annot_kws={"size": 9}) # Increased font size
            plt.title("Avg Win Rate (Color) and Avg Steps (Text): LLM vs. Game") # Clarified title
            plt.xlabel("Game")
            plt.ylabel("LLM Model")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            output_path_game = os.path.join(current_script_path, "llm_vs_game_win_rate_steps_heatmap.png") # Updated filename
            try:
                plt.savefig(output_path_game)
                print(f"LLM vs Game (with steps) heatmap saved to {output_path_game}")
            except Exception as e:
                print(f"Error saving LLM vs Game (with steps) heatmap: {e}")
        else:
            print("No win rate data for LLM vs Game heatmap.")
    else:
        print("No aggregated data for LLM vs Game heatmap.")

    # Generate heatmap per game: LLM vs Level win rates and steps
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

            if llm_level_win_rates_plot.empty:
                print(f"No win rate data to plot for game: {game_name}")
                continue
            
            # Check if there's only one level for this game after unstacking
            if len(llm_level_win_rates_plot.columns) <= 1:
                print(f"Skipping LLM vs Level heatmap for game '{game_name}' as it has only one distinct level or no level variation for comparison.")
                continue
                
            llm_level_steps_plot = llm_level_agg['average_steps'].unstack()

            annot_data_level = []
            for r_idx, row_name in enumerate(llm_level_win_rates_plot.index):
                annot_row = []
                for c_idx, col_name in enumerate(llm_level_win_rates_plot.columns):
                    # win_val = llm_level_win_rates_plot.iat[r_idx, c_idx] # Color is based on this
                    step_val = llm_level_steps_plot.iat[r_idx, c_idx]
                    if pd.notnull(step_val):
                        annot_str = f"{step_val:.0f}"
                        annot_row.append(annot_str)
                    elif pd.notnull(llm_level_win_rates_plot.iat[r_idx, c_idx]):
                        annot_row.append("N/A steps")
                    else:
                        annot_row.append("")
                annot_data_level.append(annot_row)
            
            # --- Heatmap 3: LLM vs Level (per game) ---
            h_padding3 = 3.0
            v_padding3 = 2.5
            num_rows3 = len(llm_level_win_rates_plot.index)
            num_cols3 = len(llm_level_win_rates_plot.columns)

            fig_h3 = (num_rows3 * target_cell_height_global) + v_padding3
            ideal_data_w3 = num_cols3 * target_cell_width_global
            ideal_fig_w3 = ideal_data_w3 + h_padding3
            final_fig_w3 = max(ideal_fig_w3, min_total_figure_width_global)
            
            plt.figure(figsize=(final_fig_w3, fig_h3))
            sns.heatmap(llm_level_win_rates_plot, annot=pd.DataFrame(annot_data_level, index=llm_level_win_rates_plot.index, columns=llm_level_win_rates_plot.columns), 
                        fmt="", cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={'label': 'Average Win Rate'}, annot_kws={"size": 9}) # Increased font size
            
            safe_game_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in game_name).rstrip()
            safe_game_name = safe_game_name.replace(' ', '_')

            plt.title(f"Avg Win Rate (Color) & Steps (Text): LLM vs Level for Game: {game_name}") # Clarified title
            plt.xlabel("Level")
            plt.ylabel("LLM Model")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            output_path_game_level = os.path.join(current_script_path, f"game_{safe_game_name}_llm_vs_level_steps_heatmap.png") # Updated filename
            try:
                plt.savefig(output_path_game_level)
                print(f"LLM vs Level (with steps) heatmap for game '{game_name}' saved to {output_path_game_level}")
            except Exception as e:
                print(f"Error saving LLM vs Level (with steps) heatmap for game '{game_name}': {e}")
    else:
        print("No data to generate per-game heatmaps.")


if __name__ == "__main__":
    main()
