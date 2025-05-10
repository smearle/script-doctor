import base64
import binascii as ba
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
import io
import inspect
import itertools
import json
import os
from pathlib import Path
import random
import re
import traceback
import webbrowser  # Add this import
import atexit  # Add this import
import threading  # Add this import
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from selenium import webdriver

import Levenshtein
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
import hydra
from hydra.core.config_store import ConfigStore
from PIL import Image
import numpy as np
import openai
import pandas as pd
import requests

from client import open_browser
import game_gen
from parse_lark import GAMES_DIR, MIN_GAMES_DIR, PrintPuzzleScript, RepairPuzzleScript, StripPuzzleScript, add_empty_sounds_section, preprocess_ps, TEST_GAMES
from prompts import *
from utils import extract_ps_code, gen_fewshot_examples, llm_text_query, num_tokens_from_string, save_prompts, truncate_str_to_token_len


@dataclass
class Config:
    mode: str = 'generate'
    port: int = 8000
    sweep: str = 'models'
    # Mostly for dev. When we change something client-side about the metrics that we save for a given game.
    recompute_stats: bool = False
    max_gen_attempts: int = 10
    provider: str = 'portkey'
    model: str = 'gemini-2.0-flash-exp'
    viz_feedback: bool = True
    headless: bool = False
    auto_launch_client: bool = True


@dataclass
class EvoConfig(Config):
    seed: int = 0
    port: int = 8000


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="evo", node=EvoConfig)

max_gen_attempts = None


load_dotenv()
openai_client = None
app = Flask(__name__)


@app.route('/')
def serve_doctor():
    return send_from_directory('src', 'doctor.html')

@app.route('/get_mode', methods=['GET'])
def get_mode():
    global global_cfg
    return jsonify({'mode': global_cfg.mode})

# Route to serve JavaScript files dynamically
@app.route('/<path:filename>')
def serve_js(filename):
    return send_from_directory('src', filename)


# @app.route('/save_sweep_stats', methods=['POST'])
# def save_sweep_stats():
#     data = request.json
#     stats = data['results']
#     save_dir = os.path.join('logs', data['save_dir'])
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     with open(os.path.join(save_dir, 'stats.json'), 'w') as f:
#         json.dump(stats, f, indent=4)
#     concise_stats = {}
#     for hyp_settings in stats:
#         concise_stats[hyp_settings] = []
#         for val in stats[hyp_settings]:
#             val.pop('code')
#             val.pop('min_code')
#             concise_stats[hyp_settings].append(val)
#     with open(os.path.join(save_dir, 'concise_stats.json'), 'w') as f:
#         json.dump(concise_stats, f, indent=4)

#     return jsonify({})


@app.route('/load_ideas', methods=['POST'])
def load_ideas():
    data = request.json
    brainstorm_seed = data['brainstorm_seed']
    ideas_path = os.path.join('logs', f'brainstorm_s-{brainstorm_seed}.json')
    with open(ideas_path, 'r') as f:
        ideas = json.load(f)
    return ideas

@app.route('/file_exists', methods=['POST'])
def file_exists():
    data = request.json
    file_path = data['filePath']
    exists = os.path.isfile(file_path)
    print(f"File {file_path} exists: {exists}")
    return jsonify({'exists': exists})

@app.route('/load_game_from_file', methods=['POST'])
def load_game_from_file():
    data = request.json
    game = data['game']
    game_path = os.path.join(MIN_GAMES_DIR, f'{game}.txt')
    with open(game_path, 'r') as f:
        code = f.read()
    print(code)
    return code


@app.route('/log_gen_results', methods=['POST'])
def log_gen_results():
    data = request.json
    console_text = data['console_text']
    solver_text = data['solver_text']
    save_dir = os.path.join('logs', data['save_dir'])
    n_iter = data['n_iter']
    if console_text:
        compile_output_path = os.path.join(save_dir, f'{n_iter}j_compile.txt')
        if not os.path.isfile(compile_output_path):
            with open(compile_output_path, 'w') as f:
                f.write(console_text)
        solver_output_path = os.path.join(save_dir, f'{n_iter}k_solver.txt')
        if solver_text and not os.path.isfile(solver_output_path):
            with open(solver_output_path, 'w') as f:
                f.write(solver_text)

    sols = data['sols']
    n_iter = data['n_iter']
    sols_path = os.path.join(save_dir, f'{n_iter}l_sols.json')
    if sols:
        with open(sols_path, 'w') as f:
            json.dump(sols, f, indent=4)
    gif_urls = data['gif_urls']
    
    for gif_url, level_i in gif_urls:
        # Download the gif from the url
        gif_data = base64.b64decode(gif_url.split(',')[1])
        gif_path = os.path.join(save_dir, f'{n_iter}m_sol_level-{level_i}.gif')
        
        with open(gif_path, 'wb') as f:
            f.write(gif_data)
        
        # Convert the gif to a sequence of pngs
        with Image.open(io.BytesIO(gif_data)) as img:
            img.seek(0)
            frame_number = 0
            png_dir = os.path.join(save_dir, f'{n_iter}m_sol_level-{level_i}')
            os.makedirs(png_dir, exist_ok=True)
            
            while True:
                frame_path = os.path.join(png_dir, f'frame_{frame_number}.png')
                img.save(frame_path, 'PNG')
                frame_number += 1
                try:
                    img.seek(frame_number)
                except EOFError:
                    break
    
    return jsonify({})

import lark
lark_parser = lark.Lark.open('syntax.lark', start='ps_game')

@app.route('/save_evo_gen_stats', methods=['POST'])
def save_evo_gen_stats():
    data = request.json
    stats = data['stats']
    stats_path = os.path.join('logs', data['save_dir'])
    stats_dir = Path(stats_path).parent
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    return jsonify({})

def viz_evo_stats(exp_seed=21):
    stats_dir = f'logs/evo-{exp_seed}/stats'
    gen_i = 0
    stats_exist = True
    fits = []
    while stats_exist:
        gen_i += 1
        stats_path = os.path.join(stats_dir, f'gen-{gen_i}.json')
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            pop_fits = stats['pop_fits']
            fits.append(pop_fits)
        stats_exist = os.path.exists(os.path.join(stats_dir, f'gen-{gen_i+1}.json'))
    breakpoint()

@app.route('/get_viz_feedback', methods=['POST'])
def get_viz_feedback():
    breakpoint()

@app.route('/gen_game', methods=['POST'])
def gen_game():
    data = request.json
    seed = data['seed']
    fewshot = data['fewshot']
    cot = data['cot']
    from_idea = data['from_idea']
    game_idea = data['game_idea']
    cot_prompt_text = cot_prompt if cot else ''
    log_dir = 'logs'
    save_dir = os.path.join(log_dir, data['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    code = data['code']
    gen_mode = data['gen_mode']
    parents = data['parents']
    parents = [] if parents == 'None' else parents
    compilation_success = data['compilation_success']
    console_text = data['console_text']
    solver_text = data['solver_text']
    lark_error = data['lark_error']
    n_iter = data['n_iter']
    meta_parents = data['meta_parents']
    curr_docs_prompt = docs_prompt if exp_config.docs else ''
    curr_inventive_prompt = inventive_prompt if exp_config.inventive_prompt else ''
    gen_game_output_path = os.path.join(save_dir, f'{n_iter}b_code.txt')
    gen_game_code_output_path = os.path.join(save_dir, f'{n_iter}b_code.txt')
    print(f"Generating and saving code at {gen_game_code_output_path}")
    if not os.path.isfile(gen_game_code_output_path):
        gen_game_prompt_output_path = os.path.join(save_dir, f'{n_iter}a_prompt.txt')
        if meta_parents is None:
            system_prompt = game_gen_system_prompt.format(docs_prompt=curr_docs_prompt)
        else:
            # Load the meta-parents from the scraped_games directory
            meta_parent_txt = ''
            for parent in meta_parents:
                with open(os.path.join('data/scraped_games', f'{parent}.txt'), 'r') as f:
                    meta_parent_txt += f.read() + '\n\n'
            system_prompt = evo_meta_prompt.format(docs_prompt=curr_docs_prompt, meta_parents=meta_parent_txt)
        from_idea_prompt_i = from_idea_prompt.format(game_idea=game_idea) if from_idea else ''
        if n_iter == 0:
            parents_text = '/n/n'.join([p['code'] for p in parents])
            if gen_mode == 'init':
                prompt = gen_game_prompt.format(cot_prompt=cot_prompt_text, from_idea_prompt=from_idea_prompt_i,
                                                inventive_prompt=curr_inventive_prompt)
            elif gen_mode == 'mutate':
                prompt = game_mutate_prompt.format(parents=parents_text, cot_prompt=cot_prompt_text)
            elif gen_mode == 'crossover':
                prompt = game_crossover_prompt.format(parents=parents_text, cot_prompt=cot_prompt_text)    
        elif not compilation_success:
            if lark_error is None:
                lark_error_prompt = ''
            else:
                lark_error_prompt = f"""It also resulted in the following error when we attempted to parse the code as a context free grammar using lark: ```{lark_error}```""" if lark_error is not None else ""
            
            prompt = game_compile_repair_prompt.format(code=code, console_text=console_text, cot_prompt=cot_prompt_text,
                                                       game_idea=game_idea, lark_error_prompt=lark_error_prompt,
                                                       from_idea_repair_prompt=from_idea_prompt_i)
        else:
            prompt = game_solvability_repair_prompt.format(code=code, solver_text=solver_text,
                                                           from_idea_repair_prompt=from_idea_prompt_i, cot_prompt=cot_prompt)
        # Sometimes the OpenAI reasoning model flags the prompt as potentially inappropriate. I assumme this is coming
        # from the fewshot examples, so we'll keep trying here.
        system_prompt_0 = system_prompt
        successful_query = False
        while not successful_query:
            try:
                if fewshot:
                    system_prompt = system_prompt_0 + gen_fewshot_examples(
                        system_prompt_0, prompt, max_tokens=exp_config.context_len
                    )
                successful_query = True
            except openai.BadRequestError as e:
                assert e.code == 'invalid_prompt'
                print(traceback.format_exc())
                print(f"Failed to query the model. I think because a human game in the fewshot prompt was flagged as inappropriate? Retrying...")
        save_prompts(system_prompt, prompt, gen_game_prompt_output_path)
        text = llm_text_query(system_prompt, prompt, seed, model=exp_config.model, provider=exp_config.provider)
        with open(gen_game_code_output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        with open(gen_game_code_output_path, 'r', encoding='utf-8') as f:
            text = f.read()
    code, plaintext = extract_ps_code(text)
    if 'randomDir' in code:
        skip = True
    else:
        skip = False
    # If the code we just generated has already been solved, pass it to the client so it doesn't apply the solver to it
    sols_path = os.path.join(save_dir, f'{n_iter}l_sols.json')
    if os.path.exists(sols_path):
        with open(sols_path, 'r') as f:
            sols = json.load(f)
    else:
        sols = {}

    simp_filepath = os.path.join(save_dir, f'{n_iter}b1_code_simplified.txt')
    # Now save the simplified version of the file
    simp_code = preprocess_ps(code)
    with open(simp_filepath, "w", encoding='utf-8') as file:
        print(f"Writing to {simp_filepath}")
        file.write(simp_code)
    successful_lark_parse = False
    min_code = None
    lark_error = None
    try:
        parse_tree = lark_parser.parse(simp_code)
        successful_lark_parse = True
    except lark.exceptions.ParseError as e:
        print(f"Faile to parse code with lark: {e}")
        lark_error = str(e)
    except lark.exceptions.UnexpectedCharacters as e:
        print(f"Failed to parse code with lark, UnexpectedCharacters: {e}")
        lark_error = str(e)
    if successful_lark_parse:
        lark_error = None
        min_parse_tree = StripPuzzleScript().transform(parse_tree)
        pretty_parse_tree_str = min_parse_tree.pretty()
        pretty_tree_filename = os.path.join(save_dir, f'{n_iter}b2_code_tree.txt')
        print(f"Writing pretty tree to {pretty_tree_filename}")
        with open(pretty_tree_filename, "w") as file:
            file.write(pretty_parse_tree_str)
        repaired_parse_tree = RepairPuzzleScript().transform(min_parse_tree)
        min_code = PrintPuzzleScript().transform(repaired_parse_tree)
        min_code = add_empty_sounds_section(min_code)
        min_filename = os.path.join(save_dir, f'{n_iter}b3_code_min.txt')
        with open(min_filename, "w") as file:
            file.write(min_code)

    return jsonify({
        'code': code,
        'min_code': min_code,
        'text': plaintext,
        'sols': sols,
        'skip': skip,
        'lark_error': lark_error,
    })


@app.route('/gen_game_from_plan', methods=['POST'])
def gen_game_from_plan():
    data = request.json
    seed = data['seed']
    # fewshot = data['fewshot']
    fewshot = True
    # from_idea = data['from_idea']
    game_idea = data['game_idea']
    cot_prompt_text = cot_prompt
    log_dir = 'logs'
    save_dir = os.path.join(log_dir, data['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    plan_output_path = os.path.join(save_dir, f'0b_plan.txt')
    print(f"Saving plan at: {plan_output_path}")
    curr_docs_prompt = docs_prompt if exp_config.docs else ''
    if not os.path.isfile(plan_output_path):
        plan_prompt_output_path = os.path.join(save_dir, f'0a_prompt.txt')
        plan_system_prompt = game_gen_system_prompt.format(docs_prompt=curr_docs_prompt)
        from_idea_prompt_i = from_idea_prompt.format(game_idea=game_idea)
        prompt = plan_game_prompt.format(from_idea_prompt=from_idea_prompt_i)
        if fewshot:
            plan_system_prompt += gen_fewshot_examples(plan_system_prompt, prompt, max_tokens=exp_config.context_len)
        save_prompts(plan_system_prompt, prompt, plan_prompt_output_path)
        game_plan = llm_text_query(plan_system_prompt, prompt, seed, model=exp_config.model, provider=exp_config.provider)
        with open(plan_output_path, 'w', encoding='utf-8') as f:
            f.write(game_plan)
    else:
        with open(plan_output_path, 'r', encoding='utf-8') as f:
            game_plan = f.read()
    sprites_system_prompt = game_gen_system_prompt.format(docs_prompt=curr_docs_prompt)
    if fewshot:
        sprites_system_prompt += gen_fewshot_examples(sprites_system_prompt, '', max_tokens=exp_config.context_len) 
    with open('example_sprite_names.txt', 'r', encoding='utf-8') as f:
        sprite_names = f.read()
    with open('example_sprites.json', 'r') as f:
        sprites_library = json.load(f)
    sprites_prompt = gen_sprites_prompt.format(game_plan=game_plan, sprites_library=sprite_names)
    sprites_output_path = os.path.join(save_dir, f'0c_sprites.txt')
    if not os.path.isfile(sprites_output_path):
        sprites_prompt_output_path = os.path.join(save_dir, f'0b_sprites_prompt.txt')
        save_prompts(sprites_system_prompt, sprites_prompt, sprites_prompt_output_path)
        sprites = llm_text_query(sprites_system_prompt, sprites_prompt, seed, model=exp_config.model, provider=exp_config.provider)
        with open(sprites_output_path, 'w', encoding='utf-8') as f:
            f.write(sprites)
        
    else:
        with open(sprites_output_path, 'r', encoding='utf-8') as f:
            sprites = f.read()

    match = re.search(r'OBJECTS\s*=+\s*(.*?)\s*=+\s*LEGEND\s*=+\s*(.*)```', sprites, re.DOTALL)
    objects = match.group(1)
    objects_list = objects.split('\n\n')
    # Find any objects that are just a name (a single line with a single word)
    for i, obj in enumerate(objects_list):
        if len(obj.split('\n')) == 1:
            obj = obj.strip()
            assert obj in sprites_library, f"Object {obj} not found in sprite library"
            objects_list[i] = random.choice(sprites_library[obj])
    object_legend = match.group(2)

    rules_system_prompt = game_gen_system_prompt.format(docs_prompt=curr_docs_prompt)
    if fewshot:
        rules_system_prompt += gen_fewshot_examples(rules_system_prompt, '', max_tokens=exp_config.context_len)
    rules_prompt = gen_rules_prompt.format(game_plan=game_plan, object_legend=object_legend)
    rules_output_path = os.path.join(save_dir, f'0e_rules.txt')
    if not os.path.isfile(rules_output_path):
        rules_prompt_output_path = os.path.join(save_dir, f'0d_rules_prompt.txt')
        save_prompts(sprites_system_prompt, rules_prompt, rules_prompt_output_path)
        rules = llm_text_query(sprites_system_prompt, rules_prompt, seed, model=exp_config.model, provider=exp_config.provider)
        with open(rules_output_path, 'w', encoding='utf-8') as f:
            f.write(rules)
    else:
        with open(rules_output_path, 'r', encoding='utf-8') as f:
            rules = f.read()
        
    match = re.search(r'COLLISIONLAYERS\s*=+\s*(.*?)\s*=+\s*RULES\s*=+\s*((?:(?!\n=+).)*)\s*=+\s*WINCONDITIONS\s*=+\s*((?:(?!===).)*?)\s*=*\s*```', rules, re.DOTALL)
    collision_layers = match.group(1)
    rules = match.group(2)
    win_conditions = match.group(3)
    partial_code = (
        "========\nLEGEND\n========\n"
        + object_legend
        + "\n========\nCOLLISIONLAYERS\n========\n"
        + collision_layers
        + "\n========\nRULES\n========\n"
        + rules
        + "\n========\nWINCONDITIONS\n========\n"
        + win_conditions
    )

    levels_system_prompt = game_gen_system_prompt.format(docs_prompt=curr_docs_prompt)
    if fewshot:
        levels_system_prompt += gen_fewshot_examples(levels_system_prompt, '', max_tokens=exp_config.context_len)
    levels_prompt = gen_levels_prompt.format(game_plan=game_plan, code=partial_code)
    levels_output_path = os.path.join(save_dir, f'0g_levels.txt')
    if not os.path.isfile(levels_output_path):
        levels_prompt_output_path = os.path.join(save_dir, f'0f_levels_prompt.txt')
        save_prompts(levels_system_prompt, levels_prompt, levels_prompt_output_path)
        levels = llm_text_query(levels_system_prompt, levels_prompt, seed, model=exp_config.model, provider=exp_config.provider)
        with open(levels_output_path, 'w', encoding='utf-8') as f:
            f.write(levels)
    else:
        with open(levels_output_path, 'r', encoding='utf-8') as f:
            levels = f.read()
    
    levels = re.search(r'```plaintext\s*(?:\s*=+\s*LEVELS\s*=+\s*)\s*(.*)```', levels, re.DOTALL).group(1)
    
    partial_code = (
        "========\nOBJECTS\n========\n"
        + objects
        + "\n"
        + partial_code
        + "\n========\nLEVELS\n========\n"
        + levels
    )

    finalize_system_prompt = game_gen_system_prompt.format(docs_prompt=curr_docs_prompt)
    if fewshot:
        finalize_system_prompt += gen_fewshot_examples(finalize_system_prompt, '', max_tokens=exp_config.context_len)
    finalize_prompt = gen_finalize_prompt.format(game_plan=game_plan, code=partial_code)
    finalize_output_path = os.path.join(save_dir, f'0i_code.txt')
    if not os.path.isfile(finalize_output_path):
        finalize_prompt_output_path = os.path.join(save_dir, f'0h_code_prompt.txt')
        save_prompts(finalize_system_prompt, finalize_prompt, finalize_prompt_output_path)
        code = llm_text_query(finalize_system_prompt, finalize_prompt, seed, model=exp_config.model, provider=exp_config.provider)
        with open(finalize_output_path, 'w', encoding='utf-8') as f:
            f.write(code)
    else:
        with open(finalize_output_path, 'r', encoding='utf-8') as f:
            code = f.read()
    
    code = extract_ps_code(code)[0]

    # Add an empty SOUNDS section to the code, after the LEGEND
    code_a, code_b = code.split('\n========\nCOLLISIONLAYERS')
    code = code_a + '\n========\nSOUNDS\n========\n========\nCOLLISIONLAYERS' + code_b
    
    if 'randomDir' in code:
        skip = True
    else:
        skip = False
    # If the code we just generated has already been solved, pass it to the client so it doesn't apply the solver to it
    sols_path = os.path.join(save_dir, f'{0}l_sols.json')
    if os.path.exists(sols_path):
        with open(sols_path, 'r') as f:
            sols = json.load(f)
    else:
        sols = {}
    return jsonify({
        'code': code,
        # 'text': plaintext,
        'sols': sols,
        'skip': skip,
    })


TRANSITIONS_DIR = 'transitions'

games_to_skip = set({'Broken Rigid Body'})

@app.route('/get_player_action', methods=['POST'])
def get_player_action():
    data = request.json
    obs = data['obs']
    # TODO
    # action = ...
    action = random.randint(0, 5)
    return jsonify({'action': action})

@app.route('/list_scraped_games', methods=['POST'])
def list_scraped_games():
    data = request.json
    target_dir = data['target_dir']
    games_set = set()
    games = []
    # game_files = os.listdir(os.path.join('data', target_dir))
    # random.shuffle(game_files)
    # test_game_files = [f"{test_game}.txt" for test_game in TEST_GAMES]
    # game_files = test_game_files + game_files
    with open('games_n_rules.json', 'r') as f:
        games_n_rules = json.load(f)
    games_n_rules = sorted(games_n_rules, key=lambda x: x[1])
    game_files = [game[0] for game in games_n_rules]
    for filename in game_files:
        if filename.startswith('rigid_'):
            continue
        if filename.endswith('.txt'):
            filename = filename[:-4]
            if filename not in games_set:
                if filename in games_to_skip:
                    print(f"Skipping {filename}")
                    continue
                games_set.add(filename)
        games.append(filename)
    print(games)
    return jsonify(games)

@app.route('/save_init_state', methods=['POST'])
def save_init_state():
    data = request.json
    game_rep = data['state_repr']
    game_hash = data['game_hash']
    game_level = data['game_level']
    state_hash = data['state_hash']
    im_data = data['im_data']
    im_data = base64.b64decode(im_data.split(',')[1])
    im_dir = os.path.join(TRANSITIONS_DIR, game_hash, str(game_level), 'images')
    os.makedirs(im_dir, exist_ok=True)
    im_path = os.path.join(im_dir, f'{state_hash}.png')
    if not os.path.isfile(im_path):
        with open(im_path, 'wb') as f:
            f.write(im_data)
    return jsonify({
        'status': 'success',
        'game_hash': game_hash,
        'state_hash': state_hash,
        'game_rep': game_rep,
        'im_dir':im_dir
    })
    # feed my python code the game state


@app.route('/save_transition', methods=['POST'])
def save_transition():
    data = request.json
    game_rep = data['state_repr']
    game_hash = data['game_hash']
    game_level = data['game_level']
    trans_dir = os.path.join(TRANSITIONS_DIR, game_hash, str(game_level))
    im_dir = os.path.join(trans_dir, 'images')
    
    # Create directories
    os.makedirs(trans_dir, exist_ok=True)
    os.makedirs(im_dir, exist_ok=True)
    
    # Save images
    # for i, img in enumerate([data['state1_img'], data['state2_img']]):
    img_2 = data['state2_img']
    img_data_2 = base64.b64decode(img_2.split(',')[1])
    img_hash_1 = data[f'state1_hash']
    img_hash_2 = data[f'state2_hash']
    img_path_1 = os.path.join(im_dir, f'{img_hash_1}.png')
    img_path_2 = os.path.join(im_dir, f'{img_hash_2}.png')

    if not os.path.exists(img_path_2):
        with open(img_path_2, 'wb') as f:
            f.write(img_data_2)
    
    # Save transition data
    transition = {
        'state1': data['state1_hash'],
        'state2': data['state2_hash'],
        'action': data['action']
    }
    
    with open(os.path.join(trans_dir, 'transitions.json'), 'a') as f:
        f.write(json.dumps(transition) + '\n')
        
    return jsonify({'status': 'success'})

sweep_i = 21
hypers_i = 0
hypers_ks, hypers_lst = [], []

class ExpConfig:
    seed = 0
    provider = 'portkey'
    # model = 'gemini-2.5-flash-preview-04-17'
    model = 'gemini-2.0-flash-exp'
    inventive_prompt = True
    cot = True
    fewshot = True
    docs = False
    from_idea = False
    context_len = 10_000
    viz_feedback = False

class Sweep:
    seed = list(range(10))
    context_len = [30_000]

class InventiveSweep(Sweep):
    seed = list(range(20))
    inventive_prompt = [True, False]

class FewshotSweep(Sweep):
    seed = list(range(20))
    fewshot = [True, False]
    cot = [True, False]

class FromIdeaSweep(Sweep):
    seed = list(range(20))
    from_idea = [True, False]
    cot = [True, False]

class FromIdeaSweep2(Sweep):
    seed = list(range(10))
    model = ['gpt-4o', 'o1', 'o3-mini']
    from_idea = [True, False]
    context_len = [10_000, 30_000]

class ModelSweep(Sweep):
    seed = list(range(15))
    model = ['gpt-4o', 'o1', 'o3-mini']
    context_len = [10_000]

class DocSweep(Sweep):
    # context_len = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
    context_len = [5_000, 10_000]
    docs = [True, False]
    model = ['o1', 'o3-mini']

class CtxSweep(Sweep):
    context_len = [10_000, 20_000, 30_000, 40_000, 50_000, 70_000]
    # context_len = [10_000, 30_000, 50_000, 70_000]

class CtxSweep2(Sweep):
    context_len = [10_000, 20_000, 30_000]
    model = ['o3-mini']

class VizSweep(Sweep):
    viz_feedback = [True, False]

global_cfg = None
exp_config = ExpConfig()

all_hypers = {
    'inventive': InventiveSweep(),
    'fewshot': FewshotSweep(),
    'from_idea': FromIdeaSweep(),
    'from_idea_2': FromIdeaSweep2(),
    'models': ModelSweep(),
    'docs': DocSweep(),
    'ctx': CtxSweep(),
    'ctx_2': CtxSweep2(),
}

sweep_stats = {}


class EvoConfig(ExpConfig):
    n_generations: int = 10


class QDProcess():
    def __init__(self):
        # Cre
        archive = [[] * 10] * 10
        breakpoint()

qd_process = None

@app.route('/evo_ask', methods=['POST'])
def evo_ask():
    global evo_config, qd_process
    if qd_process is None:
        qd_process = QDProcess()

@app.route('/evo_tell', methods=['POST'])
def evo_tell():
    global evo_config, qd_process
    data = request.json
    sols, gif_urls, console_text, solver_text = data['sols'], data['gif_urls'], data['console_text'], data['solver_text']
    breakpoint()


@app.route('/reset_sweep', methods=['POST'])
def reset_sweep():
    global hypers_i, sweep_stats
    hypers_i = 0
    sweep_stats = {}
    return jsonify({})

def _save_sweep_stats(save_dir, sweep_stats):
    stats_dir = os.path.join('logs', save_dir, 'stats', sweep_name)
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, 'sweep_stats.json')
    concise_stats_path = os.path.join(stats_dir, 'concise_sweep_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(sweep_stats, f, indent=4)
    concise_stats = {}
    for hyp_settings in sweep_stats:
        concise_stats[hyp_settings] = []
        for val in sweep_stats[hyp_settings]:
            val.pop('code')
            val.pop('minCode')
            concise_stats[hyp_settings].append(val)
    with open(concise_stats_path, 'w') as f:
        json.dump(concise_stats, f, indent=4)
    compute_edit_distances(stats_path, hypers_ks, hypers_lst)

def compute_edit_distances(stats_path, hyper_ks, hypers_lst):
    stats_and_dists_path = stats_path[:-5] + '_and_dists.json'
    if os.path.exists(stats_and_dists_path) and not recompute_stats:
        print(f"Edit distances already computed for {stats_path}")
    else:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        with open('example_games.json', 'r') as f:
            example_games = json.load(f)
        for hyper_vals in hypers_lst:
            print(f"Computing edit distances for {hyper_vals}")
            exp_config = get_exp_config(hyper_ks, hyper_vals)
            exp_name = get_exp_name(exp_config)
            key = os.path.join(f'sweep-{sweep_i}', exp_name)
            for game_i, game_stat in enumerate(stats[key]):
                if 'min_dist' in stats[key][game_i]:
                    # because we're also sweeping over game_i in hypers_lst
                    continue
                code = game_stat['code']
                dists = []
                for ex_game in example_games:
                    edit_distance = Levenshtein.distance(code, ex_game)
                    dists.append(edit_distance)
                game_min_dist = min(dists)
                stats[key][game_i]['min_dist'] = game_min_dist
        with open(stats_and_dists_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Saved edit distances to {stats_and_dists_path}")
    eval_sweep(stats_and_dists_path, hypers_ks, hypers_lst)

def eval_sweep(stats_and_dists_path, hyper_ks, hypers_lst):
    stats_dir = os.path.dirname(stats_and_dists_path)
    agg_stats = {}
    exp_configs = []
    with open(stats_and_dists_path, 'r') as f:
        stats = json.load(f)
    for hyper_vals in hypers_lst:
        print(f"Compiling stats for {hyper_vals}")
        exp_config = get_exp_config(hyper_ks, hyper_vals)
        exp_name = get_exp_name(exp_config)
        key = os.path.join(f'sweep-{sweep_i}', exp_name)
        if key in agg_stats:
            # because we're also sweeping over game_i in hypers_lst
            continue
        exp_configs.append(exp_config)
        first_comps = []
        first_all_solves = []
        first_any_solves = []
        min_edit_dists = []
        sol_complexities = []
        skipped = 0
        comps = 0
        all_solves = 0
        any_solves = 0
        for stat in stats[key]:
            if stat['skipped']:
                skipped += 1
                continue
            compiled_iters = stat['compiledIters']
            all_solved_iters = stat['solvedIters']
            any_solved_iters = stat['anySolvedIters']
            maxMeanSolComplexity = stat['maxMeanSolComplexity']
            sol_complexities.append(maxMeanSolComplexity)
            if compiled_iters:
                comps += 1
                first_comp = min(compiled_iters)
                # Only consider diversity metric if the game actually compiles
                min_edit_dists.append(stat['min_dist'])
            else:
                first_comp = max_gen_attempts
            if any_solved_iters:
                any_solves += 1
                first_any_solve = min(any_solved_iters)
            else:
                first_any_solve = max_gen_attempts
            if all_solved_iters:
                all_solves += 1
                first_all_solve = min(all_solved_iters)
            else:
                first_all_solve = max_gen_attempts
            first_comps.append(first_comp)
            first_any_solves.append(first_any_solve)
            first_all_solves.append(first_all_solve)
        mean_first_comp = np.mean(first_comps)
        std_first_comp = np.std(first_comps)
        mean_first_all_solve = np.mean(first_all_solves)
        std_first_all_solve = np.std(first_all_solves)
        mean_first_any_solve = np.mean(first_any_solves)
        std_first_any_solve = np.std(first_any_solves)
        pct_comp = comps / len(stats[key])
        pct_any_solve = any_solves / len(stats[key])
        pct_all_solve = all_solves / len(stats[key])
        agg_stats[key] = {
            'mean_first_comp': mean_first_comp,
            'std_first_comp': std_first_comp,
            'mean_first_all_solve': mean_first_all_solve,
            'std_first_all_solve': std_first_all_solve,
            'mean_first_any_solve': mean_first_any_solve,
            'std_first_any_solve': std_first_any_solve,
            'pct_comp': pct_comp,
            'pct_any_solve': pct_any_solve,
            'pct_all_solve': pct_all_solve,
            'mean_sol_complexity': np.mean(sol_complexities),
            'std_sol_complexity': np.std(sol_complexities),
            'mean_edit_dist': np.mean(min_edit_dists),
            'std_edit_dist': np.std(min_edit_dists),
            'skipped': skipped,
        }

    # Now make a pandas dataframe out of this
    # Convert to DataFrame
    df = pd.DataFrame(agg_stats).T

    col_header_remap = {
        'pct_any_solve': '\makecell{Any\\\\Solvable}',
        'pct_all_solve': '\makecell{All\\\\Solvable}',
        'pct_comp': 'Compiles',
    }

    row_index_header_order = ['docs', 'model', 'from_idea', 'fewshot', 'cot']
    row_index_headers_remap = {
        'model': 'Model',
        'cot': 'CoT',
        'docs': 'Docs',
        'fewshot': 'Fewshot',
        'from_idea': 'From Idea',
        'inventive_prompt': 'Inventive Prompt',
        'context_len': 'Context Length',
    }
    row_index_header_order = [row_index_headers_remap.get(k, k) for k in row_index_header_order]

    row_index_remap = {
        True: 'T',
        False: 'F',
        'gpt-4o': 'GPT-4o',
    }

    def format_row_index(n):
        if isinstance(n, int):
            # format with commas to demarcat thousands
            return f"{n:,}"
        return n

    # Extract hierarchical indices
    _hyper_ks = [k for i, k in enumerate(hyper_ks) if k != 'game_i' and len(getattr(hypers, k)) > 1]
    index_tuples = [[getattr(exp_config, k) for k in _hyper_ks] for exp_config in exp_configs]

    # Rename indices
    # index_tuples = [[row_index_rename(n) for n in t] for t in index_tuples]
    index_tuples = [[row_index_remap.get(n, n) for n in t] for t in index_tuples]
    _hyper_ks = [row_index_headers_remap.get(k, k) for k in _hyper_ks]

    r_mi = pd.MultiIndex.from_tuples(index_tuples, names=_hyper_ks)
    row_index_header_order = [k for k in row_index_header_order if k in r_mi.names] + \
                                [k for k in r_mi.names if k not in row_index_header_order]
    r_mi = r_mi.reorder_levels(row_index_header_order)
    df.index = r_mi

    first_any_solve_title = "\makecell{First\\\\Any Solve}"
    first_all_solve_title = "\makecell{First\\\\All Solve}"
    first_compile_title = "\makecell{First\\\\Compile}"

    # Format mean columns to include std as "+/-" values
    df[first_compile_title] = df.apply(
        lambda row: f"{row['mean_first_comp']:.1f} ± {row['std_first_comp']:.1f}", axis=1
    )
    df[first_any_solve_title] = df.apply(
        lambda row: f"{row['mean_first_any_solve']:.1f} ± {row['std_first_any_solve']:.1f}", axis=1
    )
    df[first_all_solve_title] = df.apply(
        lambda row: f"{row['mean_first_all_solve']:.1f} ± {row['std_first_all_solve']:.1f}", axis=1
    )

    df["Sol. Complexity"] = df.apply(
        lambda row: f"{row['mean_sol_complexity']:.1f} ± {row['std_sol_complexity']:.1f}", axis=1
    )

    df["Min. Edit Dist."] = df.apply(
        lambda row: f"{row['mean_edit_dist']:.1f} ± {row['std_edit_dist']:.1f}", axis=1
    )

    # Drop original columns
    df = df.drop(columns=[
        'mean_first_comp', 'std_first_comp', 'mean_first_all_solve', 'std_first_all_solve', 'mean_first_any_solve', 'std_first_any_solve', 
        'mean_edit_dist', 'std_edit_dist', 'skipped', 'mean_sol_complexity', 'std_sol_complexity'
    ])

    # Bold the least values in "First Compile" and "First Solve" columns
    for col in [first_compile_title, first_any_solve_title, first_all_solve_title]:
        min_value = df[col].apply(lambda x: float(x.split(" ± ")[0])).min()
        df[col] = df[col].apply(
            lambda x: f"\\textbf{{{x}}}" if float(x.split(" ± ")[0]) == min_value else x
        )

    # Bold the greatest values in "pct_comp" and "pct_solve" columns
    for col in ["pct_comp", "pct_all_solve", "pct_any_solve"]:
        # Format as percentage, but escape the percent sign
        df[col] = df[col].apply(lambda x: f"{float(x):.0%}".replace("%", "\\%"))

        # Find max value numerically before formatting
        max_value = df[col].apply(lambda x: float(x.replace("\\%", "")) / 100).max()

        df[col] = df[col].apply(
            lambda x: f"\\textbf{{{x}}}" if float(x.replace("\\%", "")) / 100 == max_value else x
        )

    for col in ["Sol. Complexity", "Min. Edit Dist."]:
        max_value = df[col].apply(lambda x: float(x.split(" ± ")[0])).max()
        
        df[col] = df[col].apply(
            lambda x: f"\\textbf{{{float(x.split(' ± ')[0]):,.0f}}} ± {float(x.split(' ± ')[1]):,.0f}"
            if float(x.split(" ± ")[0]) == max_value
            else f"{float(x.split(' ± ')[0]):,.0f} ± {float(x.split(' ± ')[1]):,.0f}"
        )

    # Rename columns to remove underscores
    df = df.rename(columns=lambda x: col_header_remap.get(x, x))
    # df = df.rename(columns=lambda x: x.replace("_", " ").title())

    df = df.sort_index()
    index_tuples, names = df.index.to_flat_index(), df.index.names
    index_tuples = [[format_row_index(n) for n in t] for t in index_tuples]
    df.index = pd.MultiIndex.from_tuples(index_tuples, names=names)

    # Save DataFrame to LaTeX
    # latex_path = "/mnt/data/modified_hierarchical_dataframe.tex"
    latex_path = os.path.join(stats_dir, f'{sweep_name}_{sweep_i}.tex')
    df.to_latex(latex_path, escape=False, multirow=True)
    # df.style.to_latex(latex_path, multirow_align='c', hrules=True, clines='all;data')

    print(json.dumps(agg_stats, indent=4))
    print(f"Saved stats to {latex_path}")


@app.route('/save_game_stats', methods=['POST'])
def save_game_stats():
    data = request.json
    exp_dir, game_dir, stats = data['expDir'], data['gameDir'], data['gameInd']
    return _save_game_stats(exp_dir, game_dir, stats)

def level_to_int_arr(level: dict):
    level_arr = []
    for x in range(level['width']):
        level_arr.append([])
        for y in range(level['height']):
            flat_idx = x * level['height'] + y
            level_arr[x].append(level['dat'][str(flat_idx)])
    return np.array(level_arr)


@app.route('/save_sol', methods=['POST'])
def save_sol():
    data = request.json
    sol_dir, level_i, sol, gif_url, end_state, timeout, obj_list = (
        data['solDir'], data['levelIdx'], data['sol'], data['dataURL'], data['bestState'], 
        data['timeout'], data['objList'])
    won, score = data['won'], data['score']
    os.makedirs(sol_dir, exist_ok=True)
    sol_path = os.path.join(sol_dir, f'level-{level_i}.json')
    end_level_arr = level_to_int_arr(end_state)
    # end_level_arr = to_binary_vectors(end_level_arr.flatten(), 8) 
    end_level_arr
    sol_dict = {
        'won': won,
        'score': score,
        'sol': sol,
        'timeout': timeout,
        'objs': obj_list,
        'state': end_level_arr.tolist(),
    }
    with open(sol_path, 'w') as f:
        json.dump(sol_dict, f, indent=4)
    print(f"Saved solution to {sol_path}")
    gif_data = base64.b64decode(gif_url.split(',')[1])
    gif_path = os.path.join(sol_dir, f'level-{level_i}_sol.gif')
    with open(gif_path, 'wb') as f:
        f.write(gif_data)
    return jsonify({})

def _save_game_stats(exp_dir, game_dir, stats):
    global sweep_stats
    game_dir = os.path.join('logs', game_dir)
    stats_path = os.path.join(game_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    if exp_dir not in sweep_stats:
        sweep_stats[exp_dir] = []
    sweep_stats[exp_dir].append(stats)
    print(f"Saved stats to {stats_path}")
    return jsonify({})

def get_exp_name(exp_config):
    return \
        (f'notInventive_' if not exp_config.inventive_prompt else '') + \
        (f'fromIdea-{int(exp_config.from_idea)}_' if exp_config.from_idea else '') + \
        f'fewshot-{int(exp_config.fewshot)}_cot-{int(exp_config.cot)}' + \
        (f'_{exp_config.model}' if exp_config.model != 'gpt-4o' else '') + \
        (f'_{exp_config.viz_feedback}' if exp_config.viz_feedback else '') + \
        (f'_docs' if exp_config.docs else '') + \
        (f'_ctx-{exp_config.context_len}' if exp_config.context_len != 30_000 else '')

def get_exp_config(hyper_ks, hyper_vals):
    exp_config = ExpConfig()
    for k, v in zip(hyper_ks, hyper_vals):
        setattr(exp_config, k, v)
    return exp_config

@app.route('/get_sweep_args', methods=['GET'])
def get_sweep_args():
    """Sends args to the client-side `genGame` function."""
    global hypers_i, exp_config, sweep_stats
    save_dir = f'sweep-{sweep_i}'
    stats_exist = True
    done = hypers_i == len(hypers_lst)

    while stats_exist and not done:
        done = hypers_i == len(hypers_lst)
        if done:
            _save_sweep_stats(save_dir, sweep_stats)
            return jsonify({'done': True})
        hyper_vals = hypers_lst[hypers_i]
        exp_config = get_exp_config(hypers_ks, hyper_vals)

        exp_dir = os.path.join(
            save_dir, 
            get_exp_name(exp_config)
        )
        game_dir = os.path.join(exp_dir, f'game-{exp_config.seed}')
        stats_path = os.path.join('logs', game_dir, 'stats.json')
        if os.path.exists(stats_path) and not recompute_stats:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                if exp_dir not in sweep_stats:
                    sweep_stats[exp_dir] = []
                sweep_stats[exp_dir].append(stats)
            hypers_i += 1
            print(f"Skipping {game_dir} because stats already exist")
        else:
            stats_exist = False

    if done:
        _save_sweep_stats(save_dir, sweep_stats)
        return jsonify({'done': True})

    hypers_i += 1
    return jsonify({
        'done': done,
        'expDir': exp_dir,
        'gameDir': game_dir,
        'fewshot': exp_config.fewshot,
        'cot': exp_config.cot,
        'gameIdx': exp_config.seed,
        'fromIdea': exp_config.from_idea,
        'fromPlan': False,
    })

def instance_to_dict(instance):
    return {k: v for k, v in inspect.getmembers(instance) if not k.startswith('__') and not inspect.ismethod(k)}

sweep_name = Config.sweep
recompute_stats = Config.recompute_stats

#LLM agents
# 
from LLM_agent import LLMAgent, ReinforcementWrapper, StateVisualizer

# 初始化智能体系统
llm_agent = LLMAgent(model_name="gpt-4o")
rl_wrapper = ReinforcementWrapper(llm_agent)

@app.route('/llm_action', methods=['POST'])
def llm_action():
    try:
        data = request.json
        state_repr = StateVisualizer.render_ascii({
            'entities': llm_agent._extract_entities(data['state'])
        })
        
        # 处理游戏状态
        processed_state = llm_agent.process_state(state_repr)
        
        # 生成决策
        action = llm_agent.choose_action(
            processed_state=processed_state,
            goal=data.get('goal', '')
        )
        
        # 记录历史
        llm_agent.update_history(action, "pending")
        
        # 强化学习更新
        if 'reward' in data:
            rl_wrapper.reinforce(data['reward'])
        
        return jsonify({
            'action': action,
            'state_hash': hash(state_repr),
            'complexity': processed_state['metrics']['complexity']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/agent_history', methods=['GET'])
def get_action_history():
    return jsonify({
        'history': llm_agent.action_history,
        'q_table': rl_wrapper.q_table
    })

@app.route('/retrain', methods=['POST'])
def retrain_agent():
    try:
        training_data = request.json
        # TODO: 实现训练逻辑
        return jsonify({'status': 'training_started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
@app.route('/get_state', methods=['GET'])
def get_state():
    game_hash = request.args.get('game_hash')
    state_hash = request.args.get('state_hash')
    game_level = request.args.get('game_level', '0')  # default to 0

    if not game_hash or not state_hash:
        return jsonify({'error': 'Missing parameters'}), 400

    # 尝试在 init 里找
    init_path = os.path.join('transitions', game_hash, str(game_level), 'images', f'{state_hash}.txt')
    if os.path.isfile(init_path):
        with open(init_path, 'r', encoding='utf-8') as f:
            return jsonify({'state': f.read()})

    # 尝试在 transition 中找（可以拓展）
    # 你可以把 transition 状态文本也存在 images 里或单独的路径
    trans_path = os.path.join('transitions', game_hash, str(game_level), 'images', f'{state_hash}.txt')
    if os.path.isfile(trans_path):
        with open(trans_path, 'r', encoding='utf-8') as f:
            return jsonify({'state': f.read()})

    return jsonify({'error': 'State not found'}), 404


@hydra.main(config_name="config", version_base="1.3")
def main(cfg: Config):
    global global_cfg
    global_cfg = cfg
    global hypers, hypers_ks, hypers_lst, sweep_name, recompute_stats, max_gen_attempts
    max_gen_attempts = cfg.max_gen_attempts
    hypers = all_hypers[cfg.sweep]
    sweep_name = cfg.sweep
    recompute_stats = cfg.recompute_stats
    sweep_dict = instance_to_dict(hypers)
    hypers_ks = list(sweep_dict)
    hypers_lst = list(itertools.product(*sweep_dict.values()))
    save_dir = f'sweep-{sweep_i}'
    stats_dir = os.path.join('logs', save_dir, 'stats', sweep_name)
    if cfg.mode == 'compute_novelty':
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, 'sweep_stats.json')
        compute_edit_distances(stats_path, hypers_ks, hypers_lst)
    elif cfg.mode == 'eval':
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, 'sweep_stats_and_dists.json')
        eval_sweep(stats_path, hypers_ks, hypers_lst)
    elif cfg.mode == 'viz_evo':
        viz_evo_stats()
    # elif cfg.mode == 'generate':
    else:

        if cfg.auto_launch_client:
            browser_thread = threading.Thread(
                target=partial(open_browser, url=f"http://127.0.0.1:{cfg.port}", headless=cfg.headless)
            )
            browser_thread.daemon = True
            browser_thread.start()

        app.run(port=cfg.port)

if __name__ == '__main__':
    main()
