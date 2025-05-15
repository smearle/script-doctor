import glob
import json
import os
import random
import re
import time

import dotenv
import jax
from lark import Lark
import numpy as np
from openai import AzureOpenAI
import requests
import tiktoken

from globals import GAMES_TO_N_RULES_PATH, GAMES_N_RULES_SORTED_PATH
from collect_games import GALLERY_GAMES_DIR
from env import PSEnv
from globals import PRIORITY_GAMES
from preprocess_games import get_tree_from_txt
from prompts import *


dotenv.load_dotenv()

def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_str_to_token_len(string: str, model_name: str, n_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(string)
    truncated_tokens = tokens[:n_tokens]
    truncated_str = encoding.decode(truncated_tokens)
    return truncated_str


def save_prompts(sys_prompt, prompt, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(
            f"SYSTEM PROMPT:\n{sys_prompt}\n\nUSER PROMPT:\n{prompt}"
        )

        
def extract_ps_code(text):
    # User a regular expression to pull out the code block
    code_block = re.search(r'(.*)```plaintext\n(.*)```(.*)', text, re.DOTALL)
    if code_block:
        plaintext = code_block.group(1) + "..." + code_block.group(3)
    else:
        # Match the code block without the final ``` delimiter, in case the the block was never closed for some reason
        code_block = re.search(r'(.*)```plaintext\n(.*)$', text, re.DOTALL)
        plaintext = code_block.group(1)
    if code_block:
        code = code_block.group(2)
        return code, plaintext
    else:
        print("No code block found in text.")
        breakpoint()
        return None, None


def gen_fewshot_examples(system_prompt, prompt, max_tokens):
    # Randomly add fewshot examples to the system prompt (within our token limit)
    with open('example_games.json', 'r') as f:
        example_games = json.load(f)
    n_tokens_avail = max_tokens - num_tokens_from_string(system_prompt, 'gpt-4o')
    fewshot_examples_prompt_i = fewshow_examples_prompt
    last_fewshot_examples_prompt_i = fewshot_examples_prompt_i
    n_games_included = 0
    while num_tokens_from_string(system_prompt + fewshot_examples_prompt_i + prompt, 'gpt-4o') < n_tokens_avail:
        last_fewshot_examples_prompt_i = fewshot_examples_prompt_i
        rand_example_i = random.randint(0, len(example_games) - 1)
        fewshot_examples_prompt_i += '\n```\n' + example_games.pop(rand_example_i) + '\n```\n'
        n_games_included += 1
    fewshot_examples_prompt_i = last_fewshot_examples_prompt_i
    print(f"Number of games included in fewshot examples: {n_games_included-1}")
    if n_games_included == 0:
        fewshow_examples_prompt_i = ''
    return fewshot_examples_prompt_i

def to_binary_vectors(arr_2d, num_bits):
    arr_2d = np.asarray(arr_2d)
    return ((arr_2d[..., None] & (1 << np.arange(num_bits)[::-1])) > 0).astype(int)


GPT4V_ENDPOINT = "https://aoai-physics.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview"
GPT4V_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}

o_endpoint = os.getenv("ENDPOINT_URL", "https://sc-pn-m898m3wl-eastus2.openai.azure.com/")
o_key = os.getenv("O3_MINI_KEY")

client = None

def llm_text_query(system_prompt, prompt, model, api_key=None, base_url=None, model_type=None):
    """
    Use Portkey API to call LLM
    
    Parameters:
        system_prompt: System prompt
        prompt: User prompt

        model: Model name, can be "o3-mini", "4o-mini", "gemini"
        
    Returns:
        LLM response text
    """
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    # Select different virtual keys based on the model parameter
    virtual_key = os.environ.get("PORTKEY_O3MINI_KEY", "")
    if model == "4o-mini":
        virtual_key = os.environ.get("PORTKEY_GPT4O_KEY", "")
    elif model == "gemini":
        virtual_key = os.environ.get("PORTKEY_VERTEX_KEY", "")
        model ="gemini-2.0-flash-exp"
    # Try using Portkey API
    try:
       
            import requests
            import json

            print(f'Querying API using model {model} with virtual key {virtual_key}...')

            # Prepare request
            url = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('PORTKEY_BEARER', '')}",
                "x-portkey-virtual-key": virtual_key
            }

            payload = {
                "model": model,
                "messages": messages
            }

            # Send request, set timeout and retry count
            max_retries = 8   # 增加最大重试次数
            retry_count = 0
            base_wait = 5     # 延长基础等待时间到5秒

            while retry_count < max_retries:
                wait_time = base_wait * (2 ** retry_count)  # 指数退避策略2^n
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=60)

                    # Check response status
                    if response.status_code == 200:
                        response_data = response.json()
                        print('Query completed successfully.')
                        return response_data['choices'][0]['message']['content']
                    elif response.status_code in [429, 504]:  # Rate limit or Gateway Timeout
                        wait_time = int(response.headers.get('Retry-After', base_wait * (2 ** retry_count)))
                        print(f"Rate limited (429), retrying in {wait_time}s ({retry_count+1}/{max_retries})")
                        time.sleep(wait_time)
                        retry_count += 1
                    else:
                        print(f"Request failed with status code: {response.status_code}")
                        print(f"Response text: {response.text}")
                        if response.status_code == 429:  # Special handling for Gemini quota limits
                            time.sleep(30)  # Additional cooldown
                        raise Exception(f"API request failed with status code: {response.status_code}")
                except requests.exceptions.Timeout:
                        print(f"Request timed out, retrying in {wait_time}s ({retry_count+1}/{max_retries})")
                        time.sleep(wait_time)
                        retry_count += 1
                except requests.exceptions.RequestException as e:
                    print(f"Request exception: {e}")
                    retry_count += 1
                    time.sleep(5)  # Wait 5 seconds before retrying

            # 增加指数退避时间并继续重试
            # 针对Vertex AI的配额限制增加更长的退避时间
            if model == "gemini":
                print("检测到Gemini模型配额限制，启用增强退避策略")
                max_retries = 5  # 减少最大重试次数但增加等待时间
                base_wait = 60  # 基础等待时间增加到60秒
                wait_time = base_wait * (2 ** retry_count)
                if retry_count >= max_retries:
                    raise Exception("Vertex AI配额限制已达最大重试次数，请稍后再试或申请增加配额")
                print(f"Gemini配额限制重试 {retry_count}/{max_retries}，等待 {wait_time} 秒")
                time.sleep(wait_time)
                retry_count += 1
            else:
                retry_count += 1
                wait_time = base_wait * (2 ** retry_count)
                print(f"Retry {retry_count} with exponential backoff: {wait_time} seconds")
                time.sleep(wait_time)
            return llm_text_query(system_prompt, prompt, model)

    except ImportError:
        # If portkey is not installed, fall back to original implementation
        print("Portkey not installed, falling back to original implementation")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if model == 'gpt-4o':
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "top_p": 0.95,
            }
            successful_query = False
            while not successful_query:
                try:
                    print('Querying openai...')
                    response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
                    response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
                    successful_query = True
                    print('Query completed.')
                except requests.RequestException as e:
                    print(f"Failed to make the request. RequestException: {e}")
                except requests.HTTPError as e:
                    print(f"HTTPError: {e}")
                time.sleep(5)

            return response.json()['choices'][0]['message']['content']

        else:
            global client
            if client is None:
                client = AzureOpenAI(  
                    azure_endpoint=o_endpoint,  
                    api_key=o_key,  
                    api_version="2024-12-01-preview",
                )
            assert model in ['o1', 'o3-mini']
            deployment = os.getenv('DEPLOYMENT_NAME', model)
            successful_query = False
            while not successful_query:
                print('Querying openai...')
                completion = client.chat.completions.create(  
                    model=deployment,
                    messages=messages,
                    max_completion_tokens=100_000,
                    stop=None,  
                    stream=False
                )
                successful_query = True
            return completion.choices[0].message.content


import imageio
from jax import numpy as jnp


def save_gif_from_states(env, states, save_path):
    frames = jax.vmap(env.render, in_axes=(0, None))(states, None)
    frames = frames.astype(np.uint8)

    scale = 10
    frames = jnp.repeat(frames, scale, axis=1)
    frames = jnp.repeat(frames, scale, axis=2)

    frames_dir = os.path.join(save_path, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    for i, js_frame in enumerate(frames):
        imageio.imsave(os.path.join(frames_dir, f'{i:03d}.png'), js_frame)

    gif_path = os.path.join(f'{save_path}.gif')
    imageio.mimsave(gif_path, frames, duration=0.1, loop=0)


def load_games_n_rules_sorted():
    with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
        games_n_rules = json.load(f)
    games_n_rules = sorted(games_n_rules, key=lambda x: x[1])
    return games_n_rules


def get_list_of_games_for_testing(all_games=True, include_random=False):
    gallery_games = glob.glob(os.path.join(GALLERY_GAMES_DIR, '*.txt'))
    gallery_games = [os.path.basename(g)[:-4] for g in gallery_games]
    if all_games:
        with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
            games_n_rules = json.load(f)
        # Sort so that at the front of the list, we have games from our priority list, then the gallery then the rest of
        # our dataset, with each subset in order of increasing complexity.
        games_in_gallery_n_rules = [(game, game in PRIORITY_GAMES, game in gallery_games, n_rules) 
                                    for game, n_rules, has_randomness in games_n_rules if not has_randomness or include_random]
        games = sorted(games_in_gallery_n_rules, key=lambda x: (not x[1], not x[2], x[3]))
        games = [g[0] for g in games]
    else:
        games = PRIORITY_GAMES
    return games

import subprocess

def get_current_commit_hash():
  """Retrieves the full hash of the current Git commit.

  Returns:
    str: The full commit hash as a string, or None if an error occurs.
  """
  try:
    full_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    full_hash_str = full_hash.decode('utf-8').strip()
    return full_hash_str
  except subprocess.CalledProcessError:
    return None

from timeit import default_timer as timer

def init_ps_env(game, level_i, max_episode_steps):
    start_time = timer()
    with open("syntax.lark", "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    tree, success, err_msg = get_tree_from_txt(parser, game, test_env_init=False)
    parse_time = timer()
    # print(f'Parsed PS file using Lark into python PSTree object in {(parse_time - start_time) / 1000} seconds.')
    env = PSEnv(tree, jit=True, level_i=level_i, max_steps=max_episode_steps, print_score=False, debug=False)
    # print(f'Initialized PSEnv in {(timer() - parse_time) / 1000} seconds.')
    return env

    
if __name__ == '__main__':
    with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
        games_n_rules = json.load(f)
    games_to_n_rules = {game: (n_rules, has_randomness) for game, n_rules, has_randomness in games_n_rules}

    with open(GAMES_TO_N_RULES_PATH, 'w') as f:
        json.dump(games_to_n_rules, f, indent=4)
