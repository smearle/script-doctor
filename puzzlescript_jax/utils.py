import glob
import importlib
import json
import math
import os
import pickle
import random
import re
import time
from functools import lru_cache
from typing import Any

import dotenv
import jax
from lark import Lark
import numpy as np
from openai import AzureOpenAI
import tiktoken

from puzzlescript_jax.globals import (
    CUSTOM_GAMES_DIR, GAMES_TO_N_RULES_PATH, GAMES_N_RULES_SORTED_PATH, PRIORITY_GAMES, GAMES_N_LEVELS_PATH, LARK_SYNTAX_PATH,
    GAMES_DIR, TREES_DIR, GALLERY_GAMES_DIR, INCREPARE_GAMES_DIR, UNIQUE_INCREPARE_GAMES_PATH,
)
from puzzlescript_jax.env import PuzzleJaxEnv
from puzzlescript_jax.gen_tree import GenPSTree
from puzzlescript_jax.preprocessing import get_tree_from_txt

game_names_remap = {
    'constellationz': 'Constellation Z',
    'limerick': 'Lime Rick',
}

dotenv.load_dotenv()

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
PORTKEY_BASE_URL = os.environ.get("PORTKEY_BASE_URL")
PORTKEY_MODEL_NAMESPACE = os.environ.get("PORTKEY_MODEL_NAMESPACE")

client = None


@lru_cache(maxsize=1)
def get_portkey_client():
    try:
        Portkey = importlib.import_module("portkey_ai").Portkey
    except ImportError as exc:
        raise RuntimeError(
            "portkey_ai is not installed. Install it to query Portkey models."
        ) from exc

    api_key = os.environ.get("PORTKEY_BEARER") or os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PORTKEY_BEARER (or PORTKEY_API_KEY) is not set. Configure it before querying Portkey models."
        )
    return Portkey(base_url=PORTKEY_BASE_URL, api_key=api_key)


def _resolve_portkey_model(model: str) -> str:
    if "/" in model or model.startswith("@"):
        return model
    if PORTKEY_MODEL_NAMESPACE:
        return f"{PORTKEY_MODEL_NAMESPACE}/{model}"
    return model


def _get_portkey_extra_headers(model_alias: str, resolved_model: str) -> dict:
    headers = {}
    portkey_config = os.environ.get("PORTKEY_CONFIG")
    if portkey_config:
        headers["x-portkey-config"] = portkey_config
        return headers

    provider = os.environ.get("PORTKEY_PROVIDER")
    if not provider:
        model_l = resolved_model.lower()
        alias_l = (model_alias or "").lower()
        if "gemini" in model_l or model_l.startswith("@vertexai/") or "vertex" in model_l:
            provider = "vertex-ai"
        elif "claude" in model_l:
            provider = "anthropic"
        else:
            provider = "openai"
        if alias_l in {"gemini", "gemini-2.5-pro"}:
            provider = "vertex-ai"
    headers["x-portkey-provider"] = provider

    virtual_key = None
    alias_l = (model_alias or "").lower()
    if alias_l == "4o-mini":
        virtual_key = os.environ.get("PORTKEY_GPT4O_KEY")
    elif alias_l in {"gemini", "gemini-2.5-pro"}:
        virtual_key = os.environ.get("PORTKEY_VERTEX_KEY")
    elif alias_l == "llama":
        virtual_key = os.environ.get("PORTKEY_LLAMA_KEY")
    else:
        virtual_key = os.environ.get("PORTKEY_O3MINI_KEY")
    if virtual_key:
        headers["x-portkey-virtual-key"] = virtual_key

    return headers


def _extract_portkey_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message")
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    chunks.append(str(item["text"]))
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content or "")

def _strip_thinking_block(text: str) -> str:
    """Strip Qwen3 ``<think>...</think>`` blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _discover_vllm_models(server_url: str, headers: dict, timeout: int) -> list[str]:
    """Return model ids exposed by the vLLM OpenAI-compatible server."""
    import requests as _requests

    try:
        resp = _requests.get(f"{server_url}/models", headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"vLLM model discovery failed (status {resp.status_code}): {resp.text[:500]}")
            return []
        data = resp.json()
        models = data.get("data", [])
        model_ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
        if model_ids:
            print(f"vLLM server exposes models: {', '.join(model_ids)}")
        return model_ids
    except _requests.exceptions.RequestException as e:
        print(f"vLLM model discovery request failed: {e}")
        return []


def _choose_vllm_model(server_url: str, headers: dict, timeout: int,
                       requested_model: str | None) -> str | None:
    """Choose a served model name, preferring the requested one when available."""
    available_models = _discover_vllm_models(server_url, headers, timeout)
    if not available_models:
        return requested_model
    if requested_model in available_models:
        return requested_model
    fallback_model = available_models[0]
    if requested_model:
        print(
            f"Requested model '{requested_model}' is unavailable; "
            f"using served model '{fallback_model}'."
        )
    return fallback_model


def _vllm_text_query(system_prompt, prompt, model_name, base_url=None, max_retries=5,
                     timeout=300, temperature=0.7, max_tokens=4096,
                     enable_thinking=None, strip_thinking=True):
    """
    Query a vLLM server via OpenAI-compatible chat completions API.

    The server URL is resolved in order:
      1. ``base_url`` argument
      2. ``VLLM_BASE_URL`` environment variable
      3. ``http://localhost:8000/v1``

    The model name sent to the server is resolved in order:
      1. ``model_name`` argument
      2. ``VLLM_MODEL`` environment variable

    Parameters:
        system_prompt: System/instruction prompt.
        prompt: User prompt.
        model_name: Model identifier (passed as-is to the server).
        base_url: Optional vLLM server base URL.
        max_retries: Number of retry attempts on transient failures.
        timeout: HTTP request timeout in seconds.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        enable_thinking: Enable Qwen3 thinking mode (True/False/None).
            When *None* the server decides (usually based on chat template).
        strip_thinking: If True (default), remove ``<think>`` blocks from
            the returned text so the caller sees only the final answer.

    Returns:
        Response text or None on failure.
    """
    import requests as _requests

    server_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    server_url = server_url.rstrip("/")
    resolved_model = (
        model_name
        or os.environ.get("VLLM_SERVED_MODEL_NAME", "")
        or os.environ.get("VLLM_MODEL", "")
    )
    if not resolved_model:
        raise ValueError(
            "No model name provided for vLLM backend. Set VLLM_MODEL or pass a "
            "model name like 'vllm-qwen3' / 'vllm-qwen3-30b'."
        )
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    payload: dict = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Qwen3 thinking-mode support (passed as extra_body by OpenAI client,
    # but with raw requests we can merge directly into the payload).
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

    req_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Single-model local vLLM servers are common in this repo. Probe `/models`
    # once up front so alias defaults do not fail when the server was started
    # with a different served model name than the CLI alias implies.
    resolved_model = _choose_vllm_model(server_url, req_headers, timeout, resolved_model)
    payload["model"] = resolved_model

    base_wait = 5
    for attempt in range(max_retries):
        try:
            print(f"Querying vLLM server at {server_url} with model {resolved_model}...")
            resp = _requests.post(
                f"{server_url}/chat/completions",
                headers=req_headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                if strip_thinking:
                    text = _strip_thinking_block(text)
                print("vLLM query completed successfully.")
                return text
            if resp.status_code == 404:
                fallback_model = _choose_vllm_model(server_url, req_headers, timeout, resolved_model)
                if fallback_model and fallback_model != resolved_model:
                    resolved_model = fallback_model
                    payload["model"] = resolved_model
                    continue
            # Log and fall through to retry
            print(f"vLLM request failed (status {resp.status_code}): {resp.text[:500]}")
        except _requests.exceptions.Timeout:
            print(f"vLLM request timed out after {timeout}s.")
        except _requests.exceptions.ConnectionError as e:
            print(f"vLLM connection error (server may be starting): {e}")
        except _requests.exceptions.RequestException as e:
            print(f"vLLM request exception: {e}")

        wait = base_wait * (2 ** min(attempt, 4))
        print(f"Retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(wait)

    print("vLLM: max retries reached. Returning None.")
    return None


# Mapping from CLI vllm aliases to HuggingFace model identifiers.
_VLLM_MODEL_ALIASES = {
    "vllm": None,              # generic – resolved from VLLM_MODEL env var at runtime
    # Qwen3 text-only models
    "vllm-qwen3": "Qwen/Qwen3-8B",
    "vllm-qwen3-4b": "Qwen/Qwen3-4B",
    "vllm-qwen3-8b": "Qwen/Qwen3-8B",
    "vllm_qwen3-8b": "Qwen/Qwen3-8B",
    "vllm-qwen3-30b": "Qwen/Qwen3-30B-A3B",
    "vllm-qwen3-32b": "Qwen/Qwen3-32B",
    "vllm-qwen3.5-27b-fp8": "Qwen/Qwen3.5-27B-FP8",
    # Llama models
    "vllm-llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "vllm-llama3-70b": "meta-llama/Llama-3.1-70B-Instruct",
    # Mistral / Mixtral
    "vllm-mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    # DeepSeek
    "vllm-deepseek": "deepseek-ai/DeepSeek-V3",
    "vllm-deepseek-r1": "deepseek-ai/DeepSeek-R1",
}


def resolve_vllm_model(alias: str) -> str:
    """Return the HF model name for a vllm-* alias, falling back to VLLM_MODEL env."""
    normalized_alias = alias.strip().lower()
    name = _VLLM_MODEL_ALIASES.get(normalized_alias)
    if name is not None:
        return name
    # For the bare 'vllm' alias or unknown sub-aliases, defer to env.
    return os.environ.get("VLLM_SERVED_MODEL_NAME") or os.environ.get("VLLM_MODEL", alias)


def llm_text_query(system_prompt, prompt, model, api_key=None, base_url=None,
                   model_type=None, enable_thinking=None):
    """
    Unified LLM text query interface.

    Supports Portkey-routed API models, direct DeepSeek/Qwen API, and local/remote
    vLLM servers (OpenAI-compatible).

    Parameters:
        system_prompt: System prompt.
        prompt: User prompt.
        model: Model alias. API models: "o3-mini", "4o-mini", "gemini", etc.
            vLLM models: any string starting with "vllm" (e.g. "vllm-qwen3").
        api_key: Unused (kept for back-compat).
        base_url: Optional override for the vLLM server URL.
        model_type: Unused (kept for back-compat).
        enable_thinking: For vLLM Qwen3 models, explicitly enable/disable
            thinking mode.  *None* lets the server decide.

    Returns:
        LLM response text (str), or None on repeated failures.
    """
    # ---- vLLM backend (local / remote open-weight models) ----
    if model.startswith("vllm"):
        resolved = resolve_vllm_model(model)
        return _vllm_text_query(
            system_prompt, prompt, model_name=resolved, base_url=base_url,
            enable_thinking=enable_thinking,
        )

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    model_alias = model
    if model == "gemini":
        model ="gemini-2.0-flash-exp"
    elif model == "gemini-2.5-pro":
        model = "gemini-2.5-pro"
    elif model == "llama":
        model = "@vertexai/meta.llama-3.1-405b-instruct-maas"
    elif model == "deepseek":
        pass  # DeepSeek will be handled separately
    elif model == "qwen":
        pass  # Qwen will be handled separately
    # Try using Portkey API, DeepSeek API or Qwen API
    try:
        import requests

        if model == "deepseek" or model == "deepseek-r1":
            model_name ='deepseek-chat' if model =="deepseek" else 'deepseek-reasoner'
            print(f'Querying DeepSeek API using model {model}...')
            url = "https://api.deepseek.com/chat/completions"
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            payload = {
                "model": model_name, # Or the specific deepseek model name you intend to use
                "messages": messages,
            }
        elif model == "qwen":
            print(f'Querying Qwen API using model qwen-plus...')
            url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            api_key = os.environ.get("QWEN_API_KEY")
            if not api_key:
                raise ValueError("QWEN_API_KEY not found in environment variables.")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            payload = {
                "model": "qwen-plus", # Defaulting to qwen-plus as per provided client code
                "messages": messages,
                # Qwen specific parameters like temperature, max_tokens can be added here if needed
                # "temperature": 0.7, # Example
            }
        else:
            print(f'Querying Portkey using model {model}...')

        max_retries = 80
        retry_count = 0
        base_wait = 10
        max_wait = 300
        portkey_timeout_seconds = 180
        use_portkey = model not in ("deepseek", "deepseek-r1", "qwen")

        if model == "gemini-2.0-flash-exp":
            print("Detected Gemini model, enabling enhanced backoff strategy")
            max_retries = 5
            base_wait = 60

        while True:
            try:
                if model == "deepseek" or model == "deepseek-r1" or model == "qwen":
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                    if response.status_code == 200:
                        response_data = response.json()
                        print('Query completed successfully.')
                        return response_data['choices'][0]['message']['content']

                    # For any other status code (including 429, 502, 504, etc.), log and prepare for a retry.
                    print(f"Request failed with status code: {response.status_code}")
                    print(f"Response text: {response.text}")

                    # Determine wait time
                    # Use Retry-After header if present (common for 429, 503), otherwise exponential backoff.
                    wait_time_default = base_wait * (2 ** retry_count)
                    wait_time = int(response.headers.get('Retry-After', wait_time_default))
                else:
                    resolved_model = _resolve_portkey_model(model)
                    extra_headers = _get_portkey_extra_headers(model_alias, resolved_model)
                    raw_response = get_portkey_client().chat.completions.create(
                        model=resolved_model,
                        messages=messages,
                        extra_headers=extra_headers,
                        timeout=portkey_timeout_seconds,
                    )
                    print('Query completed successfully.')
                    return _extract_portkey_response_text(raw_response)
                
                if use_portkey:
                    print(f"Retrying in {wait_time}s (attempt {retry_count+1}, infinite retries enabled)")
                else:
                    print(f"Retrying in {wait_time}s ({retry_count+1}/{max_retries})")
                time.sleep(wait_time)

            except requests.exceptions.Timeout:
                wait_time = min(base_wait * (2 ** min(retry_count, 6)), max_wait)
                if use_portkey:
                    print(f"Timeout after {portkey_timeout_seconds}s, retrying in {wait_time}s (attempt {retry_count+1}, infinite retries enabled)")
                else:
                    print(f"Timeout, retrying in {wait_time}s ({retry_count+1}/{max_retries})")
                time.sleep(wait_time)

            except requests.exceptions.RequestException as e:
                print(f"Request exception: {e}")
                wait_time = base_wait
                time.sleep(wait_time)

            except Exception as e:
                wait_time = min(base_wait * (2 ** min(retry_count, 6)), max_wait)
                print(f"Exception: {e}")
                print(f"Retrying in {wait_time}s (attempt {retry_count+1}, infinite retries enabled)")
                time.sleep(wait_time)

            retry_count += 1


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


def save_gif_from_states(env, states, save_path, scale=1):
    gif_path = os.path.join(f'{save_path}.gif')
    with imageio.get_writer(gif_path, mode='I', duration=0.1, loop=0) as writer:
        if isinstance(states, list):
            state_iter = iter(states)
        else:
            leaves = jax.tree.leaves(states)
            if not leaves:
                return
            n_states = int(leaves[0].shape[0])
            state_iter = (
                jax.tree.map(lambda x, idx=i: x[idx], states)
                for i in range(n_states)
            )

        for i, state in enumerate(state_iter):
            frame = env.render(state, None)
            frame = np.asarray(jax.device_get(frame), dtype=np.uint8)
            if scale != 1:
                frame = np.repeat(frame, scale, axis=0)
                frame = np.repeat(frame, scale, axis=1)
            writer.append_data(frame)


def load_games_n_rules_sorted():
    with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
        games_n_rules = json.load(f)
    games_n_rules = sorted(games_n_rules, key=lambda x: x[1])
    return games_n_rules


VALID_DATASETS = ("priority", "gallery", "pedro", "increpare")


def _strip_txt(name):
    return name[:-4] if name.endswith('.txt') else name


def _list_dir_games(directory):
    """Return sorted list of game names (no .txt) from a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(_strip_txt(f) for f in os.listdir(directory) if f.endswith('.txt'))


def _load_unique_games_json(path):
    """Load a precomputed list of unique game names from a JSON file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Unique games list not found at {path}. "
            f"Run dedup_games.py first to generate it."
        )
    with open(path, 'r') as f:
        return json.load(f)


def get_list_of_games_for_testing(dataset="pedro", include_random=False, random_order=False):
    """Return an ordered list of game names for the given dataset tier.

    Each tier is a strict superset of the previous one:

        priority  ⊂  gallery  ⊂  pedro  ⊂  increpare

    Note: these are *logical* tiers, not directory names. A tier's games may
    come from any on-disk directory (custom_games/, gallery_games/,
    data/scraped_games/, data/scraped_games_increpare/). For example,
    PRIORITY_GAMES (defined in globals.py) can reference files that live in
    any of these directories.

    Tiers:
        priority  - PRIORITY_GAMES only (a hand-picked list in globals.py)
        gallery   - priority + games from the gallery_games/ directory
        pedro     - gallery + games from data/scraped_games/ (default)
        increpare - pedro + unique (deduplicated) games from
                    data/scraped_games_increpare/
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset: {dataset!r}. Must be one of {VALID_DATASETS}")

    # --- Build the ordered game list by tier ---
    # Start with priority games (always first)
    games = list(PRIORITY_GAMES)
    seen = set(games)

    if dataset == "priority":
        if random_order:
            random.shuffle(games)
        return games

    # Gallery tier
    gallery_game_names = _list_dir_games(GALLERY_GAMES_DIR)
    for g in gallery_game_names:
        if g not in seen:
            games.append(g)
            seen.add(g)

    if dataset == "gallery":
        if random_order:
            random.shuffle(games)
        return games

    # Pedro tier: use complexity ordering from games_n_rules where available
    pedro_game_names = set(_list_dir_games(GAMES_DIR))
    if os.path.isfile(GAMES_N_RULES_SORTED_PATH):
        with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
            games_n_rules = json.load(f)
        # Insert complexity-sorted pedro games (respecting randomness filter)
        for game, n_rules, has_randomness in sorted(games_n_rules, key=lambda x: x[1]):
            if (has_randomness and not include_random):
                continue
            if game not in seen:
                games.append(_strip_txt(game))
                seen.add(_strip_txt(game))
    # Append any remaining pedro games not in games_n_rules
    for g in sorted(pedro_game_names):
        if g not in seen:
            games.append(g)
            seen.add(g)

    if dataset == "pedro":
        if random_order:
            random.shuffle(games)
        return games

    # Increpare tier: only the unique (deduplicated) games
    unique_increpare = _load_unique_games_json(UNIQUE_INCREPARE_GAMES_PATH)
    for g in unique_increpare:
        if g not in seen:
            games.append(g)
            seen.add(g)

    if random_order:
        random.shuffle(games)
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

def init_ps_env(game, level_i, max_episode_steps, vmap: bool = True):
    start_time = timer()
    parser = init_ps_lark_parser()
    tree, success, err_msg = get_tree_from_txt(parser, game, test_env_init=False)
    parse_time = timer()
    # print(f'Parsed PS file using Lark into python PSTree object in {(parse_time - start_time) / 1000} seconds.')
    env = PuzzleJaxEnv(tree, jit=True, level_i=level_i, max_steps=max_episode_steps, print_score=False, debug=False, vmap=vmap)
    # print(f'Initialized PSEnv in {(timer() - parse_time) / 1000} seconds.')
    return env

    
def init_ps_env_from_js(game, level_i, max_episode_steps, vmap: bool = True):
    """Initialize a PuzzleJaxEnv using the JS compiler as the parser.

    This requires the ``javascript`` package and Node.js to be available.
    The JS engine is used only for parsing/compilation; the resulting env
    runs purely in JAX.
    """
    import json
    from pathlib import Path

    start_time = timer()

    # Locate the game file
    game_path = os.path.join(CUSTOM_GAMES_DIR, game + '.txt')
    if not os.path.exists(game_path):
        game_path = os.path.join(GAMES_DIR, game + '.txt')
    with open(game_path, 'r', encoding='utf-8') as f:
        game_text = f.read()

    # Compile via JS
    from javascript import require
    engine_js_path = str(Path(__file__).resolve().parents[1] / "puzzlescript_nodejs" / "puzzlescript" / "engine.js")
    js_engine = require(engine_js_path).createFreshApi()
    js_engine.compile(['restart'], game_text)
    parsed_json = json.loads(str(js_engine.serializeParsedStateJSON()))
    parse_time = timer()

    env = PuzzleJaxEnv.from_js_parsed_state(
        parsed_json, jit=True, level_i=level_i,
        max_steps=max_episode_steps, print_score=False, debug=False, vmap=vmap,
    )
    return env


def init_ps_lark_parser():
    with open(LARK_SYNTAX_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    return parser

    
def level_to_int_arr(level: dict, n_objs: int):
    stride_obj = math.ceil(n_objs / 32)
    level_arr = []
    for x in range(level['width']):
        level_arr.append([])
        for y in range(level['height']):
            val = 0
            flat_idx = (x * level['height'] + y) * stride_obj
            for j in range(stride_obj):
                idx = flat_idx + j
                chunk = int(level['dat'][str(idx)]) & 0xFFFFFFFF
                val |= chunk << (32 * j)
            level_arr[x].append(val)
    if n_objs <= 32:
        dtype = np.int32
    elif n_objs <= 64:
        dtype = np.int64
    else:
        # TODO: Get more clever in order to handle this (if we must)
        raise ValueError(f"Number of objects {n_objs} exceeds 64, cannot convert to int array.")
    level_arr = np.array(level_arr, dtype=dtype)
    return level_arr

def get_n_levels_per_game(games: list[str] | None = None, *, skip_failures: bool = False) -> dict:
    """Return {game_name: n_levels}.

    Historically this function built the mapping by parsing *all* games when the cache
    file was missing. That can fail due to unsupported PuzzleScript features in some
    games (e.g. `rigid` rules) even if the caller only needs a single game.

    Args:
        games: If provided, only compute/return entries for these games.
        skip_failures: If True, skip games that fail to load/transform, printing a
            warning. Useful when enumerating many games.
    """

    cached: dict[str, int] = {}
    if os.path.exists(GAMES_N_LEVELS_PATH):
        with open(GAMES_N_LEVELS_PATH, 'r') as f:
            cached = json.load(f)
        if games is None:
            return cached

    if games is None:
        games = get_list_of_games_for_testing(dataset="pedro")

    # Determine which requested games are missing from cache.
    missing = [g for g in games if g not in cached]
    if not missing:
        return {g: cached[g] for g in games}

    updated = dict(cached)
    for game in missing:
        min_tree_path = os.path.join(TREES_DIR, game + '.pkl')
        if not os.path.exists(min_tree_path):
            msg = f"Missing preprocessed tree for game={game!r}: {min_tree_path}"
            if skip_failures:
                print(f"[get_n_levels_per_game] WARNING: {msg}; skipping.")
                continue
            raise FileNotFoundError(msg)

        try:
            with open(min_tree_path, 'rb') as f:
                tree = pickle.load(f)
            tree = GenPSTree().transform(tree)
            env = PuzzleJaxEnv(tree)
            updated[game] = len(env.levels)
        except Exception as exc:
            msg = f"Failed to compute n_levels for game={game!r}: {exc}"
            if skip_failures:
                print(f"[get_n_levels_per_game] WARNING: {msg}; skipping.")
                continue
            raise

    # Persist updated cache so later runs don't have to re-parse.
    if updated != cached:
        os.makedirs(os.path.dirname(GAMES_N_LEVELS_PATH) or '.', exist_ok=True)
        with open(GAMES_N_LEVELS_PATH, 'w') as f:
            json.dump(updated, f, indent=4)

    return {g: updated[g] for g in games if g in updated}


    
if __name__ == '__main__':
    games_n_levels = get_n_levels_per_game()
    # with open(GAMES_N_RULES_SORTED_PATH, 'r') as f:
    #     games_n_rules = json.load(f)
    # games_to_n_rules = {game: (n_rules, has_randomness) for game, n_rules, has_randomness in games_n_rules}

    # with open(GAMES_TO_N_RULES_PATH, 'w') as f:
    #     json.dump(games_to_n_rules, f, indent=4)
