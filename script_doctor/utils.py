import json
import random

import tiktoken

from script_doctor.prompts import fewshow_examples_prompt


def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



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