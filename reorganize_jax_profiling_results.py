
import json
import os

from globals import JAX_PROFILING_RESULTS_DIR
from profile_rand_jax import get_level_int, get_step_int, get_vmap


results_path = os.path.join('data', 'jax_fps_NVIDIA_GeForce_RTX_4090.json')

with open(results_path, 'r') as file:
    data = json.load(file)

for device, d_data in data.items():
    device_results_dir = os.path.join(JAX_PROFILING_RESULTS_DIR, device)
    os.makedirs(device_results_dir, exist_ok=True)
    for n_step_str, s_data in d_data.items():
        step_results_dir = os.path.join(device_results_dir, n_step_str)
        os.makedirs(step_results_dir, exist_ok=True)
        n_steps = get_step_int(n_step_str)
        for game, g_data in s_data.items():
            game_results_dir = os.path.join(step_results_dir, game)
            os.makedirs(game_results_dir, exist_ok=True)
            for level_str, l_data in g_data.items():
                level_results_path = os.path.join(game_results_dir, f'{level_str}.json')
                json.dump(l_data, open(level_results_path, 'w'), indent=4)
