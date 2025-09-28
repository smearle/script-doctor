import copy
import os

import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np

from conf.config import EnjoyConfig
from puzzlejax.env import PuzzleJaxEnv, PJState
# from envs.probs.problem import get_loss
# from eval import get_eval_name, init_config_for_eval
from purejaxrl.wrappers import LogWrapper
from train import init_checkpointer
from utils_rl import get_env_params_from_config, get_exp_dir, init_network, init_ps_env, init_config


@hydra.main(version_base="1.3", config_path='./conf', config_name='enjoy_config')
def main_enjoy(enjoy_config: EnjoyConfig):
    enjoy_config = init_config(enjoy_config)

    exp_dir = enjoy_config._exp_dir
    if enjoy_config.random_agent:
        # Save the gif of random agent behavior here. For debugging.
        os.makedirs(exp_dir, exist_ok=True)
        steps_prev_complete = 0
    else:
        if not os.path.exists(exp_dir):
            exit(f"Experiment directory {exp_dir} does not exist")
        print(f'Loading checkpoint from {exp_dir}')
        checkpoint_manager, restored_ckpt, wandb_run_id = init_checkpointer(enjoy_config)
        runner_state = restored_ckpt['runner_state']
        network_params = runner_state.train_state.params
        steps_prev_complete = restored_ckpt['steps_prev_complete'] 


    if enjoy_config.render_ims:
        frames_dir = os.path.join(exp_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

    best_frames_dir = os.path.join(exp_dir, 'best_frames')
    os.makedirs(best_frames_dir, exist_ok=True)

    env: PuzzleJaxEnv

    # Preserve config as it was during training, for future reference (i.e. naming output of enjoy/eval)
    train_config = copy.deepcopy(enjoy_config)

    enjoy_config = init_config(enjoy_config)
    env = init_ps_env(enjoy_config)
    env_params = get_env_params_from_config(env, enjoy_config)
    env = LogWrapper(env)
    network = init_network(env, env_params, enjoy_config)

    rng = jax.random.PRNGKey(enjoy_config.eval_seed)
    rng_reset = jax.random.split(rng, enjoy_config.n_enjoy_envs)

    # obs, env_state = env.reset(rng, env_params)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        rng_reset, env_params
    )

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act = jax.random.split(rng)
        if enjoy_config.random_agent:
            action = env.action_space(env_params).sample(rng_act)[None, None, None, None]
        else:
            # obs = jax.tree_map(lambda x: x[None, ...], obs)
            action = network.apply(network_params, obs)[
                0].sample(seed=rng_act)
        rng_step = jax.random.split(rng, enjoy_config.n_enjoy_envs)
        # obs, env_state, reward, done, info = env.step(
        #     rng_step, env_state, action[..., 0], env_params
        # )
        obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            rng_step, env_state, action, env_params

        )
        frames = jax.vmap(env.render, in_axes=(0))(env_state.env_state)
        # frame = env.render(env_state)
        rng = jax.random.split(rng)[0]
        # Can't concretize these values inside jitted function (?)
        # So we add the stats on cpu later (below)
        # frame = render_stats(env, env_state, frame)
        return (rng, obs, env_state), (env_state, reward, done, info, frames)

    step_env = jax.jit(step_env)
    print('Scanning episode steps:')
    _, (states, rewards, dones, infos, frames) = jax.lax.scan(
        step_env, (rng, obs, env_state), None,
        length=enjoy_config.n_eps*env.max_steps)  # *at least* this many eps (maybe more if change percentage or whatnot)
    states: PJState
    n_total_eps = enjoy_config.n_eps * enjoy_config.n_enjoy_envs
    n_wins = min(n_total_eps, int(jnp.sum(dones)) - n_total_eps)
    mean_wins = n_wins / n_total_eps

    # Save a json with stats
    stats = {
        'n_wins': int(n_wins),
        'mean_wins': float(mean_wins),
        'n_eps': enjoy_config.n_eps * enjoy_config.n_enjoy_envs,
    }
    with open(os.path.join(exp_dir, 'eval_stats.json'), 'w') as f:
        import json
        json.dump(stats, f, indent=4)
    
    assert frames.shape[1] == enjoy_config.n_enjoy_envs and frames.shape[0] == enjoy_config.n_eps * env.max_steps, \
        "`frames` has wrong shape"

    # Save gifs.
    print('Rendering frames:')
    for env_idx in range(enjoy_config.n_enjoy_envs):
        # ep_frames = frames[ep_is*env.max_steps:(ep_is+1)*env.max_steps]

        for ep_idx in range(enjoy_config.n_eps):

            net_ep_idx = env_idx * enjoy_config.n_eps + ep_idx

            new_ep_frames = []

            for i in range(ep_idx * env.max_steps, (ep_idx + 1) * env.max_steps):
                frame = frames[i, env_idx]
                
                new_ep_frames.append(frame)

                if enjoy_config.render_ims:
                    # Save frame as png
                    png_name = os.path.join(frames_dir,
                        f"frame_ep-{net_ep_idx}_step-{i}" + \
                        f"{('_randAgent' if enjoy_config.random_agent else '')}.png")
                    imageio.v3.imwrite(png_name, frame)
                    # imageio.imwrite(png_name, frame)
                new_ep_frames.append(frame)

            ep_frames = new_ep_frames

            frame_shapes = [frame.shape for frame in ep_frames]
            max_frame_w, max_frame_h = max(frame_shapes, key=lambda x: x[0])[0], \
                max(frame_shapes, key=lambda x: x[1])[1]
            # Pad frames to be same size
            new_ep_frames = []
            for frame in ep_frames:
                frame = np.pad(frame, ((0, max_frame_w - frame.shape[0]),
                                        (0, max_frame_h - frame.shape[1]),
                                        (0, 0)), constant_values=0)
                frame[:, :, 3] = 255
                new_ep_frames.append(frame)
            ep_frames = new_ep_frames

            # cum_rewards = jnp.cumsum(jnp.array(
            #   rewards[ep_is*env.rep.max_steps:(ep_is+1)*env.rep.max_steps]))
            gif_name = os.path.join(
                f"{exp_dir}",
                f"anim_step-{steps_prev_complete}" + \
                f"_ep-{net_ep_idx}" + \
                f"{('_randAgent' if enjoy_config.random_agent else '')}" + \
                ".gif"
            )
            imageio.v3.imwrite(
                gif_name,
                ep_frames,
                # Not sure why but the frames are too slow otherwise (compared to 
                # when captured in `train.py`). Are we saving extra frames?
                duration=enjoy_config.gif_frame_duration / 2
            )
            print(f'Saved gif to {gif_name}')


if __name__ == '__main__':
    main_enjoy()
