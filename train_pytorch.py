"""PPO training using batched PuzzleScript backends with PyTorch.

Usage:
    python train_pytorch.py backend=cpp game=sokoban_basic level=0 n_envs=64 total_timesteps=5000000
    python train_pytorch.py backend=nodejs game=sokoban_basic level=0 n_envs=64 total_timesteps=5000000
"""
import glob
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import hydra
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from javascript import require
from omegaconf import OmegaConf

from torchinfo import summary as torch_summary
from tqdm import tqdm
import wandb
from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_cpp import CppBatchedPuzzleScriptEnv, CppPuzzleScriptEnv, Renderer
from puzzlescript_nodejs.rl_env import NodeJSBatchedPuzzleEnv
from puzzlescript_nodejs.utils import compile_game


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_JS_PATH = os.path.join(ROOT_DIR, "puzzlescript_nodejs", "puzzlescript", "engine.js")


@dataclass
class TrainPytorchConfig:
    backend: str = "cpp"
    game: str = "sokoban_basic"
    level: int = -1
    max_episode_steps: int = 200
    cpp_num_threads: int = 0

    # PPO hyperparams
    lr: float = 2.5e-4
    n_envs: int = 64
    num_steps: int = 128
    total_timesteps: int = int(5e7)
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    norm_adv: bool = True

    # Architecture
    model: str = "conv"
    hidden_dims: Tuple[int, ...] = (128, 128)

    seed: int = 1

    # Logging
    wandb_mode: str = "online"
    wandb_project: str = "puzzlejax_ppo"
    wandb_entity: str = ""
    log_freq: int = 1
    save_freq: int = 50

    # Rendering
    render_freq: int = 50
    n_render_eps: int = 3
    gif_frame_duration: float = 0.1

    overwrite: bool = False


cs = ConfigStore.instance()
cs.store(name="train_pytorch_config", node=TrainPytorchConfig)
cs.store(name="train_cpp_config", node=TrainPytorchConfig)


def get_exp_dir(cfg: TrainPytorchConfig) -> str:
    if cfg.backend == "cpp":
        thread_slug = f"cpp-threads-{cfg.cpp_num_threads if cfg.cpp_num_threads > 0 else 'auto'}"
        backend_slug = os.path.join("backend-cpp", thread_slug)
    else:
        backend_slug = f"backend-{cfg.backend}"
    return os.path.join(
        "rl_logs_pytorch",
        *(part for part in [backend_slug, cfg.game, f"level-{cfg.level}"] if part),
        f"n-envs-{cfg.n_envs}_{cfg.model}-{'-'.join(str(h) for h in cfg.hidden_dims)}_seed-{cfg.seed}",
    )


def compile_game_json(game: str) -> tuple[str, str]:
    parser = init_ps_lark_parser()
    js_engine = require(ENGINE_JS_PATH)
    compile_game(parser, js_engine, game, 0)
    compiled_json = str(js_engine.serializeCompiledStateJSON())
    sprite_json = str(js_engine.serializeSpriteDataJSON())
    return compiled_json, sprite_json


def find_latest_checkpoint(exp_dir: str) -> Optional[str]:
    pattern = os.path.join(exp_dir, "agent_step*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None

    def _step(path: str) -> int:
        base = os.path.basename(path)
        return int(base.replace("agent_step", "").replace(".pt", ""))

    ckpts.sort(key=_step)
    return ckpts[-1]


def make_renderer(sprite_json: str, json_str: str) -> Renderer:
    renderer = Renderer()
    renderer.load_sprite_data(sprite_json)
    renderer.load_render_config(json_str)
    return renderer


@torch.no_grad()
def render_eval_episodes_cpp(
    agent: nn.Module,
    json_str: str,
    level_i: int,
    max_episode_steps: int,
    n_eps: int,
    renderer: Renderer,
    device: torch.device,
) -> list[list[np.ndarray]]:
    """Returns a list of episodes, each a list of frames."""
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_episode_steps)
    n_objs, h, w = env.observation_shape
    episodes = []
    for _ in range(n_eps):
        if level_i < 0:
            env.set_level(int(np.random.randint(env.num_levels)))
        obs, _info = env.reset()
        renderer.reset_viewport(env._engine.get_width(), env._engine.get_height())
        frames = [renderer.render_engine(env._engine)]
        done = False
        truncated = False
        while not done and not truncated:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            logits, _ = agent(obs_t)
            action = logits.argmax(dim=-1).item()
            obs, _reward, done, truncated, _info = env.step(action)
            frames.append(renderer.render_engine(env._engine))
        episodes.append(frames)
    return episodes


def save_and_log_gif(
    episodes: list[list[np.ndarray]],
    exp_dir: str,
    update: int,
    global_step: int,
    duration: float,
) -> None:
    for ep_i, frames in enumerate(episodes):
        gif_path = os.path.join(exp_dir, f"update-{update}_step-{global_step}_ep-{ep_i}.gif")
        imageio.mimsave(gif_path, frames, duration=duration, loop=0)
        wandb.log({f"video/ep_{ep_i}": wandb.Video(gif_path, format="gif")}, step=global_step)
        print(f"  Saved render gif: {gif_path}")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ConvAgent(nn.Module):
    def __init__(
        self,
        n_objs: int,
        height: int,
        width: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        h1 = hidden_dims[0] if len(hidden_dims) > 0 else 128
        h2 = hidden_dims[1] if len(hidden_dims) > 1 else h1

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(n_objs, h1, kernel_size=7, stride=2, padding=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(h1, h1, kernel_size=7, stride=2, padding=3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_objs, height, width)
            conv_out_size = self.conv(dummy).shape[1]

        self.shared = nn.Sequential(
            layer_init(nn.Linear(conv_out_size, h2)),
            nn.ReLU(),
            layer_init(nn.Linear(h2, h1)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(h1, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(h1, 1), std=1.0)

    def forward(self, x):
        x = x.float()
        h = self.conv(x)
        h = self.shared(h)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_value(self, x):
        x = x.float()
        h = self.conv(x)
        h = self.shared(h)
        return self.critic(h).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class MLPAgent(nn.Module):
    def __init__(
        self,
        n_objs: int,
        height: int,
        width: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        in_dim = n_objs * height * width
        h1 = hidden_dims[0] if len(hidden_dims) > 0 else 128
        h2 = hidden_dims[1] if len(hidden_dims) > 1 else h1

        self.shared = nn.Sequential(
            layer_init(nn.Linear(in_dim, h1)),
            nn.ReLU(),
            layer_init(nn.Linear(h1, h2)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(h2, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(h2, 1), std=1.0)

    def forward(self, x):
        x = x.float().flatten(1)
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_value(self, x):
        x = x.float().flatten(1)
        h = self.shared(x)
        return self.critic(h).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


def create_batched_env(cfg: TrainPytorchConfig):
    if cfg.backend not in {"cpp", "nodejs"}:
        raise ValueError(f"Unsupported backend: {cfg.backend}. Expected one of: cpp, nodejs")

    if cfg.backend == "cpp":
        json_str, sprite_json = compile_game_json(cfg.game)
        env = CppBatchedPuzzleScriptEnv(
            json_str=json_str,
            batch_size=cfg.n_envs,
            level_indices=[cfg.level] * cfg.n_envs,
            max_episode_steps=cfg.max_episode_steps,
            num_threads=cfg.cpp_num_threads,
        )
        return env, {"json_str": json_str, "sprite_json": sprite_json}

    if cfg.backend == "nodejs":
        env = NodeJSBatchedPuzzleEnv(
            game=cfg.game,
            level_i=cfg.level,
            batch_size=cfg.n_envs,
            max_episode_steps=cfg.max_episode_steps,
        )
        json_str, sprite_json = compile_game_json(cfg.game)
        return env, {"json_str": json_str, "sprite_json": sprite_json}


def render_eval(
    cfg: TrainPytorchConfig,
    agent: nn.Module,
    exp_dir: str,
    update: int,
    global_step: int,
    render_ctx: dict[str, object],
    device: torch.device,
) -> None:
    if cfg.backend in {"cpp", "nodejs"}:
        episodes = render_eval_episodes_cpp(
            agent,
            render_ctx["json_str"],
            cfg.level,
            cfg.max_episode_steps,
            cfg.n_render_eps,
            render_ctx["renderer"],
            device,
        )
        if episodes:
            save_and_log_gif(episodes, exp_dir, update, global_step, cfg.gif_frame_duration)
        return

    raise ValueError(f"Unsupported backend: {cfg.backend}")


def train(cfg: TrainPytorchConfig) -> None:
    logging.getLogger().setLevel(logging.WARNING)

    exp_dir = get_exp_dir(cfg)
    if cfg.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backend: {cfg.backend}")
    print(f"Creating batched environment for game: {cfg.game} ...")

    env = None
    try:
        env, render_ctx = create_batched_env(cfg)
        if cfg.backend in {"cpp", "nodejs"}:
            render_ctx["renderer"] = make_renderer(render_ctx["sprite_json"], render_ctx["json_str"])

        obs_shape = env.observation_shape
        n_objs = obs_shape[1]
        height = obs_shape[2]
        width = obs_shape[3]
        n_actions = env.num_actions
        print(f"Obs shape: {obs_shape}, n_actions: {n_actions}")
        if cfg.backend == "cpp" and hasattr(env, "num_threads"):
            print(f"C++ batched env threads: {env.num_threads}")

        batch_size = cfg.n_envs * cfg.num_steps
        minibatch_size = batch_size // cfg.num_minibatches
        num_updates = cfg.total_timesteps // batch_size
        print(f"num_updates={num_updates}, batch_size={batch_size}, minibatch_size={minibatch_size}")

        if cfg.model == "conv":
            agent = ConvAgent(n_objs, height, width, n_actions, cfg.hidden_dims).to(device)
        else:
            agent = MLPAgent(n_objs, height, width, n_actions, cfg.hidden_dims).to(device)
        torch_summary(agent, input_size=(1, n_objs, height, width), dtypes=[torch.float32])

        optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

        start_update = 1
        global_step = 0
        wandb_run_id = None

        ckpt_path = find_latest_checkpoint(exp_dir)
        if ckpt_path is not None:
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            agent.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = ckpt["global_step"]
            start_update = ckpt["update"] + 1
            print(f"  Restored global_step={global_step:,}, resuming from update {start_update}")

            wandb_id_path = os.path.join(exp_dir, "wandb_run_id.txt")
            if os.path.exists(wandb_id_path):
                with open(wandb_id_path, "r") as f:
                    wandb_run_id = f.read().strip()

        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb_mode,
            dir=exp_dir,
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
        )
        with open(os.path.join(exp_dir, "wandb_run_id.txt"), "w") as f:
            f.write(run.id)

        obs_buf = torch.zeros((cfg.num_steps, cfg.n_envs, n_objs, height, width), dtype=torch.uint8)
        actions_buf = torch.zeros((cfg.num_steps, cfg.n_envs), dtype=torch.long)
        logprobs_buf = torch.zeros((cfg.num_steps, cfg.n_envs))
        rewards_buf = torch.zeros((cfg.num_steps, cfg.n_envs))
        dones_buf = torch.zeros((cfg.num_steps, cfg.n_envs))
        values_buf = torch.zeros((cfg.num_steps, cfg.n_envs))

        ep_returns = np.zeros(cfg.n_envs, dtype=np.float32)
        ep_lengths = np.zeros(cfg.n_envs, dtype=np.int32)
        start_time = time.time()

        obs_np = env.reset()
        next_obs = torch.from_numpy(obs_np).to(device)
        next_done = torch.zeros(cfg.n_envs, device=device)

        n_levels = env.num_levels
        csv_path = os.path.join(exp_dir, "progress.csv")
        if start_update == 1:
            win_cols = ",".join(f"level-{i}-win" for i in range(n_levels))
            sol_cols = ",".join(f"level-{i}-min_sol_len" for i in range(n_levels))
            with open(csv_path, "w") as f:
                f.write(f"timestep,ep_return,ep_return_max,ep_length,fps,{win_cols},{sol_cols}\n")

        pbar = tqdm(range(start_update, num_updates + 1), initial=start_update - 1,
                    total=num_updates, desc="Training", unit="update", dynamic_ncols=True, position=0)
        stats_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
        for update in pbar:
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                lr_now = frac * cfg.lr
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

            update_finished_returns = []
            update_finished_lengths = []
            update_finished_levels = []
            update_finished_wins = []

            for step in range(cfg.num_steps):
                global_step += cfg.n_envs
                obs_buf[step] = next_obs.cpu()
                dones_buf[step] = next_done.cpu()

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)

                actions_buf[step] = action.cpu()
                logprobs_buf[step] = logprob.cpu()
                values_buf[step] = value.cpu()

                obs_np, rewards_np, dones_np, _truncated_np, infos = env.step(action.cpu().numpy())
                rewards_buf[step] = torch.from_numpy(rewards_np)

                ep_returns += rewards_np
                ep_lengths += 1
                finished = dones_np
                if np.any(finished):
                    finished_returns = ep_returns[finished]
                    finished_lengths = ep_lengths[finished]
                    finished_levels = np.asarray(infos["level_i"], dtype=np.int32)[finished]
                    finished_wins = np.asarray(infos["won"], dtype=bool)[finished]
                    update_finished_returns.append(finished_returns.copy())
                    update_finished_lengths.append(finished_lengths.copy())
                    update_finished_levels.append(finished_levels.copy())
                    update_finished_wins.append(finished_wins.copy())
                    ep_returns[finished] = 0.0
                    ep_lengths[finished] = 0

                next_obs = torch.from_numpy(obs_np).to(device)
                next_done = torch.from_numpy(dones_np.astype(np.float32)).to(device)

            if (update % cfg.log_freq == 0 or update == 1) and update_finished_returns:
                finished_returns = np.concatenate(update_finished_returns)
                finished_lengths = np.concatenate(update_finished_lengths)
                finished_levels = np.concatenate(update_finished_levels)
                finished_wins = np.concatenate(update_finished_wins)
                mean_ret = finished_returns.mean()
                max_ret = finished_returns.max()
                mean_len = finished_lengths.mean()
                fps = global_step / (time.time() - start_time)
                wandb_payload = {
                    "ep_return": mean_ret,
                    "ep_return_max": max_ret,
                    "ep_length": mean_len,
                    "fps": fps,
                }
                if finished_levels.size > 0:
                    for level_i in np.unique(finished_levels):
                        level_mask = finished_levels == level_i
                        level_returns = finished_returns[level_mask]
                        level_wins = finished_wins[level_mask]
                        wandb_payload[f"level/{int(level_i)}/ep_return"] = float(level_returns.mean())
                        wandb_payload[f"level/{int(level_i)}/win_rate"] = float(level_wins.mean())
                stats_bar.set_description_str(
                    f"  step={global_step:,} ret={mean_ret:.2f}/{max_ret:.2f} "
                    f"len={mean_len:.0f} FPS={fps:,.0f}"
                )
                wandb.log(wandb_payload, step=global_step)

                level_wins = np.zeros(n_levels, dtype=np.int32)
                level_sol_lens = np.full(n_levels, np.nan)
                for level_i in np.unique(finished_levels):
                    lmask = finished_levels == level_i
                    won_mask = lmask & finished_wins.astype(bool)
                    if won_mask.any():
                        level_wins[int(level_i)] = 1
                        level_sol_lens[int(level_i)] = float(finished_lengths[won_mask].min())
                sol_len_strs = ["" if np.isnan(v) else str(int(round(v))) for v in level_sol_lens]
                with open(csv_path, "a") as f:
                    f.write(f"{global_step},{mean_ret},{max_ret},{mean_len},{fps},"
                            + ",".join(str(w) for w in level_wins) + ","
                            + ",".join(sol_len_strs) + "\n")

            with torch.no_grad():
                next_value = agent.get_value(next_obs).cpu()

            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.cpu()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

            b_obs = obs_buf.reshape((-1, n_objs, height, width))
            b_logprobs = logprobs_buf.reshape(-1)
            b_actions = actions_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)

            b_inds = np.arange(batch_size)
            clipfracs = []

            for _epoch in range(cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_obs = b_obs[mb_inds].to(device)
                    mb_actions = b_actions[mb_inds].to(device)

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                    logratio = newlogprob - b_logprobs[mb_inds].to(device)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        clipfracs.append(((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item())

                    mb_advantages = b_advantages[mb_inds].to(device)
                    if cfg.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    mb_returns = b_returns[mb_inds].to(device)
                    mb_values = b_values[mb_inds].to(device)
                    newvalue_clipped = mb_values + (newvalue - mb_values).clamp(-cfg.clip_eps, cfg.clip_eps)
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_loss_clipped = (newvalue_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

            if update % cfg.log_freq == 0:
                y_pred = b_values.numpy()
                y_true = b_returns.numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                wandb.log(
                    {
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "value_loss": v_loss.item(),
                        "policy_loss": pg_loss.item(),
                        "entropy": entropy_loss.item(),
                        "approx_kl": approx_kl,
                        "clipfrac": np.mean(clipfracs),
                        "explained_variance": explained_var,
                    },
                    step=global_step,
                )

            if update % cfg.save_freq == 0 or update == num_updates:
                ckpt_path = os.path.join(exp_dir, f"agent_step{global_step}.pt")
                torch.save(
                    {
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "update": update,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            if cfg.render_freq > 0 and (update == start_update or update % cfg.render_freq == 0 or update == num_updates):
                agent.eval()
                render_eval(cfg, agent, exp_dir, update, global_step, render_ctx, device)
                agent.train()

        wandb.finish()
        print(f"Training complete. Total steps: {global_step:,}")
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


@hydra.main(version_base="1.3", config_path="conf", config_name="train_pytorch")
def main(cfg: TrainPytorchConfig):
    train(cfg)


if __name__ == "__main__":
    main()
