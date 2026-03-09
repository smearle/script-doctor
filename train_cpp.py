"""PPO training using the C++ PuzzleScript engine (batched) with PyTorch.

Usage:
    python train_cpp.py game=sokoban_basic level=0 n_envs=64 total_timesteps=5000000
"""
import glob
import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, field
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

import wandb
from puzzlejax.utils import init_ps_lark_parser
from puzzlescript_cpp import CppBatchedPuzzleScriptEnv, CppPuzzleScriptEnv, Renderer
from puzzlescript_nodejs.utils import compile_game


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_JS_PATH = os.path.join(ROOT_DIR, "puzzlescript_nodejs", "puzzlescript", "engine.js")


@dataclass
class TrainCppConfig:
    game: str = "sokoban_basic"
    level: int = 0
    max_episode_steps: int = 200

    # PPO hyperparams
    lr: float = 2.5e-4
    n_envs: int = 64
    num_steps: int = 128  # rollout length
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
    model: str = "conv"  # "conv" or "mlp"
    hidden_dims: Tuple[int, ...] = (128, 128)

    seed: int = 1

    # Logging
    wandb_mode: str = "disabled"  # "online", "offline", "disabled"
    wandb_project: str = "puzzlescript_ppo_cpp"
    wandb_entity: str = ""
    log_freq: int = 1  # log every N updates
    save_freq: int = 50  # save checkpoint every N updates

    # Rendering
    render_freq: int = 50  # render gifs every N updates (0 to disable)
    n_render_eps: int = 3  # how many episodes to render
    gif_frame_duration: float = 0.1

    overwrite: bool = False


cs = ConfigStore.instance()
cs.store(name="train_cpp_config", node=TrainCppConfig)


# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

def get_exp_dir(cfg: TrainCppConfig) -> str:
    return os.path.join(
        "rl_logs_cpp",
        cfg.game,
        f"level-{cfg.level}",
        f"n-envs-{cfg.n_envs}_{cfg.model}-{'-'.join(str(h) for h in cfg.hidden_dims)}_seed-{cfg.seed}",
    )


# ---------------------------------------------------------------------------
# Compile game to JSON for the C++ engine
# ---------------------------------------------------------------------------

def compile_game_json(game: str) -> tuple[str, str]:
    """Compile game and return (compiled_json, sprite_data_json)."""
    parser = init_ps_lark_parser()
    js_engine = require(ENGINE_JS_PATH)
    compile_game(parser, js_engine, game, 0)
    compiled_json = str(js_engine.serializeCompiledStateJSON())
    sprite_json = str(js_engine.serializeSpriteDataJSON())
    return compiled_json, sprite_json


# ---------------------------------------------------------------------------
# Checkpoint resume helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(exp_dir: str) -> Optional[str]:
    """Find the latest agent_step*.pt checkpoint in exp_dir."""
    pattern = os.path.join(exp_dir, "agent_step*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    # Extract step numbers and pick the largest
    def _step(p):
        base = os.path.basename(p)
        # agent_step<N>.pt
        return int(base.replace("agent_step", "").replace(".pt", ""))
    ckpts.sort(key=_step)
    return ckpts[-1]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def make_renderer(sprite_json: str) -> Renderer:
    """Create and initialise a C++ Renderer from sprite data JSON."""
    renderer = Renderer()
    renderer.load_sprite_data(sprite_json)
    return renderer


@torch.no_grad()
def render_eval_episodes(
    agent: nn.Module,
    json_str: str,
    level_i: int,
    max_episode_steps: int,
    n_eps: int,
    renderer: Renderer,
    device: torch.device,
) -> list[np.ndarray]:
    """Run n_eps greedy episodes and collect rendered frames."""
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_episode_steps)
    n_objs, h, w = env.observation_shape
    all_frames = []
    for _ in range(n_eps):
        obs, info = env.reset()
        frames = [renderer.render_obs(obs, n_objs, h, w)]
        done = False
        truncated = False
        while not done and not truncated:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            logits, _ = agent(obs_t)
            action = logits.argmax(dim=-1).item()
            obs, reward, done, truncated, info = env.step(action)
            frames.append(renderer.render_obs(obs, n_objs, h, w))
        all_frames.extend(frames)
    return all_frames


def save_and_log_gif(
    frames: list[np.ndarray],
    exp_dir: str,
    update: int,
    global_step: int,
    duration: float,
):
    """Save a gif and log it to wandb."""
    gif_path = os.path.join(exp_dir, f"update-{update}_step-{global_step}.gif")
    imageio.mimsave(gif_path, frames, duration=duration, loop=0)
    wandb.log({"video": wandb.Video(gif_path, format="gif")}, step=global_step)
    print(f"Saved render gif: {gif_path}")


# ---------------------------------------------------------------------------
# PyTorch agent
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ConvAgent(nn.Module):
    """Conv + MLP actor-critic for multihot grid observations."""

    def __init__(self, n_objs: int, height: int, width: int, n_actions: int,
                 hidden_dims: Tuple[int, ...] = (128, 128)):
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

        # Compute conv output size
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
        # x: (batch, n_objs, H, W) uint8 -> float
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
    """Simple MLP actor-critic (flattens observation)."""

    def __init__(self, n_objs: int, height: int, width: int, n_actions: int,
                 hidden_dims: Tuple[int, ...] = (128, 128)):
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


# ---------------------------------------------------------------------------
# PPO Training loop
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="conf", config_name="train_cpp")
def main(cfg: TrainCppConfig):
    logging.getLogger().setLevel(logging.WARNING)

    exp_dir = get_exp_dir(cfg)

    # Handle overwrite
    if cfg.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)

    # Seeding
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Compile game & create batched env ----------------------------
    print(f"Compiling game: {cfg.game} ...")
    json_str, sprite_json = compile_game_json(cfg.game)
    renderer = make_renderer(sprite_json)

    print("Creating batched environment ...")
    level_indices = [cfg.level] * cfg.n_envs
    env = CppBatchedPuzzleScriptEnv(
        json_str=json_str,
        batch_size=cfg.n_envs,
        level_indices=level_indices,
        max_episode_steps=cfg.max_episode_steps,
    )

    obs_shape = env.observation_shape  # (batch, n_objs, H, W)
    n_objs = obs_shape[1]
    height = obs_shape[2]
    width = obs_shape[3]
    n_actions = env.num_actions
    print(f"Obs shape: {obs_shape}, n_actions: {n_actions}")

    # ---- Derive PPO constants -----------------------------------------
    batch_size = cfg.n_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_timesteps // batch_size
    print(f"num_updates={num_updates}, batch_size={batch_size}, minibatch_size={minibatch_size}")

    # ---- Init agent & optimizer ---------------------------------------
    if cfg.model == "conv":
        agent = ConvAgent(n_objs, height, width, n_actions, cfg.hidden_dims).to(device)
    else:
        agent = MLPAgent(n_objs, height, width, n_actions, cfg.hidden_dims).to(device)

    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    # ---- Checkpoint resume --------------------------------------------
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

        # Try to restore wandb run id for seamless resume
        wandb_id_path = os.path.join(exp_dir, "wandb_run_id.txt")
        if os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb_run_id = f.read().strip()

    # ---- WandB --------------------------------------------------------
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb_mode,
        dir=exp_dir,
        name=f"{cfg.game}_l{cfg.level}_s{cfg.seed}",
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
    )
    # Persist wandb run id for future resumes
    with open(os.path.join(exp_dir, "wandb_run_id.txt"), "w") as f:
        f.write(run.id)

    # ---- Storage buffers (on CPU, move to device for updates) ---------
    obs_buf = torch.zeros((cfg.num_steps, cfg.n_envs, n_objs, height, width), dtype=torch.uint8)
    actions_buf = torch.zeros((cfg.num_steps, cfg.n_envs), dtype=torch.long)
    logprobs_buf = torch.zeros((cfg.num_steps, cfg.n_envs))
    rewards_buf = torch.zeros((cfg.num_steps, cfg.n_envs))
    dones_buf = torch.zeros((cfg.num_steps, cfg.n_envs))
    values_buf = torch.zeros((cfg.num_steps, cfg.n_envs))

    # ---- Episode tracking ---------------------------------------------
    ep_returns = np.zeros(cfg.n_envs, dtype=np.float32)
    ep_lengths = np.zeros(cfg.n_envs, dtype=np.int32)

    # ---- Training loop ------------------------------------------------
    start_time = time.time()

    obs_np = env.reset()  # (batch, n_objs, H, W) uint8
    next_obs = torch.from_numpy(obs_np).to(device)
    next_done = torch.zeros(cfg.n_envs, device=device)

    # CSV progress log (append if resuming, otherwise write header)
    csv_path = os.path.join(exp_dir, "progress.csv")
    if start_update == 1:
        with open(csv_path, "w") as f:
            f.write("global_step,ep_return_mean,ep_return_max,ep_length_mean,fps\n")

    for update in range(start_update, num_updates + 1):
        # Anneal learning rate
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            lr_now = frac * cfg.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        # Aggregate episode completions across the full rollout so we log once per update.
        update_finished_returns = []
        update_finished_lengths = []

        # ---- Rollout phase --------------------------------------------
        for step in range(cfg.num_steps):
            global_step += cfg.n_envs
            obs_buf[step] = next_obs.cpu()
            dones_buf[step] = next_done.cpu()

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            actions_buf[step] = action.cpu()
            logprobs_buf[step] = logprob.cpu()
            values_buf[step] = value.cpu()

            # Step the C++ batched env
            obs_np, rewards_np, dones_np, truncated_np, infos = env.step(action.cpu().numpy())

            rewards_buf[step] = torch.from_numpy(rewards_np)

            # Track episode stats
            ep_returns += rewards_np
            ep_lengths += 1
            finished = dones_np
            if np.any(finished):
                finished_returns = ep_returns[finished]
                finished_lengths = ep_lengths[finished]
                update_finished_returns.append(finished_returns.copy())
                update_finished_lengths.append(finished_lengths.copy())

                ep_returns[finished] = 0.0
                ep_lengths[finished] = 0

            next_obs = torch.from_numpy(obs_np).to(device)
            next_done = torch.from_numpy(dones_np.astype(np.float32)).to(device)

        if (update % cfg.log_freq == 0 or update == 1) and update_finished_returns:
            finished_returns = np.concatenate(update_finished_returns)
            finished_lengths = np.concatenate(update_finished_lengths)
            mean_ret = finished_returns.mean()
            max_ret = finished_returns.max()
            mean_len = finished_lengths.mean()
            fps = global_step / (time.time() - start_time)
            print(
                f"update={update}, global_step={global_step:,}, "
                f"ep_return mean={mean_ret:.3f} max={max_ret:.3f}, "
                f"ep_length={mean_len:.1f}, FPS={fps:,.0f}"
            )
            wandb.log({
                "ep_return": mean_ret,
                "ep_return_max": max_ret,
                "ep_length": mean_len,
                "fps": fps,
            }, step=global_step)

            with open(csv_path, "a") as f:
                f.write(f"{global_step},{mean_ret},{max_ret},{mean_len},{fps}\n")

        # ---- Compute GAE & returns ------------------------------------
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

        # ---- Flatten batch --------------------------------------------
        b_obs = obs_buf.reshape((-1, n_objs, height, width))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # ---- PPO update epochs ----------------------------------------
        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, mb_actions
                )
                logratio = newlogprob - b_logprobs[mb_inds].to(device)
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item())

                mb_advantages = b_advantages[mb_inds].to(device)
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
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

        # ---- Logging --------------------------------------------------
        if update % cfg.log_freq == 0:
            y_pred = b_values.numpy()
            y_true = b_returns.numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            wandb.log({
                "learning_rate": optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "approx_kl": approx_kl,
                "clipfrac": np.mean(clipfracs),
                "explained_variance": explained_var,
            }, step=global_step)

        # ---- Checkpoint -----------------------------------------------
        if update % cfg.save_freq == 0 or update == num_updates:
            ckpt_path = os.path.join(exp_dir, f"agent_step{global_step}.pt")
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # ---- Render eval episodes as gifs -----------------------------
        if cfg.render_freq > 0 and (update % cfg.render_freq == 0 or update == num_updates):
            agent.eval()
            frames = render_eval_episodes(
                agent, json_str, cfg.level, cfg.max_episode_steps,
                cfg.n_render_eps, renderer, device,
            )
            agent.train()
            if frames:
                save_and_log_gif(frames, exp_dir, update, global_step, cfg.gif_frame_duration)

    wandb.finish()
    print(f"Training complete. Total steps: {global_step:,}")


if __name__ == "__main__":
    main()
