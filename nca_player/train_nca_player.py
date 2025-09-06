from abc import ABC, abstractmethod
from copy import copy
from functools import partial
import os
from pdb import set_trace as TT
import shutil
import sys
from typing import OrderedDict

from einops.layers.torch import Reduce
import hydra
from omegaconf import OmegaConf
import torch as th
from torch import nn
from torch import functional as F
import torchinfo
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conf.config import NCAConfig
from gen_nca_targets import PSStateData
from globals import NCA_DATA_DIR
from validate_sols import JS_SOLS_DIR


class PathfindingNN(ABC, nn.Module):
    def __init__(self, cfg: NCAConfig, n_in_chan: int):
        """A Neural Network for pathfinding.
        
        The underlying state of the maze is given through `reset()`, then concatenated with the input at all subsequent
        passes through the network until the next reset.
        """
        nn.Module.__init__(self)
        self.n_step = 0
        self.cfg = cfg

        # The initial/actual state of the board (so that we can use walls to simulate path-flood).
        self.initial_level = None

        # Reserve some channels for concatenating the input maze if using skip connections.
        self.n_out_chan = cfg.n_hid_chan + (n_in_chan if not cfg.skip_connections else 0)


    def add_initial_maze(self, x):
        # FIXME: skip/initial maze and x should be in the same order each time.
        if self.cfg.skip_connections:
            # Concatenate the underlying state of the maze with the input along the channel dimension.
            x = th.cat([self.initial_level, x], dim=1)
        else:
            # Overwrite the additional hidden channels with the underlying state of the maze.
            x[:, self.cfg.n_hid_chan: self.cfg.n_hid_chan + self.initial_level.shape[1]] = self.initial_level

        return x

    def forward(self, x):

        # This forward pass will iterate through a single layer (or all layers if passing dummy input via `torchinfo`).
        for _ in range(1 if not self.is_torchinfo_dummy else self.cfg.n_layers):
            if self.cfg.skip_connections or self.n_step == 0:
                x = self.add_initial_maze(x)

            x = self.forward_layer(x, self.n_step % len(self.layers))
            self.n_step += 1

        return x

    def forward_layer(self, x, i):
        # assert hasattr(self, 'layers'), "Subclass of PathfindingNN must have `layers` attribute."
        x = self.layers[i](x)

        return x

    def reset(self, initial_maze, is_torchinfo_dummy=False, **kwargs):
        """Store the initia maze to concatenate with later activations."""
        self.is_torchinfo_dummy = is_torchinfo_dummy
        self.initial_level = initial_maze
        self.n_step = 0

    def seed(self, batch_size, level_width: int, level_height: int):
        # NOTE: I think the effect of `sparse_update` here is to only update the last layer. Will break non-shared-weight
        #   networks. Should implement this more explicitly in the future.
        width, height = level_width, level_height

        if self.cfg.skip_connections:
            n_chan = self.cfg.n_hid_chan
        else:
            n_chan = self.cfg.n_hid_chan + self.cfg.n_in_chan

        x = th.zeros(batch_size, n_chan, width, height, requires_grad=False)
        # x = th.zeros(batch_size, n_chan, cfg.height + 2, cfg.width + 2, requires_grad=not cfg.sparse_update)

        return x

class NCA(PathfindingNN):
    def __init__(self, cfg: NCAConfig, n_in_chan: int):
        """A Neural Cellular Automata model for pathfinding over grid-based mazes.
        
        Args:
            n_in_chan: Number of input channels in the onehot-encoded input.
            n_hid_chan: Number of channels in the hidden layers.
            drop_diagonals: Whether to drop diagonals in the 3x3 input patches to each conv layer.
            """
        super().__init__(cfg, n_in_chan=n_in_chan)
        n_in_chan, n_hid_chan = n_in_chan, cfg.n_hid_chan
        # Number of hidden channels, also number of writable channels the the output. (None-maze channels at the input.)
        self.n_hid_chan = n_hid_chan    

        conv2d = nn.Conv2d
        max_pool = partial(MaxPools, cfg=cfg)
        
        assert cfg.kernel_size % 2 == 1

        def _make_conv():
            # This layer applies a dense layer to each 3x3 block of the input.
            conv = conv2d(
                n_hid_chan + n_in_chan, 
                # If we're not repeatedly feeding the input maze, replace this with some extra hidden channels.
                self.n_out_chan,
                kernel_size=cfg.kernel_size, 
                padding= (cfg.kernel_size - 1) // 2
            )
            return conv

        supp_modules = ()
        if cfg.max_pool:
            supp_modules = (max_pool,)

        if not cfg.shared_weights:
            modules = [nn.Sequential(OrderedDict([
                (f'conv_{i}', _make_conv()), 
                *[(f'supp_{i}', s()) for i, s in enumerate(supp_modules)],
                (f'relu_{i}', nn.ReLU())])) for i in range(cfg.n_layers - 1)
            ]
            modules += [nn.Sequential(OrderedDict([
                (f'conv_{cfg.n_layers - 1}', _make_conv()),
                *[(f'supp_{cfg.n_layers - 1}', s()) for i, s in enumerate(supp_modules)],
            ]))]
        else:
            conv_0 = _make_conv()
            modules = [nn.Sequential(conv_0, *(s() for s in supp_modules), nn.ReLU()) for _ in range(cfg.n_layers - 1)]
            modules += [nn.Sequential(conv_0, *(s() for s in supp_modules))]

        self.layers = nn.ModuleList(modules)

class MaxPools(nn.Module):
    """Take the max over the entire 2D input, outputting a scalar."""
    def __init__(self, cfg):
        super().__init__()
        # path_chan is where we'll be expecting the model to output the target path.
        self.spatial_pool_chan = cfg.path_chan + 1
        self.chan_pool_chan = cfg.path_chan + 2
        # self.map_pool_layer = nn.MaxPool2d((cfg.width, cfg.height))
        self.chan_pool_layer = Reduce('b c h w -> b 1 h w', 'max')

    def forward(self, x):
        map_kernel_size = x.shape[-2:]
        y_spatial = th.max_pool2d(x[:, self.spatial_pool_chan].clone(), map_kernel_size)
        # y_spatial = nn.MaxPool2d(map_kernel_size)(x[:, self.spatial_pool_chan].clone())
        x[:, self.spatial_pool_chan] += y_spatial
        # chan_kernel_size = (x.shape[1], 1, 1)
        # y_chan = th.max_pool3d(x, chan_kernel_size)
        y_chan = self.chan_pool_layer(x.clone())
        x[:, self.chan_pool_chan: self.chan_pool_chan + 1] = x[:, self.chan_pool_chan: self.chan_pool_chan + 1] + y_chan

        return x


import os
from pdb import set_trace as TT
import pickle
import PIL

from matplotlib import animation, image, pyplot as plt
import numpy as np
# from ray.util.multiprocessing import Pool
from timeit import default_timer as timer
import torch as th
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from tqdm import tqdm
import wandb

# up, down, left, right, action
N_ACTIONS = 5

class Logger():
    def __init__(self):
        self.loss_log = []
        self.discrete_loss_log = []
        self.val_stats_log = {}
        self.n_step = 0

    def log(self, loss, discrete_loss):
        """Log training loss (once per update step)."""
        self.loss_log.append(loss)
        self.discrete_loss_log.append(discrete_loss)
        self.n_step += 1

    def log_val(self, val_stats):
        """Log validation loss."""
        self.val_stats_log[self.n_step] = val_stats

    def get_val_stat(self, k):
        return {i: self.val_stats_log[i][k] for i in self.val_stats_log}

def to_target_frames(x):
    return x[:, -N_ACTIONS:, :, :]

def get_discrete_loss(x, target_frames, cfg=None):
    paths = to_target_frames(x).round()
    err = paths - target_frames
    loss = err.square()  #.float().mean()    #+ overflow_loss
    return loss


def get_ce_loss(paths, target_frames, cfg=None):
    loss = nn.BCELoss()(nn.Sigmoid()(paths), target_frames.float())
    return loss


def train(model: PathfindingNN, opt: th.optim.Optimizer, ps_state_data: PSStateData, logger: Logger, cfg: NCAConfig):
    tb_writer = None
    loss_fn = get_ce_loss
    print("Done unpacking mazes.")
    minibatch_size = min(cfg.minibatch_size, cfg.n_data)
    lr_sched = th.optim.lr_scheduler.MultiStepLR(opt, [10000], 0.1)
    logger.last_time = timer()
    multihot_levels = th.Tensor(ps_state_data.multihot_levels)
    target_frames = th.Tensor(ps_state_data.target_frames)
    level_ims = np.array(ps_state_data.level_ims)
    batch_size = multihot_levels.shape[0]
    hid_states = model.seed(batch_size=batch_size, level_width=multihot_levels.shape[2], level_height=multihot_levels.shape[3])


    for i in tqdm(range(logger.n_step, cfg.n_updates)):
        if tb_writer is None:
            tb_writer = SummaryWriter(log_dir=cfg._log_dir)
        with th.no_grad():
            # Randomly select indices of data-points on which to train during this update step (i.e., minibatch)
            replace = batch_size < minibatch_size
            batch_idx = np.random.choice(hid_states.shape[0], minibatch_size, replace=replace)
            render_batch_idx = batch_idx[:cfg.render_minibatch_size]

            # x = th.zeros(x_maze.shape[0], n_chan, x_maze.shape[2], x_maze.shape[3])
            # x[:, :4, :, :] = x_maze
            x0 = multihot_levels[batch_idx].clone()
            target_paths_mini_batch = target_frames[batch_idx]

            e0 = None
            ef = None
            model.reset(x0, e0=e0, edge_feats=ef)

        # TODO: move initial auxiliary state to model? Probably a better way...
        x = hid_states[batch_idx]

        # else:
        # FYI: this is from the differentiating NCA textures notebook. Weird checkpointing behavior indeed! See the
        #   comments about the `sparse_updates` arg. -SE
        # The following line is equivalent to this code:
        # for k in range(cfg.n_layers):
            # x = model(x)
        # It uses gradient checkpointing to save memory, which enables larger
        # batches and longer CA step sequences. Surprisingly, this version
        # is also ~2x faster than a simple loop, even though it performs
        # the forward pass twice!
        # x = th.utils.checkpoint.checkpoint_sequential([model]*cfg.n_layers, 32, x)

        n_subepisodes = cfg.n_layers // cfg.loss_interval
        loss = 0

        for _ in range(n_subepisodes):

            # Hackish way of storing gradient only at last "chunk" (with checkpointing), or not.
            if cfg.sparse_update:
                x = th.utils.checkpoint.checkpoint_sequential([model]*cfg.loss_interval, min(16, cfg.loss_interval), x)
            else:
                for _ in range(cfg.loss_interval):
                    x = model(x)

            # loss += get_mse_loss(x, target_paths_mini_batch)
            out_paths = to_target_frames(x)
            batch_errs = loss_fn(out_paths, target_paths_mini_batch)
            # err = (out_paths - target_paths_mini_batch).square()
            loss = loss + batch_errs.mean()

        loss = loss / n_subepisodes
        discrete_loss = get_discrete_loss(x, target_paths_mini_batch).float().mean()

        with th.no_grad():

            # FIXME: why are we resetting here??
            model.reset(x0, e0=e0, edge_feats=ef)

            loss.backward()

            # for p in model.parameters():
            for name, p in model.named_parameters():
                p.grad /= (p.grad.norm()+1e-8)     # normalize gradients 

            opt.step()
            opt.zero_grad()

            # lr_sched.step()
            tb_writer.add_scalar("training/loss", loss.item(), i)
            if cfg.wandb:
                wandb.log({"training/loss": loss.item()}, step=i)
            logger.log(loss=loss.item(), discrete_loss=discrete_loss.item())
                    
            key_ckp_step = ((i > 0) and (i % 50000 == 0))
            last_step = (i == cfg.n_updates - 1)
            if i % cfg.save_interval == 0 or last_step or key_ckp_step:
                log_dir = cfg._log_dir
                if key_ckp_step:
                    log_dir = os.path.join(cfg._log_dir, f"iter_{i}")
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                save(model, opt, logger, log_dir)
                # print(f'Saved CA and optimizer state dicts and maze archive to {log_dir}')

            if i % cfg.log_interval == 0 or last_step:
                log(logger, lr_sched, cfg)
            
            if i % cfg.log_interval == 0 or last_step:
                render_paths = np.hstack(to_target_frames(x[:cfg.render_minibatch_size]).cpu())
                render_maze_ims = np.hstack(level_ims[render_batch_idx])
                target_path_ims = np.hstack(target_frames[render_batch_idx].cpu())
                breakpoint()
                if cfg.manual_log or last_step or key_ckp_step:
                    vis_train(logger, render_maze_ims, render_paths, target_path_ims, render_batch_idx, cfg._log_dir)
                images = np.vstack((
                    render_maze_ims, 
                    np.tile(render_paths[...,None], (1, 1, 3))*255, 
                    np.tile(target_path_ims[...,None], (1, 1, 3))*255,
                    )) 
                tb_images = images.astype(np.uint8).transpose(2,0,1)
                tb_writer.add_image("examples", np.array(tb_images), i)
                if cfg.wandb:
                    images = wandb.Image(images, caption="Top: Input, Middle: Output, Bottom: Target")
                    wandb.log({"examples": images}, step=i)

def field_to_hsv(probs):
    moves = probs[..., :4]
    action_p = probs[..., 4]
    # vector expectation of direction (soft)
    # map [down,left,right,up] -> vectors
    dirs = np.array([[0,1], [-1,0], [1,0], [0,-1]], dtype=np.float32)  # (4,2)
    v = moves @ dirs  # (H,W,2)
    mag = np.linalg.norm(v, axis=-1)  # 0..1
    ang = np.arctan2(-v[...,1], v[...,0])  # [-pi, pi], negative y=up
    hue = (np.degrees(ang) % 360) / 360.0  # 0..1

    # Confidence ideas:
    maxp = probs.max(axis=-1)
    entropy = -np.sum(probs * np.clip(np.log(probs + 1e-9), -50, 50), axis=-1) / np.log(5)
    conf = maxp * (1 - entropy)  # 0..1

    # Map to HSV: hue from direction, saturation/value from confidence
    H = hue
    S = np.clip(mag, 0, 1) * 0.9         # only moves add chroma
    V = 0.3 + 0.7 * conf                 # brighter if confident

    # For Action: desaturate & add center-dot glyph later
    S = np.where(action_p > 0.5, 0.0, S)

    return np.stack([H, S, V], axis=-1), action_p

def log(logger, lr_sched, cfg):
    fps = cfg.n_layers * cfg.minibatch_size * cfg.log_interval / (timer() - logger.last_time)
    print('step_n:', len(logger.loss_log),
        ' loss: {:.6e}'.format(logger.loss_log[-1]),
        ' fps: {:,.2f}'.format(fps), 
        ' lr:', lr_sched.get_last_lr(), # end=''
        )

def vis_train(logger, render_maze_ims, render_path_ims, target_path_ims, render_batch_idx, log_dir):
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplot(411)
    # smooth_loss_log = smooth(logger.loss_log, 10)
    loss_log = np.array(logger.loss_log)
    discrete_loss_log = logger.discrete_loss_log
    discrete_loss_log = np.where(np.array(discrete_loss_log) == 0, 1e-8, discrete_loss_log)
    plt.plot(loss_log, '.', alpha=0.1, label='loss')
    plt.plot(discrete_loss_log, '.', alpha=0.1, label='discrete loss')
    val_loss = logger.get_val_stat('losses')
    val_discrete_loss = logger.get_val_stat('disc_losses')
    plt.plot(list(val_loss.keys()), [v[0] for v in list(val_loss.values())], '.', alpha=1.0, label='val loss')
    plt.plot(list(val_discrete_loss.keys()), [v[0] for v in list(val_discrete_loss.values())], '.', alpha=1.0, label='val discrete loss')
    plt.legend()
    plt.yscale('log')
    # plt.ylim(np.min(np.hstack((loss_log, discrete_loss_log))), logger.loss_log[0])
    plt.ylim(np.min(np.hstack((loss_log, discrete_loss_log))), 
             np.max(np.hstack((loss_log, discrete_loss_log, [v[0] for v in list(val_loss.values())]))))
        # imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
        # imshow(np.hstack(imgs))
    plt.subplot(412)
    # Remove ticks and labels.
    plt.xticks([])
    plt.yticks([])
    plt.imshow(render_maze_ims)
    plt.subplot(413)
    plt.imshow(render_path_ims)    #, vmin=-1.0, vmax=1.0)
    plt.subplot(414)
    plt.imshow(target_path_ims)
        # plt.imshow(np.hstack(x[...,-2:-1,:,:].permute([0,2,3,1]).cpu()))
        # plt.imshow(np.hstack(ca.x0[...,0:1,:,:].permute([0,2,3,1]).cpu()))
        # print(f'path activation min: {render_paths.min()}, max: {render_paths.max()}')
    plt.savefig(f'{log_dir}/training_progress.png')
    plt.close()
    logger.last_time = timer()


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def render_trained(model: PathfindingNN, maze_data, cfg: NCAConfig, pyplot_animation=True, name=''):
    """Generate a video showing the behavior of the trained NCA on mazes from its training distribution.
    """
    model.eval()
    mazes_onehot, mazes_discrete, edges, edge_feats, target_paths = maze_data.mazes_onehot, maze_data.mazes_discrete, \
        maze_data.edges, maze_data.edge_feats, maze_data.target_paths
    if cfg.task == 'traveling':
        target_paths = maze_data.target_travelings
    if N_RENDER_CHANS is None:
        n_render_chans = model.n_out_chan
    else:
        n_render_chans = min(N_RENDER_CHANS, model.n_out_chan)

    render_minibatch_size = min(cfg.render_minibatch_size, mazes_onehot.shape[0])
    path_lengths = target_paths.sum((1, 2))
    path_chan = mazes_discrete[0].max() + 1
    mazes_discrete = mazes_discrete.cpu()
    target_paths = target_paths.cpu()
    path_chan = path_chan.cpu()
    # mazes_discrete = th.where((mazes_discrete != Tiles.DEST) & (target_paths == 1), path_chan, mazes_discrete)
    mazes_discrete = th.where((mazes_discrete == Tiles.EMPTY) & (target_paths == 1), path_chan, mazes_discrete)

    # Render most complex mazes first.
    batch_idxs = path_lengths.sort(descending=True)[1]
    # batch_idxs = th.arange(mazes_onehot.shape[0])

    # batch_idx = np.random.choice(mazes_onehot.shape[0], render_minibatch_size, replace=False)
    bi = 0

    global N_RENDER_EPISODES
    if N_RENDER_EPISODES is None:
        N_RENDER_EPISODES = len(batch_idxs)

    model_has_oracle = model == "BfsNCA"

    if RENDER_WEIGHTS:
        for name, p in model.named_parameters():
            if "weight" in name:
                # (in_chan, height, width, out_chan)
                im = p.data.permute(0, 2, 3, 1)
                # (in_chan * height, width, out_chan)
                im = th.vstack([wi for wi in im])
                # (out_chan, width, in_chan * height)
                im = im.permute(2, 1, 0)
                im = th.vstack([wi for wi in im])
                im = (im - im.min()) / (im.max() - im.min())
                # im = PIL.Image.fromarray(im.cpu().numpy() * 255)
                # im.show()
                # im.save(open(os.path.join(cfg.log_dir, 'weights.png')))
        return

    # Render live and indefinitely using cv2.
    if RENDER_TYPE == 0:

        def render_loop_cv2(bi, writer=None):
            # Create large window
            # cv2.namedWindow('maze', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('model', cv2.WINDOW_NORMAL)
            if not cfg.headless:
                cv2.namedWindow('pathfinding', cv2.WINDOW_NORMAL)
            render_dir = os.path.join(cfg._log_dir, f"render{name}")
            if SAVE_PNGS:
                if os.path.exists(render_dir):
                    shutil.rmtree(render_dir)
                os.mkdir(render_dir)
            # cv2.resize('pathfinding', (2000, 2000))
            frame_i = 0

            for ep_i in range(N_RENDER_EPISODES):
                bi, frame_i, x = render_ep_cv2(bi, model, mazes_onehot, target_paths, batch_idxs, n_render_chans, render_dir, cfg, 
                    model_has_oracle=model_has_oracle, writer=writer, edges=edges, edge_feats=edge_feats, frame_i=frame_i)
            
        if SAVE_GIF:
            with imageio.get_writer(os.path.join(cfg._log_dir, f"behavior{name}.gif"), mode='I', duration=0.1) as writer:
                render_loop_cv2(bi, writer)
        else:
            render_loop_cv2(bi)


    # Save a video with pyplot.
    elif RENDER_TYPE == 1:
        plt.axis('off')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10), gridspec_kw={'width_ratios': [1, 10]})
        

        for i in range(N_RENDER_EPISODES):
            x, maze_imgs, bi = reset_ep_render(bi, model, mazes_onehot, target_paths, batch_idxs, render_minibatch_size, 
                cfg, edges, edge_feats)
            bw_imgs = get_imgs(x, n_render_chans, cfg)
            global im1, im2
            mat1 = ax1.matshow(maze_imgs)
            mat2 = ax2.matshow(bw_imgs)
            plt.tight_layout()

            # def init():
            #     im1 = ax1.imshow(maze_imgs)
            #     im2 = ax2.imshow(bw_imgs)

            #     return im1, im2

            xs = []
            oracle_outs = []

            xs.append(x)
            oracle_outs.append(model.oracle_out) if model_has_oracle else None

            for j in range(cfg.n_layers):
                x = model.forward(x)
                xs.append(x)
                oracle_outs.append(model.oracle_out) if oracle_outs else None


        # FIXME: We shouldn't need to call `imshow` from scratch here!!
        def animate(i):
            xi = xs[i]
            oracle_out = oracle_outs[i] if oracle_outs else None
            bw_imgs = get_imgs(xi, n_render_chans, cfg, oracle_out=oracle_out)
            if i % cfg.n_layers == 0:
                # ax1.imshow(maze_imgs)
                mat1.set_array(bw_imgs)
            # ax2.imshow(bw_imgs)
            mat2.set_array(bw_imgs)
            # ax1.figure.canvas.draw()
            # ax2.figure.canvas.draw()

            return mat1, mat2,


        anim = animation.FuncAnimation(
        fig, animate, interval=1, frames=cfg.n_layers, blit=True, save_count=50)
        anim.save(f'{cfg._log_dir}/path_nca_anim{name}.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

        
def load_dataset(cfg: NCAConfig):
    game_dir = os.path.join(NCA_DATA_DIR, cfg.game)
    level_pkl_path = os.path.join(game_dir, f"level-{cfg.level}.pkl")
    with open(level_pkl_path, "rb") as f:
        ps_data = pickle.load(f)
    return ps_data


ca_state_fname = 'ca_state_dict.pt'
opt_state_fname = 'opt_state_dict.pt'
logger_fname = 'logger.pk'


def backup_file(fname):
    if os.path.isfile(fname):
        shutil.copyfile(fname, fname + '.bkp')


def delete_backup(fname):
    if os.path.isfile(fname + '.bkp'):
        os.remove(fname + '.bkp')


def save(ca, opt, logger, log_dir):
    model_path = f'{log_dir}/{ca_state_fname}'
    optimizer_path = f'{log_dir}/{opt_state_fname}'
    backup_file(model_path)
    backup_file(optimizer_path)
    th.save(ca.state_dict(), model_path)
    th.save(opt.state_dict(), optimizer_path)
    delete_backup(model_path)
    delete_backup(optimizer_path)
    with open(f'{log_dir}/{logger_fname}', 'wb') as f:
        pickle.dump(logger, f)


def load(model, opt, cfg):
    if not th.cuda.is_available():
        map_location = th.device('cpu')
    else:
        map_location = None
    try:
        model.load_state_dict(th.load(f'{cfg.log_dir}/{ca_state_fname}', map_location=map_location))
        opt.load_state_dict(th.load(f'{cfg.log_dir}/{opt_state_fname}', map_location=map_location))
    except Exception:  #FIXME: lol
        model.load_state_dict(th.load(f'{cfg.log_dir}/{opt_state_fname}', map_location=map_location))
        opt.load_state_dict(th.load(f'{cfg.log_dir}/{ca_state_fname}', map_location=map_location))
    logger = pickle.load(open(f'{cfg.log_dir}/{logger_fname}', 'rb'))
    print(f'Loaded CA and optimizer state dict, maze archive, and logger from {cfg.log_dir}.')


    return model, opt, logger


@hydra.main(config_path=None, config_name="nca_config")
def main_experiment(cfg: NCAConfig = None, cfg_path: str = None):
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # TODO: Re-enable running this script/function directly? (i.e. main hydra.main.) Currently always going through 
    #   `run_batch.py`.
    cfg._exp_name = f"{cfg.game}_lr-{cfg.lr}_layers-{cfg.n_layers}_hid-{cfg.n_hid_chan}_s-{cfg.seed}"
    cfg._log_dir = os.path.join('logs_nca', cfg._exp_name)
    # Validate and set full experiment name if this has not been done already (i.e. if running this script directly or 
    # launching via SLURM).
    # if not hasattr(load_cfg, 'full_exp_name'):
        # cfg = Config()
        # [setattr(cfg, k, v) for k, v in vars(load_cfg).items() if not k.startswith('_')]
        # cfg.set_exp_name()
    os.system('nvidia-smi -L')
    if th.cuda.is_available():
            print('Using GPU/CUDA.')
            th.set_default_tensor_type('torch.cuda.FloatTensor')
            cfg.device = "cuda"
    else:
            print('Not using GPU/CUDA, using CPU.')
        
    print(f"Running experiment with config:\n {OmegaConf.to_yaml(cfg)}")
    
    print("Loading data...")
    try:
        ps_data_train = load_dataset(cfg)
    except FileNotFoundError as e:
        print("No maze data files found. Run `python mazes.py` to generate the dataset.")
        raise
    print("Done loading data.")


    multihot_levels = ps_data_train.multihot_levels
    level_ims = np.array(ps_data_train.level_ims)
    target_frames = ps_data_train.target_frames

    model: PathfindingNN = NCA(cfg, n_in_chan=multihot_levels.shape[1])
    # Set a dummy initial maze state to probe model. Model will default to edges for full grid.
    n_in_chan = multihot_levels.shape[1]
    level_width = multihot_levels.shape[2]
    level_height = multihot_levels.shape[3]
    x0 = th.zeros(cfg.minibatch_size, n_in_chan, level_width, level_height)
    model.reset(x0, is_torchinfo_dummy=True)

    if not (cfg.render or cfg.evaluate):
        dummy_input = model.seed(batch_size=cfg.minibatch_size, level_width=level_width, level_height=level_height)
        torchinfo.summary(model, input_size=dummy_input.shape)

    loaded = False
    if cfg.load:
        try:
            model, opt, logger = load(model, opt, cfg)
            loaded = True
        except FileNotFoundError as e:
            print("Failed to load, with error:\n", e)
            if cfg.evaluate:
                print("Skipping evaluation.")
                return
            else:
                print("Attempting to start experiment from scratch (not overwriting).")

    opt = th.optim.Adam(model.parameters(), cfg.lr)

    if not loaded:
        if cfg.overwrite and os.path.exists(cfg._log_dir):
            shutil.rmtree(cfg._log_dir, ignore_errors=True)
        try:
            os.makedirs(cfg._log_dir)
            logger = Logger()
        except FileExistsError as e:
            raise FileExistsError(f"Experiment log folder {cfg._log_dir} already exists. Use `load=True` or "
            "`overwrite=True` command line arguments to load or overwrite it.")

    if cfg.wandb:
        hyperparam_cfg = {k: v for k, v in vars(cfg).items() if k not in set({'log_dir', 'exp_name'})}
        wandb.login()
        wandb.init(
            project='pathfinding-nca', 
            # name=cfg.exp_name, 
            # id=cfg.exp_name,
            config=hyperparam_cfg,
            # resume="allow" if cfg.load else None,
        )

    # Save a dictionary of the config to a json for future reference.
    # json.dump(cfg.__dict__, open(f'{cfg.log_dir}/config.json', 'w'))
    yaml.dump(OmegaConf.to_yaml(cfg), open(f'{cfg._log_dir}/config.yaml', 'w'))

    # assert cfg.path_chan == mazes_discrete.max() + 1  # Wait we don't even use this??

    if cfg.render:
        with th.no_grad():
            render_trained(model, ps_data_train, cfg)

    else:
        print("Beginning to train.")
        train(model, opt, ps_data_train, logger, cfg)

    if cfg.wandb:
        wandb.finish()


if __name__ == '__main__':
    main_experiment()