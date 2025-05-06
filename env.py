from dataclasses import dataclass
from functools import partial
import os
from timeit import default_timer as timer
from typing import Dict, Iterable, List, Optional, Tuple, Union

import PIL
import chex
from einops import rearrange
import flax
import imageio
import jax
from jax.experimental import checkify
import jax.numpy as jnp
from lark import Lark
import numpy as np

from conf.config import Config
from env_render import render_solid_color, render_sprite
from jax_utils import stack_leaves
from marl.spaces import Box
from parse_lark import get_tree_from_txt
from ps_game import LegendEntry, PSGameTree, Rule, WinCondition
from spaces import Discrete


# Whether to print out a bunch of stuff, etc.
DEBUG = False
# DEBUG = True

# per-object movement forces that can be applied: left, right, up, down, action
N_MOVEMENTS = 5


# @partial(jax.jit, static_argnums=(0))
def disambiguate_meta(obj, detected_meta_objs, pattern_meta_objs, obj_to_idxs):
    """In the right pattern of rules, we may have a meta-object (mapping to a corresponding meta-object in the 
    left pattern). This function uses the `meta_objs` dictionary returned by the detection function to project
    the correct object during rule application."""
    if obj in obj_to_idxs:
        return obj_to_idxs[obj]
    elif obj in detected_meta_objs:
        return detected_meta_objs[obj]
    else:
        # assert obj in pattern_meta_objs, f"Meta-object `{obj}` not found in meta_objs or pattern_meta_objs."
        if obj not in pattern_meta_objs:
            print(f"When compiling a meta-object projection rule, the meta-object `{obj}` in the output pattern is " 
                  "not found in the return of the compiled detection function (either in meta_objs or pattern_meta_objs).")
            breakpoint()
        return pattern_meta_objs[obj]
    # FIXME: as above, but jax (this is broken. is it necessary?)
    # obj_idx = jax.lax.select_n(
    #     jnp.array([obj in obj_to_idxs, obj in meta_objs, obj in pattern_meta_objs]),
    #     jnp.array([obj_to_idxs.get(obj, -1), meta_objs.get(obj, -1), pattern_meta_objs.get(obj, -1)]),
    # )
    # return obj_idx
    

def level_to_multihot(level):
    pass

def assign_vecs_to_objs(collision_layers, atomic_obj_names):
    n_lyrs = len(collision_layers)
    n_objs = len(atomic_obj_names)
    coll_masks = np.zeros((n_lyrs, n_objs), dtype=bool)
    objs_to_idxs = {}
    # vecs = np.eye(n_objs, dtype=np.uint8)
    # obj_vec_dict = {}
    j = 0
    objs = []
    atomic_objs_to_idxs = {obj: i for i, obj in enumerate(atomic_obj_names)}
    for i, layer in enumerate(collision_layers):
        for obj in layer:
            objs.append(obj)
            # obj_vec_dict[obj] = vecs[i]
            if obj in atomic_objs_to_idxs:
                obj_idx = atomic_objs_to_idxs[obj]
                objs_to_idxs[obj] = obj_idx
            else:
                obj_idx = j
                j += 1
            objs_to_idxs[obj] = obj_idx
            coll_masks[i, obj_idx] = 1
    return objs, objs_to_idxs, coll_masks

def process_legend(legend):
    char_legend = {}
    char_legend_inverse = {}
    meta_objs = {}
    conjoined_tiles = {}
    for k, v in legend.items():
        k = k.split(' ')[0]
        v: LegendEntry
        if v.operator is None:
            assert len(v.obj_names) == 1
            v_obj = v.obj_names[0]
            k_obj = k.strip()
            # TODO: instead of this hack, check set of objects to see
            # if k in obj_to_idxs:
            if len(k) == 1:
                char_legend[v_obj] = k_obj
                char_legend_inverse[k_obj] = v_obj
            else:
                meta_objs[k_obj] = [v_obj]
        elif v.operator == 'or':
            meta_objs[k.strip()] = v.obj_names
        elif v.operator == 'and':
            conjoined_tiles[k.strip()] = v.obj_names
        else: raise Exception('Invalid LegendEntry operator.')


    for k, v in meta_objs.items():
        # Replace with actual atomic object name if this value has already been established as a shorthand for
        # an actual atomic object
        v_1 = []
        for v_obj in v:
            if v_obj in char_legend_inverse:
                v_1.append(char_legend_inverse[v_obj])
            else:
                v_1.append(v_obj)
        meta_objs[k] = v_1 
    
    return char_legend, meta_objs, conjoined_tiles

def expand_collision_layers(collision_layers, meta_objs, char_to_obj):
    # Preprocess collision layers to replace joint objects with their sub-objects
    # TODO: could do this more elegantly using `expand_meta_objs`, right?
    # for i, l in enumerate(collision_layers):
    #     j = 0
    #     for o in l:
    #         if o in meta_objs:
    #             subtiles = meta_objs[o]
    #             l = l[:j] + subtiles + l[j+1:]
    #             collision_layers[i] = l
    #             # HACK: we could do this more efficiently
    #             expand_collision_layers(collision_layers, meta_objs=meta_objs)
    #             j += len(subtiles)
    #         else:
    #             j += 1
    cl = []
    for l in collision_layers:
        l_1 = []
        for o in l:
            l_1 += expand_meta_objs([o], meta_objs, char_to_obj)
        cl.append(l_1)
    return cl

def expand_meta_objs(tile_list: List, meta_objs, char_to_obj):
    assert isinstance(tile_list, list), f"tile_list should be a list, got {type(tile_list)}"
    expanded_meta_objs = []
    for mo in tile_list:
        if mo in meta_objs:
            expanded_meta_objs += expand_meta_objs(meta_objs[mo], meta_objs, char_to_obj)
        elif mo in char_to_obj:
            expanded_meta_objs.append(char_to_obj[mo])
        # elif mt in obj_to_idxs:
        else:
            expanded_meta_objs.append(mo)
        # else:
        #     raise Exception(f'Invalid meta-tile `{mt}`.')
    return expanded_meta_objs

def get_meta_channel(lvl, obj_idxs):
    return jnp.any(lvl[jnp.array(obj_idxs)], axis=0)

def compute_manhattan_dists(lvl, src, trg):
    n_cells = np.prod(lvl.shape[1:])
    src_channel = get_meta_channel(lvl, src)
    trg_channel = get_meta_channel(lvl, trg)
    src_coords = jnp.argwhere(src_channel, size=n_cells, fill_value=-1)
    trg_coords = jnp.argwhere(trg_channel, size=n_cells, fill_value=-1)
    src_coords = rearrange(src_coords, 'n c -> n 1 c')
    trg_coords = rearrange(trg_coords, 'n c -> 1 n c')
    dists = jnp.abs(src_coords - trg_coords)
    dists = jnp.sum(dists, axis=-1)
    # Exclude any dists corresponding to cells without src/trg nodes
    dists = jnp.where(jnp.all(src_coords == -1, axis=-1), jnp.nan, dists)
    dists = jnp.where(jnp.all(trg_coords == -1, axis=-1), jnp.nan, dists)
    jax.debug.breakpoint()
    return dists

def compute_sum_of_manhattan_dists(lvl, src, trg):
    dists = compute_manhattan_dists(lvl, src, trg)
    # Get minimum of each source to any target
    dists = jnp.nanmin(dists, axis=0)
    dists = jnp.where(jnp.isnan(dists), 0, dists)
    sum_dist = jnp.sum(dists, axis=0).astype(np.int32)
    return sum_dist

def compute_min_manhattan_dist(lvl, src, trg):
    dists = compute_manhattan_dists(lvl, src, trg)
    dists = jnp.where(jnp.isnan(dists), jnp.inf, dists)
    min_dist = jnp.min(dists, axis=0).astype(np.int32)
    return min_dist

def gen_check_win(win_conditions: Iterable[WinCondition], obj_to_idxs, meta_objs, char_to_obj, jit=True):

    # @partial(jax.jit, static_argnums=(1, 2))
    def check_all(lvl, src, trg):
        src_channel = get_meta_channel(lvl, src)
        trg_channel = get_meta_channel(lvl, trg)
        # There can be no source objects that do not overlap target objects
        win = ~jnp.any(src_channel & ~trg_channel)
        score = jnp.count_nonzero(src_channel & trg_channel)
        heuristic = compute_sum_of_manhattan_dists(lvl, src, trg)
        return win, score, -heuristic

    # @partial(jax.jit, static_argnums=(1, 2))
    def check_some(lvl, src, trg):
        src_channel = get_meta_channel(lvl, src)
        trg_channel = get_meta_channel(lvl, trg)
        win = jnp.any(src_channel & trg_channel)
        score = win.astype(np.int32)
        heuristic = compute_min_manhattan_dist(lvl, src, trg)
        return win, score, -heuristic

    def check_some_exist(lvl, src):
        src_channel = get_meta_channel(lvl, src)
        win = jnp.any(src_channel)
        score = win.astype(np.int32)
        heuristic = score.astype(np.int32)
        return win, score, heuristic

    # @partial(jax.jit, static_argnums=(1,))
    def check_none(lvl, src):
        src_channel = get_meta_channel(lvl, src)
        win = ~jnp.any(src_channel)
        score = -jnp.count_nonzero(src_channel)
        heuristic = score
        return win, score, heuristic

    # @partial(jax.jit, static_argnums=(1,))
    def check_any(lvl, src):
        src_channel = get_meta_channel(lvl, src)
        win = jnp.any(src_channel)
        score = jnp.count_nonzero(src_channel)
        heuristic = compute_min_manhattan_dist(lvl, src, trg)
        return win, score, -heuristic

    funcs = []
    for win_condition in win_conditions:
        src, trg = win_condition.src_obj, win_condition.trg_obj
        if src in obj_to_idxs:
            src = [obj_to_idxs[src]]
        else:
            src_objs = expand_meta_objs([src], meta_objs, char_to_obj)
            src = [obj_to_idxs[obj] for obj in src_objs]
        if trg is not None:
            if trg in obj_to_idxs:
                trg = [obj_to_idxs[trg]]
            else:
                trg_objs = expand_meta_objs([trg], meta_objs, char_to_obj)
                trg = [obj_to_idxs[obj] for obj in trg_objs]
        if win_condition.quantifier == 'all':
            func = partial(check_all, src=src, trg=trg)
        elif win_condition.quantifier in ['some']:
            if trg is not None:
                func = partial(check_some, src=src, trg=trg)
            else:
                func = partial(check_some_exist, src=src)
        elif win_condition.quantifier == 'no':
            func = partial(check_none, src=src)
        elif win_condition.quantifier == 'any':
            func = partial(check_any, src=src)
        else:
            breakpoint()
            raise Exception('Invalid quantifier.')
        funcs.append(func)

    # @partial(jax.jit)
    def check_win(lvl):

        if len(funcs) == 0:
            return False, 0
        
        def apply_win_condition_func(i, lvl):
            # FIXME: can't jit this list of functions... when the funcs are jitted theselves??
            return jax.lax.switch(i, funcs, lvl)

        if jit:
            wins, scores, heuristics = jax.vmap(apply_win_condition_func, in_axes=(0, None))(jnp.arange(len(funcs)), lvl)
        else:
            func_returns = [f(lvl) for f in funcs]
            wins, scores, heuristics = zip(*func_returns)
            wins, scores, heuristics = np.array(wins), np.array(scores), np.array(heuristics)
        return jnp.all(wins), scores.sum(), heuristics.sum()

    return check_win


@flax.struct.dataclass
class ObjFnReturn:
    # detected object/force indices
    detected: jnp.ndarray
    active: bool = False
    force_idx: int = -1
    moving_idx: int = -1
    obj_idx: int = None


@flax.struct.dataclass
class CellFnReturn:
    # A list of indices of objects that were detected
    # Return these so that we can remove them in output cells (before projecting output pattern)
    # (this doesn't include detecting `no` i.e. the absence of an object)
    detected: np.ndarray
    # Return the force index so that, in default, non-direction-specific rules, we can project the correct force to output cells
    force_idx: np.ndarray
    # A dictionary of the objects detected, mapping meta-object names to sub-object indices
    detected_meta_objs: dict
    detected_moving_idx: int = None

@flax.struct.dataclass
class KernelFnReturn:
    detected_meta_objs: dict
    detected_moving_idx: Optional[int] = None

@flax.struct.dataclass
class PatternFnReturn:
    detected_meta_objs: dict
    detected_moving_idx: int = None

def is_rel_force_in_kernel(k):
    for c in k:
        for o in c:
            if o in ['>', 'v', '^', '<']:
                return True
    return False

def gen_subrules_meta(rule: Rule, n_objs, obj_to_idxs, meta_objs, coll_mat, rule_name, char_to_obj, jit=True):
    idxs_to_objs = {v: k for k, v in obj_to_idxs.items()}
    has_right_pattern = len(rule.right_kernels) > 0

    def is_obj_forceless(obj_idx, m_cell):
        # note that `action` does not count as a force
        return jnp.sum(jax.lax.dynamic_slice(m_cell, (n_objs + (obj_idx * N_MOVEMENTS),), (4,))) == 0

    ### Functions for detecting regular atomic objects
    # @partial(jax.jit, static_argnames='obj_idx')
    def detect_obj_in_cell(m_cell, obj_idx):
        # active = m_cell[obj_idx] == 1 & is_obj_forceless(obj_idx, m_cell)
        detected = jnp.zeros_like(m_cell)
        active = m_cell[obj_idx] == 1
        detected = jax.lax.cond(
            active,
            lambda: detected.at[obj_idx].set(1),
            lambda: detected,
        )
        # jax.lax.cond(
        #     active,
        #     lambda: jax.debug.print('detected obj_idx: {obj_idx}', obj_idx=obj_idx),
        #     lambda: None,
        # )
        return ObjFnReturn(active=active, detected=detected, obj_idx=obj_idx)

    # @partial(jax.jit, static_argnames=('obj_idx'))
    def detect_no_obj_in_cell(m_cell, obj_idx):
        active = m_cell[obj_idx] == 0
        detected = jnp.zeros_like(m_cell)
        return ObjFnReturn(active=active, detected=detected, obj_idx=-1)

    # @partial(jax.jit, static_argnames=('obj_idx', 'force_idx'))
    def detect_force_on_obj(m_cell, obj_idx, force_idx):
        obj_is_present = m_cell[obj_idx] == 1
        force_is_present = m_cell[n_objs + (obj_idx * N_MOVEMENTS) + force_idx] == 1
        # force_idx = np.argwhere(m_cell[n_objs + (obj_idx * N_MOVEMENTS):n_objs + (obj_idx * N_MOVEMENTS) + 4] == 1)
        # assert len(force_idx) <= 1
        active = obj_is_present & force_is_present
        is_detected = np.zeros(m_cell.shape, dtype=bool)
        is_detected[obj_idx] = 1
        is_detected[n_objs + (obj_idx * N_MOVEMENTS) + force_idx] = 1
        detected = jax.lax.cond(
            active,
            lambda: is_detected,
            lambda: np.zeros(m_cell.shape, bool),
        )
        obj_idx = jax.lax.select(
            active,
            obj_idx,
            -1,
        )
        force_idx = jax.lax.select(
            active,
            force_idx,
            -1,
        )
        # jax.lax.cond(
        #     active,
        #     lambda: jax.debug.print('detected force_idx {force_idx} on obj_idx: {obj_idx}', obj_idx=obj_idx, force_idx=force_idx),
        #     lambda: None,
        # )
        return ObjFnReturn(active=active, detected=detected, force_idx=force_idx, obj_idx=obj_idx)

    ### Functions for detecting meta-objects
    # @partial(jax.jit, static_argnums=())
    def detect_any_objs_in_cell(m_cell, objs_vec):
        """Given a multi-hot vector indicating a set of objects, return the index of the object contained in this cell."""
        # m_cell_forceless_objs = jax.vmap(is_obj_forceless, in_axes = (0, None))(jnp.arange(n_objs), m_cell)
        # m_cell_forceless = m_cell.at[:n_objs].set(m_cell_forceless_objs * m_cell[:n_objs])
        obj_idx = jnp.argwhere(objs_vec[:n_objs] * m_cell[:n_objs] > 0, size=1, fill_value=-1)[0, 0]
        detected = jnp.zeros(m_cell.shape, bool)
        is_detected = detected.at[obj_idx].set(1)
        active = obj_idx != -1
        # obj_idx = jax.lax.select(
        #     active,
        #     detected_vec_idx,
        #     -1,
        # )
        detected = jax.lax.cond(
            active,
            lambda: is_detected,
            lambda: detected,
        )
        # jax.lax.cond(
        #     active,
        #     lambda: jax.debug.print('detected obj_idx: {obj_idx}', obj_idx=obj_idx),
        #     lambda: None,
        # )
        return ObjFnReturn(active=active, detected=detected, obj_idx=obj_idx)

    # @partial(jax.jit, static_argnums=())
    def detect_no_objs_in_cell(m_cell, objs_vec):
        active = ~jnp.any(objs_vec * m_cell)
        detected = np.zeros(m_cell.shape, bool)
        return ObjFnReturn(active=active, detected=detected, obj_idx=-1)

    # @partial(jax.jit, static_argnames=())
    def detect_force_on_meta(m_cell, obj_idxs, force_idx):
        dummy_force_obj_vec = jnp.zeros(n_objs + n_objs * N_MOVEMENTS, dtype=bool)

        def force_obj_vec_fn(obj_idx):
            force_obj_vec = dummy_force_obj_vec.at[obj_idx].set(1)
            force_obj_vec = force_obj_vec.at[n_objs + obj_idx * N_MOVEMENTS + force_idx].set(1)
            return force_obj_vec
        
        force_obj_vecs = jax.vmap(force_obj_vec_fn)(obj_idxs)

        obj_activations = jnp.sum(jnp.array(force_obj_vecs) * m_cell[None], axis=1) 
        active = jnp.any(obj_activations == 2)

        obj_idxs = jnp.array(obj_idxs)
        obj_idx = jax.lax.select(
            active,
            obj_idxs[jnp.argwhere(obj_activations == 2, size=1)[0][0]],
            -1,
        )
        is_detected = jnp.zeros(m_cell.shape, dtype=bool)
        is_detected = is_detected.at[obj_idx].set(1)
        is_detected = is_detected.at[n_objs + (obj_idx * N_MOVEMENTS) + force_idx].set(1)
        detected = jax.lax.cond(
            active,
            lambda: is_detected,
            lambda: np.zeros(m_cell.shape, dtype=bool),
        )
        force_idx = jax.lax.select(
            active,
            force_idx,
            -1,
        )

        # jax.lax.cond(
        #     active,
        #     lambda: jax.debug.print('active: {active}. detected {detected}', active=active, detected=detected),
        #     lambda: None,
        # )

        # FIXME: wtf?
        active = ((obj_idx != -1) & active)

        return ObjFnReturn(active=active, detected=detected, obj_idx=obj_idx, force_idx=force_idx)

    # @partial(jax.jit, static_argnames=('obj_idxs'))
    def detect_stationary_meta(m_cell, obj_idxs):
        # TODO: vmap this?

        def detect_stationary_obj(m_cell, obj_idx):
            # if not m_cell[obj_idx]:
            #     continue
            # if not is_obj_forceless(obj_idx, m_cell):
            #     continue
            # detected = detected.at[obj_idx].set(1)

            detected = jnp.zeros(m_cell.shape, dtype=bool)

            obj_is_present = m_cell[obj_idx] == 1
            obj_is_forceless = is_obj_forceless(obj_idx, m_cell)
            obj_active = obj_is_present & obj_is_forceless

            detected = jax.lax.select(
                obj_active,
                detected.at[obj_idx].set(1),
                detected,
            )
            # note that this takes the last-detected active sub-object
            active_obj_idx = jax.lax.select(
                obj_active,
                obj_idx,
                -1,
            )
            return detected, active_obj_idx

        detecteds, active_obj_idxs = jax.vmap(detect_stationary_obj, in_axes=(None, 0))(m_cell, obj_idxs)
        detected = jnp.any(detecteds, axis=0)
        active_obj_i = jnp.argwhere(active_obj_idxs != -1, size=1, fill_value=-1)[0][0]
        active_obj_idx = jax.lax.select(
            active_obj_i != -1,
            jnp.array(obj_idxs)[active_obj_i],
            -1,
        )

        active = active_obj_idx != -1
        if DEBUG:
            jax.lax.cond(
                active,
                lambda: jax.debug.print('detected stationary obj_idx: {obj_idx}', obj_idx=active_obj_idx),
                lambda: None,
            )
        return ObjFnReturn(active=active, detected=detected, obj_idx=active_obj_idx)

    # @partial(jax.jit, static_argnums=())
    def detect_moving_meta(m_cell, obj_idxs):
        # TODO: vmap this?
        active_obj_idx = -1
        active_force_idx = -1
        detected = jnp.zeros(m_cell.shape, dtype=bool)

        for obj_idx in obj_idxs:

            obj_is_present = m_cell[obj_idx] == 1
            obj_forces = jax.lax.dynamic_slice(m_cell, (n_objs + (obj_idx * N_MOVEMENTS),), (4,))
            force_idx = jnp.argwhere(obj_forces, size=1, fill_value=-1)[0, 0]
            obj_active = obj_is_present & (force_idx != -1)

            active_detected = detected.at[obj_idx].set(1)
            active_detected = active_detected.at[n_objs + (obj_idx * N_MOVEMENTS) + force_idx].set(1)

            detected = jax.lax.select(
                obj_active,
                active_detected,
                detected,
            )
            # note that this takes the last-detected active sub-object
            active_obj_idx = jax.lax.select(
                obj_active,
                obj_idx,
                active_obj_idx,
            )
            active_force_idx = jax.lax.select(
                obj_active,
                force_idx,
                active_force_idx,
            )
        active = active_obj_idx != -1
        if DEBUG:
            jax.lax.cond(
                active,
                lambda: jax.debug.print('detected stationary obj_idx: {obj_idx}', obj_idx=active_obj_idx),
                lambda: None,
            )
        return ObjFnReturn(active=active, detected=detected, obj_idx=active_obj_idx, moving_idx=active_force_idx)

    dirs_to_force_idx = {
        'up': 3,
        'right': 2,
        'down': 1,
        'left': 0,
    }
    rel_dirs_to_force_idx_offsets = {
        '>': 0,
        'v': 3,
        '^': 1,
        '<': 2,
    }

    def gen_cell_detection_fn(l_cell, right_force_idx):
        """Produce a function to detect whether all objects/conditions in a cell (within a kernel, within the left pattern of a rule) are present.
        So for the rule `[> Player | Crate] -> [> Player | > Crate]`, this will return a function that detects, for the first cell in the left pattern,
        if the player is present, and has a force applied to it.
            l_cell: the kernel cell to detect
            right_force_idx: the force index corresponding to `>` given some rotation
        """
        fns = []
        if DEBUG:
            print('l cell 1:', l_cell)
        l_cell = l_cell.split(' ')
        if DEBUG:
            print('l cell 2:', l_cell)
        no, force, directional_force, stationary, action, moving = False, False, False, False, False, False
        obj_names = []
        for obj in l_cell:
            obj = obj.lower()
            if obj == 'no':
                no = True
            elif obj in ['>', '<', '^', 'v']:
                force = True
                force_idx = (rel_dirs_to_force_idx_offsets[obj] + right_force_idx) % 4
            elif obj in ['up', 'down', 'left', 'right']:
                force = True
                force_idx = dirs_to_force_idx[obj]
            elif obj == 'stationary':
                stationary = True
            elif obj == 'action':
                force = True
                force_idx = 4
            elif obj == 'moving':
                moving = True
            else:
                obj_names.append(obj)
                sub_objs = expand_meta_objs([obj], meta_objs, char_to_obj)
                obj_idxs = np.array([obj_to_idxs[so] for so in sub_objs])
                obj_vec = np.zeros((n_objs + n_objs * N_MOVEMENTS), dtype=bool)
                obj_vec[obj_idxs] = 1
                if obj in char_to_obj:
                    obj = char_to_obj[obj]
                # TODO: we can remove these functions to individual objects and apply the more abstract meta-tile versions instead
                if len(obj_idxs) == 1:
                # if obj in obj_to_idxs:
                    obj_idx = obj_to_idxs[obj]
                    if no:
                        fns.append(partial(detect_no_obj_in_cell, obj_idx=obj_idx))
                        no = False
                    elif force:
                        fns.append(partial(detect_force_on_obj, obj_idx=obj_idx, force_idx=force_idx))
                        force = False
                    elif stationary:
                        fns.append(partial(detect_stationary_meta, obj_idxs=obj_idxs))
                        stationary = False
                    elif moving:
                        fns.append(partial(detect_moving_meta, obj_idxs=obj_idxs))
                    else:
                        fns.append(partial(detect_obj_in_cell, obj_idx=obj_idx))
                elif obj in meta_objs:
                    if no:
                        if DEBUG:
                            print(l_cell)
                        fns.append(partial(detect_no_objs_in_cell, objs_vec=obj_vec))
                        no = False
                    elif force:
                        fns.append(partial(detect_force_on_meta, obj_idxs=obj_idxs, force_idx=force_idx))
                        force = False
                    elif stationary:
                        fns.append(partial(detect_stationary_meta, obj_idxs=obj_idxs))
                        stationary = False
                    elif moving:
                        fns.append(partial(detect_moving_meta, obj_idxs=obj_idxs))
                        moving = False
                    else:
                        fns.append(partial(detect_any_objs_in_cell, objs_vec=obj_vec))
                else:
                    raise Exception(f'Invalid object `{obj}` in rule.')
        
        # @partial(jax.jit)
        def detect_cell(m_cell):

            def apply_cell_fn_switch(i):
                return jax.lax.switch(i, fns, m_cell)

            # detect_obj_outs: List[ObjFnReturn] = [fn(m_cell=m_cell) for fn in fns]
            detect_obj_outs: ObjFnReturn = jax.vmap(apply_cell_fn_switch, in_axes=0)(jnp.arange(len(fns)))
            activated = jnp.all(detect_obj_outs.active, axis=0)
            detected = jnp.any(detect_obj_outs.detected, axis=0)
            force_idx = detect_obj_outs.force_idx[jnp.argwhere(detect_obj_outs.force_idx != -1, size=1)]
            detected_meta_objs = dict(list(zip(obj_names, detect_obj_outs.obj_idx)))
            # detected_meta_objs = {k: fn_out.obj_idx for k, fn_out in zip(obj_names, detect_obj_outs)}

            # for i, detect_obj_out in enumerate(detect_obj_outs):
            #     # if detect_obj_out.obj_idx != -1:
            #     #     jax.debug.print('obj_idx: {obj_idx}', obj_idx=detect_obj_out.obj_idx)
            #     #     detected = detected.at[detect_obj_out.obj_idx].set(1)
            #     detected = detected | detect_obj_out.detected
            #     if detect_obj_out.force_idx is not None:
            #         detected = detected.at[n_objs + N_MOVEMENTS * detect_obj_out.obj_idx + detect_obj_out.force_idx].set(1)
            #         if force_idx is None:
            #             force_idx = detect_obj_out.force_idx

            # NOTE: What to do about multiple `moving` objects?
            # moving_idxs = [f.moving_idx for f in detect_obj_outs if f.moving_idx is not None]
            # detected_moving_idx = moving_idxs[0] if len(moving_idxs) > 0 else None
            detected_moving_idx = detect_obj_outs.moving_idx[jnp.argwhere(detect_obj_outs.moving_idx != -1, size=1, fill_value=-1)]

            # jax.lax.cond(
            #     activated,
            #     lambda: jax.debug.print('activated: {activated}\ndetected: {detected}\ndetect_obj_outs: {detect_obj_outs}', activated=activated, detected=detected, detect_obj_outs=detect_obj_outs),
            #     lambda: None,
            # )

            ret = CellFnReturn(
                detected=detected,
                force_idx=force_idx,
                detected_meta_objs=detected_meta_objs,
                detected_moving_idx=detected_moving_idx,
            )
            # jax.debug.print('detected: {detected}', detected=detected)
            return activated, ret

        return detect_cell

    def remove_colliding_objs(m_cell, obj_idx, coll_mat):
        # If any objects on the same collision layer are present in the cell, remove them
        # FIXME: We should make this collision matrix static...
        coll_mat = jnp.array(coll_mat)
        coll_vec = coll_mat[:, obj_idx]
        coll_vec = coll_vec.at[obj_idx].set(0)
        # print('various shapes lol', m_cell.shape, n_objs, obj_idx, coll_vec.shape, coll_mat.shape)
        m_cell = m_cell.at[:n_objs].set(m_cell[:n_objs] * ~coll_vec)
        return m_cell
    
    if jit:
        remove_colliding_objs = jax.jit(remove_colliding_objs)
    else:
        remove_colliding_objs = remove_colliding_objs

    # @partial(jax.jit, static_argnums=(3))
    def project_obj(m_cell, cell_detect_out: CellFnReturn, pattern_detect_out: PatternFnReturn, obj):
        meta_objs = cell_detect_out.detected_meta_objs
        pattern_meta_objs = pattern_detect_out.detected_meta_objs
        if DEBUG:
            jax.debug.print('meta objs: {meta_objs}', meta_objs=meta_objs)
        obj_idx = disambiguate_meta(obj, meta_objs, pattern_meta_objs, obj_to_idxs)
        if DEBUG:
            jax.debug.print('projecting obj {obj}, disambiguated index: {obj_idx}', obj=obj, obj_idx=obj_idx)
        if not jit:
            if obj_idx == -1:
                breakpoint()
        m_cell = m_cell.at[obj_idx].set(1)
        m_cell = remove_colliding_objs(m_cell, obj_idx, coll_mat)
        return m_cell

    # @partial(jax.jit, static_argnums=(3))
    def project_no_obj(m_cell, cell_detect_out: CellFnReturn, pattern_detect_out: PatternFnReturn, obj):
        meta_objs = cell_detect_out.detected_meta_objs
        pattern_meta_objs = pattern_detect_out.detected_meta_objs
        obj_idx = disambiguate_meta(obj, meta_objs, pattern_meta_objs, obj_to_idxs)
        m_cell = m_cell.at[obj_idx].set(0)
        jax.lax.dynamic_update_slice(
            m_cell, jnp.zeros(n_objs, dtype=bool), (n_objs + obj_idx * N_MOVEMENTS,)
        )
        return m_cell

    # @partial(jax.jit, static_argnums=(3))
    def project_no_meta(m_cell: chex.Array, cell_detect_out, pattern_detect_out, obj: int):
        sub_objs = expand_meta_objs([obj], meta_objs, char_to_obj)
        obj_idxs = np.array([obj_to_idxs[so] for so in sub_objs])
        for obj_idx in obj_idxs:
            m_cell = m_cell.at[obj_idx].set(0)
            jax.lax.dynamic_update_slice(
                m_cell, jnp.zeros(N_MOVEMENTS, dtype=bool), (n_objs + obj_idx * N_MOVEMENTS,)
            )
        return m_cell

    # @partial(jax.jit, static_argnums=(3, 4))
    def project_force_obj(m_cell, cell_detect_out: CellFnReturn, pattern_detect_out: PatternFnReturn, obj, force_idx):
        meta_objs = cell_detect_out.detected_meta_objs
        pattern_meta_objs = pattern_detect_out.detected_meta_objs
        obj_idx = disambiguate_meta(obj, meta_objs, pattern_meta_objs, obj_to_idxs)
        # Add the object
        m_cell = m_cell.at[obj_idx].set(1)
        # Add force to the object
        m_cell = m_cell.at[n_objs + (obj_idx * N_MOVEMENTS) + force_idx].set(1)
        m_cell = remove_colliding_objs(m_cell, obj_idx, coll_mat)

        if jit:
            if DEBUG:
                jax.debug.print('project_force_on_obj: {obj_idx}', obj_idx=obj_idx)

        else:
            pass
            # jax.debug.print('project_force_on_obj: {obj_idx}', obj_idx=obj_idx)
            # TODO: jax this
            # obj_name = idxs_to_objs[obj_idx]
            # jax.debug.print('project_force_on_obj: {obj_name}', obj_name=obj_name)
            # print(f'project_force_on_obj: {obj_idx}')

        return m_cell

    # @partial(jax.jit, static_argnums=(3))
    def project_moving_obj(m_cell, cell_detect_out: CellFnReturn, pattern_detect_out: PatternFnReturn, obj):
        meta_objs = cell_detect_out.detected_meta_objs
        pattern_meta_objs = pattern_detect_out.detected_meta_objs
        obj_idx = disambiguate_meta(obj, meta_objs, pattern_meta_objs, obj_to_idxs)
        if cell_detect_out.detected_moving_idx is not None:
            force_idx = pattern_detect_out.detected_moving_idx
        else:
            assert pattern_detect_out.detected_moving_idx is not None, f'No moving index found in rule {rule_name}'
            force_idx = pattern_detect_out.detected_moving_idx
        m_cell = m_cell.at[obj_idx].set(1)
        m_cell = m_cell.at[n_objs + (obj_idx * N_MOVEMENTS) + force_idx].set(1)
        m_cell = remove_colliding_objs(m_cell, obj_idx, coll_mat)
        return m_cell

    def project_stationary_obj(m_cell, cell_detect_out: CellFnReturn, pattern_detect_out: PatternFnReturn, obj):
        meta_objs = cell_detect_out.detected_meta_objs
        pattern_meta_objs = pattern_detect_out.detected_meta_objs
        obj_idx = disambiguate_meta(obj, meta_objs, pattern_meta_objs, obj_to_idxs)
        m_cell = m_cell.at[obj_idx].set(1)
        m_cell = m_cell.at[n_objs + (obj_idx * N_MOVEMENTS): n_objs + ((obj_idx + 1) * N_MOVEMENTS)].set(1)
        m_cell = remove_colliding_objs(m_cell, obj_idx, coll_mat)
        return m_cell

    def gen_cell_projection_fn(r_cell, right_force_idx):
        fns = []
        if r_cell is None:
            r_cell = []
        else:
            r_cell = r_cell.split(' ')
        no, force, moving, stationary = False, False, False, False
        for obj in r_cell:
            obj = obj.lower()
            if obj in char_to_obj:
                obj = char_to_obj[obj]
            if obj == 'no':
                no = True
            elif obj in ['>', '<', '^', 'v']:
                force = True
                force_idx = (rel_dirs_to_force_idx_offsets[obj] + right_force_idx) % 4
            elif obj in ['up', 'down', 'left', 'right']:
                force = True
                force_idx = dirs_to_force_idx[obj]
            elif obj == 'action':
                force = True
                force_idx = 4
            elif obj == 'moving':
                moving = True
            elif obj == 'stationary':
                stationary = True
            # ignore sound effects (which can exist incide rules (?))
            elif obj.startswith('sfx'):
                continue
            elif (obj in obj_to_idxs) or (obj in meta_objs):
                if no:
                    if obj in obj_to_idxs:
                        fns.append(partial(project_no_obj, obj=obj))
                    elif obj in meta_objs:
                        fns.append(partial(project_no_meta, obj=obj))
                    else:
                        raise Exception(f'Invalid object `{obj}` in rule.')
                    no = False
                elif force:
                    fns.append(partial(project_force_obj, obj=obj, force_idx=force_idx))
                    force = False
                elif moving:
                    fns.append(partial(project_moving_obj, obj=obj))
                    moving = False
                elif stationary:
                    fns.append(partial(project_stationary_obj, obj=obj))
                else:
                    fns.append(partial(project_obj, obj=obj))
            else:
                raise Exception(f'Invalid object `{obj}` in rule.')
        
        # @partial(jax.jit, static_argnums=())
        def project_cell(m_cell, cell_detect_out, pattern_detect_out):
            m_cell = m_cell & ~cell_detect_out.detected
            assert len(m_cell.shape) == 1, f'Invalid cell shape {m_cell.shape}'
            for proj_fn in fns:
                m_cell = proj_fn(m_cell=m_cell, cell_detect_out=cell_detect_out, pattern_detect_out=pattern_detect_out)
            removed_something = jnp.any(cell_detect_out.detected)
            # jax.lax.cond(
            #     removed_something,
            #     lambda: jax.debug.print('removing detected: {det}', det=detect_out.detected),
            #     lambda: None
            # )
            # jax.debug.print('      removing detected: {det}', det=cell_detect_out.detected)
            return m_cell

        return project_cell

    def gen_rotated_rule_fn(lps, rps, rot, r_command):

        def gen_rotated_kernel_fns(lp, rp, rot):
            lp, rp = np.array(lp), np.array(rp)
            # is_horizontal = np.all([lps[i].shape[0] == 1 for i in range(len(lps))])
            # is_vertical = np.all([lps[i].shape[1] == 1 for i in range(len(lps))])
            # is_single = np.all([lps[i].shape[0] == 1 and lps[i].shape[1] == 1 for i in range(len(lps))])
            lp_is_horizontal = lp.shape[0] == 1 and lp.shape[1] > 1
            lp_is_vertical = lp.shape[0] > 1 and lp.shape[1] == 1
            lp_is_single = lp.shape[0] == 1 and lp.shape[1] == 1
            # Here we map force directions to corresponding rule rotations
            rot_to_force = [1, 2, 3, 0]
            in_patch_shape = lp.shape
            # has_rp = len(rps) > 0
            lp, rp = np.array(lp), np.array(rp)
            # has_right_pattern = len()
            force_idx = rot_to_force[rot]
            # Here we use the assumption that rules are 1D.
            # TODO: generalize rules to 2D...?
            if lp_is_horizontal:
                lp = lp[0, :]
            elif lp_is_vertical:
                lp = lp[:, 0]
            elif lp_is_single:
                lp = lp[0, 0]
            else:
                raise Exception('Invalid rule shape.')

            if has_right_pattern:
                rp_is_horizontal = rp.shape[0] == 1 and rp.shape[1] > 1
                rp_is_vertical = rp.shape[0] > 1 and rp.shape[1] == 1
                rp_is_single = rp.shape[0] == 1 and rp.shape[1] == 1
                if rp_is_horizontal:
                    rp = rp[0, :]
                elif rp_is_vertical:
                    rp = rp[:, 0]
                elif rp_is_single:
                    rp = rp[0, 0]
                else:
                    raise Exception('Invalid rule shape.')
            is_line_detector = False
            cell_detection_fns = []
            cell_projection_fns = []
            if DEBUG:
                print(f'lp, rp: {lp}, {rp}')
            # FIXME, HACK
            if isinstance(lp, str):
                lp = np.array([lp])
            if isinstance(rp, str):
                rp = np.array([rp])
            if DEBUG:
                print(f'lp2, rp2: {lp}, {rp}')
                print(f'has right pattern: {has_right_pattern}')
            if len(lp.shape) == 1:
                for i, l_cell in enumerate(lp):
                    if l_cell == '...':
                        is_line_detector = True
                        cell_detection_fns.append('...')
                        # TODO
                        exit()
                    if l_cell is not None:
                        cell_detection_fns.append(gen_cell_detection_fn(l_cell, force_idx))
                    else:
                        cell_detection_fns.append(
                            lambda m_cell: (True, CellFnReturn(detected=jnp.zeros_like(m_cell), force_idx=None, detected_meta_objs={}))
                        )
            # FIXME: why is the normal way broken here?
            # if has_right_pattern:
            if has_right_pattern and rp is None:
                rp = np.array([None] * len(lp))
            if has_right_pattern:
                for i, r_cell in enumerate(rp):
                    if r_cell == '...':
                        assert is_line_detector, f"`...` not found in left pattern of rule {rule_name}"
                        cell_projection_fns.append('...')
                    else:
                        cell_projection_fns.append(gen_cell_projection_fn(r_cell, force_idx))

            def detect_kernel(lvl):
                n_chan = lvl.shape[1]
                # @jax.jit
                def detect_cells(in_patch):
                    cell_outs_patch = []
                    patch_active = True
                    for i, cell_fn in enumerate(cell_detection_fns):
                        in_patch = in_patch.reshape((n_chan, *in_patch_shape))
                        if lp_is_vertical:
                            m_cell = in_patch[:, i, 0]
                        if lp_is_horizontal:
                            m_cell = in_patch[:, 0, i]
                        if lp_is_single:
                            m_cell = in_patch[:, 0, 0]
                        cell_active, cell_out = cell_fn(m_cell=m_cell)
                        patch_active = patch_active & cell_active
                        cell_outs_patch.append(cell_out)
                    return patch_active, cell_outs_patch

                def detect_line(in_line):
                    for i, cell_fn in enumerate(cell_detection_fns):
                        cell_fn_activs, cell_fn_rets = jax.vmap(cell_fn)(in_line)
                        breakpoint()

                if not is_line_detector:

                    patches = jax.lax.conv_general_dilated_patches(
                        lvl, in_patch_shape, window_strides=(1, 1), padding='VALID',
                    )
                    assert patches.shape[0] == 1
                    patches = patches[0]
                    patches = rearrange(patches, "c h w -> h w c")

                    if jit:
                        kernel_activations, cell_detect_outs = jax.vmap(jax.vmap(detect_cells))(patches)

                    else:
                        kernel_activations = jnp.zeros((patches.shape[0], patches.shape[1]), dtype=bool)
                        # What's up with this???
                        cell_detect_outs = [[None for _ in range(patches.shape[0])] for _ in range(patches.shape[1])]
                        for i in range(patches.shape[0]):
                            for j in range(patches.shape[1]):
                                patch = patches[i, j]
                                patch_active, cell_out = detect_cells(patch)
                                kernel_activations = kernel_activations.at[i, j].set(patch_active)
                                cell_detect_outs[j][i] = cell_out
                        cell_detect_outs = stack_leaves(stack_leaves(cell_detect_outs))

                else:
                    # TODO:
                    breakpoint()

                # eliminate all but one activation
                kernel_detected = kernel_activations.sum() > 0
                if DEBUG:
                    jax.lax.cond(
                        # True,
                        kernel_detected,
                        lambda: jax.debug.print('      Rule {rule_name} left kernel {lp} detected: {kernel_detected}',
                                                rule_name=rule_name, kernel_detected=kernel_detected, lp=lp),
                        lambda: None,
                    )
                cancelled = False
                detected_kernel_meta_objs = {}
                detected_kernel_moving_idx = None
                for cell_detect_out in cell_detect_outs:
                    detected_kernel_meta_objs.update(cell_detect_out.detected_meta_objs)
                    if cell_detect_out.detected_moving_idx is not None:
                        detected_kernel_moving_idx = cell_detect_out.detected_moving_idx
                kernel_detect_out = KernelFnReturn(
                    detected_meta_objs=detected_kernel_meta_objs,
                    detected_moving_idx=detected_kernel_moving_idx,
                )
                return kernel_activations, cell_detect_outs, kernel_detect_out


            def project_kernel(lvl, kernel_activations, 
                               cell_detect_outs: List[CellFnReturn],
                               kernel_detect_outs: List[KernelFnReturn], 
                               pattern_detect_out: List[PatternFnReturn]
                            ):
                n_tiles = np.prod(lvl.shape[-2:])
                # Ensure we always have some invalid coordinates so that the loop will break even when all tiles are active
                if jit:
                    kernel_activ_xys = jnp.argwhere(kernel_activations == 1, size=n_tiles+1, fill_value=-1)
                else:
                    kernel_activ_xys = np.argwhere(kernel_activations == 1)
                kernel_activ_xy_idx = 0
                kernel_activ_xy = kernel_activ_xys[kernel_activ_xy_idx]

                # @jax.jit
                def project_cells_at(xy, lvl):
                    # patch_activations_xy = jnp.zeros_like(patch_activations)
                    # patch_activations_xy = patch_activations_xy.at[*xy].set(1)
                    # detect_outs = detect_outs[first_a[0]][first_a[1]]
                    cell_detect_outs_xy = [jax.tree.map(lambda x: x[xy[0]][xy[1]], cell_detect_out) for 
                        cell_detect_out in cell_detect_outs]
                    # cell_detect_outs_xy = stack_leaves(cell_detect_outs_xy)

                    # pattern_detect_outs_xy = [jax.tree_map(lambda x: x[xy[0]][xy[1]], pattern_detect_out)]
                    pattern_detect_outs_xy = pattern_detect_out


                    # pattern_meta_objs = {}
                    # moving_idx = None
                    # FIXME: Shouldn't this be a jnp array at this point when `jit=True`? And yet it's a list...?
                    # for k in detect_outs_xy.meta_objs:
                    #     pattern_meta_objs[k] = detect_outs_xy.meta_objs[k].max()
                    # for cell_fn_out in detect_outs_xy:
                    #     pattern_meta_objs.update(cell_fn_out.meta_objs)
                    #     if cell_fn_out.moving_idx is not None:
                    #         moving_idx = cell_fn_out.moving_idx
                    # rule_fn_out = PatternFnReturn(meta_objs=pattern_meta_objs, moving_idx=moving_idx)

                    # Apply projection functions to the affected cells
                    out_cell_idxs = np.indices(in_patch_shape)
                    out_cell_idxs = rearrange(out_cell_idxs, "xy h w -> h w xy")
                    if lp_is_vertical:
                        out_cell_idxs = out_cell_idxs[:, 0]
                    elif lp_is_horizontal:
                        out_cell_idxs = out_cell_idxs[0, :]
                    elif lp_is_single:
                        out_cell_idxs = out_cell_idxs[0, :]

                    # Either the rule has no right pattern, or it should detect as many cells as there are cell projection functions
                    if not (~has_right_pattern or (len(cell_detect_outs) == len(out_cell_idxs) == len(cell_projection_fns))):
                        print(f"Warning: rule {rule} with has_right_pattern {has_right_pattern} results in len(cell_detect_outs) {len(cell_detect_outs)} != len(out_cell_idxs) {len(out_cell_idxs)} != len(cell_projection_fns) {len(cell_projection_fns)}")
                        breakpoint()
                    init_lvl = lvl

                    #TODO: vmap this. But then we risk overlapping? But we do here, too.
                    for i, (out_cell_idx, cell_proj_fn) in enumerate(zip(out_cell_idxs, cell_projection_fns)):
                    # def apply_cell_proj_fn(lvl, i):
                        # FIXME: a cell at position `i` may not exist in the input kernel!! So here, it's just referring to the ``last'' cell in the input (?)
                        cell_detect_out_i = cell_detect_outs_xy[i]
                        pattern_detect_out_i = pattern_detect_outs_xy
                        cell_xy = out_cell_idx + xy
                        m_cell = lvl[0, :, *cell_xy]
                        m_cell = jnp.array(m_cell)
                        m_cell = cell_proj_fn(m_cell, cell_detect_out=cell_detect_out_i, pattern_detect_out=pattern_detect_out_i)
                        # m_cell = jax.lax.switch(i, cell_projection_fns, m_cell, cell_detect_out,
                        #                         pattern_detect_outs_xy)
                        lvl = lvl.at[0, :, *cell_xy].set(m_cell)
                        # return lvl, None
                    
                    # lvl, _ = jax.lax.scan(apply_cell_proj_fn, lvl, jnp.arange(len(out_cell_idxs)))

                    lvl_changed = jnp.any(lvl != init_lvl)
                    if DEBUG:
                        jax.debug.print('      at position {xy}, the level changed: {lvl_changed}', xy=xy, lvl_changed=lvl_changed)
                    # if not jit:
                    #     print('\n' + multihot_to_desc(lvl[0], obj_to_idxs=obj_to_idxs, n_objs=n_objs))
                    return lvl

                def project_kernel_at_xy(carry):               
                    kernel_activ_xy_idx = carry[0]
                    kernel_activ_xy = kernel_activ_xys[kernel_activ_xy_idx]
                    # jax.debug.print('      kernel_activ_xys: {kernel_activ_xys}', kernel_activ_xys=kernel_activ_xys)
                    # jax.debug.print('      projecting kernel at position index {kernel_activ_xy_idx}, position {xy}', xy=kernel_activ_xy, kernel_activ_xy_idx=kernel_activ_xy_idx)
                    lvl = carry[1]
                    lvl = project_cells_at(kernel_activ_xy, lvl)

                    kernel_activ_xy_idx += 1
                    return kernel_activ_xy_idx, lvl


                if jit:
                    _, lvl = jax.lax.while_loop(
                        lambda carry: jnp.all(kernel_activ_xys[carry[0]] != -1),  
                        lambda carry: project_kernel_at_xy(carry),
                        (kernel_activ_xy_idx, lvl),
                    )
                else:
                    carry = (kernel_activ_xy_idx, lvl)
                    while kernel_activ_xy_idx < len(kernel_activ_xys):
                        carry = project_kernel_at_xy(carry)
                        kernel_activ_xy_idx = carry[0]
                    lvl = carry[1]
                return lvl

            return detect_kernel, project_kernel


        if not has_right_pattern:
            rps = [None] * len(lps)
        if DEBUG:
            print('rps', rps)
            print('lps', lps)
        kernel_fns = [gen_rotated_kernel_fns(lp, rp, rot) for lp, rp in zip(lps, rps)]
        kernel_detection_fns, kernel_projection_fns = zip(*kernel_fns)

        def detect_pattern(lvl):
            kernel_activations: List[chex.Array] = []
            cell_detect_outs: List[CellFnReturn] = []
            kernel_detect_outs: List[KernelFnReturn] = []
            for kernel_detection_fn in kernel_detection_fns:
                kernel_activations_i, cell_detect_outs_i, kernel_detect_out_i = kernel_detection_fn(lvl)
                kernel_activations.append(kernel_activations_i)
                cell_detect_outs.append(cell_detect_outs_i)
                kernel_detect_outs.append(kernel_detect_out_i)

            # pad activations to the same shape
            max_kernel_activation_shape = max([k.shape for k in kernel_activations])
            kernel_activations = [jnp.pad(k, ((0, max_kernel_activation_shape[0] - k.shape[0]), (0, max_kernel_activation_shape[1] - k.shape[1]))) for k in kernel_activations]
            kernel_activations = jnp.stack(kernel_activations, axis=0)

            detected_pattern_meta_objs = {}
            detected_pattern_moving_idx = None
            for kernel_detect_out in kernel_detect_outs:
                # Each kernel has detected meta-objects at different coordinates on the board.
                # To get pattern-wide meta-objs, take any detected meta-object index that is not -1 (indicating no meta-object was detected) 
                # (We can take the max here because we assume that if a meta-tile in the right pattern is not specified in the corresponding left kernel, it is only specified once in the rest of the left pattern)
                # FIXME: Shouldn't `detected_moving_idx` also be an array (not an int) then?
                boardwide_kernel_meta_objs = {k: v.max() for k, v in kernel_detect_out.detected_meta_objs.items()}
                detected_pattern_meta_objs.update(boardwide_kernel_meta_objs)
                if kernel_detect_out.detected_moving_idx is not None:
                    detected_pattern_moving_idx = kernel_detect_out.detected_moving_idx
            pattern_out = PatternFnReturn(
                detected_meta_objs=detected_pattern_meta_objs,
                detected_moving_idx=detected_pattern_moving_idx,
            )
            return kernel_activations, cell_detect_outs, kernel_detect_outs, pattern_out

        def apply_pattern(lvl):
            kernel_activations, cell_detect_outs, kernel_detect_outs, pattern_detect_out = detect_pattern(lvl)
            pattern_detected = jnp.all(jnp.sum(kernel_activations, axis=(1,2)) > 0)

            def project_kernels(lvl, kernel_activations, kernel_detect_outs):
                # TODO: use a jax.lax.switch
                for i, kernel_projection_fn in enumerate(kernel_projection_fns):
                    if DEBUG:
                        jax.debug.print('      projecting kernel {i}', i=i)
                    lvl = kernel_projection_fn(
                        lvl, kernel_activations[i], cell_detect_outs[i], kernel_detect_outs[i], pattern_detect_out)
                return lvl

            cancel, restart, again = False, False, False
            if has_right_pattern:
                if jit:
                    next_lvl = jax.lax.cond(
                        pattern_detected,
                        project_kernels,
                        lambda lvl, pattern_activations, pattern_detect_outs: lvl,
                        lvl, kernel_activations, kernel_detect_outs,
                    )
                else:
                    if pattern_detected:
                        next_lvl = project_kernels(lvl, kernel_activations, kernel_detect_outs)
                    else:
                        next_lvl = lvl

                rule_applied = jnp.any(next_lvl != lvl)
            else:
                next_lvl = lvl
                rule_applied = False
                # assert rule.command is not None
                if rule.command is None and not np.all([r is None for r in rps]):
                    print(rps)
                    breakpoint()

                if rule.command == 'cancel':
                    cancel = pattern_detected
                elif rule.command == 'restart':
                    restart = pattern_detected
            if rule.command == 'again':
                again = rule_applied
                if DEBUG:
                    jax.debug.print('applying the {command} command', command=rule.command)
            return next_lvl, rule_applied, cancel, restart, again

        return apply_pattern

    rule_fns = []
    if DEBUG:
        print('RULE PATTERNS', rule.left_kernels, rule.right_kernels)


    l_kerns, r_kerns = rule.left_kernels, rule.right_kernels
    # Replace any empty lists in lp and rp with a None
    l_kerns = [[[None] if len(l) == 0 else [' '.join(l)] for l in kernel] for kernel in l_kerns]
    if DEBUG:
        print('lps', l_kerns)
    # lps = np.array(lps)

    # rp, rp_rot = None, None
    if has_right_pattern:
    #     rp = rule.right_patterns[i]
        r_kerns = [[[None] if len(r) == 0 else [' '.join(r)] for r in kernel] for kernel in r_kerns]
        # r_kerns = np.array(r_kerns)
    else:
        r_kerns = None

    if 'horizontal' in rule.prefixes:
        rots = [1, 3]
    elif 'left' in rule.prefixes:
        rots = [3]
    elif 'right' in rule.prefixes:
        rots = [1]
    elif 'vertical' in rule.prefixes:
        rots = [0, 2]
    elif 'up' in rule.prefixes:
        rots = [2]
    elif 'down' in rule.prefixes:
        rots = [0]
    # TODO: Remove unnecessary rotated variations of single-cell kernels
    # elif np.any(np.array([len(kernel) for kernel in l_kerns]) > 1):
    else:
        if np.all(np.array([len(lp) for lp in rule.left_kernels]) == 1) and not \
            np.any([is_rel_force_in_kernel(lp) for lp in rule.left_kernels]):
            rots = [0]
        else:
            rots = [0, 1, 2, 3]
    # else:
    #     rots = [0]
    first_lp = np.array(l_kerns[0])
    if first_lp.shape[0] == 1 and first_lp.shape[1] == 1:
        # print(f'LPS: {lps}')
        # print(f'RPS: {rps}')
        rule_fns.append(gen_rotated_rule_fn(l_kerns, r_kerns, 0, rule.command))
    for rot in rots:
        # rotate the patterns
        l_kerns_rot = []
        for kern in l_kerns:
            kern = np.rot90(kern, rot, axes=(0, 1))
            l_kerns_rot.append(kern)

        r_kerns_rot = None
        if has_right_pattern:
            r_kerns_rot = []
            for kern in r_kerns:
                kern = np.rot90(kern, rot, axes=(0, 1))
                r_kerns_rot.append(kern)

        rule_fns.append(gen_rotated_rule_fn(l_kerns_rot, r_kerns_rot, rot, rule.command))

    return rule_fns
        
def player_has_moved(lvl_0, lvl_1, obj_to_idxs, meta_objs, char_to_obj):
    player_sub_objs = expand_meta_objs(['player'], meta_objs, char_to_obj)
    player_idxs = [obj_to_idxs[obj] for obj in player_sub_objs]
    player_moved = False
    for player_idx in player_idxs:
        player_i_moved = jnp.sum(lvl_0[player_idx] != lvl_1[player_idx]) > 0
        player_moved = jnp.logical_or(player_moved, player_i_moved)
    if DEBUG:
        jax.debug.print('player moved: {player_moved}', player_moved=player_moved)
    return player_moved

def loop_rule_grp(carry, grp_i, block_i, n_prior_rules_arr, n_rules_per_grp_arr, all_rule_fns, obj_to_idxs, n_objs,
                  jit):
    lvl, grp_applied_prev, grp_app_i, cancelled, restart, again = carry

    grp_applied = True

    def apply_rule_grp(carry):
        lvl, _, grp_app_i, cancelled, restart, again = carry
        grp_app_i += 1
        grp_applied = False

        def loop_rule_fn(carry):
            lvl, rule_i, rule_applied_prev, rule_app_i, cancelled_prev, restart_prev, again_prev = carry

            def apply_rule_fn(carry):
                init_lvl, rule_applied_prev, rule_app_i, cancelled_prev, restart_prev, again_prev = carry
                if jit:
                    # lvl, rule_applied, cancelled, restart, again = jax.lax.switch(rule_i, rule_grp, init_lvl)
                    lvl, rule_applied, cancelled, restart, again = jax.lax.switch(
                        n_prior_rules_arr[block_i, grp_i] + rule_i, all_rule_fns, init_lvl)
                else:
                    # lvl, rule_applied, cancelled, restart, again = rule_grp[rule_i](init_lvl)
                    lvl, rule_applied, cancelled, restart, again = \
                        all_rule_fns[n_prior_rules_arr[block_i, grp_i] + rule_i](init_lvl)
                rule_had_effect = jnp.any(lvl != init_lvl)
                print(f'      rule {rule_i} of group {grp_i} had effect: {rule_had_effect}')
                if DEBUG:
                    if jit:
                        jax.debug.print('    rule {rule_i} had effect: {rule_had_effect}', rule_i=rule_i, rule_had_effect=rule_had_effect)
                    else:
                        if rule_had_effect:
                            print(f'      rule {rule_i} of group {grp_i} had effect: True')
                rule_app_i += 1
                again = again_prev | again
                restart = restart_prev | restart
                cancelled = cancelled_prev | cancelled
                return lvl, rule_had_effect, rule_app_i, cancelled, restart, again

            rule_applied = True
            rule_app_i = 0
            if jit:
                lvl, rule_applied, rule_app_i, cancelled, restart, again = jax.lax.while_loop(
                    cond_fun=lambda x: x[1] & ~x[3] & ~x[4],
                    body_fun=lambda x: apply_rule_fn(x),
                    init_val=(lvl, rule_applied, rule_app_i, cancelled_prev, restart_prev, again_prev),
                )
            else:
                cancelled = cancelled_prev
                restart = restart_prev
                again = again_prev
                while rule_applied and not cancelled and not restart:
                    lvl, rule_applied, rule_app_i, cancelled, restart, again = apply_rule_fn(
                        (lvl, rule_applied, rule_app_i, cancelled_prev, restart_prev, again_prev))

            rule_applied = rule_app_i > 1
            rule_applied = rule_applied | rule_applied_prev
            if DEBUG:
                if jit:
                    jax.debug.print('    rule {rule_i} applied: {rule_applied}', rule_i=rule_i, rule_applied=rule_applied)
                else:
                    # if rule_applied:
                    print(f'      rule {rule_i} applied: {rule_applied}')
            
            rule_i += 1
            return lvl, rule_i, rule_applied, rule_app_i, cancelled, restart, again

        rule_i = 0
        rule_app_i = 0
        grp_applied = False
        n_rules_in_grp = n_rules_per_grp_arr[block_i, grp_i]
        if jit:
            # For each rule in the group, apply it as many times as possible.
            lvl, rule_i, grp_applied, _, cancelled, restart, again = jax.lax.while_loop(
                cond_fun=lambda x: (x[1] < n_rules_in_grp) & ~x[4] & ~x[5],
                body_fun=loop_rule_fn,
                init_val=(lvl, rule_i, grp_applied, rule_app_i, cancelled, restart, again),
            )
        else:
            while rule_i < n_rules_in_grp and not cancelled and not restart:
                lvl, rule_i, grp_applied, rule_app_i, cancelled, restart, again = loop_rule_fn(
                    (lvl, rule_i, grp_applied, rule_app_i, cancelled, restart, again))

        if DEBUG:
            if jit:
                jax.debug.print('group {grp_i} applied: {grp_applied}', grp_i=grp_i, grp_applied=grp_applied)
            else:
                if grp_applied:
                    print(f'group {grp_i} applied')
        # print('rule: {rule}', rule=tree_rules[0][0].rules[grp_i])
        if not jit:
            if grp_applied:
                if DEBUG:
                    print('\n' + multihot_to_desc(lvl[0], obj_to_idxs, n_objs))
        # force_is_present = jnp.sum(lvl[0, n_objs:n_objs + (n_objs * N_MOVEMENTS)]) > 0
        # jax.debug.print('force is present: {force_is_present}', force_is_present=force_is_present)
        # jax.debug.print('lvl:\n{lvl}', lvl=lvl)
        return lvl, grp_applied, grp_app_i, cancelled, restart, again

    grp_app_i = 0
    if jit:
        lvl, grp_applied, grp_app_i, cancelled, restart, grp_again = jax.lax.while_loop(
            cond_fun=lambda x: x[1] & ~x[3] & ~x[4],
            body_fun=apply_rule_grp,
            init_val=(lvl, grp_applied, grp_app_i, cancelled, restart, again),
        )
    else:
        while grp_applied and not cancelled and not restart:
            lvl, grp_applied, grp_app_i, cancelled, restart, grp_again = apply_rule_grp((lvl, grp_applied, grp_app_i, cancelled, restart, again))

    again = again | grp_again
    grp_applied = grp_app_i > 1
    block_applied = grp_applied_prev | grp_applied

    return (lvl, block_applied, grp_app_i, cancelled, restart, again), None

        
def gen_tick_fn(obj_to_idxs, coll_mat, tree_rules, meta_objs, jit, n_objs, char_to_obj):
    if len(tree_rules) == 0:
        pass
    else:
        rule_blocks = []
        late_rule_grps = []
        for rule_block in tree_rules:

            # FIXME: what's with this unnecessary list?
            assert len(rule_block) == 1
            rule_block = rule_block[0]

            looping = rule_block.looping
            rule_grps = []
            for rule in rule_block.rules:
                sub_rule_fns = gen_subrules_meta(rule, n_objs, obj_to_idxs, meta_objs, coll_mat, rule_name=str(rule), 
                                                 char_to_obj=char_to_obj, jit=jit)
                if not 'late' in rule.prefixes:
                    rule_grps.append(sub_rule_fns)
                else:
                    late_rule_grps.append(sub_rule_fns)
            rule_blocks.append((looping, rule_grps))

    rule_blocks.append((False, [gen_move_rules(obj_to_idxs, coll_mat, n_objs, jit=jit)]))
    # Can we have loops in late rules? I hope not.
    rule_blocks.append((False, late_rule_grps))

    all_rule_fns = [rule_fn for looping, rule_grps in rule_blocks for rule_grp in rule_grps for rule_fn in rule_grp]
    n_rules_counted = 0
    # n_prior_rules = {}
    max_n_grps = max([len(rule_grps) for _, rule_grps in rule_blocks])
    n_prior_rules_arr = np.zeros((len(rule_blocks), max_n_grps), dtype=jnp.int32)
    n_rules_per_grp_arr = np.zeros((len(rule_blocks), max_n_grps), dtype=jnp.int32)
    for rule_block_i, rule_block in enumerate(rule_blocks):
        _, rule_grps = rule_block
        for rule_grp_i, rule_grp in enumerate(rule_grps):
            # n_prior_rules[(rule_block_i, rule_grp_i)] = n_rules_counted
            n_prior_rules_arr[rule_block_i, rule_grp_i] = n_rules_counted
            n_rules_per_grp_arr[rule_block_i, rule_grp_i] = len(rule_grp)
            n_rules_counted += len(rule_grp)
    n_prior_rules_arr = jnp.array(n_prior_rules_arr)
    n_rules_per_grp_arr = jnp.array(n_rules_per_grp_arr)

    def tick_fn(lvl):
        lvl_changed = False
        cancelled = False
        restart = False
        again = False
        lvl = lvl[None]

        if not jit:
            if DEBUG:
                print('\n' + multihot_to_desc(lvl[0], obj_to_idxs, n_objs))

        def apply_turn(carry):
            lvl, _, turn_app_i, cancelled, restart, turn_again = carry
            turn_app_i += 1
            turn_applied = False

            for block_i, (looping, rule_grps) in enumerate(rule_blocks):
                # n_prior_rules

                _loop_rule_grp = partial(loop_rule_grp, block_i=block_i, n_prior_rules_arr=n_prior_rules_arr,
                                        n_rules_per_grp_arr=n_rules_per_grp_arr, all_rule_fns=all_rule_fns,
                                        obj_to_idxs=obj_to_idxs, n_objs=n_objs, jit=jit)

                def apply_rule_block(carry):
                    lvl, _, block_app_i, cancelled, restart, again = carry
                    block_app_i += 1
                    block_applied = False

                    if jit:
                        (lvl, block_applied, block_app_i, cancelled, restart, again), _ = jax.lax.scan(
                            _loop_rule_grp,
                            init=(lvl, block_applied, block_app_i, cancelled, restart, again),
                            xs=jnp.arange(len(rule_grps)),
                        )
                    else:
                        for grp_i in range(len(rule_grps)):
                            carry, _ = _loop_rule_grp((lvl, block_applied, block_app_i, cancelled, restart, again), grp_i)
                            lvl, block_applied, block_app_i, cancelled, restart, again = carry

                    return lvl, block_applied, block_app_i, cancelled, restart, again

                block_applied = True
                block_app_i = 0
                block_again = False
                init_carry = (lvl, block_applied, block_app_i, cancelled, restart, block_again)
                if not jit:
                    print(f'Level when applying block {block_i}:\n', multihot_to_desc(lvl[0], obj_to_idxs, n_objs))
                if looping:
                    if jit:
                        lvl, block_applied, block_app_i, cancelled, restart, block_again = jax.lax.while_loop(
                            cond_fun=lambda x: x[1] & ~x[3] & ~x[4],
                            body_fun=apply_rule_block,
                            init_val=init_carry,
                        )
                    else:
                        carry = init_carry
                        while block_applied and not cancelled and not restart:
                            carry = apply_rule_block(carry)
                            lvl, block_applied, block_app_i, cancelled, restart, block_again = carry
                    block_applied = block_app_i > 1
                else:
                    lvl, block_applied, block_app_i, cancelled, restart, block_again = apply_rule_block(init_carry)
                if DEBUG:
                    if jit:
                        jax.debug.print('block {block_i} applied: {block_applied}', block_i=block_i, block_applied=block_applied)
                    else:
                        if block_applied:
                            print(f'block {block_i} applied')
                        else:
                            print(f'block {block_i} not applied')
                turn_again = block_again
                
                turn_applied = turn_applied | block_applied

            return lvl, turn_applied, turn_app_i, cancelled, restart, turn_again

        turn_applied = True
        turn_app_i = 0
        turn_again = True

        init_carry = (lvl, turn_applied, turn_app_i, cancelled, restart, turn_again)
        if jit:
            lvl, turn_applied, turn_app_i, cancelled, restart, again = jax.lax.while_loop(
                lambda x: x[1] & x[5],
                lambda x: apply_turn(x),
                init_carry
            )
        else:
            carry = init_carry
            while turn_applied and turn_again:
                carry = apply_turn(carry)
                lvl, turn_applied, turn_app_i, cancelled, restart, turn_again = carry

        lvl_changed = turn_app_i > 1

        # if not jit:
        #     print('\nLevel after applying rules:\n', multihot_to_desc(lvl[0], obj_to_idxs, n_objs))
        #     print('grp_applied:', grp_applied)
        #     print('cancelled:', cancelled)

        return lvl[0], lvl_changed, turn_app_i, cancelled, restart

    if jit:
        tick_fn = jax.jit(tick_fn)
    return tick_fn

def gen_move_rules(obj_to_idxs, coll_mat, n_objs, jit=True):
    rule_fns = []
    coll_mat = coll_mat.astype(np.int8)
    for obj, idx in obj_to_idxs.items():
        if obj == 'Background':
            continue
        coll_vector = coll_mat[idx]

        left_rule_in = np.zeros((n_objs + (n_objs * 5), 1, 2), dtype=np.int8)
        # object must be present in right cell
        left_rule_in[idx, 0, 1] = 1
        # leftward force
        left_rule_in[n_objs + (idx * 5) + 0, 0, 1] = 1
        # absence of collidable tiles in the left cell
        left_rule_in[:n_objs, 0, 0] = -coll_vector
        left_rule_out = np.zeros_like(left_rule_in)
        # object moves to left cell
        left_rule_out[idx, 0, 0] = 1
        # remove anything that was present in the input (i.e. the object and force)
        left_rule_out -= np.clip(left_rule_in, 0, 1)
        left_rule = np.stack((left_rule_in, left_rule_out), axis=0)

        # FIXME: Actually this is the down rule and vice versa, and we've relabelled actions accordingly. Hack. Can't figure out what's wrong here.
        # Something to do with flipping output kernel?
        up_rule_in = np.zeros((n_objs + (n_objs * 5), 2, 1), dtype=np.int8)
        # object must be present in lower cell
        up_rule_in[idx, 0] = 1
        # upward force
        up_rule_in[n_objs + (idx * N_MOVEMENTS) + 1, 0] = 1
        # absence of collidable tiles in the upper cell
        up_rule_in[:n_objs, 1, 0] = -coll_vector
        up_rule_out = np.zeros_like(up_rule_in)
        # object moves to upper cell
        up_rule_out[idx, 1] = 1
        # remove anything that was present in the input
        up_rule_out -= np.clip(up_rule_in, 0, 1)
        up_rule = np.stack((up_rule_in, up_rule_out), axis=0)

        right_rule_in = np.zeros((n_objs + (n_objs * N_MOVEMENTS), 1, 2), dtype=np.int8)
        # object must be present in left cell
        right_rule_in[idx, 0, 0] = 1
        # rightward force
        right_rule_in[n_objs + (idx * N_MOVEMENTS) + 2, 0, 0] = 1
        # absence of collidable tiles in the right cell
        right_rule_in[:n_objs, 0, 1] = -coll_vector
        right_rule_out = np.zeros_like(right_rule_in)
        # object moves to right cell
        right_rule_out[idx, 0, 1] = 1
        # remove anything that was present in the input
        right_rule_out -= np.clip(right_rule_in, 0, 1)
        right_rule = np.stack((right_rule_in, right_rule_out), axis=0)

        down_rule_in = np.zeros((n_objs + (n_objs * N_MOVEMENTS), 2, 1), dtype=np.int8)
        # object must be present in upper cell
        down_rule_in[idx, 1] = 1
        # downward force
        down_rule_in[n_objs + (idx * N_MOVEMENTS) + 3, 1] = 1
        # absence of collidable tiles in the lower cell
        down_rule_in[:n_objs, 0, 0] = -coll_vector
        down_rule_out = np.zeros_like(down_rule_in)
        # object moves to lower cell
        down_rule_out[idx, 0] = 1
        # remove anything that was present in the input
        down_rule_out -= np.clip(down_rule_in, 0, 1)
        down_rule = np.stack((down_rule_in, down_rule_out), axis=0)

        # rules += [left_rule, right_rule, up_rule, down_rule]
        rules = [left_rule, right_rule, up_rule, down_rule]
        rule_names = [f"{obj}_move_{j}" for j in ['left', 'right', 'down', 'up']]
        rule_fns += [partial(apply_move_rule, move_rule=rule, jit=jit, rule_name=rule_name)
                     for rule, rule_name in zip(rules, rule_names)]
    return rule_fns

def apply_move_rule(lvl, move_rule, jit=True, rule_name=None):
    inp = move_rule[0]
    ink = inp[None]
    lvl = lvl.astype(np.float32)
    ink = ink.astype(np.float32)
    # jax.debug.print('lvl: {lvl}', lvl=(lvl.min(), lvl.max(), lvl.shape))
    if jit:
        activations = jax.lax.conv(lvl, ink, window_strides=(1,1), padding='VALID')[0, 0]
    else:
        # TODO
        activations = jax.lax.conv(lvl, ink, window_strides=(1,1), padding='VALID')[0, 0]

    thresh_act = np.sum(np.clip(inp, 0, 1))
    bin_activations = (activations == thresh_act).astype(np.float32)

    non_zero_activation = jnp.argwhere(bin_activations != 0, size=1, fill_value=-1)[0]
    # jax.lax.cond(jnp.all(non_zero_activation == -1), lambda _: None, 
    #              lambda _: jax.debug.print(
    #                  'Non-zero activation detected: {non_zero_activation}. Rule_i: {rule_name}',
    #                  non_zero_activation=non_zero_activation, rule_name=rule_name), None)
    # if non_zero_activations.size > 0:
    #     print(f"Non-zero activations detected: {non_zero_activations}. Rule_i: {rule_name}")
        # if rule_i == 1:
        #     breakpoint()

    outp = move_rule[1]
    outk = outp[:, None]
    outk = outk.astype(np.float32)
    # flip along width and height
    outk = np.flip(outk, axis=(2, 3))
    bin_activations = bin_activations[None, None]
    changed = bin_activations.sum() > 0
    if jit:
        out = jax.lax.conv_transpose(bin_activations, outk, (1, 1), padding='VALID',
                                            dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    else:
        # TODO
        out = jax.lax.conv_transpose(bin_activations, outk, (1, 1), padding='VALID',
                                            dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    if DEBUG:
        if jit:
            jax.lax.cond(
                changed,
                lambda _: jax.debug.print('rule {rule_name} applied: {changed}', rule_name=rule_name, changed=changed),
                lambda _: None,
                None
            )    # if out.sum() != 0:
        else:
            if changed:
                print(f'rule {rule_name} applied')
            else:
                print(f'rule {rule_name} not applied')
    #     print('lvl')
    #     print(lvl[0,2])
    #     print('out')
    #     print(out[0,2])
    #     breakpoint()
    lvl += out
    lvl = lvl.astype(bool)

    return lvl, changed, False, False, False

    

@flax.struct.dataclass
class PSState:
    multihot_level: np.ndarray
    win: bool
    restart: bool
    init_heuristic: int
    prev_heuristic: int
    step_i: int

@flax.struct.dataclass
class PSParams:
    level: chex.Array

@flax.struct.dataclass
class PSObs:
    multihot_level: chex.Array
    flat_obs: Optional[chex.Array] = None



class PSEnv:
    def __init__(self, tree: PSGameTree, jit: bool = True, level_i: int = 0, max_steps: int = 1000):
        self.jit = jit
        self.title = tree.prelude.title
        self.tree = tree
        self.levels = tree.levels
        self.level_i = level_i
        self.require_player_movement = tree.prelude.require_player_movement
        obj_to_char, meta_objs, joint_tiles = process_legend(tree.legend)
        self.meta_objs = meta_objs
        self.max_steps = max_steps

        # Add to the legend any objects to whose keys are specified in their object definition
        for obj_key, obj in tree.objects.items():
            obj_key = obj.legend_key
            if obj_key is not None:
                obj_to_char[obj.name] = obj_key
        self.char_to_obj = char_to_obj = {v: k for k, v in obj_to_char.items()}

        collision_layers = expand_collision_layers(tree.collision_layers, meta_objs, char_to_obj)
        atomic_obj_names = [name for layer in collision_layers for name in layer]
        # atomic_obj_names = [name for name in tree.objects.keys()]
        # atomic_obj_names = [name for name in atomic_obj_names]
        self.atomic_obj_names = atomic_obj_names
        objs, self.obj_to_idxs, coll_masks = assign_vecs_to_objs(collision_layers, atomic_obj_names)
        for obj, sub_objs in meta_objs.items():
            # Meta-objects that are actually just alternate names.
            if DEBUG:
                print(f'sub_objs {sub_objs}')
            sub_objs = expand_meta_objs(sub_objs, meta_objs, char_to_obj)
            if len(sub_objs) == 1 and (obj not in self.obj_to_idxs):
                self.obj_to_idxs[obj] = self.obj_to_idxs[sub_objs[0]]
        self.n_objs = len(atomic_obj_names)
        coll_mat = np.einsum('ij,ik->jk', coll_masks, coll_masks, dtype=bool)
        self.tick_fn = gen_tick_fn(self.obj_to_idxs, coll_mat, tree.rules, meta_objs, jit=self.jit, n_objs=self.n_objs,
                                   char_to_obj=char_to_obj)
        self.check_win = gen_check_win(tree.win_conditions, self.obj_to_idxs, meta_objs, self.char_to_obj, jit=self.jit)
        if 'player' in self.obj_to_idxs:
            self.player_idxs = [self.obj_to_idxs['player']]
        else:
            player_objs = expand_meta_objs(['player'], meta_objs, char_to_obj)
            self.player_idxs = [self.obj_to_idxs[p] for p in player_objs]
        self.player_idxs = np.array(self.player_idxs)
        if DEBUG:
            print(f'player_idxs: {self.player_idxs}')
        sprite_stack = []
        if DEBUG:
            print(atomic_obj_names)
            print(self.obj_to_idxs)
        # for obj_name in self.obj_to_idxs:
        for obj_key in atomic_obj_names:
            if obj_key not in tree.objects:
                breakpoint()
            obj = tree.objects[obj_key]
            if obj.sprite is not None:
                if DEBUG:
                    print(f'rendering pixel sprite for {obj_key}')
                im = render_sprite(obj.colors, obj.sprite)

            else:
                assert len(obj.colors) == 1
                if DEBUG:
                    print(f'rendering solid color for {obj_key}')
                im = render_solid_color(obj.colors[0])

            if DEBUG:
                temp_dir = 'scratch'
                os.makedirs(temp_dir, exist_ok=True)
                sprite_path = os.path.join(temp_dir, f'sprite_{obj_key}.png')

                # Size the image up a bunch
                im_s = PIL.Image.fromarray(im)
                im_s = im_s.resize((im_s.size[0] * 10, im_s.size[1] * 10), PIL.Image.NEAREST)
                im_s.save(sprite_path)

            sprite_stack.append(im)
        self.sprite_stack = np.array(sprite_stack)
        char_legend = {v: k for k, v in obj_to_char.items()}
        # Generate vectors to detect atomic objects
        self.obj_vecs = np.eye(self.n_objs, dtype=bool)
        joint_obj_vecs = []
        self.chars_to_idxs = {obj_to_char[k]: v for k, v in self.obj_to_idxs.items() if k in obj_to_char}
        self.chars_to_idxs.update({k: v for k, v in self.obj_to_idxs.items() if len(k) == 1})

        for jo, subobjects in joint_tiles.items():
            vec = np.zeros(self.n_objs, dtype=bool)
            subobjects = expand_meta_objs(subobjects, meta_objs, char_to_obj)
            for so in subobjects:
                if DEBUG:
                    print(so)
                vec += self.obj_vecs[self.obj_to_idxs[so]]
            assert jo not in self.chars_to_idxs
            self.chars_to_idxs[jo] = self.obj_vecs.shape[0]
            self.obj_vecs = np.concatenate((self.obj_vecs, vec[None]), axis=0)

        if self.jit:
            self.step = jax.jit(self.step)
            self.reset = jax.jit(self.reset)
            self.apply_player_force = jax.jit(self.apply_player_force)
        self.joint_tiles = joint_tiles

        multihot_level = self.get_level(level_i)
        self.observation_space = Box(low=0, high=1, shape=multihot_level.shape)
        self.action_space = Discrete(5)

    def gen_dummy_obs(self, params):
        return PSObs(
            multihot_level=jnp.zeros(self.observation_space.shape)[None],
            flat_obs=None,
        )

    def char_level_to_multihot(self, level):
        int_level = np.vectorize(lambda x: self.chars_to_idxs[x])(level)
        multihot_level = self.obj_vecs[int_level]
        multihot_level = rearrange(multihot_level, "h w c -> c h w")

        # Add a default background object everywhere
        background_sub_objs = expand_meta_objs(['background'], self.meta_objs, self.char_to_obj)
        sub_objs = background_sub_objs
        bg_obj = background_sub_objs[0]
        bg_idx = self.obj_to_idxs[bg_obj]
        multihot_level[bg_idx] = 1

        multihot_level = multihot_level.astype(bool)
        return multihot_level

    @partial(jax.jit, static_argnums=(0, 2))
    def render(self, state: PSState, cv2=True):
        lvl = state.multihot_level
        level_height, level_width = lvl.shape[1:]
        sprite_height, sprite_width = self.sprite_stack.shape[1:3]
        im = np.zeros((level_height * sprite_height, level_width * sprite_width, 4), dtype=np.uint8)
        im_lyrs = []
        for i, sprite in enumerate(self.sprite_stack):
            sprite_stack_i = np.stack(
                (np.zeros_like(sprite), sprite)
            )
            lyr = lvl[i]
            im_lyr = jnp.array(sprite_stack_i)[lyr.astype(int)]
            im_lyr = rearrange(im_lyr, "lh lw sh sw c -> (lh sh) (lw sw) c")
            overwrite_mask = im_lyr[:, :, 3] == 255
            im = jnp.where(jnp.repeat(overwrite_mask[:, :, None], 4, 2), im_lyr, im)

        if cv2:
            # swap the red and blue channels
            im = im[:, :, [2, 1, 0, 3]]

        return im

    def reset(self, key, params: PSParams):
        lvl = params.level
        again = False
        _, _, init_heuristic = self.check_win(lvl)
        if self.tree.prelude.run_rules_on_level_start:
            state = PSState(
                multihot_level=lvl,
                win = jnp.array(False),
                restart = jnp.array(False),
                step_i = 0,
                init_heuristic = init_heuristic,
                prev_heuristic = init_heuristic,
            )
            lvl = self.apply_player_force(-1, state)
            lvl, _, _, _, _ = self.tick_fn(lvl)
            lvl = lvl[:self.n_objs]
        state = PSState(
            multihot_level=lvl,
            win = jnp.array(False),
            restart = jnp.array(False),
            step_i = 0,
            init_heuristic = init_heuristic,
            prev_heuristic = init_heuristic,
        )
        obs = self.get_obs(state)
        return obs, state

    def get_obs(self, state):
        obs = PSObs(
            multihot_level=state.multihot_level,
            flat_obs=None,
        )
        return obs

    @partial(jax.jit, static_argnums=(0))
    def apply_player_force(self, action, state: PSState):
        multihot_level = state.multihot_level
        # add a dummy object at the front
        force_map = jnp.zeros((N_MOVEMENTS * (multihot_level.shape[0] + 1), *multihot_level.shape[1:]), dtype=bool)
        
        def place_force(force_map, action):
            player_int_mask = (self.player_idxs[...,None,None] + 1) * multihot_level[self.player_idxs]
            # turn the int mask into coords, by flattening it, and appending it with xy coords
            xy_coords = jnp.indices(force_map.shape[1:])
            xy_coords = xy_coords[:, None].repeat(len(self.player_idxs), axis=1)
            player_int_mask = player_int_mask * N_MOVEMENTS + action
            player_int_mask = jnp.concatenate((player_int_mask[None], xy_coords), axis=0)
            player_coords = player_int_mask.reshape(3, -1).T
            force_map = force_map.at[tuple(player_coords.T)].set(1)
            # force_map_sum = force_map.sum()
            # jax.debug.print('force_map: {force_map}', force_map=force_map)
            # jax.debug.print('force map sum: {force_map_sum}', force_map_sum=force_map_sum)
            return force_map

        # apply movement (<4) and/or action (if not noaction)
        apply_force = (action != -1) & ((action < 4) | (~self.tree.prelude.noaction))

        force_map = jax.lax.cond(
            apply_force,
            place_force,
            lambda force_map, _: force_map,
            force_map, action
        )
        # remove the dummy object
        force_map = force_map[N_MOVEMENTS:]

        lvl = jnp.concatenate((multihot_level, force_map), axis=0)

        return lvl

    # @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: PSState,
        action: int,
        params: Optional[PSParams] = None,
    ) -> Tuple[chex.Array, PSState, float, bool, dict]:
        """Performs step transitions in the environment."""
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, action, params
        )
        obs_re, state_re = self.reset(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        # Generalizing this to flax dataclass observations
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )
        return obs, state, reward, done, info



    # @partial(jax.jit, static_argnums=(0))
    def step_env(self, key, state: PSState, action, params: Optional[PSParams] = None):
        init_lvl = state.multihot_level.copy()
        lvl = self.apply_player_force(action, state)

        # def cond_fun(loop_state):
        #     lvl, lvl_changed, n_apps, cancelled = loop_state
        #     return jax.numpy.logical_and(lvl_changed, n_apps < 100) & ~cancelled
            
        # def body_fun(carry):
        #     lvl, lvl_changed_last, n_apps, cancelled = carry
        #     new_lvl, lvl_changed, cancelled = self.rule_fn(lvl)
        #     return (new_lvl, lvl_changed, n_apps + 1, cancelled)
        
        # # Initial state for the while loop
        # init_state = (lvl, True, 0, False)

        # if self.jit:
        #     final_lvl, _, _, cancelled = jax.lax.while_loop(cond_fun, body_fun, init_state)
        
        # else:
        #     while cond_fun(init_state):
        #         init_state = body_fun(init_state)
        #     final_lvl, _, _, cancelled = init_state

        # Actually, just apply the rule function once
        cancelled = False
        restart = False
        final_lvl, tick_applied, turn_app_i, cancelled, restart = self.tick_fn(lvl)

        accept_lvl_change = ((not self.require_player_movement) or 
                             player_has_moved(init_lvl, final_lvl, self.obj_to_idxs, self.meta_objs, self.char_to_obj)) & ~cancelled
        # jax.debug.print('accept level change: {accept_lvl_change}', accept_lvl_change=accept_lvl_change) 
        final_lvl = jax.lax.select(
            accept_lvl_change,
            final_lvl,
            lvl,
        )
        multihot_level = final_lvl[:self.n_objs]
        win, score, heuristic = self.check_win(multihot_level)
        jax.debug.print('heuristic: {heuristic}, score: {score}', heuristic=heuristic, score=score)
        # reward = (heuristic - state.init_heuristic) / jnp.abs(state.init_heuristic)
        reward = heuristic - state.prev_heuristic

        done = win | ((state.step_i + 1) >= self.max_steps)
        info = {}
        state = PSState(
            multihot_level=multihot_level,
            win=win,
            restart=restart,
            step_i=state.step_i + 1,
            init_heuristic=state.init_heuristic,
            prev_heuristic=heuristic,
        )
        obs = self.get_obs(state)
        if DEBUG:
            jax.debug.print('episode done: {done}', done=done)
            jax.debug.print('step_i: {step_i}', step_i=state.step_i)
            jax.debug.print('max_steps: {max_steps}', max_steps=self.max_steps)
            jax.debug.print('win: {win}', win=win)
            jax.debug.print('score: {score}', score=score)
            jax.debug.print('heuristic: {heuristic}', heuristic=heuristic)
        return obs, state, reward, done, info

    def get_level(self, level_idx):
        level = self.levels[level_idx][0]
        # Convert the level to a multihot representation and render it
        multihot_level = self.char_level_to_multihot(level)
        return multihot_level

def multihot_to_desc(multihot_level, obj_to_idxs, n_objs):
    """Converts a multihot array to a 2D list of descriptions.
    
    Args:
        multihot_level: A multihot array of shape [n_objects + n_forces, height, width].
        obj_to_idxs: Dictionary mapping object names to their indices.
    
    Returns:
        A 2D list where each cell contains a string describing all objects and forces present.
    """
    height, width = multihot_level.shape[1:]
    
    # Create a reverse mapping from indices to object names
    idxs_to_obj = {idx: obj for obj, idx in obj_to_idxs.items()}
    
    # Create the description grid
    desc_grid = []
    for h in range(height):
        row = []
        for w in range(width):
            cell_desc = []
            
            # Check which objects are active in this cell
            for obj_idx in range(n_objs):
                if multihot_level[obj_idx, h, w] > 0:
                    obj_name = idxs_to_obj[obj_idx]
                    obj_desc = obj_name
                    
                    # Check if there's a force applied to this object
                    force_names = ["left", "down", "right", "up", "action"]
                    forces = []
                    for f_idx, force_name in enumerate(force_names):
                        force_channel = n_objs + (obj_idx * N_MOVEMENTS) + f_idx
                        if force_channel < multihot_level.shape[0] and multihot_level[force_channel, h, w] > 0:
                            forces.append(f"force {force_name}")
                    
                    if forces:
                        obj_desc += f" ({', '.join(forces)})"
                    
                    cell_desc.append(obj_desc)
            
            row.append(", ".join(cell_desc) if cell_desc else "empty")
        desc_grid.append(row)
    
    # Find the maximum width for each column
    column_widths = []
    for w in range(width):
        max_width = 0
        for h in range(height):
            max_width = max(max_width, len(desc_grid[h][w]))
        column_widths.append(max_width)
    
    # Add padding to each cell
    for h in range(height):
        for w in range(width):
            desc_grid[h][w] = desc_grid[h][w].ljust(column_widths[w])
    
    # Join rows with consistent spacing
    desc_str = "\n".join([" || ".join(row) for row in desc_grid])
    
    return desc_str


def init_ps_env(config: Config, verbose: bool = False) -> PSEnv:
    start_time = timer()
    game = config.game
    level_i = config.level_i
    with open("syntax.lark", "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    tree = get_tree_from_txt(parser, game)
    parse_time = timer()
    print(f'Parsed PS file using Lark into python PSTree object in {(parse_time - start_time) / 1000} seconds.')
    env = PSEnv(tree, jit=True, level_i=level_i, max_steps=config.max_episode_steps)
    print(f'Initialized PSEnv in {(timer() - parse_time) / 1000} seconds.')
    return env