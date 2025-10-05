from dataclasses import dataclass
from functools import partial
import itertools
import logging
import os
from timeit import default_timer as timer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import PIL
import chex
from einops import rearrange
import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

from env_render import render_solid_color, render_sprite
from puzzlejax.env_utils import N_MOVEMENTS, multihot_to_desc, N_FORCES, ACTION
from jax_utils import stack_leaves
from ps_game import LegendEntry, PSGameTree, PSObject, Rule, WinCondition
from gymnax.environments.spaces import Discrete, Box


logger = logging.getLogger(__name__)

# Whether to print out a bunch of stuff, etc.
DEBUG = False
PRINT_SCORE = True
# DEBUG = True


# @partial(jax.jit, static_argnums=(0))
def disambiguate_meta(obj, cell_meta_objs, kernel_meta_objs, pattern_meta_objs, obj_to_idxs):
    """In the right pattern of rules, we may have a meta-object (mapping to a corresponding meta-object in the 
    left pattern). This function uses the `meta_objs` dictionary returned by the detection function to project
    the correct object during rule application."""
    if obj in obj_to_idxs:
        return obj_to_idxs[obj]
    elif obj in cell_meta_objs:
        return cell_meta_objs[obj]
    elif obj in kernel_meta_objs:
        return kernel_meta_objs[obj]
    elif obj in pattern_meta_objs:
        return pattern_meta_objs[obj]
    else:
        # raise Exception(f"When compiling a meta-object projection rule, the meta-object {obj} in the output pattern is " 
        #         "not found in the return of the compiled detection function (either in meta_objs or pattern_meta_objs).")
        # TODO: I think we can just do this in general.
        return None

    # FIXME: as above, but jax (this is broken. is it necessary?)
    # obj_idx = jax.lax.select_n(
    #     jnp.array([obj in obj_to_idxs, obj in meta_objs, obj in pattern_meta_objs]),
    #     jnp.array([obj_to_idxs.get(obj, -1), meta_objs.get(obj, -1), pattern_meta_objs.get(obj, -1)]),
    # )
    # return obj_idx
    

def assign_vecs_to_objs(collision_layers, atomic_obj_names):
    n_lyrs = len(collision_layers)
    n_objs = len(atomic_obj_names)
    coll_masks = np.zeros((n_lyrs, n_objs), dtype=bool)
    objs_to_idxs = {}
    # vecs = np.eye(n_objs, dtype=np.uint8)
    # obj_vec_dict = {}
    # j = 0
    objs = []
    obj_idxs_to_force_idxs_dict = {}
    # atomic_objs_to_idxs = {obj: i for i, obj in enumerate(atomic_obj_names)}
    # Need to check for dupes
    atomic_objs_to_idxs = {}
    i = 0
    for obj in atomic_obj_names:
        if obj in atomic_objs_to_idxs:
            continue
        atomic_objs_to_idxs[obj] = i
        i += 1

    for i, layer in enumerate(collision_layers):
        for obj in layer:
            if obj in objs_to_idxs:
                continue
            objs.append(obj)
            # obj_vec_dict[obj] = vecs[i]
            # if obj in atomic_objs_to_idxs:
            obj_idx = atomic_objs_to_idxs[obj]
            objs_to_idxs[obj] = obj_idx
            # else:
            #     obj_idx = j
            #     j += 1
            # objs_to_idxs[obj] = obj_idx
            coll_masks[i, obj_idx] = 1
            obj_idxs_to_force_idxs_dict[obj_idx] = n_objs + i * N_FORCES
    obj_idxs_to_force_idxs = np.zeros(len(obj_idxs_to_force_idxs_dict), dtype=int) 
    for i in obj_idxs_to_force_idxs_dict.keys():
        obj_idxs_to_force_idxs[i] = obj_idxs_to_force_idxs_dict[i]
    return objs, objs_to_idxs, coll_masks, obj_idxs_to_force_idxs

def process_legend(legend):
    objs_to_chars = {}
    meta_objs = {}
    conjoined_tiles = {}
    chars_to_objs = {}
    for k, v in legend.items():
        v: LegendEntry
        if v.operator is None:
            assert len(v.obj_names) == 1
            v_obj = v.obj_names[0]
            k_obj = k.strip()
            # TODO: instead of this hack, check set of objects to see
            # if k in obj_to_idxs:
            if len(k) == 1:
                objs_to_chars[v_obj] = k_obj
                chars_to_objs[k_obj] = v_obj
            else:
                meta_objs[k_obj] = [v_obj]
        elif v.operator.lower() == 'or':
            if k.strip() in v.obj_names:
                logger.error(f"You can't define object {k.strip().upper()} in terms of itself!")
                continue
            meta_objs[k.strip()] = v.obj_names
        elif v.operator.lower() == 'and':
            k = k.strip().lower()
            conjoined_tiles[k] = v.obj_names
        else:
            raise Exception('Invalid LegendEntry operator.')

    for k, v in meta_objs.items():
        # Replace with actual atomic object name if this value has already been established as a shorthand for
        # an actual atomic object
        v_1 = []
        for v_obj in v:
            if v_obj in chars_to_objs:
                v_1.append(chars_to_objs[v_obj])
            else:
                v_1.append(v_obj)
        meta_objs[k] = v_1 
    
    return objs_to_chars, meta_objs, conjoined_tiles, chars_to_objs

def expand_collision_layers(collision_layers, meta_objs, char_to_obj, tree_obj_names):
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
    seen = set()
    cl = []
    # If an object appears multiple times, it belongs to the last collision layer in which we saw it
    for l in collision_layers[::-1]:
        l_1 = []
        for o in l:
            sub_objs = expand_meta_objs([o], meta_objs, char_to_obj)
            # If our meta-object is also atomic (e.g. in `Bug_Exterminator`), and does not recursively refer to itself,
            # we still need to include it in the collision layers.
            if o in tree_obj_names and o not in sub_objs:
                sub_objs = [o] + sub_objs
            for so in sub_objs:
                if so not in seen:
                    seen.add(so)
                    l_1.append(so)
                else:
                    logger.warn(f"Object {so} appears multiple times in collision layers.")
        cl.append(l_1)
    return cl[::-1]

def expand_meta_objs(tile_list: List, meta_objs, char_to_obj):
    assert isinstance(tile_list, list), f"tile_list should be a list, got {type(tile_list)}"
    expanded_meta_objs = []
    seen = set()
    stack = list(tile_list)[::-1]  # So that we start expanding in the right order
    atomic_obj_names = char_to_obj.values()

    while stack:
        # Expand depth-first
        mo = stack.pop(-1)
        if mo in meta_objs and mo not in seen:
            stack.extend(meta_objs[mo])  # defer expanding sub-elements
            seen.add(mo)  # mark the meta-object as seen
        elif mo in char_to_obj:
            obj = char_to_obj[mo]
            if obj not in seen:
                seen.add(obj)
                if obj in meta_objs:
                    stack.extend(meta_objs[obj])
                else:
                    expanded_meta_objs.append(obj)
        else:
            if mo not in seen:
                seen.add(mo)
                expanded_meta_objs.append(mo)
            # Deals with `background = background or yardline` in Touchdown Heroes
            elif mo in atomic_obj_names:
                expanded_meta_objs.append(mo)
    expanded_meta_objs = expanded_meta_objs[::-1]  # Reverse to maintain original order
    return expanded_meta_objs

def get_meta_channel(lvl, obj_idxs):
    """ Return a boolean array indicating whether any of the specified object indices are present in each cell of the level."""
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
    dists = jnp.where(jnp.all(src_coords == -1, axis=-1), np.nan, dists)
    dists = jnp.where(jnp.all(trg_coords == -1, axis=-1), np.nan, dists)
    return dists

def compute_sum_of_manhattan_dists(lvl, src, trg):
    dists = compute_manhattan_dists(lvl, src, trg)
    # Get minimum of each source to any target
    dists = jnp.nanmin(dists, axis=1)
    dists = jnp.where(jnp.isnan(dists), 0, dists)
    sum_dist = jnp.sum(dists, axis=0).astype(np.int32)
    return sum_dist

def compute_min_manhattan_dist(lvl, src, trg):
    dists = compute_manhattan_dists(lvl, src, trg)
    dists = jnp.where(jnp.isnan(dists), jnp.iinfo(np.int32).max, dists)
    min_dist = jnp.min(dists).astype(np.int32)
    return min_dist

def check_all(lvl, src, trg):
    src_channel = get_meta_channel(lvl, src)
    if trg is None:
        return True, 0, 0
    trg_channel = get_meta_channel(lvl, trg)
    win = ~jnp.any(src_channel & ~trg_channel)
    score = jnp.count_nonzero(src_channel & trg_channel)
    heuristic = compute_sum_of_manhattan_dists(lvl, src, trg)
    return win, score, -heuristic

def check_some_on(lvl, src, trg):
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

def check_none(lvl, src):
    src_channel = get_meta_channel(lvl, src)
    win = ~jnp.any(src_channel)
    score = -jnp.count_nonzero(src_channel)
    heuristic = score
    return win, score, heuristic

def check_none_on(lvl, src, trg):
    src_channel = get_meta_channel(lvl, src)
    trg_channel = get_meta_channel(lvl, trg)
    win = ~jnp.any(src_channel & trg_channel)
    score = -jnp.count_nonzero(src_channel & trg_channel)
    heuristic = score
    return win, score, heuristic

# def check_any(lvl, src):
#     src_channel = get_meta_channel(lvl, src)
#     win = jnp.any(src_channel)
#     score = jnp.count_nonzero(src_channel)
#     heuristic = compute_min_manhattan_dist(lvl, src, trg)
#     return win, score, -heuristic

def check_win(lvl, funcs, jit):
    if len(funcs) == 0:
        return False, 0, 0

    def apply_win_condition_func(i, lvl):
        return jax.lax.switch(i, funcs, lvl)

    if jit:
        wins, scores, heuristics = jax.vmap(apply_win_condition_func, in_axes=(0, None))(jnp.arange(len(funcs)), lvl)
    else:
        func_returns = [f(lvl) for f in funcs]
        wins, scores, heuristics = zip(*func_returns)
        wins, scores, heuristics = np.array(wins), np.array(scores), np.array(heuristics)
    return jnp.all(wins), scores.sum(), heuristics.sum()

def gen_check_win(win_conditions: Iterable[WinCondition], obj_to_idxs, meta_objs, char_to_obj, jit=True):
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
        elif win_condition.quantifier in ['some', 'any']:
            if trg is not None:
                func = partial(check_some_on, src=src, trg=trg)
            else:
                func = partial(check_some_exist, src=src)
        elif win_condition.quantifier == 'no' and trg is None:
            func = partial(check_none, src=src)
        elif win_condition.quantifier == 'no':
            func = partial(check_none_on, src=src, trg=trg)
        # elif win_condition.quantifier == 'any':
        #     func = partial(check_any, src=src, trg=trg)
        else:
            raise Exception('Invalid quantifier.')
        funcs.append(func)

    from functools import partial as _partial
    return _partial(check_win, funcs=funcs, jit=jit)


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
    # An array of detected objects, as many as there are overlapping objects in the cell
    # Note that we don't necessarily know these at compile-time because of meta-objects
    # A dictionary of the objects detected, mapping meta-object names to sub-object indices
    detected_meta_objs: dict
    detected_obj_idxs: Optional[chex.Array] = None
    detected_moving_idx: Optional[None] = None

@flax.struct.dataclass
class KernelFnReturn:
    detected_meta_objs: dict
    detected_moving_idx: Optional[int] = None

@flax.struct.dataclass
class LineKernelFnReturn:
    per_line_subkernel_activations: chex.Array
    subkernel_detect_outs: List[KernelFnReturn]
    detected_meta_objs: dict
    detected_moving_idx: Optional[int] = None

@flax.struct.dataclass
class PatternFnReturn:
    """We should only have to fall back to this in multi-kernel patterns. Otherwise, these attributes can be
    disambiguated at runtime by the KernelFnReturn."""
    detected_meta_objs: dict
    detected_moving_idx: int = None

def is_rel_force_in_kernel(k):
    for c in k:
        for rule_content in c:
            for o in rule_content.split(' '):
                if o in ['>', 'v', '^', '<']:
                    return True
    return False

def is_perp_or_par_in_pattern(p):
    for k in p:
        for c in k:
            for object_with_modifier in c:
                modifier = object_with_modifier.split(' ')[0]
                if modifier.lower() in ['perpendicular', 'parallel']:
                    return True
    return False

def get_command_from_pattern(p, objs_to_idxs):
    detected_commands = []
    new_p = []
    for k in p:
        new_k = []
        for c in k:
            new_c = []
            for object_with_modifier in c:
                mod_obj = object_with_modifier.split(' ')
                obj = mod_obj[-1]
                if len(mod_obj) == 1:
                    if obj.lower() in ['win', 'cancel', 'restart', 'again'] and obj.lower() not in objs_to_idxs:
                        detected_commands.append(obj.lower())
                    else:
                        new_c.append(obj)
                else:
                    new_c.append(object_with_modifier)
            new_k.append(new_c)
        new_p.append(new_k)
    assert len(detected_commands) <= 1, (f"Detected multiple commands in pattern: {detected_commands}. "
                                          "Does this make sense?")
    return new_p, detected_commands[0] if len(detected_commands) > 0 else None

def gen_perp_par_subrules(l_kerns, r_kerns):
    new_patterns = [[None, None], [None, None]]
    for i, p in enumerate((l_kerns, r_kerns)):
        new_kerns_a = []
        new_kerns_b = []
        for k in p:
            l_kern_a = []
            l_kern_b = []
            for c in k:
                c_a = []
                c_b = []
                for object_with_modifier in c:
                    object_with_modifier_tpl = object_with_modifier.split(' ')
                    if len(object_with_modifier_tpl) == 2:
                        modifier, obj = object_with_modifier_tpl
                        modifier = modifier.lower()
                        obj = obj.lower()
                        if modifier in ['perpendicular']:
                            c_a.append(f'^ {obj}')
                            c_b.append(f'v {obj}')
                            continue
                        elif modifier in ['parallel']:
                            c_a.append(f'> {obj}')
                            c_b.append(f'< {obj}')
                            continue
                    c_a.append(object_with_modifier)
                    c_b.append(object_with_modifier)
                l_kern_a.append(c_a)
                l_kern_b.append(c_b)
            new_kerns_a.append(l_kern_a)
            new_kerns_b.append(l_kern_b)
        # The first and second patterns are the two directional variants of the parallel/perpendicular rule.
        new_patterns[0][i] = (new_kerns_a)
        new_patterns[1][i] = (new_kerns_b)
    return new_patterns

def player_has_moved(player_idxs, lvl_0, lvl_1, obj_to_idxs, meta_objs, char_to_obj):
    # player_sub_objs = expand_meta_objs(['player'], meta_objs, char_to_obj)
    # player_idxs = [obj_to_idxs[obj] for obj in player_sub_objs]
    player_moved = False
    for player_idx in player_idxs:
        player_i_moved = jnp.sum(lvl_0[player_idx] != lvl_1[player_idx]) > 0
        player_moved = jnp.logical_or(player_moved, player_i_moved)
    if DEBUG:
        jax.debug.print('player moved: {player_moved}', player_moved=player_moved)
    return player_moved

def expand_joint_objs(objs, joint_tiles):
    """Expand a list of objects to include all the objects in the joint tiles."""
    expanded_objs = []
    for obj in objs:
        if obj in joint_tiles:
            expanded_objs.extend(expand_joint_objs(joint_tiles[obj], joint_tiles))
        else:
            expanded_objs.append(obj)
    return expanded_objs

def expand_joint_objs_in_pattern(pattern, joint_tiles):
    new_pattern = []
    for kernel in pattern:
        new_kernel = []
        for cell in kernel:
            new_cell = []
            for rule_content in cell:
                modifier_obj = rule_content.split(' ')
                if len(modifier_obj) == 1:
                    modifier = None
                    obj = modifier_obj[0]
                elif len(modifier_obj) == 2:
                    modifier = modifier_obj[0]
                    obj = modifier_obj[1]
                else:
                    raise Exception(f'Invalid rule_content: {rule_content}. Lark parsing issue?')
                obj = obj.lower()
                if obj in joint_tiles:
                    sub_objs = expand_joint_objs([obj], joint_tiles)
                    for so in sub_objs:
                        if modifier is not None:
                            new_cell.append(f'{modifier} {so}')
                        else:
                            new_cell.append(so)
                else:
                    new_cell.append(rule_content)
            new_kernel.append(new_cell)
        new_pattern.append(new_kernel)
    return new_pattern

@flax.struct.dataclass
class PJState:
    multihot_level: np.ndarray
    win: bool
    score: int
    heuristic: int
    restart: bool
    init_heuristic: int
    prev_heuristic: int
    step_i: int
    rng: chex.PRNGKey

@flax.struct.dataclass
class PJParams:
    level: chex.Array

@flax.struct.dataclass
class PSObs:
    multihot_level: chex.Array
    flat_obs: Optional[chex.Array] = None


def get_alts_to_names(objects):
    alts_to_names = {}
    names_to_alts = {}
    for obj_key, obj in objects.items():
        obj: PSObject
        if obj.alt_names is not None:
            for alt_name in obj.alt_names:
                alts_to_names[alt_name] = obj_key
            if len(obj.alt_names) > 0:
                names_to_alts[obj_key] = obj.alt_names
    return alts_to_names, names_to_alts


def gen_obj_force_masks(n_objs, obj_idxs_to_force_idxs, n_layers):
    # Generate a mask corresponding to the channels denoting the forces that can be applied to a given object
    obj_force_masks = np.zeros((n_objs, n_objs + n_layers * N_FORCES + 1), dtype=bool)
    for i in range(n_objs):
        obj_force_mask = np.zeros((n_objs + n_layers * N_FORCES + 1,), dtype=bool)
        force_idx = obj_idxs_to_force_idxs[i]
        obj_force_mask[force_idx: force_idx + N_FORCES] = 1
        obj_force_masks[i] = obj_force_mask
    return jnp.array(obj_force_masks)

MAX_LOOPS = 200

@flax.struct.dataclass
class LoopRuleBlockState:
    lvl: chex.Array
    applied: bool
    block_app_i: int
    cancelled: bool
    restart: bool
    again: bool
    win: bool
    rng: chex.PRNGKey
    block_i: int
    app_i: int


@flax.struct.dataclass
class RuleState:
    """Returned by an atomic rule (`apply_pattern`)"""
    lvl: chex.Array
    applied: bool
    cancelled: bool
    restart: bool
    again: bool
    win: bool
    rng: chex.PRNGKey

@flax.struct.dataclass
class LoopRuleState:
    """Carried across calls within rule loops."""
    lvl: chex.Array
    applied: bool
    cancelled: bool
    restart: bool
    again: bool
    win: bool
    rng: chex.PRNGKey
    app_i: int  # Number of times a rule was applied in this loop
    rule_i: int  # The index (in the group) of the rule we're applying.
    grp_i: int
    block_i: int

@flax.struct.dataclass
class RuleBlockState:
    lvl: chex.Array
    applied: bool  # Tracks whether any group in the block was applied.
    cancelled: bool
    restart: bool
    again: bool
    win: bool
    rng: chex.PRNGKey
    grp_i: int
    block_i: int


@flax.struct.dataclass
class RuleGroupState:
    """Carried across rule loops (`loop_rule_fn`) within a group.
    `applied` tracks whether any rule in the group was applied.
    """
    lvl: chex.Array
    applied: bool  # Tracks whether any rule in the group was applied (during a looping application of that rule)
    cancelled: bool
    restart: bool
    again: bool
    win: bool
    rng: chex.PRNGKey
    rule_i: int  # The index (in the group) of the rule we're applying.
    grp_i: int
    block_i: int


@flax.struct.dataclass
class LoopRuleGroupState:
    lvl: chex.Array
    applied: bool  # Tracks whether any group in the block was applied.
    cancelled: bool
    restart: bool
    again: bool
    win: bool
    rng: chex.PRNGKey
    app_i: int  # Number of times this group was applied
    grp_i: int
    block_i: int


class PuzzleJaxEnv:
    def __init__(self, tree: PSGameTree, jit: bool = True, level_i: int = 0, max_steps: int = np.inf,
                 debug: bool = False, print_score: bool = True, vmap: bool = True):
        global DEBUG, PRINT_SCORE
        DEBUG, PRINT_SCORE = debug, print_score
        self.jit = jit
        self.vmap = vmap
        self.title = tree.prelude.title
        self._has_randomness = False
        self.tree = tree
        self.levels = tree.levels
        self.level_i = level_i
        self.require_player_movement = tree.prelude.require_player_movement
        if DEBUG:
            print(f"Processing legend for {self.title}")
        obj_to_char, meta_objs, joint_tiles, legend_chars_to_objs = process_legend(tree.legend)
        self.joint_tiles = joint_tiles

        alts_to_names, names_to_alts = get_alts_to_names(tree.objects)
        # alts_to_names, names_to_alts = {}, {}

        self.meta_objs = meta_objs
        self.max_steps = max_steps  
        self.state_history = []  
        self.total_reward = 0  

        # Add to the legend any objects to whose (single-character) keys are specified in their object definition
        for obj_name, obj in tree.objects.items():
            obj_key = obj.legend_key
            if obj_key is not None:
                obj_to_char[obj.name] = obj_key
                if obj.alt_names is not None:
                    for alt_name in obj.alt_names:
                        obj_to_char[alt_name] = obj_key
        self.char_to_obj = {v: k for k, v in obj_to_char.items()}

        if DEBUG:
            print(f"Expanding collision layers for {self.title}")
        self.collision_layers = collision_layers = expand_collision_layers(tree.collision_layers, meta_objs,
                                                                           self.char_to_obj, list(tree.objects.keys()))
        atomic_obj_names = [name for layer in collision_layers for name in layer]
        # dedupe
        # atomic_obj_names = [name for name in tree.objects.keys()]
        # atomic_obj_names = [name for name in atomic_obj_names]
        self.atomic_obj_names = atomic_obj_names
        self.n_objs = len(atomic_obj_names)
        self.n_objs_per_layer = np.zeros((len(self.collision_layers),), dtype=int)
        self.n_objs_prior_to_layer = np.zeros((len(self.collision_layers)), dtype=int)
        self.layer_masks = np.zeros((len(self.collision_layers), self.n_objs), dtype=bool)
        for i, l in enumerate(self.collision_layers):
            self.n_objs_per_layer[i] = len(l)
            self.n_objs_prior_to_layer[i+1:] += len(l)
            n_objs_prior_to_layer_i = self.n_objs_prior_to_layer[i]
            self.layer_masks[i, n_objs_prior_to_layer_i: n_objs_prior_to_layer_i + len(l)] = True
        objs, self.objs_to_idxs, coll_masks, self.obj_idxs_to_force_idxs = assign_vecs_to_objs(collision_layers, atomic_obj_names)

        # Make sure all alternates/names (depending on which was specified in the collision layers) are in the
        # objs_to_idxs dict.
        for obj_key in names_to_alts:
            alt_names = names_to_alts[obj_key]
            if obj_key not in self.objs_to_idxs:
                # Then one alt must be in the objs_to_idxs dict...
                for alt_name in alt_names:
                    if alt_name in self.objs_to_idxs:
                        self.objs_to_idxs[obj_key] = self.objs_to_idxs[alt_name]
                        break
            # Now make sure all the alts are in the objs_to_idxs dict
            for alt_name in alt_names:
                if alt_name not in self.objs_to_idxs:
                    self.objs_to_idxs[alt_name] = self.objs_to_idxs[obj_key]

        self.obj_force_masks = gen_obj_force_masks(self.n_objs, self.obj_idxs_to_force_idxs, len(self.collision_layers))
        
        for obj, sub_objs in meta_objs.items():
            # Meta-objects that are actually just alternate names.
            if DEBUG:
                print(f'sub_objs {sub_objs}')
            sub_objs = expand_meta_objs(sub_objs, meta_objs, self.char_to_obj)
            if len(sub_objs) == 1 and (obj not in self.objs_to_idxs):
                self.objs_to_idxs[obj] = self.objs_to_idxs[sub_objs[0]]
        self.coll_mat = np.einsum('ij,ik->jk', coll_masks, coll_masks, dtype=bool)
        if DEBUG:
            print(f"Generating tick function for {self.title}")
        if DEBUG:
            print(f"Generating check win function for {self.title}")
        self.check_win = gen_check_win(tree.win_conditions, self.objs_to_idxs, meta_objs, self.char_to_obj, jit=self.jit)
        if 'player' in self.objs_to_idxs:
            self.player_idxs = [self.objs_to_idxs['player']]
        elif 'player' in meta_objs:
            player_objs = expand_meta_objs(['player'], meta_objs, self.char_to_obj)
            self.player_idxs = [self.objs_to_idxs[p] for p in player_objs]
        elif 'player' in joint_tiles: 
            sub_objs = joint_tiles['player']
            self.player_idxs = [self.objs_to_idxs[sub_obj] for sub_obj in sub_objs]
        elif 'player' in alts_to_names:
            self.player_idxs = [self.objs_to_idxs[alts_to_names['player']]]
        else: 
            raise ValueError("Cannot figure out what indices to assign to player.")
        self.player_idxs = np.array(self.player_idxs)
        if DEBUG:
            print(f'player_idxs: {self.player_idxs}')
        sprite_stack = []
        if DEBUG:
            print(atomic_obj_names)
            print(self.objs_to_idxs)
        # for obj_name in self.obj_to_idxs:
        for obj_key in atomic_obj_names:
            if obj_key not in tree.objects:
                if obj_key in alts_to_names:
                    meta_objs[obj_key] = [alts_to_names[obj_key]]
                    obj_key = alts_to_names[obj_key]

                elif obj_key in names_to_alts:
                    for alt_name in names_to_alts[obj_key]:
                        if alt_name not in meta_objs:
                            meta_objs[alt_name] = [obj_key]
                        else:
                            meta_objs[alt_name].append(obj_key)
                        if alt_name in tree.objects:
                            obj_key = alt_name
                else:
                    raise ValueError(f"Object {obj_key} not found in tree.objects")
            obj = tree.objects[obj_key]
            if obj.sprite is not None:
                if DEBUG:
                    print(f'rendering pixel sprite for {obj_key}')
                im = render_sprite(obj.colors, obj.sprite)

            else:
                # assert len(obj.colors) == 1
                if len(obj.colors) != 1:
                    logger.warning(f"Object {obj_key} has more than one color, but no sprite. Using first color: {obj.colors[0]}.")
                if DEBUG:
                    print(f'rendering solid color for {obj_key}')
                im = render_solid_color(obj.colors[0])

            # Size the image up a bunch
            im_s = PIL.Image.fromarray(im)
            im_s = im_s.resize((50, 50), PIL.Image.NEAREST)
            im = np.array(im_s)

            if DEBUG:
                temp_dir = 'scratch'
                os.makedirs(temp_dir, exist_ok=True)
                sprite_path = os.path.join(temp_dir, f'sprite_{obj_key}.png')

                im_s.save(sprite_path)

            sprite_stack.append(im)
        self.sprite_stack = np.array(sprite_stack)
        char_legend = {v: k for k, v in obj_to_char.items()}
        # Generate vectors to detect atomic objects
        self.obj_vecs = np.eye(self.n_objs, dtype=bool)
        joint_obj_vecs = []
        self.chars_to_idxs = {obj_to_char[k]: v for k, v in self.objs_to_idxs.items() if k in obj_to_char}
        self.chars_to_idxs.update({k: v for k, v in self.objs_to_idxs.items() if len(k) == 1})

        # Automatically assign unused ASCII characters to joint objects
        ascii_chars = set(chr(i) for i in range(32, 127))
        # Remove already used characters
        ascii_chars -= set(self.chars_to_idxs.keys())

        # Generate vectors to detect joint objects
        for jo, subobjects in joint_tiles.items():
            vec = np.zeros(self.n_objs, dtype=bool)
            subobjects = expand_meta_objs(subobjects, meta_objs, self.char_to_obj)
            for so in subobjects:
                if DEBUG:
                    print(so)
                vec += self.obj_vecs[self.objs_to_idxs[so]]
            joint_idx = self.obj_vecs.shape[0]  # index before append
            self.obj_vecs = np.concatenate((self.obj_vecs, vec[None]), axis=0)
            self.objs_to_idxs[jo] = joint_idx
            if jo in obj_to_char:
                jo_char = obj_to_char[jo]
                self.chars_to_idxs[jo_char] = joint_idx
            elif len(jo) == 1:
                self.chars_to_idxs[jo] = joint_idx

        for char in legend_chars_to_objs:
            if char not in self.chars_to_idxs:
                obj = legend_chars_to_objs[char]
                if obj in self.objs_to_idxs:
                    self.chars_to_idxs[char] = self.objs_to_idxs[obj]
                else:
                    logger.warning(f"Object {obj} not found in objs_to_idxs. Presumably it's a meta-object. "
                                   f"Mapping the character `{char}` the first sub-object's index instead. Hopefully "
                                   " it's not actually used in any level definitions.")
                    sub_objs = expand_meta_objs([obj], meta_objs, self.char_to_obj)
                    self.chars_to_idxs[char] = self.objs_to_idxs[sub_objs[0]]

        if self.jit:
            self.step = jax.jit(self.step)
            self.step_env = jax.jit(self.step_env)
            self.reset = jax.jit(self.reset)
            self.apply_player_force = jax.jit(self.apply_player_force)
            self.render = jax.jit(self.render, static_argnums=(1,))
        self.joint_tiles = joint_tiles

        multihot_level = self.get_level(level_i)
        self.observation_space = Box(low=0, high=1, shape=multihot_level.shape)
        self.action_space = Discrete(5)

    def has_randomness(self):
        return self._has_randomness

    def gen_dummy_obs(self, params):
        return PSObs(
            multihot_level=jnp.zeros(self.observation_space.shape)[None],
            flat_obs=None,
        )

    def char_level_to_multihot(self, level):
        int_level = np.vectorize(lambda x: self.chars_to_idxs[x.lower()])(level)
        multihot_level = self.obj_vecs[int_level]
        multihot_level = rearrange(multihot_level, "h w c -> c h w")

        # Add a default background object everywhere
        background_sub_objs = expand_meta_objs(['background'], self.meta_objs, self.char_to_obj)
        if 'background' in background_sub_objs:
            bg_obj = 'background'
        else:
            flat_int_level = int_level.flatten()
            # bg_obj = background_sub_objs[0]
            for bg_obj in background_sub_objs:
                bg_obj_int = self.objs_to_idxs[bg_obj]
                if bg_obj_int in flat_int_level:
                    break
        bg_idx = self.objs_to_idxs[bg_obj]
        multihot_level[bg_idx] = 1

        multihot_level = multihot_level.astype(bool)
        return multihot_level

    # @partial(jax.jit, static_argnums=(0, 2))
    def render(self, state: PJState, cv2=True):
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

    def reset(self, rng, params: PJParams) -> Tuple[chex.Array, PJState]:
        lvl = params.level
        self.tick_fn = self.gen_tick_fn(lvl.shape[1:])
        again = False
        win, score, init_heuristic = self.check_win(lvl)
        if PRINT_SCORE:
            jax.debug.print('heuristic: {heuristic}, score: {score}, win: {win}', heuristic=init_heuristic, score=score, win=win)
        state = PJState(
            multihot_level=lvl,
            win=jnp.array(False),
            score=jnp.array(0, dtype=jnp.int32),
            heuristic=init_heuristic,
            restart=jnp.array(False),
            step_i=jnp.array(0, dtype=jnp.int32),
            init_heuristic=init_heuristic,
            prev_heuristic=init_heuristic,
            rng=rng,
        )
        if self.tree.prelude.run_rules_on_level_start:
            lvl = self.apply_player_force(-1, state)
            lvl, _, _, _, _, _, rng = self.tick_fn(rng, lvl)
            lvl = lvl[:self.n_objs]
            state = state.replace(multihot_level=lvl, rng=rng)
        obs = self.get_obs(state)
        return obs, state

    def get_obs(self, state):
        obs = PSObs(
            multihot_level=state.multihot_level,
            flat_obs=None,
        )
        return obs

    # @partial(jax.jit, static_argnums=(0))
    def apply_player_force(self, action, state: PJState):
        multihot_level = state.multihot_level
        # Add a dummy collision layer at the front. Add one final channel to mark the player's effect.
        force_map = np.zeros((N_FORCES * (len(self.collision_layers) + 1) + 1, *multihot_level.shape[1:]), dtype=bool)
        
        def place_force(force_map, action):
            # This is a map-shape array of the obj-indices corresponding to the player objects active on these respective cells.
            # The `+ 1` here moves us past the dummy collision layer.
            player_int_mask = (self.player_idxs[...,None,None] + 1) * multihot_level[self.player_idxs].astype(int)

            # Turn the int mask into coords, by flattening it, and appending it with xy coords
            xy_coords = np.indices(force_map.shape[1:])
            xy_coords = xy_coords[:, None].repeat(len(self.player_idxs), axis=1)

            # Create a new dictionary mapping objects to force channels, which adds a dummy collision layer at the front
            # Also, ignore the indices corresponding to object channels since we're dealing directly with the force map.
            obj_idxs_to_force_idxs = np.concat((np.array([0]), self.obj_idxs_to_force_idxs + N_FORCES - self.n_objs))
            obj_idxs_to_force_idxs = jnp.array(obj_idxs_to_force_idxs)

            # This is a map-shaped array of the force-indices (and xy indices) that should be applied, given the player objects at these cells.
            player_force_mask = obj_idxs_to_force_idxs[player_int_mask] + action

            player_force_mask = jnp.concatenate((player_force_mask[None], xy_coords), axis=0)
            player_coords = player_force_mask.reshape(3, -1).T
            force_map = force_map.at[tuple(player_coords.T)].set(True)

            # Similarly, activate the player_effect channel (the last channel) wherever this action is applied.
            player_effect_mask = (player_int_mask > 0).astype(int) * (force_map.shape[0] - 1)
            player_effect_mask = jnp.concatenate((player_effect_mask[None], xy_coords), axis=0)
            player_effect_mask = player_effect_mask.reshape(3, -1).T
            force_map = force_map.at[tuple(player_effect_mask.T)].set(True)

            # force_map_sum = force_map.sum()
            # jax.debug.print('force_map: {force_map}', force_map=force_map)
            # jax.debug.print('force map sum: {force_map_sum}', force_map_sum=force_map_sum)
            return force_map

        # apply movement (<4) and/or action (if not noaction)
        should_apply_force = (action != -1) & (action < 4) | (not self.tree.prelude.noaction)

        if self.jit:
            force_map = jax.lax.cond(
                should_apply_force,
                place_force,
                lambda force_map, _: force_map,
                force_map, action
            )
        else:
            if should_apply_force:
                force_map = place_force(force_map, action)
        # remove the dummy collision layer
        force_map = force_map[N_FORCES:]

        lvl = jnp.concatenate((multihot_level, force_map), axis=0)

        return lvl

    # @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: PJState,
        action: int,
        params: Optional[PJParams] = None,
    ) -> Tuple[chex.Array, PJState, float, bool, dict]:
        """Performs step transitions in the environment."""
        key, key_reset = jax.random.split(key)
        if self.jit:
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
        else:
            obs, state, reward, done, info = self.step_env(key, state, action, params)
            if done:
                obs, state = self.reset(key_reset, params)
        return obs, state, reward, done, info



    # @partial(jax.jit, static_argnums=(0))
    def step_env(self, rng, state: PJState, action, params: Optional[PJParams] = None):
        init_lvl = state.multihot_level.copy()
        lvl = self.apply_player_force(action, state)

        # Actually, just apply the rule function once
        cancelled = False
        restart = False
        final_lvl, tick_applied, turn_app_i, cancelled, restart, tick_win, rng = self.tick_fn(rng, lvl)

        accept_lvl_change = ((not self.require_player_movement) or 
                             player_has_moved(self.player_idxs, init_lvl, final_lvl, self.objs_to_idxs, self.meta_objs, self.char_to_obj)) & ~cancelled
        # if DEBUG:
        #     jax.debug.print('accept level change: {accept_lvl_change}', accept_lvl_change=accept_lvl_change) 

        final_lvl = jax.lax.select(
            accept_lvl_change,
            final_lvl,
            lvl,
        )
        multihot_level = final_lvl[:self.n_objs]

        multihot_level = jax.lax.cond(
            restart,
            lambda: self.reset(rng, params)[1].multihot_level,
            lambda: multihot_level
        )

        win, score, heuristic = self.check_win(multihot_level)
        win = win | tick_win
        if PRINT_SCORE:
            jax.debug.print('heuristic: {heuristic}, score: {score}, win: {win}', heuristic=heuristic, score=score, win=win)

        # reward = (heuristic - state.init_heuristic) / jnp.abs(state.init_heuristic)
        reward = heuristic - state.prev_heuristic
        # reward += 10 if win else 0
        reward = jax.lax.select(win, reward + 1, reward)
        reward = reward.astype(float) - 0.01

        done = win | ((state.step_i + 1) >= self.max_steps)
        info = {}
        state = PJState(
            multihot_level=multihot_level,
            win=win,
            score=score,
            heuristic=heuristic,
            restart=restart,
            step_i=state.step_i + 1,
            init_heuristic=state.init_heuristic,
            prev_heuristic=heuristic,
            rng=rng,
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
        multihot_level = jnp.array(multihot_level)
        return multihot_level

    def gen_subrules_meta(self, rule: Rule, rule_name: str, lvl_shape: Tuple[int, int],):
        if 'random' in rule.prefixes:
            # TODO: Randomize the order of rule rotations within a group
            self._has_randomness = True
        has_right_pattern = len(rule.right_kernels) > 0

        def is_meta_subobj_forceless(obj_idx, m_cell):
            """ `obj_idx` is dynamic."""
            return jnp.sum(jax.lax.dynamic_slice(
                m_cell, (jnp.array(self.obj_idxs_to_force_idxs)[obj_idx],), (N_FORCES,))) == 0

        def is_obj_forceless(obj_idx, m_cell):
            """ `obj_idx` is static."""
            obj_force_mask = self.obj_force_masks[obj_idx]
            return jnp.sum(m_cell[obj_force_mask]) == 0

        ### Functions for detecting regular atomic objects
        # @partial(jax.jit, static_argnames='obj_idx')
        def detect_obj_in_cell(m_cell, obj_idx):
            # active = m_cell[obj_idx] == 1 & is_obj_forceless(obj_idx, m_cell)
            detected = jnp.zeros_like(m_cell)
            active = m_cell[obj_idx]

            def mark_obj_as_detected():
                new_detected = detected.at[obj_idx].set(True)
                # Actually, we don't mark force as detected, since we need to transfer forces if object transforms.
                # obj_force_mask = self.obj_force_masks[obj_idx]
                # new_detected = jnp.where(obj_force_mask, m_cell, new_detected)
                return new_detected

            detected = jax.lax.cond(
                active,
                mark_obj_as_detected,
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
            obj_is_present = m_cell[obj_idx]
            force_is_present = m_cell[self.obj_idxs_to_force_idxs[obj_idx] + force_idx]
            active = obj_is_present & force_is_present
            is_detected = np.zeros(m_cell.shape, dtype=bool)
            is_detected[obj_idx] = 1
            is_detected[self.obj_idxs_to_force_idxs[obj_idx] + force_idx] = 1
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
            obj_idx = jnp.argwhere(objs_vec[:self.n_objs] * m_cell[:self.n_objs] > 0, size=1, fill_value=-1)[0, 0]
            detected = jnp.zeros(m_cell.shape, bool)

            def mark_obj_as_detected():
                new_detected = detected.at[obj_idx].set(True)
                # detected_forces = jax.lax.dynamic_slice(
                #     m_cell,
                #     (self.n_objs + (obj_idx * N_FORCES),),
                #     (N_FORCES,)
                # )
                # new_detected = jax.lax.dynamic_update_slice(
                #     new_detected,
                #     detected_forces,
                #     (self.n_objs + (obj_idx * N_FORCES),)
                # )
                return new_detected

            active = obj_idx != -1
            # obj_idx = jax.lax.select(
            #     active,
            #     detected_vec_idx,
            #     -1,
            # )
            detected = jax.lax.cond(
                active,
                mark_obj_as_detected,
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
        def detect_force_on_meta(m_cell: chex.Array, obj_idxs, force_idx):
            dummy_force_obj_vec = jnp.zeros(self.n_objs + len(self.collision_layers) * N_FORCES + 1, dtype=bool)

            def force_obj_vec_fn(obj_idx):
                force_obj_vec = dummy_force_obj_vec.at[obj_idx].set(True)
                force_obj_vec = force_obj_vec.at[jnp.array(self.obj_idxs_to_force_idxs)[obj_idx] + force_idx].set(True)
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
            is_detected = is_detected.at[obj_idx].set(True)
            is_detected = is_detected.at[jnp.array(self.obj_idxs_to_force_idxs)[obj_idx] + force_idx].set(True)
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

            # FIXME: hm?
            active = ((obj_idx != -1) & active)

            return ObjFnReturn(active=active, detected=detected, obj_idx=obj_idx, force_idx=force_idx)

        def detect_stationary_obj(m_cell, obj_idx):
            """Version of the above that assumes `obj_idx` is static."""

            detected = jnp.zeros(m_cell.shape, dtype=bool)

            obj_is_present = m_cell[obj_idx] == 1
            obj_is_forceless = is_meta_subobj_forceless(obj_idx, m_cell)
            obj_active = obj_is_present & obj_is_forceless

            detected = jax.lax.select(
                obj_active,
                detected.at[obj_idx].set(True),
                detected,
            )
            # note that this takes the last-detected active sub-object
            active_obj_idx = jax.lax.select(
                obj_active,
                obj_idx,
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

        # @partial(jax.jit, static_argnames=('obj_idxs'))
        def detect_stationary_meta(m_cell, obj_idxs):
            # TODO: vmap this?

            def detect_stationary_meta_subobj(m_cell, obj_idx):
                # if not m_cell[obj_idx]:
                #     continue
                # if not is_obj_forceless(obj_idx, m_cell):
                #     continue
                # detected = detected.at[obj_idx].set(1)

                detected = jnp.zeros(m_cell.shape, dtype=bool)

                obj_is_present = m_cell[obj_idx] == 1
                obj_is_forceless = is_meta_subobj_forceless(obj_idx, m_cell)
                obj_active = obj_is_present & obj_is_forceless

                detected = jax.lax.select(
                    obj_active,
                    detected.at[obj_idx].set(True),
                    detected,
                )
                # note that this takes the last-detected active sub-object
                active_obj_idx = jax.lax.select(
                    obj_active,
                    obj_idx,
                    -1,
                )
                return detected, active_obj_idx

            detecteds, active_obj_idxs = jax.vmap(detect_stationary_meta_subobj, in_axes=(None, 0))(m_cell, obj_idxs)
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
        def detect_moving_meta(m_cell, obj_idxs, vertical=False, horizontal=False, orthogonal=False):
            # TODO: vmap this?
            active_obj_idx = -1
            active_force_idx = -1
            detected = jnp.zeros(m_cell.shape, dtype=bool)

            for obj_idx in obj_idxs:

                obj_is_present = m_cell[obj_idx] == 1
                obj_forces = jax.lax.dynamic_slice(m_cell, (jnp.array(self.obj_idxs_to_force_idxs)[obj_idx],), (N_FORCES,))
                # obj_force_mask = self.obj_force_masks[obj_idx]
                # obj_forces = m_cell[obj_force_mask]
                if vertical:
                    vertical_mask = np.array([0, 1, 0, 1, 0], dtype=bool)
                    obj_forces = jnp.logical_and(obj_forces, vertical_mask)
                elif horizontal:
                    horizontal_mask = np.array([1, 0, 1, 0, 0], dtype=bool)
                    obj_forces = jnp.logical_and(obj_forces, horizontal_mask)
                elif orthogonal:
                    orthogonal_mask = np.array([1, 1, 1, 1, 0], dtype=bool)
                    obj_forces = jnp.logical_and(obj_forces, orthogonal_mask)
                force_idx = jnp.argwhere(obj_forces, size=1, fill_value=-1)[0, 0]
                obj_active = obj_is_present & (force_idx != -1)

                active_detected = detected.at[obj_idx].set(True)
                active_detected = active_detected.at[self.obj_idxs_to_force_idxs[obj_idx] + force_idx].set(True)

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
                    lambda: jax.debug.print('detected moving_meta obj_idx: {obj_idx}, force_idx: {moving_idx}',
                                            obj_idx=active_obj_idx, moving_idx=active_force_idx),
                    lambda: None,
                )
            return ObjFnReturn(active=active, detected=detected, obj_idx=active_obj_idx, moving_idx=active_force_idx)

        def detect_moving_obj(m_cell, obj_idx, vertical=False, horizontal=False, orthogonal=False):
            # TODO: vmap this?
            active_obj_idx = -1
            active_force_idx = -1
            detected = jnp.zeros(m_cell.shape, dtype=bool)


            obj_is_present = m_cell[obj_idx] == 1
            obj_forces = jax.lax.dynamic_slice(m_cell, (self.obj_idxs_to_force_idxs[obj_idx],), (N_FORCES,))
            # obj_force_mask = self.obj_force_masks[obj_idx]
            # obj_forces = m_cell[obj_force_mask]
            if vertical:
                vertical_mask = np.array([0, 1, 0, 1, 0], dtype=bool)
                obj_forces = jnp.logical_and(obj_forces, vertical_mask)
            elif horizontal:
                horizontal_mask = np.array([1, 0, 1, 0, 0], dtype=bool)
                obj_forces = jnp.logical_and(obj_forces, horizontal_mask)
            elif orthogonal:
                orthogonal_mask = np.array([1, 1, 1, 1, 0], dtype=bool)
                obj_forces = jnp.logical_and(obj_forces, orthogonal_mask)
            force_idx = jnp.argwhere(obj_forces, size=1, fill_value=-1)[0, 0]
            obj_active = obj_is_present & (force_idx != -1)

            active_detected = detected.at[obj_idx].set(True)
            active_detected = active_detected.at[self.obj_idxs_to_force_idxs[obj_idx] + force_idx].set(True)

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
                    lambda: jax.debug.print('detected moving_meta obj_idx: {obj_idx}, force_idx: {moving_idx}',
                                            obj_idx=active_obj_idx, moving_idx=active_force_idx),
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
            no, force, orthogonal, stationary, action, moving, vertical, horizontal = \
                False, False, False, False, False, False, False, False
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
                    # Detect movement or action (yes, OG PS is kind of weird for this)
                    moving = True
                elif obj in ['orthogonal', 'orthoganal']:
                    # Detect movement only (no, not the same as `perpendicular`).
                    orthogonal = True
                elif obj == 'vertical':
                    vertical = True
                elif obj == 'horizontal':
                    horizontal = True
                else:
                    sub_objs = expand_meta_objs([obj], self.meta_objs, self.char_to_obj)
                    obj_idxs = np.array([self.objs_to_idxs[so] for so in sub_objs])
                    obj_vec = np.zeros((self.n_objs + len(self.collision_layers) * N_FORCES + 1), dtype=bool)
                    obj_vec[obj_idxs] = 1
                    if obj in self.char_to_obj:
                        obj = self.char_to_obj[obj]
                    obj_names.append(obj)
                    # TODO: we can remove these functions to individual objects and apply the more abstract meta-tile versions instead
                    if len(obj_idxs) == 1:
                    # if obj in obj_to_idxs:
                        obj_idx = self.objs_to_idxs[obj]
                        if no:
                            fns.append(partial(detect_no_obj_in_cell, obj_idx=obj_idx))
                            no = False
                        elif force:
                            fns.append(partial(detect_force_on_obj, obj_idx=obj_idx, force_idx=force_idx))
                            force = False
                        elif stationary:
                            fns.append(partial(detect_stationary_obj, obj_idx=obj_idx))
                            stationary = False
                        elif moving:
                            fns.append(partial(detect_moving_obj, obj_idx=obj_idx))
                            moving = False
                        elif orthogonal:
                            fns.append(partial(detect_moving_obj, obj_idx=obj_idx, orthogonal=True))
                            orthogonal = False
                        elif vertical:
                            fns.append(partial(detect_moving_obj, obj_idx=obj_idx, vertical=True))
                            vertical = False
                        elif horizontal:
                            fns.append(partial(detect_moving_obj, obj_idx=obj_idx, horizontal=True))
                            horizontal = False
                        else:
                            fns.append(partial(detect_obj_in_cell, obj_idx=obj_idx))
                    elif obj in self.meta_objs:
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
                        elif orthogonal:
                            fns.append(partial(detect_moving_meta, obj_idxs=obj_idxs, orthogonal=True))
                            orthogonal = False
                        elif vertical:
                            fns.append(partial(detect_moving_meta, obj_idxs=obj_idxs, vertical=True))
                            vertical = False
                        elif horizontal:
                            fns.append(partial(detect_moving_meta, obj_idxs=obj_idxs, horizontal=True))
                            horizontal = False
                        else:
                            fns.append(partial(detect_any_objs_in_cell, objs_vec=obj_vec))
                    else:
                        raise Exception(f'Invalid object `{obj}` in rule.')
            
            # @partial(jax.jit)
            def detect_cell(m_cell):

                # def apply_cell_fn_switch(i):
                #     return jax.lax.switch(i, fns, m_cell)
                # detect_obj_outs: ObjFnReturn = jax.vmap(apply_cell_fn_switch, in_axes=0)(jnp.arange(len(fns)))
                detect_obj_outs: List[ObjFnReturn] = stack_leaves([fn(m_cell=m_cell) for fn in fns])

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
                #         detected = detected.at[n_objs + N_FORCES * detect_obj_out.obj_idx + detect_obj_out.force_idx].set(1)
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

                cell_ret = CellFnReturn(
                    detected=detected,
                    detected_obj_idxs=detect_obj_outs.obj_idx,
                    force_idx=force_idx,
                    detected_meta_objs=detected_meta_objs,
                    detected_moving_idx=detected_moving_idx,
                )
                # jax.debug.print('detected: {detected}', detected=detected)
                return activated, cell_ret

            return detect_cell

        def remove_colliding_objs(m_cell, obj_idx, coll_mat):
            # If any objects on the same collision layer are present in the cell, remove them
            # FIXME: We should make this collision matrix static...
            coll_mat = jnp.array(coll_mat)
            coll_vec = coll_mat[:, obj_idx]
            coll_vec = coll_vec.at[obj_idx].set(False)
            # print('various shapes lol', m_cell.shape, n_objs, obj_idx, coll_vec.shape, coll_mat.shape)
            m_cell = m_cell.at[:self.n_objs].set(m_cell[:self.n_objs] * ~coll_vec)
            return m_cell
        
        if self.jit:
            remove_colliding_objs = jax.jit(remove_colliding_objs)
        else:
            remove_colliding_objs = remove_colliding_objs

        # @partial(jax.jit, static_argnums=(3))
        def project_obj(rng, m_cell, obj_pos: int, cell_i: int, cell_detect_out: CellFnReturn, kernel_detect_out: KernelFnReturn,
                        pattern_detect_out: PatternFnReturn, obj, random=False):
            """
            Project an object into a cell in the output pattern.
                m_cell: an n_channels-size vector of all object and per-object force activations and player effect at the
                    current cell
                obj_pos: the position of the object in the cell
                cell_i: the index of the cell relative to its position in the kernel
                cell_detect_out: the output of the corrsponding cell detection function
            """
            if cell_detect_out.detected_obj_idxs is None:
                detected_obj_idx = -1
            else:
                detected_obj_idx = cell_detect_out.detected_obj_idxs[obj_pos]
            detected_meta_objs = cell_detect_out.detected_meta_objs
            kernel_meta_objs = kernel_detect_out.detected_meta_objs
            pattern_meta_objs = pattern_detect_out.detected_meta_objs
            if DEBUG:
                jax.debug.print('        meta objs: {meta_objs}', meta_objs=detected_meta_objs)
            if random:
                sub_objs = expand_meta_objs([obj], self.meta_objs, self.char_to_obj)
                obj_idxs = np.array([self.objs_to_idxs[so] for so in sub_objs])
                obj_idx = jax.random.choice(rng, obj_idxs, shape=(1,), replace=False)[0]
                rng, _ = jax.random.split(rng)
            else:
                if obj in self.objs_to_idxs:
                    obj_idx = self.objs_to_idxs[obj]
                else:
                    obj_idx = disambiguate_meta(obj, detected_meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)
            if DEBUG:
                jax.debug.print('        projecting obj {obj}, disambiguated index: {obj_idx}', obj=obj, obj_idx=obj_idx)
            if not self.jit:
                if obj_idx == -1:
                    raise RuntimeError(f'Object `{obj}` not found in cell {cell_i}.')
            m_cell = m_cell.at[obj_idx].set(True)

            # Actually, forces of transformed objects are just left there, on the relevant collision layer, I think.
            # def transfer_force(m_cell, obj_idx, detected_obj_idx):
            #     # Reassign any forces belonging to the detected object to the new object
            #     # First, identify the forces of the detected object.
            #     detected_forces = jax.lax.dynamic_slice(
            #         m_cell, (jnp.array(self.obj_idxs_to_force_idxs)[detected_obj_idx],), (N_FORCES,)
            #     )
                
            #     # obj_force_mask = self.obj_force_masks[detected_obj_idx]
            #     # detected_forces = m_cell[obj_force_mask]
            #     # Then remove them from the detected object.
            #     m_cell = jax.lax.dynamic_update_slice(
            #         m_cell, jnp.zeros(N_FORCES, dtype=bool), (jnp.array(self.obj_idxs_to_force_idxs)[detected_obj_idx],),
            #     )
            #     # m_cell = m_cell.at[obj_force_mask].set(0)
            #     # Then copy them to the new object.
            #     m_cell = jax.lax.dynamic_update_slice(
            #         m_cell, detected_forces, (jnp.array(self.obj_idxs_to_force_idxs)[obj_idx],),
            #     )
            #     # m_cell = jnp.where(obj_force_mask, detected_forces, m_cell)
            #     return m_cell

            # m_cell = jax.lax.cond(
            #     detected_obj_idx != -1,
            #     transfer_force,
            #     lambda m_cell, _, __: m_cell,
            #     m_cell, obj_idx, detected_obj_idx
            # )

            m_cell = remove_colliding_objs(m_cell, obj_idx, self.coll_mat)
            return rng, m_cell

        # @partial(jax.jit, static_argnums=(3))
        def project_no_obj(rng, m_cell, obj_pos, cell_i, cell_detect_out: CellFnReturn,
                           kernel_detect_out: KernelFnReturn,
                        pattern_detect_out: PatternFnReturn, obj: str):
            obj_idx = self.objs_to_idxs[obj]
            object_was_present = m_cell[obj_idx]
            # Remove the object
            m_cell = m_cell.at[obj_idx].set(False)
            # Remove any existing forces from the object
            force_mask = self.obj_force_masks[obj_idx]
            # m_cell = m_cell.at[force_mask].set(False)
            # If the object was present, then we remove any forces on the relevant collision layer.
            m_cell = jax.lax.select(
                object_was_present,
                jnp.where(force_mask, False, m_cell),
                m_cell,
            )
            return rng, m_cell

        # @partial(jax.jit, static_argnums=(3))
        def project_no_meta(rng, m_cell: chex.Array, obj_pos, cell_i, cell_detect_out, kernel_detect_out,
                            pattern_detect_out, obj: str):
            cell_meta_objs = cell_detect_out.detected_meta_objs
            kernel_meta_objs = kernel_detect_out.detected_meta_objs
            pattern_meta_objs = pattern_detect_out.detected_meta_objs
            obj_idx = disambiguate_meta(obj, cell_meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)
            if obj_idx is None:
                sub_objs = expand_meta_objs([obj], self.meta_objs, self.char_to_obj)
                obj_idxs = [self.objs_to_idxs[o] for o in sub_objs]
            else:
                obj_idxs = [obj_idx]
            for obj_idx in obj_idxs:
                object_was_present = m_cell[obj_idx]
                m_cell = m_cell.at[obj_idx].set(False)
                # Remove any existing forces from the object
                # Here we need to use a dynamic update slice, since we don't know the obj_idx at compile time
                m_cell = jax.lax.select(
                    object_was_present,
                    jax.lax.dynamic_update_slice(
                        m_cell, jnp.zeros(N_FORCES, dtype=bool), (jnp.array(self.obj_idxs_to_force_idxs)[obj_idx],),
                    ),
                    m_cell,
                )
            return rng, m_cell

        # @partial(jax.jit, static_argnums=(3, 4))
        def project_force_obj(
                rng, m_cell, obj_pos: int, cell_i: int, obj_idx: int, force_idx: int, cell_detect_out: CellFnReturn,
                kernel_detect_out: KernelFnReturn, pattern_detect_out: PatternFnReturn):
            # Add the object
            m_cell = m_cell.at[obj_idx].set(True)
            if force_idx is None:
                # Generate random movement.
                force_idx = jax.random.randint(rng, (1,), 0, N_FORCES-1)[0]
                rng, _ = jax.random.split(rng)

            force_mask = self.obj_force_masks[obj_idx]
            # Remove any existing forces from the object and add the new one
            m_cell = jnp.where(force_mask, False, m_cell)
            m_cell = m_cell.at[self.obj_idxs_to_force_idxs[obj_idx] + force_idx].set(True)
            # Also remove player action mask if it exists.
            m_cell = m_cell.at[-1].set(False)

            m_cell = remove_colliding_objs(m_cell, obj_idx, self.coll_mat)

            if self.jit:
                if DEBUG:
                    jax.debug.print('project_force_on_obj: {obj_idx}', obj_idx=obj_idx)

            else:
                pass
                # jax.debug.print('project_force_on_obj: {obj_idx}', obj_idx=obj_idx)
                # TODO: jax this
                # obj_name = idxs_to_objs[obj_idx]
                # jax.debug.print('project_force_on_obj: {obj_name}', obj_name=obj_name)
                # print(f'project_force_on_obj: {obj_idx}')

            return rng, m_cell

        # @partial(jax.jit, static_argnums=(3, 4))
        def project_force_meta(
                rng, m_cell, obj_pos: int, cell_i: int, obj: str, cell_detect_out: CellFnReturn,
                kernel_detect_out: KernelFnReturn, pattern_detect_out: PatternFnReturn, force_idx):
            meta_objs = cell_detect_out.detected_meta_objs
            kernel_meta_objs = kernel_detect_out.detected_meta_objs
            pattern_meta_objs = pattern_detect_out.detected_meta_objs
            obj_idx = disambiguate_meta(obj, meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)
            # Add the object
            m_cell = m_cell.at[obj_idx].set(True)
            if force_idx is None:
                # Generate random movement.
                force_idx = jax.random.randint(rng, (1,), 0, N_FORCES-1)[0]
                rng, _ = jax.random.split(rng)
            # Remove any existing forces from the object
            # m_cell = jax.lax.dynamic_update_slice(
            #     m_cell, jnp.zeros(N_FORCES, dtype=bool), (self.n_objs + obj_idx * N_FORCES,)
            # )
            # force_mask = self.obj_force_masks[obj_idx]
            # m_cell = jnp.where(force_mask, False, m_cell)

            # Place the new force
            # m_cell = m_cell.at[n_objs + (obj_idx * N_FORCES) + force_idx].set(1)

            # Remove any existing forces from the object and add the new one
            force_arr = jnp.zeros(N_FORCES, dtype=bool)
            force_arr = force_arr.at[force_idx].set(True)
            m_cell = jax.lax.dynamic_update_slice(
                m_cell, force_arr, (jnp.array(self.obj_idxs_to_force_idxs)[obj_idx],),
            )
            # Also remove player action mask
            m_cell = m_cell.at[-1].set(False)

            m_cell = remove_colliding_objs(m_cell, obj_idx, self.coll_mat)

            if self.jit:
                if DEBUG:
                    jax.debug.print('project_force_on_obj: {obj_idx}', obj_idx=obj_idx)

            else:
                pass
                # jax.debug.print('project_force_on_obj: {obj_idx}', obj_idx=obj_idx)
                # TODO: jax this
                # obj_name = idxs_to_objs[obj_idx]
                # jax.debug.print('project_force_on_obj: {obj_name}', obj_name=obj_name)
                # print(f'project_force_on_obj: {obj_idx}')

            return rng, m_cell

        # @partial(jax.jit, static_argnums=(3))
        # TODO: add kernel detect out
        def project_moving_obj(
                rng, m_cell, obj_pos: int, cell_i: int, cell_detect_out: CellFnReturn,
                kernel_detect_out: KernelFnReturn, pattern_detect_out: PatternFnReturn, obj):
            meta_objs = cell_detect_out.detected_meta_objs
            kernel_meta_objs = kernel_detect_out.detected_meta_objs
            pattern_meta_objs = pattern_detect_out.detected_meta_objs
            obj_idx = disambiguate_meta(obj, meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)

            # Look for detected force index in corresponding input cell, then kernel, then pattern.
            # This should never end up as -1 (i.e. there should be a corresponding movement index somewhere in the LHS).
            cell_moving_idx = cell_detect_out.detected_moving_idx
            kernel_moving_idx = kernel_detect_out.detected_moving_idx
            pattern_moving_idx = pattern_detect_out.detected_moving_idx
            if cell_moving_idx is None:
                cell_moving_idx = jnp.array([[-1]])
            if kernel_moving_idx is None:
                kernel_moving_idx = jnp.array([[-1]])
            if pattern_moving_idx is None:
                pattern_moving_idx = jnp.array([[-1]])

            force_idx = cell_moving_idx
            force_idx = jax.lax.select(
                force_idx == -1,
                kernel_moving_idx,
                force_idx,
            )
            force_idx = jax.lax.select(
                force_idx == -1,
                pattern_moving_idx,
                force_idx,
            )
                
            m_cell = m_cell.at[obj_idx].set(True)

            # Remove any existing forces from the object and add the new one
            force_arr = jnp.zeros(N_FORCES, dtype=bool)
            force_arr = force_arr.at[force_idx].set(True)
            # m_cell = m_cell.at[self.n_objs + (obj_idx * N_FORCES) + force_idx].set(1)
            m_cell = jax.lax.dynamic_update_slice(
                m_cell, force_arr, (jnp.array(self.obj_idxs_to_force_idxs)[obj_idx],),
            )

            if DEBUG:
                jax.debug.print('project_moving_obj, obj_idx: {obj_idx}, force_idx: {force_idx}',
                                obj_idx=obj_idx, force_idx=force_idx)
            m_cell = remove_colliding_objs(m_cell, obj_idx, self.coll_mat)
            return rng, m_cell

        def project_stationary_obj(rng, m_cell: chex.Array, obj_pos: int, cell_i: int, cell_detect_out: CellFnReturn,
                                   kernel_detect_out: KernelFnReturn, pattern_detect_out: PatternFnReturn, obj):
            meta_objs = cell_detect_out.detected_meta_objs
            kernel_meta_objs = kernel_detect_out.detected_meta_objs
            pattern_meta_objs = pattern_detect_out.detected_meta_objs
            if obj in self.objs_to_idxs:
                obj_idx = self.objs_to_idxs[obj]
                force_idx = self.obj_idxs_to_force_idxs[obj_idx]
                m_cell = m_cell.at[force_idx: (force_idx + 1) * N_FORCES].set(False)
            else:
                obj_idx = disambiguate_meta(obj, meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)
                m_cell = jax.lax.dynamic_update_slice(
                    m_cell, jnp.zeros(N_FORCES, dtype=bool), (jnp.array(self.obj_idxs_to_force_idxs)[obj_idx],)
                )
            m_cell = m_cell.at[obj_idx].set(True)
            m_cell = remove_colliding_objs(m_cell, obj_idx, self.coll_mat)
            return rng, m_cell

        def gen_cell_projection_fn(r_cell, right_force_idx):
            fns = []
            if r_cell is None:
                r_cell = []
            else:
                r_cell = r_cell.split(' ')
            no, force, moving, stationary, random, vertical, horizontal, random_dir, orthogonal = \
                False, False, False, False, False, False, False, False, False
            for obj in r_cell:
                obj = obj.lower()
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
                elif obj in ['orthogonal', 'orthoganal']:
                    orthogonal = True
                elif obj == 'stationary':
                    stationary = True
                elif obj == 'random':
                    random = True
                    self._has_randomness = True
                elif obj == 'horizontal':
                    horizontal = True
                elif obj == 'vertical':
                    vertical = True
                elif obj == 'randomdir':
                    random_dir = True
                    self._has_randomness = True
                # ignore sound effects (which can exist incide rules (?))
                elif obj.startswith('sfx'):
                    continue
                else:
                    if obj in self.char_to_obj:
                        obj = self.char_to_obj[obj]
                    if (obj in self.objs_to_idxs) or (obj in self.meta_objs):
                        if no:
                            if obj in self.objs_to_idxs:
                                fns.append(partial(project_no_obj, obj=obj))
                            elif obj in self.meta_objs:
                                fns.append(partial(project_no_meta, obj=obj))
                            else:
                                raise Exception(f'Invalid object `{obj}` in rule.')
                            no = False
                        elif force:
                            if obj in self.objs_to_idxs:
                                obj_idx = self.objs_to_idxs[obj]
                                fns.append(partial(project_force_obj, obj_idx=obj_idx, force_idx=force_idx))
                            else:
                                fns.append(partial(project_force_meta, obj=obj, force_idx=force_idx))
                            force = False
                        elif moving:
                            fns.append(partial(project_moving_obj, obj=obj))
                            moving = False
                        elif orthogonal:
                            fns.append(partial(project_moving_obj, obj=obj))
                            orthogonal = False
                        elif vertical:
                            fns.append(partial(project_moving_obj, obj=obj))
                            vertical = False
                        elif horizontal:
                            fns.append(partial(project_moving_obj, obj=obj))
                            horizontal = False
                        elif stationary:
                            fns.append(partial(project_stationary_obj, obj=obj))
                            stationary = False
                        elif random:
                            fns.append(partial(project_obj, obj=obj, random=True))
                            random = False
                        elif random_dir:
                            fns.append(partial(project_force_meta, obj=obj, force_idx=None))
                            random_dir = False
                        else:
                            fns.append(partial(project_obj, obj=obj))
                    else:
                        raise Exception(f'Invalid object `{obj}` in rule.')
            
            # @partial(jax.jit, static_argnums=())
            def project_cell(rng, m_cell, cell_i, cell_detect_out, kernel_detect_out, pattern_detect_out):
                # This is hackish. Instead, we should probably expand rules out to include `no objA` if objA is present
                # on the left side but not the right.
                m_cell = m_cell & ~cell_detect_out.detected
                assert len(m_cell.shape) == 1, f'Invalid cell shape {m_cell.shape}'
                for obj_pos, proj_fn in enumerate(fns):
                    rng, m_cell = proj_fn(rng=rng, obj_pos=obj_pos, cell_i=cell_i, m_cell=m_cell, cell_detect_out=cell_detect_out,
                                        kernel_detect_out=kernel_detect_out, pattern_detect_out=pattern_detect_out)
                # removed_something = jnp.any(cell_detect_out.detected)
                # jax.lax.cond(
                #     removed_something,
                #     lambda: jax.debug.print('removing detected: {det}', det=detect_out.detected),
                #     lambda: None
                # )
                # jax.debug.print('      removing detected: {det}', det=cell_detect_out.detected)
                return rng, m_cell

            return project_cell

        def gen_rotated_rule_fn(lps, rps, rot, r_command):

            def gen_rotated_line_kernel_fns(lp, rp, rot):
                lp_is_horizontal = lp.shape[0] == 1 and lp.shape[1] > 1
                lp_is_vertical = lp.shape[0] > 1 and lp.shape[1] == 1
                assert lp_is_horizontal or lp_is_vertical, f'Invalid kernel shape {lp.shape} for a kernel including a line detector.'
                lp_subkernels = [[]]
                for l_cell_row in lp:
                    for l_cell in l_cell_row:
                        if l_cell == '...':
                            lp_subkernels.append([])
                        else:
                            lp_subkernels[-1].append(l_cell)
                    rp_subkernels = [[]]
                if rp is not None:
                    for r_cell_row in rp:
                        for r_cell in r_cell_row:
                            if r_cell == '...':
                                rp_subkernels.append([])
                            else:
                                rp_subkernels[-1].append(r_cell)
                else:
                    rp_subkernels = [[None] * len(lp_subkernels[i]) for i in range(len(lp_subkernels))]
                # Now put these subkernels back in the correct shape
                lp_subkernels = [np.array(lps) for lps in lp_subkernels]
                rp_subkernels = [np.array(rps) for rps in rp_subkernels]
                if lp_is_horizontal:
                    lp_subkernels = [lps[None] for lps in lp_subkernels]
                    rp_subkernels = [rps[None] for rps in rp_subkernels]
                    row_len = np.sum([np.sum([len(c) for c in k if c[0] != '...']) for k in lp_subkernels])
                    if row_len > lvl_shape[1]:
                        return None, None
                elif lp_is_vertical:
                    lp_subkernels = [lps[:, None] for lps in lp_subkernels]
                    rp_subkernels = [rps[:, None] for rps in rp_subkernels]
                    col_len = np.sum([np.sum([len(c) for c in k if c[0] != '...']) for k in lp_subkernels])
                    if col_len > lvl_shape[0]:
                        return None, None
                
                subkernel_detection_fns = []
                subkernel_projection_fns = []
                for lp, rp in zip(lp_subkernels, rp_subkernels):
                    subkernel_detection_fn, subkernel_projection_fn = gen_rotated_kernel_fns(lp, rp, rot)
                    if subkernel_detection_fn is None and subkernel_projection_fn is None:
                        return None, None
                        
                    subkernel_detection_fns.append(subkernel_detection_fn)
                    subkernel_projection_fns.append(subkernel_projection_fn)
                
                detect_kernel = partial(
                    self.detect_line_kernel,
                    subkernel_detection_fns=subkernel_detection_fns,
                    lp_is_horizontal=lp_is_horizontal,
                    lp_is_vertical=lp_is_vertical,
                    rule_name=rule_name,
                    rot=rot,
                    subkernel_sizes=[max(lp.shape) for lp in lp_subkernels],
                )
                project_kernel = partial(
                    self.project_line_kernel,
                    subkernel_projection_fns=subkernel_projection_fns,
                    lp_is_horizontal=lp_is_horizontal,
                    lp_is_vertical=lp_is_vertical,
                    random='random' in rule.prefixes,
                )
                return detect_kernel, project_kernel


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
                    row_len = lp.shape[0]
                    if row_len > lvl_shape[1]:
                        return None, None
                elif lp_is_vertical:
                    lp = lp[:, 0]
                    col_height = lp.shape[0]
                    if col_height > lvl_shape[0]:
                        return None, None
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
                if lp is None:
                    cell_detection_fns.append(
                        lambda m_cell: (
                            True,  # detected by default
                            CellFnReturn(detected=jnp.zeros_like(m_cell), force_idx=None, detected_meta_objs={})  # empty cell return
                        )
                    )
                elif len(lp.shape) == 1:
                    for i, l_cell in enumerate(lp):
                        if l_cell == '...':
                            is_line_detector = True
                            cell_detection_fns.append('...')
                            # TODO
                            raise NotImplementedError('Line detector not implemented yet')    
                        if l_cell is not None:
                            cell_detection_fns.append(gen_cell_detection_fn(l_cell, force_idx))
                        else:
                            cell_detection_fns.append(
                                lambda m_cell: (
                                    True,
                                    CellFnReturn(detected=jnp.zeros_like(m_cell), force_idx=None, detected_meta_objs={}),
                                )
                            )
                # FIXME: why is the normal way broken here?
                # if has_right_pattern:
                if has_right_pattern and rp is None and lp is not None:
                    rp = np.array([None] * len(lp))
                if has_right_pattern:
                    if rp is not None:
                        for i, r_cell in enumerate(rp):
                            if r_cell == '...':
                                # assert is_line_detector, f"`...` not found in left pattern of rule {rule_name}"
                                if not is_line_detector:
                                    logger.warn(f'`...` found in right pattern of rule {rule_name}, but not in left pattern. Removing it on the right side for now.')
                                else:
                                    cell_projection_fns.append('...')
                            else:
                                cell_projection_fns.append(gen_cell_projection_fn(r_cell, force_idx))


                detect_kernel = partial(
                    self.detect_kernel,
                    in_patch_shape=in_patch_shape,
                    cell_detection_fns=cell_detection_fns,
                    lp_is_horizontal=lp_is_horizontal,
                    lp_is_vertical=lp_is_vertical,
                    lp_is_single=lp_is_single,
                    lp=lp, rule_name=rule_name,
                )

                project_kernel = partial(
                    self.project_kernel,
                    in_patch_shape=in_patch_shape,
                    cell_projection_fns=cell_projection_fns,
                    lp_is_horizontal=lp_is_horizontal,
                    lp_is_vertical=lp_is_vertical,
                    lp_is_single=lp_is_single,
                    has_right_pattern=has_right_pattern,
                    rule_name=rule_name,
                    random='random' in rule.prefixes,
                )
                return detect_kernel, project_kernel


            if not has_right_pattern:
                rps = [None] * len(lps)
            if DEBUG:
                print('rps', rps)
                print('lps', lps)

            # kernel_fns = [gen_rotated_kernel_fns(lp, rp, rot) for lp, rp in zip(lps, rps)]
            kernel_fns = []
            for lp, rp in zip(lps, rps):
                if is_line_detector_in_kernel(lp):
                    det_proj_fns = gen_rotated_line_kernel_fns(lp, rp, rot)
                    if det_proj_fns == (None, None):
                        return None
                    kernel_fns.append(gen_rotated_line_kernel_fns(lp, rp, rot))
                else:
                    det_proj_fns = gen_rotated_kernel_fns(lp, rp, rot)
                    if det_proj_fns == (None, None):
                        return None
                    kernel_fns.append(gen_rotated_kernel_fns(lp, rp, rot))
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
                    # FIXME: We should be stacking then maxing these too. Currently we might overwrite a meta-obj dict entry
                    # with one from a kernel where the meta-obj is not detected? (Wait, is this ever a thing?)
                    boardwide_kernel_meta_objs = {k: v.max() for k, v in kernel_detect_out.detected_meta_objs.items()}
                    detected_pattern_meta_objs.update(boardwide_kernel_meta_objs)

                    # Propagate the detected moving index across kernels.
                    detected_pattern_moving_idxs = [
                        kernel_detect_out.detected_moving_idx for kernel_detect_out in kernel_detect_outs
                        if kernel_detect_out.detected_moving_idx is not None]
                    if len(detected_pattern_moving_idxs) == 0:
                        # No detected moving idxs in the kernel
                        detected_pattern_moving_idx = None
                    else:
                        # If these tensors do not all have the same shape, then pad them with `-1s` as necessary
                        max_detected_pattern_moving_idx_shape = max([k.shape for k in detected_pattern_moving_idxs])     
                        detected_pattern_moving_idxs = [
                            jnp.pad(k, 
                                    ((0, max_detected_pattern_moving_idx_shape[0] - k.shape[0]),
                                     (0, max_detected_pattern_moving_idx_shape[1] - k.shape[1]),
                                     (0, 0),
                                     (0, 0),
                                     )) 
                            for k in detected_pattern_moving_idxs]

                        detected_pattern_moving_idxs = jnp.stack(detected_pattern_moving_idxs, axis=0)
                        detected_pattern_moving_idx = jnp.max(detected_pattern_moving_idxs, axis=0)
                        # Now we have a board-shaped map of all the moving indices detected by *any* kernel.
                        # For a pattern-wide (presumed multi-kernel) function return, we can take the max of these indices to 
                        # get one board-wide detected moving index.
                        detected_pattern_moving_idx = jnp.max(detected_pattern_moving_idx, axis=(0,1))

                pattern_out = PatternFnReturn(
                    detected_meta_objs=detected_pattern_meta_objs,
                    detected_moving_idx=detected_pattern_moving_idx,
                )
                return kernel_activations, cell_detect_outs, kernel_detect_outs, pattern_out

            def apply_pattern(rng, lvl):
                kernel_activations, cell_detect_outs, kernel_detect_outs, pattern_detect_out = detect_pattern(lvl)
                pattern_detected = jnp.all(jnp.sum(kernel_activations, axis=(1,2)) > 0)

                def project_kernels(rng, lvl, kernel_activations, kernel_detect_outs):
                    for i, kernel_projection_fn in enumerate(kernel_projection_fns):
                        if DEBUG:
                            jax.debug.print('        projecting kernel {i}', i=i)
                        rng, lvl = kernel_projection_fn(
                            rng, lvl, kernel_activations[i], cell_detect_outs[i], kernel_detect_outs[i], pattern_detect_out)
                    return rng, lvl

                cancel, restart, again, win = False, False, False, False
                if has_right_pattern:
                    if self.jit:
                        rng, next_lvl = jax.lax.cond(
                            pattern_detected,
                            project_kernels,
                            lambda rng, lvl, pattern_activations, pattern_detect_outs: (rng, lvl),
                            rng, lvl, kernel_activations, kernel_detect_outs,
                        )
                    else:
                        if pattern_detected:
                            rng, next_lvl = project_kernels(rng, lvl, kernel_activations, kernel_detect_outs)
                        else:
                            next_lvl = lvl

                    rule_applied = jnp.any(next_lvl != lvl)
                else:
                    next_lvl = lvl
                    rule_applied = False
                    # assert rule.command is not None
                    if r_command is None and not np.all([r is None for r in rps]):
                        print(rps)

                    if r_command == 'cancel':
                        cancel = pattern_detected
                    elif r_command == 'restart':
                        restart = pattern_detected
                if r_command == 'again':
                    # Again will be applied as long as left pattern is detected, until the entire turn has no effect 
                    # on the level.
                    again = pattern_detected
                    if DEBUG:
                        jax.debug.print('      applying the {command} command: {rule_applied}', command=r_command, rule_applied=rule_applied)
                elif r_command == 'win':
                    win = pattern_detected

                rule_state = RuleState(
                    lvl=next_lvl,
                    applied=rule_applied,
                    cancelled=cancel,
                    restart=restart,
                    again=again,
                    win=win,
                    rng=rng
                )

                return rule_state

            return apply_pattern

        rule_fns = []
        if DEBUG:
            print('RULE PATTERNS', rule.left_kernels, rule.right_kernels)


        l_kerns, r_kerns = rule.left_kernels, rule.right_kernels

        # Expand joint objects in kernels into the corresponding subtitles (repeating object modifiers as necessary)
        l_kerns, r_kerns = expand_joint_objs_in_pattern(l_kerns, self.joint_tiles), \
            expand_joint_objs_in_pattern(r_kerns, self.joint_tiles)

        # Expand into appropriate subrules (with relative forces) if perpendicular or parallel keywords are present.
        if is_perp_or_par_in_pattern(l_kerns):
            pattern_tpls = gen_perp_par_subrules(l_kerns, r_kerns)
        else:
            pattern_tpls = [(l_kerns, r_kerns)]

        new_pattern_tpls = []
        for l_kerns, r_kerns in pattern_tpls:
            # This is not actually syntactically correct, but OG PS admits it.
            r_kerns, in_rule_command = get_command_from_pattern(r_kerns, self.objs_to_idxs)
            new_pattern_tpls.append((l_kerns, r_kerns))
            if in_rule_command is not None:
                if rule.command is not None:
                    assert rule.command == in_rule_command, (f"Rule {rule_name} has the {in_rule_command} command in the "
                    f"right pattern, but the command in the rule definition is {rule.command}. ")
                rule.command = in_rule_command
        pattern_tpls = new_pattern_tpls

        for l_kerns, r_kerns in pattern_tpls:
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
                rots = [3, 1]
            elif 'left' in rule.prefixes:
                rots = [3]
            elif 'right' in rule.prefixes:
                rots = [1]
            elif 'vertical' in rule.prefixes:
                rots = [2, 0]
            elif 'up' in rule.prefixes:
                rots = [2]
            elif 'down' in rule.prefixes:
                rots = [0]
            # TODO: Remove unnecessary rotated variations of single-cell kernels
            # elif np.any(np.array([len(kernel) for kernel in l_kerns]) > 1):
            else:
                if np.all(np.array([len(lp) for lp in rule.left_kernels]) == 1) and not \
                    np.any([is_rel_force_in_kernel(lp) for lp in rule.left_kernels]) and not \
                    np.any([is_rel_force_in_kernel(rp) for rp in rule.right_kernels]):
                    rots = [0]
                else:
                    rots = [2, 0, 3, 1]
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
                rule_fn = gen_rotated_rule_fn(l_kerns_rot, r_kerns_rot, rot, 
                                              r_command=rule.command.lower() if rule.command is not None else None)
                if rule_fn is None:
                    continue
                rule_fns.append(rule_fn)

        return rule_fns
            

    def gen_tick_fn(self, lvl_shape):
        rule_blocks = []
        late_rule_grps = []
        for rule_block in self.tree.rules:

            # FIXME: what's with this unnecessary list?
            assert len(rule_block) == 1
            rule_block = rule_block[0]

            looping = rule_block.looping
            rule_grps = []
            last_subrule_fns_were_late = None
            for rule in rule_block.rules:
                sub_rule_fns = self.gen_subrules_meta(rule, rule_name=str(rule), lvl_shape=lvl_shape)
                if '+' in rule.prefixes:
                    # I'm not actually clear on how PS handles these groups combining late/non-late rules, so have just
                    # taken a best guess here (which seems to agree with game `Teh_Interwebs`).
                    if last_subrule_fns_were_late is None:
                        logger.warn(
                            (f'Initial rule has `+` prefix, but no rule precedes it, so ignoring `+` and adding this rule'
                            ' as a new rule group.')
                        )
                        rule_grps.append(sub_rule_fns)
                    if 'late' in rule.prefixes:
                        if last_subrule_fns_were_late:
                            late_rule_grps[-1].extend(sub_rule_fns)
                        else:
                            logger.warn(
                                (f'Attempting to add `late` rule to a non-late rule. Ignoring `+` and creating a new '
                                '`late` rule group.')
                            )
                            late_rule_grps.append(sub_rule_fns)
                            last_subrule_fns_were_late = True
                    else:
                        if last_subrule_fns_were_late:
                            logger.warn(
                                (f'Attempting to add `+` non-late rule to a late rule. Ignoring `+` and creating a new '
                                'non-late rule group.')
                            )
                            rule_grps.append(sub_rule_fns)
                            last_subrule_fns_were_late = False
                        else:
                            rule_grps[-1].extend(sub_rule_fns)
                elif 'late' in rule.prefixes:
                    late_rule_grps.append(sub_rule_fns)
                    last_subrule_fns_were_late = True
                else:
                    rule_grps.append(sub_rule_fns)
                    last_subrule_fns_were_late = False
            rule_blocks.append((looping, rule_grps))

        _move_rule_fn = partial(self.apply_movement, coll_mat=self.coll_mat,
                                n_objs=self.n_objs, obj_force_masks=self.obj_force_masks, jit=self.jit)
        rule_blocks.append((False, [[_move_rule_fn]]))
        # Can we have loops in late rules? I hope not.
        rule_blocks.append((False, late_rule_grps))

        all_rule_fns = [rule_fn for looping, rule_grps in rule_blocks for rule_grp in rule_grps for rule_fn in rule_grp]
        n_rules_counted = 0
        # n_prior_rules = {}
        max_n_grps = max([len(rule_grps) for _, rule_grps in rule_blocks])
        n_prior_rules_arr = np.zeros((len(rule_blocks), max_n_grps), dtype=jnp.int32)
        n_rules_per_grp_arr = np.zeros((len(rule_blocks), max_n_grps), dtype=jnp.int32)
        n_grps_per_block_arr = np.zeros((len(rule_blocks),), dtype=jnp.int32)
        for rule_block_i, rule_block in enumerate(rule_blocks):
            _, rule_grps = rule_block
            n_grps_per_block_arr[rule_block_i] = len(rule_grps)
            for rule_grp_i, rule_grp in enumerate(rule_grps):
                # n_prior_rules[(rule_block_i, rule_grp_i)] = n_rules_counted
                n_prior_rules_arr[rule_block_i, rule_grp_i] = n_rules_counted
                n_rules_per_grp_arr[rule_block_i, rule_grp_i] = len(rule_grp)
                n_rules_counted += len(rule_grp)
        n_prior_rules_arr = jnp.array(n_prior_rules_arr)
        n_rules_per_grp_arr = jnp.array(n_rules_per_grp_arr)
        n_grps_per_block_arr = jnp.array(n_grps_per_block_arr)
        blocks_are_looping_lst = jnp.array([looping for looping, _ in rule_blocks])

        def tick_fn(rng, lvl):
            lvl_changed = False
            cancelled = False
            restart = False
            again = False
            lvl = lvl[None]

            if not self.jit:
                if DEBUG:
                    print('\n' + multihot_to_desc(lvl[0], self.objs_to_idxs, self.n_objs, obj_idxs_to_force_idxs=self.obj_idxs_to_force_idxs))

            def apply_turn(carry):
                init_lvl, _, turn_app_i, cancelled, restart, turn_again, win, rng = carry
                lvl = init_lvl
                turn_app_i += 1
                applied = False
                block_again = False

                # for block_i, (looping, rule_grps) in enumerate(rule_blocks):
                    # n_prior_rules
                
                _loop_rule_block = partial(
                    self.loop_rule_block,
                    ### COMPILE VS RUNTIME ###
                    # n_prior_rules_arr=n_prior_rules_arr, n_rules_per_grp_arr=n_rules_per_grp_arr, all_rule_fns=all_rule_fns,
                    # n_grps_per_block_arr=n_grps_per_block_arr, blocks_are_looping_lst=blocks_are_looping_lst,
                    ### COMPILE VS RUNTIME ###
                )

                ### COMPILE VS RUNTIME ###
                for block_i, (looping, rule_block) in enumerate(rule_blocks):
                    carry = (lvl, applied, block_again, cancelled, restart, win, rng, block_i)
                    lvl, applied, block_again, cancelled, restart, win, rng, block_i = _loop_rule_block(
                        carry=carry,
                        rule_block=rule_block,
                        looping=looping,
                    )
                    if DEBUG:
                        jax.debug.print(
                            'apply_turn: block {block_i} applied: {applied}. again: {again}',
                            block_i=block_i, applied=applied, again=block_again)

                # block_i = 0
                # carry = (lvl, applied, again, cancelled, restart, win, rng, block_i)
                # if self.jit:
                #     # Apply (loop) each block in sequence.
                #     carry = jax.lax.while_loop(
                #         cond_fun=lambda x: ~x[3] & ~x[4] & (x[7] < len(rule_blocks)),
                #         body_fun=_loop_rule_block,
                #         init_val=carry,
                #     )
                #     # carry, _ = jax.lax.scan(
                #     #     loop_rule_block,
                #     #     init=init_carry,
                #     #     xs=jnp.arange(len(rule_blocks)),
                #     # )
                # else:
                #     while not cancelled and not restart and block_i < len(rule_blocks):
                #         carry = _loop_rule_block(carry)
                #         lvl, applied, again, cancelled, restart, win, rng, block_i = carry
                #         if DEBUG:
                #             print(f'      block {block_i} applied: {applied}. again: {again}')

                # lvl, applied, again, cancelled, restart, win, rng, block_i = carry
                ### COMPILE VS RUNTIME ###

                turn_applied = jnp.any(lvl != init_lvl)

                return lvl, turn_applied, turn_app_i, cancelled, restart, block_again, win, rng

            turn_applied = True
            turn_app_i = 0
            turn_again = True
            win = False

            carry = (lvl, turn_applied, turn_app_i, cancelled, restart, turn_again, win, rng)
            if not self.jit:
                while (turn_again and turn_applied):
                    carry = apply_turn(carry)
                    lvl, turn_applied, turn_app_i, cancelled, restart, turn_again, win, rng = carry
            else:
                carry = jax.lax.while_loop(
                    cond_fun=lambda x: ~x[3] & ~x[4] & (x[5] & x[1]),
                    body_fun=apply_turn,
                    init_val=carry,
                )
            lvl, turn_applied, turn_app_i, cancelled, restart, turn_again, win, rng = carry

            # if not jit:
            #     print('\nLevel after applying rules:\n', multihot_to_desc(lvl[0], obj_to_idxs, n_objs))
            #     print('grp_applied:', grp_applied)
            #     print('cancelled:', cancelled)

            return lvl[0], turn_applied, turn_app_i, cancelled, restart, win, rng

        if self.jit:
            tick_fn = jax.jit(tick_fn)
        return tick_fn

    def detect_line_kernel(self, lvl: chex.Array, subkernel_detection_fns: List[Callable], lp_is_vertical: bool,
                           lp_is_horizontal: bool, rule_name: str, rot: int, subkernel_sizes: List[int]):
        """
            subkernel_detection_fns: List of functions that detect the subkernels in the line kernel. These need to be detected in sequence, along some line.
        """
        subkernel_activations = []
        subkernel_cell_detect_outs = []
        subkernel_detect_outs = []
        for subkernel_detection_fn in subkernel_detection_fns:
            subkernel_activations_i, cell_detect_outs_i, subkernel_detect_out_i = subkernel_detection_fn(lvl)
            subkernel_activations.append(subkernel_activations_i)
            subkernel_cell_detect_outs.append(cell_detect_outs_i)
            subkernel_detect_outs.append(subkernel_detect_out_i)

        if not np.all([k.shape == subkernel_activations[0].shape for k in subkernel_activations]):
            # Then pad the subkernel activations to the same shape
            max_shape = max([k.shape for k in subkernel_activations])
            subkernel_activations = [jnp.pad(k, ((0, max_shape[0] - k.shape[0]), (0, max_shape[1] - k.shape[1]))) 
                                     for k in subkernel_activations]
        subkernel_activations = jnp.stack(subkernel_activations, axis=0)

        if lp_is_vertical:
            dim = 2
            one_hot_masks = gen_one_hot_masks(*subkernel_activations[:, :, 0].shape, rot=rot, spacings=subkernel_sizes[:-1])

        elif lp_is_horizontal:
            dim = 1
            one_hot_masks = gen_one_hot_masks(*subkernel_activations[:, 0].shape, rot=rot, spacings=subkernel_sizes[:-1])

        # (n_possible_lines, n_subkernels, height, width)
        per_line_subkernel_activations = jnp.zeros((one_hot_masks.shape[0], *subkernel_activations.shape))

        if not self.jit:
            for i in range(subkernel_activations.shape[dim]):
                if dim == 1:
                    line_activations = subkernel_activations[:, i]
                    per_line_subkernel_activations = per_line_subkernel_activations.at[:, :, i].set(
                        mask_to_valid_sequences(line_activations, jit=self.jit, one_hot_masks=one_hot_masks))
                elif dim == 2:
                    line_activations = subkernel_activations[:, :, i]
                    per_line_subkernel_activations = per_line_subkernel_activations.at[:, :, :, i].set(
                        mask_to_valid_sequences(line_activations, jit=self.jit, one_hot_masks=one_hot_masks))
        else:
            per_line_subkernel_activations = jax.vmap(mask_to_valid_sequences, in_axes=(dim, None, None))(
                subkernel_activations, one_hot_masks, True)
            if dim == 1:
                per_line_subkernel_activations = jnp.transpose(per_line_subkernel_activations, (1, 2, 0, 3))
            elif dim == 2:
                per_line_subkernel_activations = jnp.transpose(per_line_subkernel_activations, (1, 2, 3, 0))

        detected_kernel_meta_objs = {}
        detected_kernel_moving_idx = None

        for subkernel_detect_out in subkernel_detect_outs:
            # Each kernel has detected meta-objects at different coordinates on the board.
            # To get pattern-wide meta-objs, take any detected meta-object index that is not -1 (indicating no meta-object was detected) 
            # (We can take the max here because we assume that if a meta-tile in the right pattern is not specified in the corresponding left kernel, it is only specified once in the rest of the left pattern)
            # FIXME: We should be stacking then maxing these too. Currently we might overwrite a meta-obj dict entry
            # with one from a kernel where the meta-obj is not detected? (Wait, is this ever a thing?)
            boardwide_kernel_meta_objs = {k: v.max() for k, v in subkernel_detect_out.detected_meta_objs.items()}
            detected_kernel_meta_objs.update(boardwide_kernel_meta_objs)

            # Propagate the detected moving index across kernels.
            detected_kernel_moving_idxs = [
                subkernel_detect_out.detected_moving_idx for subkernel_detect_out in subkernel_detect_outs
                if subkernel_detect_out.detected_moving_idx is not None]
            if len(detected_kernel_moving_idxs) == 0:
                # No detected moving idxs in the kernel
                detected_kernel_moving_idx = None
            else:
                # If these tensors do not all have the same shape, then pad them with `-1s` as necessary
                max_detected_kernel_moving_idx_shape = max([k.shape for k in detected_kernel_moving_idxs])     
                detected_kernel_moving_idxs = [
                    jnp.pad(k, 
                            ((0, max_detected_kernel_moving_idx_shape[0] - k.shape[0]),
                                (0, max_detected_kernel_moving_idx_shape[1] - k.shape[1]),
                                (0, 0),
                                (0, 0),
                                )) 
                    for k in detected_kernel_moving_idxs]

                detected_kernel_moving_idxs = jnp.stack(detected_kernel_moving_idxs, axis=0)
                detected_kernel_moving_idx = jnp.max(detected_kernel_moving_idxs, axis=0)
        
        kernel_detect_out = LineKernelFnReturn(
            per_line_subkernel_activations=per_line_subkernel_activations,
            subkernel_detect_outs=subkernel_detect_outs,
            detected_meta_objs=detected_kernel_meta_objs,
            detected_moving_idx=detected_kernel_moving_idx,
        )
        kernel_activations = per_line_subkernel_activations.any(axis=(0,1))

        return kernel_activations, subkernel_cell_detect_outs, kernel_detect_out


    def project_line_kernel(self, rng, lvl, kernel_activations, subkernel_cell_detect_outs, kernel_detect_out: LineKernelFnReturn, pattern_detect_out,
                            subkernel_projection_fns: List[Callable], lp_is_vertical: bool, lp_is_horizontal: bool,
                            random: bool = False):

        valid_line_idxs = jnp.argwhere(kernel_detect_out.per_line_subkernel_activations.any(axis=(1,2,3)),
                                       size=kernel_detect_out.per_line_subkernel_activations.shape[0]+1, fill_value=-1)

        def project_line_i(carry):
            lvl, line_applied, i = carry
            init_lvl = lvl
            line_idx = valid_line_idxs[i][0]
            subkernel_activations = kernel_detect_out.per_line_subkernel_activations[line_idx]
            for subkern_i, subkernel_projection_fn in enumerate(subkernel_projection_fns):
                _, lvl = subkernel_projection_fn(
                    rng, lvl, subkernel_activations[subkern_i], subkernel_cell_detect_outs[subkern_i],
                    kernel_detect_out.subkernel_detect_outs[subkern_i], pattern_detect_out
                )
            line_applied = jnp.any(lvl != init_lvl)
            i = i + 1
            # jax.debug.print('line_applied: {line_applied}, i: {i}, line_idx: {line_idx}', line_applied=line_applied, i=i, line_idx=line_idx)
            return lvl, line_applied, i
        
        init_carry = (lvl, False, 0)
        if self.jit:
            lvl, line_applied, i = jax.lax.while_loop(
                lambda x: (~x[1]) & (valid_line_idxs[x[2],0] != -1),
                project_line_i,
                init_carry,
            )
        else:
            line_applied, i = init_carry[1], init_carry[2]
            carry = init_carry
            while (not line_applied) and (valid_line_idxs[i,0] != -1):
                carry = project_line_i(carry)
                lvl, line_applied, i = carry

        return rng, lvl

    def detect_kernel(self, lvl, cell_detection_fns, lp_is_vertical, lp_is_horizontal, lp_is_single, in_patch_shape, lp,
                      rule_name: str):
        n_chan = lvl.shape[1]
        # @jax.jit
        def detect_cells(in_patch: chex.Array):
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

        patches = jax.lax.conv_general_dilated_patches(
            lvl, in_patch_shape, window_strides=(1, 1), padding='VALID',
        )
        assert patches.shape[0] == 1
        patches = patches[0]
        # patches = rearrange(patches, "c h w -> h w c")
        patches = patches.transpose(1, 2, 0)

        if self.jit and self.vmap:
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
        for cell_detect_out in cell_detect_outs:
            detected_kernel_meta_objs.update(cell_detect_out.detected_meta_objs)

        # Kernel-wide detected moving idxs are only fallen back on if a given input cell has no detected moving
        # index. In this case, we assume there is only one detected moving index in the kernel, so we can take 
        # the max to propagate these values across cells.
        cell_detected_moving_idxs = [cell_detect_out.detected_moving_idx for cell_detect_out in cell_detect_outs
                                    if cell_detect_out.detected_moving_idx is not None]
        if len(cell_detected_moving_idxs) == 0:
            # No detected moving idxs in the kernel
            detected_kernel_moving_idx = None
        else:
            cell_detected_moving_idxs = jnp.stack(cell_detected_moving_idxs, axis=0)
            detected_kernel_moving_idx = jnp.max(cell_detected_moving_idxs, axis=0)

        kernel_detect_out = KernelFnReturn(
            detected_meta_objs=detected_kernel_meta_objs,
            detected_moving_idx=detected_kernel_moving_idx,
        )
        if kernel_activations.shape[0] == 0:
            kernel_activations = jnp.pad(kernel_activations, ((0, 1), (0, 0)), constant_values=False)
        return kernel_activations, cell_detect_outs, kernel_detect_out


    def project_kernel(self, rng, lvl, kernel_activations, 
                    cell_detect_outs: List[CellFnReturn],
                    kernel_detect_outs: List[KernelFnReturn], 
                    pattern_detect_out: List[PatternFnReturn],
                    cell_projection_fns: List[Callable],
                    lp_is_vertical: bool, lp_is_horizontal: bool, lp_is_single: bool,
                    has_right_pattern: bool, in_patch_shape: Tuple[int, int], rule_name: str,
                    random: bool = False,
                    ):
        n_tiles = np.prod(lvl.shape[-2:])
        # Ensure we always have some invalid coordinates so that the loop will break even when all tiles are active
        if self.jit:
            kernel_activ_xys = jnp.argwhere(kernel_activations, size=n_tiles+1, fill_value=-1)
            if random:
                kernel_activ_xys = jax.random.permutation(rng, kernel_activ_xys)
                # Sort so that -1s are at the end
                kernel_activ_xys_bin = jnp.where(kernel_activ_xys == -1, 0, 1)
                sorted_idxs = jnp.argsort(kernel_activ_xys_bin, axis=0)[:, 0][::-1]
                kernel_activ_xys = jnp.take(kernel_activ_xys, sorted_idxs, axis=0)
        else:
            kernel_activ_xys = np.argwhere(kernel_activations)
            if random:
                kernel_activ_xys = np.random.permutation(kernel_activ_xys)
        kernel_activ_xy_idx = 0
        # kernel_activ_xy = kernel_activ_xys[kernel_activ_xy_idx]

        # @jax.jit
        def project_cells_at(rng, xy, lvl):
            cell_detect_outs_xy = [jax.tree.map(lambda x: x[xy[0]][xy[1]], cell_detect_out) for 
                cell_detect_out in cell_detect_outs]

            kernel_detect_outs_xy = jax.tree.map(lambda x: x[xy[0]][xy[1]], kernel_detect_outs)

            pattern_detect_outs_xy = pattern_detect_out

            # Apply projection functions to the affected cells
            out_cell_idxs = np.indices(in_patch_shape)
            # out_cell_idxs = rearrange(out_cell_idxs, "xy h w -> h w xy")
            out_cell_idxs = out_cell_idxs.transpose(1, 2, 0)
            if lp_is_vertical:
                out_cell_idxs = out_cell_idxs[:, 0]
            elif lp_is_horizontal:
                out_cell_idxs = out_cell_idxs[0, :]
            elif lp_is_single:
                out_cell_idxs = out_cell_idxs[0, :]

            # Either the rule has no right pattern, or it should detect as many cells as there are cell projection functions
            if not (~has_right_pattern or (len(cell_detect_outs) == len(out_cell_idxs) == len(cell_projection_fns))):
                raise RuntimeError(f"Warning: rule {rule_name} with has_right_pattern {has_right_pattern} results in len(cell_detect_outs) {len(cell_detect_outs)} != len(out_cell_idxs) {len(out_cell_idxs)} != len(cell_projection_fns) {len(cell_projection_fns)}")
            init_lvl = lvl

            #TODO: vmap this. But then we risk overlapping? But we do here, too.
            for i, (out_cell_idx, cell_proj_fn) in enumerate(zip(out_cell_idxs, cell_projection_fns)):
            # def apply_cell_proj_fn(lvl, i):
                # FIXME: a cell at position `i` may not exist in the input kernel!! So here, it's just referring to the ``last'' cell in the input (?)
                cell_detect_out_i = cell_detect_outs_xy[i]
                pattern_detect_out_i = pattern_detect_outs_xy
                cell_xy = out_cell_idx + xy
                if DEBUG:
                    jax.debug.print('        projecting cell {i} at position {cell_xy}', i=i, cell_xy=cell_xy)
                m_cell = lvl[0, :, *cell_xy]
                rng, m_cell = cell_proj_fn(
                    rng=rng, m_cell=m_cell, cell_i=i, cell_detect_out=cell_detect_out_i,
                    kernel_detect_out=kernel_detect_outs_xy,
                    pattern_detect_out=pattern_detect_out_i)
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
            return rng, lvl

        def project_kernel_at_xy(carry):               
            rng, kernel_activ_xy_idx, lvl = carry
            kernel_activ_xy = kernel_activ_xys[kernel_activ_xy_idx]
            # jax.debug.print('        kernel_activ_xys: {kernel_activ_xys}', kernel_activ_xys=kernel_activ_xys)
            if DEBUG:
                jax.debug.print('        projecting kernel at position index {kernel_activ_xy_idx}, position {xy}', xy=kernel_activ_xy, kernel_activ_xy_idx=kernel_activ_xy_idx)
            rng, lvl = project_cells_at(rng, kernel_activ_xy, lvl)

            kernel_activ_xy_idx += 1
            return rng, kernel_activ_xy_idx, lvl


        carry = (rng, kernel_activ_xy_idx, lvl)
        if self.jit:
            rng, _, lvl = jax.lax.while_loop(
                lambda carry: jnp.all(kernel_activ_xys[carry[1]] != -1),  
                lambda carry: project_kernel_at_xy(carry),
                carry,
            )
        else:
            while kernel_activ_xy_idx < len(kernel_activ_xys):
                carry = project_kernel_at_xy(carry)
                rng, kernel_activ_xy_idx, lvl = carry
        return rng, lvl


    def apply_movement(self, rng, lvl, coll_mat, n_objs, obj_force_masks, jit=True):
        coll_mat = jnp.array(coll_mat, dtype=bool)
        # Upper bound on the number of forces that might exist in the level at any given time.
        n_layers = len(self.collision_layers)
        max_possible_forces = n_layers * lvl.shape[2] * lvl.shape[3]
        force_arr = lvl[0, n_objs:-1]
        # Mask out all forces corresponding to ACTION.
        force_mask = np.ones((force_arr.shape[0],), dtype=bool)
        force_mask[ACTION::N_FORCES] = 0
        force_arr = force_arr[force_mask]
        # Rearrange the array, since we want to apply force to the "first" objects spatially on the map.
        # force_arr = rearrange(force_arr, "c h w -> w h c")
        force_arr = force_arr.transpose(2, 1, 0)
        # Get the first x,y,c coordinates where force is present.
        coords = jnp.argwhere(force_arr, size=max_possible_forces+1, fill_value=-1)
        # force_idxs = coords[:, 2] % (N_FORCES - 1)

        def remove_invalid_force(carry):
            lvl, i = carry
            y, x, c = coords[i]
            coll_layer_idx = c // (N_FORCES - 1)
            layer_obj_mask = jnp.array(self.layer_masks)[coll_layer_idx]
            obj_idx = jnp.argwhere(
                jnp.where(layer_obj_mask, lvl[0, :self.n_objs, x, y], False), size=1, fill_value=-1
            )[0,0]
            obj_exists = obj_idx != -1
            new_lvl = jax.lax.dynamic_update_slice(
                lvl,
                jnp.zeros((1, N_FORCES, 1, 1), dtype=bool),
                (0, n_objs + (coll_layer_idx * N_FORCES), x, y)
            )
            lvl = jax.lax.select(
                ~obj_exists,
                new_lvl,
                lvl
            )
            i += 1
            return lvl, i

        def attempt_move(carry):
            # NOTE: This depends on movement forces preceding any other forces (per object) in the channel dimension.
            lvl, _, _, i = carry
            y, x, c = coords[i]
            # Get the obj idx on which the force is applied.
            # First get the collision layer idx.
            coll_layer_idx = c // N_MOVEMENTS
            # Check that the coordinates are not null, and that the force is actually present at the coordinates
            # (since we may have removed it if not corresponding to an object).
            is_force_present = (x != -1) & (
                jnp.any(jax.lax.dynamic_slice(
                    lvl, (0, n_objs + (coll_layer_idx * N_FORCES), x, y), (1, N_MOVEMENTS, 1, 1)
                )))
            # Then find the active object in this collition layer.
            n_layer_objs = jnp.array(self.n_objs_per_layer)[coll_layer_idx]
            n_prior_objs = jnp.array(self.n_objs_prior_to_layer)[coll_layer_idx]
            layer_obj_mask = jnp.array(self.layer_masks)[coll_layer_idx]
            # obj_idx = n_prior_objs + jnp.argwhere(lvl[0, n_prior_objs: n_prior_objs + n_layer_objs, x, y])
            obj_idx = jnp.argwhere(
                jnp.where(layer_obj_mask, lvl[0, :self.n_objs, x, y], False), size=1, fill_value=-1
            )[0,0]
            obj_exists = obj_idx != -1
            # Determine where the object would move and whether such a move would be legal.
            forces_to_deltas = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
            delta = forces_to_deltas[c % N_MOVEMENTS]
            x_1, y_1 = x + delta[0], y + delta[1]
            would_collide = jnp.any(lvl[0, :n_objs, x_1, y_1] * coll_mat[obj_idx])
            out_of_bounds = (x_1 < 0) | (x_1 >= lvl.shape[2]) | (y_1 < 0) | (y_1 >= lvl.shape[3])
            can_move = obj_exists & is_force_present & ~would_collide & ~out_of_bounds
            # Now, in the new level, move the object in the direction of the force.
            new_lvl = lvl.at[0, obj_idx, x, y].set(False)
            new_lvl = new_lvl.at[0, obj_idx,  x_1, y_1].set(True)

            # And remove any forces that were applied to the object before it moved.
            new_lvl = jax.lax.dynamic_update_slice(
                new_lvl,
                jnp.zeros((1, N_FORCES, 1, 1), dtype=bool),
                (0, n_objs + (coll_layer_idx * N_FORCES), x, y)
            )
            # Use the force mask instead
            # obj_force_mask = obj_force_masks[obj_idx]
            # new_lvl = new_lvl.at[0, :, x, y].set(
            #     jnp.where(obj_force_mask, 0, new_lvl[0, :, x, y])
            # )

            lvl = jax.lax.select(can_move, new_lvl, lvl)
            i += 1
            # if DEBUG:
            #     jax.debug.print('      at position {xy}, the object {obj} moved to {new_xy}', xy=(x, y), obj=obj_idx, new_xy=(x_1, y_1))
            #     jax.debug.print('      would collide: {would_collide}, out of bounds: {out_of_bounds}, can_move: {can_move}',
            #                     would_collide=would_collide, out_of_bounds=out_of_bounds, can_move=can_move)
            return lvl, can_move, rng, i

        init_carry = (lvl, 0)

        # Iterate through forces and remove them if they don't correspond to an object.
        if jit:
            lvl, i = jax.lax.while_loop(
                lambda carry: (coords[carry[1], 0] != -1),
                lambda carry: remove_invalid_force(carry),
                init_carry,
            )
        else:
            i = init_carry[1]
            carry = init_carry
            while (coords[i, 0] != -1):
                lvl, i = remove_invalid_force(carry)
                carry = (lvl, i)

        init_carry = (lvl, False, rng, 0)

        # Iterate through possible moves until we apply one, or run out of possible moves.
        if jit:
            lvl, can_move, rng, i = jax.lax.while_loop(
                lambda carry: (coords[carry[3], 0] != -1),
                lambda carry: attempt_move(carry),
                init_carry,
            )
        else:
            i = init_carry[3]
            can_move = init_carry[1]
            carry = init_carry
            while (coords[i, 0] != -1) and not can_move:
                lvl, can_move, rng, i = attempt_move(carry)
                carry = (lvl, can_move, rng, i)
        
        if DEBUG:
            jax.debug.print('      applied movement: {can_move}', can_move=can_move)
        rule_state = RuleState(
            lvl=lvl,
            applied=can_move,
            cancelled=False,
            restart=False,
            again=False,
            win=False,
            rng=rng
        )
        return rule_state

    def apply_rule_fn(
            self,
            loop_rule_state: LoopRuleState, 
            ### COMPILE VS RUNTIME ###
            # all_rule_fns, n_prior_rules_arr, 
            rule_fn,
            ### COMPILE VS RUNTIME ###
            ):
        """Apply an atomic rule once to the level."""

        prev_loop_rule_state = loop_rule_state
        rule_i, grp_i, block_i = loop_rule_state.rule_i, loop_rule_state.grp_i, loop_rule_state.block_i
        rng, lvl = prev_loop_rule_state.rng, prev_loop_rule_state.lvl
        # Now we apply the atomic rule function.
        rule_state: RuleState

        # if DEBUG:
        #     jax.debug.print('      applying rule {loop_rule_sttate.rule_i} of group {grp_i}, block {block_i}')

        ### COMPILE VS RUNTIME ###
        # if jit:
        #     rule_state = jax.lax.switch(
        #         n_prior_rules_arr[block_i, grp_i] + rule_i, all_rule_fns, rng, lvl)
        # else:
        #     rule_state = all_rule_fns[n_prior_rules_arr[block_i, grp_i] + rule_i](rng, lvl)

        rule_state = rule_fn(rng, lvl)
        ### COMPILE VS RUNTIME ###

        # FIXME: Should just be able to use rule_state.applied
        # rule_had_effect = rule_state.applied
        rule_had_effect = jnp.any(rule_state.lvl != lvl)

        # HACK (?): Random rules do not count as having had effect (otherwise we'll end up in an infinite loop)
        applied = rule_had_effect & jnp.all(rng == rule_state.rng)
        again = rule_state.again | prev_loop_rule_state.again
        restart = rule_state.restart | prev_loop_rule_state.restart
        cancelled = rule_state.cancelled | prev_loop_rule_state.cancelled
        win = rule_state.win | prev_loop_rule_state.win

        if DEBUG:
            jax.debug.print('      apply_rule_fn: rule {rule_i} had effect: {rule_had_effect}. again: {again}', rule_i=rule_i, rule_had_effect=rule_had_effect, again=again)
            if not self.jit:
                if rule_had_effect:
                    print(f'Level state after rule {rule_i} application:\n{multihot_to_desc(rule_state.lvl[0], objs_to_idxs=self.objs_to_idxs, n_objs=self.n_objs, obj_idxs_to_force_idxs=self.obj_idxs_to_force_idxs)}')

        loop_rule_state = LoopRuleState(
            lvl=rule_state.lvl,
            applied=applied,
            again=again,
            cancelled=cancelled,
            restart=restart,
            win=win,
            app_i=loop_rule_state.app_i + 1,
            rng=rule_state.rng,
            rule_i=rule_i,
            grp_i=grp_i,
            block_i=block_i,
        )

        return loop_rule_state

    def loop_rule_fn(
            self,
            rule_group_state: RuleGroupState, 
            ### COMPILE VS RUNTIME ###
            # all_rule_fns, n_prior_rules_arr,
            rule_fn,
            ### COMPILE VS RUNTIME ###
        ):
        """Repeatedly apply an atomic rule to the level until it no longer has an effect. This function is called in 
        sequence on each rule in the group."""

        _apply_rule_fn = partial(
            self.apply_rule_fn, 
            ### COMPILE VS RUNTIME ###
            # all_rule_fns=all_rule_fns, n_prior_rules_arr=n_prior_rules_arr, rule_fn=rule_grp[rule_i],
            rule_fn=rule_fn,
            ### COMPILE VS RUNTIME ###
            )
        
        loop_rule_state = LoopRuleState(
            lvl=rule_group_state.lvl,
            applied=True,
            cancelled=False,
            restart=False,
            again=rule_group_state.again,
            win=rule_group_state.win,
            rng=rule_group_state.rng,
            app_i=0,
            rule_i=rule_group_state.rule_i,
            grp_i=rule_group_state.grp_i,
            block_i=rule_group_state.block_i,
        )
        loop_rule_state: LoopRuleState

        if self.jit:
            # For the current rule in the group, apply it as many times as possible.
            loop_rule_state = jax.lax.while_loop(
                cond_fun=lambda loop_rule_state: loop_rule_state.applied & ~loop_rule_state.cancelled & ~loop_rule_state.restart \
                    & (loop_rule_state.app_i < MAX_LOOPS),
                body_fun=lambda loop_rule_state: _apply_rule_fn(loop_rule_state),
                init_val=loop_rule_state,
            )
        else:
            while loop_rule_state.applied and not loop_rule_state.cancelled and not loop_rule_state.restart:
                loop_rule_state = _apply_rule_fn(loop_rule_state)

        rule_applied = loop_rule_state.app_i > 1
        again = rule_group_state.again | loop_rule_state.again
        win = rule_group_state.win | loop_rule_state.win
        if DEBUG:
            jax.debug.print('    loop_rule_rn: rule {rule_i} applied: {rule_applied}. again: {again}. win: {win}',
                            rule_i=rule_group_state.rule_i, rule_applied=rule_applied, again=again, win=win)
        grp_applied = rule_applied | loop_rule_state.applied

        rule_group_state = RuleGroupState(
            lvl=loop_rule_state.lvl,
            applied=grp_applied,
            cancelled=rule_group_state.cancelled | loop_rule_state.cancelled,
            restart=rule_group_state.restart | loop_rule_state.restart,
            again=again,
            win=rule_group_state.win | loop_rule_state.win,
            rng=loop_rule_state.rng,
            rule_i=rule_group_state.rule_i + 1,
            grp_i=rule_group_state.grp_i,
            block_i=rule_group_state.block_i,
        )

        return rule_group_state

    def apply_rule_grp(
            self,
            loop_group_state: LoopRuleGroupState,
            ### COMPILE VS RUNTIME ###
            # all_rule_fns, n_prior_rules_arr, n_rules_per_grp_arr,
            rule_grp,
            ### COMPILE VS RUNTIME ###
            ):
        """Iterate through each rule in the group. Loop it until it no longer has an effect."""

        _loop_rule_fn = partial(
            self.loop_rule_fn, 
            ### COMPILE VS RUNTIME ###
            # all_rule_fns=all_rule_fns, n_prior_rules_arr=n_prior_rules_arr,
            ### COMPILE VS RUNTIME ###
            )

        block_i, grp_i = loop_group_state.block_i, loop_group_state.grp_i
        init_lvl = loop_group_state.lvl

        ### COMPILE VS RUNTIME ###
        # n_rules_in_grp = n_rules_per_grp_arr[block_i, grp_i]
        ### COMPILE VS RUNTIME ###

        rule_group_state = RuleGroupState(
            lvl=loop_group_state.lvl,
            applied=False,
            cancelled=loop_group_state.cancelled,
            restart=loop_group_state.restart,
            again=loop_group_state.again,
            win=loop_group_state.win,
            rng=loop_group_state.rng,
            rule_i=0,
            grp_i=grp_i,
            block_i=block_i,
        )

        # ### COMPILE VS RUNTIME ###
        # if jit:
        #     # Iterate through each rule in the group (and loop it). (Note that this is effectively a for loop.)
        #     rule_group_state = jax.lax.while_loop(
        #         cond_fun=lambda rule_group_state: (rule_group_state.rule_i < n_rules_in_grp) & ~rule_group_state.cancelled & ~rule_group_state.restart,
        #         body_fun=_loop_rule_fn,
        #         init_val=rule_group_state,
        #     )
        #     # We can't use scan because n_rules_in_grp is a jnp array, as are block_i and grp_i, which index into this
        #     # array.
        #     # rule_group_state = jax.lax.scan(
        #     #     _loop_rule_fn,
        #     #     init=rule_group_state,
        #     #     length=n_rules_in_grp,
        #     # )

        # else:
        #     while rule_group_state.rule_i < n_rules_in_grp and not rule_group_state.cancelled and not rule_group_state.restart:
        #         rule_group_state = _loop_rule_fn(rule_group_state)

        for rule_fn in rule_grp:
            rule_group_state = _loop_rule_fn(rule_group_state, rule_fn)
        # ### COMPILE VS RUNTIME ###

        again = loop_group_state.again | rule_group_state.again
        win = loop_group_state.win | rule_group_state.win
        restart = loop_group_state.restart | rule_group_state.restart
        cancelled = loop_group_state.cancelled | rule_group_state.cancelled
        lvl = rule_group_state.lvl

        if DEBUG:
            jax.debug.print('  apply_rule_grp: group {grp_i} applied: {grp_applied}. again: {again}', grp_i=grp_i, grp_applied=rule_group_state.applied, again=again)

            if not self.jit:
                if loop_group_state.applied:
                    print('Level state after rule group application:\n' + multihot_to_desc(lvl[0], self.objs_to_idxs, self.n_objs,
                                                                                           obj_idxs_to_force_idxs=self.obj_idxs_to_force_idxs))

        loop_group_state = LoopRuleGroupState(
            lvl=lvl,
            # applied=rule_group_state.applied,
            # FIXME: Shouldn't have to do this...
            # (if we don't `test_electrician` breaks, for example)
            applied=np.any(lvl != init_lvl),
            cancelled=rule_group_state.cancelled,
            restart=rule_group_state.restart,
            again=again,
            win=win,
            rng=rule_group_state.rng,
            app_i=loop_group_state.app_i + 1,
            grp_i=loop_group_state.grp_i,
            block_i=loop_group_state.block_i,
        )
        return loop_group_state

    def loop_rule_grp(
            self,
            rule_block_state: RuleBlockState, 
            ### COMPILE VS RUNTIME ###
            # n_prior_rules_arr, n_rules_per_grp_arr, all_rule_fns,
            rule_grp,
            ### COMPILE VS RUNTIME ###
            ):
        """Given a rule group, repeatedly attempt to apply the group (by looping each rule in sequence) until the group no longer has an effect."""
        lvl = rule_block_state.lvl
        grp_applied_prev = rule_block_state.applied
        cancelled = rule_block_state.cancelled
        restart = rule_block_state.restart
        prev_again = rule_block_state.again
        rng = rule_block_state.rng
        win = rule_block_state.win
        grp_i = rule_block_state.grp_i
        block_i = rule_block_state.block_i

        grp_applied = True

        _apply_rule_grp = partial(
            self.apply_rule_grp,
            ### COMPILE VS RUNTIME ###
            # all_rule_fns=all_rule_fns, n_prior_rules_arr=n_prior_rules_arr, n_rules_per_grp_arr=n_rules_per_grp_arr,
            rule_grp=rule_grp,
            ### COMPILE VS RUNTIME ###
        )

        grp_app_i = 0
        win = False

        loop_rule_group_state = LoopRuleGroupState(
            lvl=lvl,
            applied=True,
            cancelled=cancelled,
            restart=restart,
            again=prev_again,
            win=win,
            rng=rng,
            app_i=grp_app_i,
            grp_i=grp_i,
            block_i=block_i,
        )

        if self.jit:
            loop_rule_group_state = jax.lax.while_loop(
                cond_fun=lambda x: x.applied & ~x.cancelled & ~x.restart & (x.app_i < MAX_LOOPS),
                body_fun=_apply_rule_grp,
                init_val=loop_rule_group_state,
            )
        else:
            while loop_rule_group_state.applied and not loop_rule_group_state.cancelled and not loop_rule_group_state.restart and loop_rule_group_state.app_i < MAX_LOOPS:
                loop_rule_group_state = \
                    _apply_rule_grp(loop_rule_group_state)
                if DEBUG:
                    print(f'     group {grp_i} applied for the {loop_rule_group_state.app_i}th time')

        lvl = loop_rule_group_state.lvl
        again = prev_again | loop_rule_group_state.again
        grp_applied = loop_rule_group_state.app_i > 1
        block_applied = grp_applied_prev | grp_applied

        rule_block_state = RuleBlockState(
            lvl=lvl,
            applied=block_applied,
            # cancelled=loop_rule_group_state.cancelled,  # Actually, this should be enough...
            cancelled=cancelled | loop_rule_group_state.cancelled,
            # restart=loop_rule_group_state.restart,  # This should be enough also...
            restart=restart | loop_rule_group_state.restart,
            again=again,
            win=loop_rule_group_state.win | rule_block_state.win,
            rng=loop_rule_group_state.rng,
            grp_i=grp_i + 1,
            block_i=block_i,
        )

        # return (lvl, block_applied, grp_app_i, cancelled, restart, again, win, rng), None
        return rule_block_state


    def apply_rule_block(
            self,
            loop_rule_block_state: LoopRuleBlockState, 
            ### COMPILE VS RUNTIME ###
            # n_prior_rules_arr, n_rules_per_grp_arr, n_grps_per_block_arr, all_rule_fns,
            rule_block,
            ### COMPILE VS RUNTIME ###
        ):

        lvl, block_app_i, cancelled, restart, prev_again, win, rng, block_i = \
            loop_rule_block_state.lvl, loop_rule_block_state.block_app_i, loop_rule_block_state.cancelled, \
            loop_rule_block_state.restart, loop_rule_block_state.again, loop_rule_block_state.win, \
            loop_rule_block_state.rng, loop_rule_block_state.block_i

        block_app_i += 1
        block_applied = False
        win = False

        _loop_rule_grp = partial(
            self.loop_rule_grp, 
            ### COMPILE VS RUN-TIME ###
            # n_prior_rules_arr=n_prior_rules_arr, n_rules_per_grp_arr=n_rules_per_grp_arr, all_rule_fns=all_rule_fns,
            ### COMPILE VS RUN-TIME ###
        )

        rule_block_state = RuleBlockState(
            lvl=lvl,
            applied=block_applied,
            cancelled=cancelled,
            restart=restart,
            again=prev_again,
            win=win,
            rng=rng,
            grp_i=0,
            block_i=block_i,
        )
        ### COMPILE VS RUN-TIME ###
        # n_rule_grps = n_grps_per_block_arr[block_i]
        # if jit:
        #     rule_block_state = jax.lax.while_loop(
        #         cond_fun=lambda x: (x.grp_i < n_rule_grps) & ~x.cancelled & ~x.restart,
        #         body_fun=_loop_rule_grp,
        #         init_val=rule_block_state,
        #     )
        #     lvl, block_applied, cancelled, restart, again, win, rng = \
        #         rule_block_state.lvl, rule_block_state.applied, rule_block_state.cancelled, \
        #         rule_block_state.restart, rule_block_state.again, rule_block_state.win, rule_block_state.rng
        # else:
        #     again = False  # In case there are no groups in this block
        #     while rule_block_state.grp_i < n_rule_grps and not cancelled and not restart:
        #         rule_block_state = _loop_rule_grp(rule_block_state)
        #         rule_block_state: RuleBlockState
        #         lvl, block_applied, cancelled, restart, again, win, rng = \
        #             rule_block_state.lvl, rule_block_state.applied, rule_block_state.cancelled, \
        #             rule_block_state.restart, rule_block_state.again, rule_block_state.win, rule_block_state.rng

        for grp_i, rule_grp in enumerate(rule_block):
            rule_block_state = _loop_rule_grp(rule_block_state, rule_grp=rule_grp)
        lvl, block_applied, cancelled, restart, again, win, rng = \
            rule_block_state.lvl, rule_block_state.applied, rule_block_state.cancelled, \
            rule_block_state.restart, rule_block_state.again, rule_block_state.win, rule_block_state.rng
        ### COMPILE VS RUN-TIME ###

        again = prev_again | again

        if DEBUG:
            jax.debug.print('apply_rule_block: block {block_i} applied: {block_applied}. again: {again}', block_i=block_i, block_applied=block_applied, again=again)
            if not self.jit:
                if block_applied:
                    print(f'Level state after rule block {block_i} application:\n{multihot_to_desc(lvl[0], objs_to_idxs=self.objs_to_idxs, n_objs=self.n_objs,
                                                                                                   obj_idxs_to_force_idxs=self.obj_idxs_to_force_idxs)}')

        rule_block_state = LoopRuleBlockState(
            lvl=lvl,
            applied=block_applied,
            block_app_i=block_app_i,
            cancelled=cancelled,
            restart=restart,
            again=again,
            win=win,
            rng=rng,
            block_i=block_i,
            app_i=loop_rule_block_state.app_i + 1,
        )

        return rule_block_state

    def loop_rule_block(
            self,
            carry,
            ### COMPILE VS RUN-TIME ###
            # all_rule_fns, n_prior_rules_arr, n_rules_per_grp_arr, n_grps_per_block_arr, blocks_are_looping_lst,
            rule_block, looping,
            ### COMPILE VS RUN-TIME ###
        ):
        """This function is called on each rule block in sequence. It either applies the rule block once (if not a looping 
        block) or else applies a given rule block until it no longer has an
        effect or is otherwise interrupted."""

        lvl, applied, prev_again, prev_cancelled, prev_restart, prev_win, rng, block_i = carry

        # looping = blocks_are_looping_lst[block_i]

        _apply_rule_block = partial(
            self.apply_rule_block,
            ### COMPILE VS RUN-TIME ###
            # n_prior_rules_arr=n_prior_rules_arr, n_rules_per_grp_arr=n_rules_per_grp_arr, all_rule_fns=all_rule_fns,
            # n_grps_per_block_arr=n_grps_per_block_arr,
            rule_block=rule_block,
            ### COMPILE VS RUN-TIME ###
        )

        block_applied = True
        block_app_i = 0

        if not self.jit:
            print(f'Level when applying block {block_i}:\n', multihot_to_desc(lvl[0], self.objs_to_idxs, self.n_objs,
                                                                              obj_idxs_to_force_idxs=self.obj_idxs_to_force_idxs))

        loop_block_state = LoopRuleBlockState(
            lvl=lvl,
            applied=block_applied,
            block_app_i=block_app_i,
            cancelled=prev_cancelled,
            restart=prev_restart,
            again=prev_again,
            win=prev_win,
            rng=rng,
            block_i=block_i,
            app_i=0,
        )
        ### COMPILE VS RUN-TIME ###
        if looping:
            if self.jit:
                loop_block_state = jax.lax.while_loop(
                    cond_fun=lambda x: x.applied & ~x.cancelled & ~x.restart & (x.app_i < MAX_LOOPS),
                    body_fun=_apply_rule_block,
                    init_val=loop_block_state,
                )
            else:
                while loop_block_state.applied and not loop_block_state.cancelled and not loop_block_state.restart:
                    loop_block_state = _apply_rule_block(loop_block_state)
        else:
            loop_block_state = _apply_rule_block(loop_block_state)

        # if jit:
        #     def apply_block_loop():
        #         return jax.lax.while_loop(
        #             cond_fun=lambda x: x.applied & ~x.cancelled & ~x.restart,
        #             body_fun=_apply_rule_block,
        #             init_val=loop_block_state,
        #         )

        #     def apply_block():
        #         return _apply_rule_block(loop_block_state)
            
        #     loop_block_state = jax.lax.cond(
        #         looping,
        #         apply_block_loop,
        #         apply_block,
        #     )
        # else:
        #     if looping:
        #         while loop_block_state.applied and not loop_block_state.cancelled and not loop_block_state.restart:
        #             loop_block_state = _apply_rule_block(loop_block_state)
        #     else:
        #         loop_block_state = _apply_rule_block(loop_block_state)
        ### COMPILE VS RUN-TIME ###
        lvl, block_applied, block_app_i, cancelled, restart, block_again, win, rng, block_i = \
            loop_block_state.lvl, loop_block_state.applied, loop_block_state.block_app_i, \
            loop_block_state.cancelled, loop_block_state.restart, loop_block_state.again, \
            loop_block_state.win, loop_block_state.rng, loop_block_state.block_i

        block_applied = block_app_i > 1
        # else:
        #     lvl, block_applied, block_app_i, cancelled, restart, block_again, win, rng, block_i = \
        #         _apply_rule_block(init_carry)
        applied = applied | block_applied
        again = prev_again | block_again
        restart = prev_restart | restart
        cancelled = prev_cancelled | cancelled
        win = prev_win | win

        if DEBUG:
            if self.jit:
                jax.debug.print('loop_rule_block: block {block_i} applied: {block_applied}. again: {again}', block_i=block_i, block_applied=block_applied, again=again)
            else:
                if block_applied:
                    print(f'block {block_i} applied: True')
                else:
                    print(f'block {block_i} applied: False')


        return lvl, applied, again, cancelled, restart, win, rng, block_i + 1

def is_line_detector_in_kernel(kernel):
    # Note that the kernel has been rotated, and is 2D
    for cell_row in kernel:
        for cell in cell_row:
            if cell == '...':
                return True
    return False

def generate_index_sequences(N, M, rot):
    # assert N <= M, f"N ({N}) must be less than or equal to M ({M})"
    
    valid_indices = list(range(N))
    result = []

    # Probably a faster way to do this from "within" this combination-generating function.
    positions_lst = list(itertools.combinations(range(M), N))
    # positions_lst = sorted(positions_lst, key=lambda x: abs(x[0] - x[1]))

    if rot == 2:
        positions_lst = sorted(positions_lst, key=lambda x: (x[1], abs(x[0] - x[1])))

    elif rot == 3:
        positions_lst = sorted(positions_lst, key=lambda x: (x[1], abs(x[0] - x[1])))

    # Step 1: Choose positions in the M-length vector to place the N valid indices
    for positions in positions_lst:
        # Step 2: Generate increasing permutations of valid indices (fixed order)
        for values in itertools.permutations(valid_indices,):
            if list(values) != sorted(values):
                continue  # skip non-increasing permutations
            arr = np.full(M, -1, dtype=int)
            arr[list(positions)] = values
            result.append(arr)
    
    return np.array(result)


def one_hot_sequences(sequences, num_classes):
    M = sequences.shape[1]
    result = []

    for seq in sequences:
        one_hot = np.zeros((M, num_classes), dtype=int)
        for i, val in enumerate(seq):
            if val != -1:
                one_hot[i, val] = 1
        result.append(one_hot)
    
    return np.array(result, dtype=bool)


def gen_one_hot_masks(N: int, M: int, rot: int, spacings: List[int]):
    if N > M:
        return np.zeros((1, N, M), dtype=bool)  # No valid sequences if M < N
    # Generate valid index sequences
    sequences = generate_index_sequences(N, M, rot=rot)
    one_hot_masks = one_hot_sequences(sequences, num_classes=N)  # shape: (K, M, N)

    # Transpose to match binary_matrix shape (N, M)
    one_hot_masks = one_hot_masks.transpose(0, 2, 1)  # shape: (K, N, M)

    oh_mask_start_idxs = np.argmax(one_hot_masks, axis=2)  # Get the first position of each sub-pattern in each mask
    oh_mask_diffs = np.diff(oh_mask_start_idxs, axis=1)
    oh_masks_are_valid = np.all(oh_mask_diffs >= np.array(spacings), axis=1)
    oh_mask_idxs = np.argwhere(oh_masks_are_valid) 
    one_hot_masks = one_hot_masks[oh_mask_idxs[:, 0]]  # Filter masks based on spacings
    return one_hot_masks


def mask_to_valid_sequences(binary_matrix, one_hot_masks=None, jit=True):
    N, M = binary_matrix.shape

    if one_hot_masks is None:
        one_hot_masks = gen_one_hot_masks(N, M)

    init_valid_mask = jnp.zeros_like(binary_matrix)

    if not jit:
        valid_masks = np.zeros_like(one_hot_masks)
        for mask_i, mask in enumerate(one_hot_masks):
            if np.all((mask & binary_matrix) == mask):  # check if mask is a subset
                valid_masks[mask_i] = mask
        valid_mask_v = np.stack(valid_masks, axis=0)
    else:
        def body_fun(mask):
            detected = jnp.all((mask & binary_matrix) == mask)
            valid_mask = jnp.where(detected, mask, jnp.zeros_like(mask))
            return valid_mask

        valid_mask_v = jax.vmap(body_fun, in_axes=(0,))(one_hot_masks)

    return valid_mask_v


if __name__ == "__main__":
    binary_matrix = np.array([
        [1, 0, 0, 0],  # row for index 0
        [1, 1, 0, 1]   # row for index 1
    ])

    masked = mask_to_valid_sequences(binary_matrix)
    print(masked)
