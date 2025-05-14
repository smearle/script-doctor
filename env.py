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

from conf.config import RLConfig
from env_render import render_solid_color, render_sprite
from jax_utils import stack_leaves
from marl.spaces import Box
from ps_game import LegendEntry, PSGameTree, PSObject, Rule, WinCondition
from spaces import Discrete


# Whether to print out a bunch of stuff, etc.
DEBUG = False
PRINT_SCORE = True
# DEBUG = True

# Per-object forces forces that can be applied: left, right, up, down; action.
N_FORCES = 5
N_MOVEMENTS = 4
ACTION = 4

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
        elif v.operator.lower() == 'or':
            meta_objs[k.strip()] = v.obj_names
        elif v.operator.lower() == 'and':
            conjoined_tiles[k.strip()] = v.obj_names
        else:
            raise Exception('Invalid LegendEntry operator.')

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
    min_dist = jnp.min(dists).astype(np.int32)
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
            raise Exception('Invalid quantifier.')
        funcs.append(func)

    # @partial(jax.jit)
    def check_win(lvl):

        if len(funcs) == 0:
            return False, 0, 0
        
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
    # An array of detected objects, as many as there are overlapping objects in the cell
    # Note that we don't necessarily know these at compile-time because of meta-objects
    # A dictionary of the objects detected, mapping meta-object names to sub-object indices
    detected_meta_objs: dict
    detected_obj_idxs: Optional[chex.Array] = None
    detected_moving_idx: Optional[int] = None

@flax.struct.dataclass
class KernelFnReturn:
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
                if modifier.lower() in ['perpendicular', 'orthogonal', 'parallel']:
                    return True
    return False

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
                    modifier = object_with_modifier.split(' ')[0].lower()
                    if modifier.lower() in ['perpendicular', 'orthogonal']:
                        c_a.append('^')
                        c_b.append('v')
                    elif modifier.lower() in ['parallel']:
                        c_a.append('>')
                        c_b.append('<')
                    else:
                        c_a.append(object_with_modifier)
                        c_b.append(object_with_modifier)
                l_kern_a.append(c_a)
                l_kern_b.append(c_b)
            new_kerns_a.append(l_kern_a)
            new_kerns_b.append(l_kern_b)
        new_patterns[0][i] = (new_kerns_a)
        new_patterns[1][i] = (new_kerns_b)
    return new_patterns

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

def apply_rule_fn(
        loop_rule_state: LoopRuleState, 
        ### COMPILE VS RUNTIME ###
        # all_rule_fns, n_prior_rules_arr, 
          rule_fn,
        ### COMPILE VS RUNTIME ###
        obj_to_idxs, n_objs, jit):
    """Apply an atomic rule once to the level."""

    prev_loop_rule_state = loop_rule_state
    rule_i, grp_i, block_i = loop_rule_state.rule_i, loop_rule_state.grp_i, loop_rule_state.block_i
    rng, lvl = prev_loop_rule_state.rng, prev_loop_rule_state.lvl
    # Now we apply the atomic rule function.
    rule_state: RuleState

    if DEBUG:
        jax.debug.print('      applying rule {rule_i} of group {grp_i}, block {block_i}')

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
        if not jit:
            if rule_had_effect:
                print(f'Level state after rule {rule_i} application:\n{multihot_to_desc(rule_state.lvl[0], objs_to_idxs=obj_to_idxs, n_objs=n_objs)}')

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


def loop_rule_fn(
        rule_group_state: RuleGroupState, 
        ### COMPILE VS RUNTIME ###
        # all_rule_fns, n_prior_rules_arr,
        rule_fn,
        ### COMPILE VS RUNTIME ###
        obj_to_idxs, n_objs, jit):
    """Repeatedly apply an atomic rule to the level until it no longer has an effect. This function is called in 
    sequence on each rule in the group."""

    _apply_rule_fn = partial(
        apply_rule_fn, 
        ### COMPILE VS RUNTIME ###
        # all_rule_fns=all_rule_fns, n_prior_rules_arr=n_prior_rules_arr, rule_fn=rule_grp[rule_i],
        rule_fn=rule_fn,
        ### COMPILE VS RUNTIME ###
        jit=jit, obj_to_idxs=obj_to_idxs, n_objs=n_objs)
    
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

    if jit:
        # For the current rule in the group, apply it as many times as possible.
        loop_rule_state = jax.lax.while_loop(
            cond_fun=lambda loop_rule_state: loop_rule_state.applied & ~loop_rule_state.cancelled & ~loop_rule_state.restart,
            body_fun=lambda loop_rule_state: _apply_rule_fn(loop_rule_state),
            init_val=loop_rule_state,
        )
    else:
        while loop_rule_state.applied and not loop_rule_state.cancelled and not loop_rule_state.restart:
            loop_rule_state = _apply_rule_fn(loop_rule_state)

    rule_applied = loop_rule_state.app_i > 1
    again = rule_group_state.again | loop_rule_state.again
    if DEBUG:
        jax.debug.print('    loop_rule_rn: rule {rule_i} applied: {rule_applied}. again: {again}', rule_i=rule_group_state.rule_i, rule_applied=rule_applied, again=again)
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

def apply_rule_grp(
        loop_group_state: LoopRuleGroupState,
        ### COMPILE VS RUNTIME ###
        # all_rule_fns, n_prior_rules_arr, n_rules_per_grp_arr,
        rule_grp,
        ### COMPILE VS RUNTIME ###
        obj_to_idxs, n_objs, jit):
    """Iterate through each rule in the group. Loop it until it no longer has an effect."""

    _loop_rule_fn = partial(
        loop_rule_fn, 
        ### COMPILE VS RUNTIME ###
        # all_rule_fns=all_rule_fns, n_prior_rules_arr=n_prior_rules_arr,
        ### COMPILE VS RUNTIME ###
        jit=jit, obj_to_idxs=obj_to_idxs, n_objs=n_objs)

    block_i, grp_i = loop_group_state.block_i, loop_group_state.grp_i

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

        if not jit:
            if loop_group_state.applied:
                print('Level state after rule group application:\n' + multihot_to_desc(lvl[0], obj_to_idxs, n_objs))

    loop_group_state = LoopRuleGroupState(
        lvl=lvl,
        applied=rule_group_state.applied,
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

def loop_rule_grp(
        rule_block_state: RuleBlockState, 
        ### COMPILE VS RUNTIME ###
        # n_prior_rules_arr, n_rules_per_grp_arr, all_rule_fns,
        rule_grp,
        ### COMPILE VS RUNTIME ###
        obj_to_idxs, n_objs, jit):
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
        apply_rule_grp,
        ### COMPILE VS RUNTIME ###
        # all_rule_fns=all_rule_fns, n_prior_rules_arr=n_prior_rules_arr, n_rules_per_grp_arr=n_rules_per_grp_arr,
        rule_grp=rule_grp,
        ### COMPILE VS RUNTIME ###
        obj_to_idxs=obj_to_idxs, n_objs=n_objs, jit=jit)

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

    if jit:
        loop_rule_group_state = jax.lax.while_loop(
            cond_fun=lambda x: x.applied & ~x.cancelled & ~x.restart & (x.app_i < 200),
            body_fun=_apply_rule_grp,
            init_val=loop_rule_group_state,
        )
    else:
        while loop_rule_group_state.applied and not loop_rule_group_state.cancelled and not loop_rule_group_state.restart and loop_rule_group_state.app_i < 200:
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
        win=loop_rule_group_state.win | win,
        rng=loop_rule_group_state.rng,
        grp_i=grp_i + 1,
        block_i=block_i,
    )

    # return (lvl, block_applied, grp_app_i, cancelled, restart, again, win, rng), None
    return rule_block_state

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

def apply_rule_block(
        rule_block_state: LoopRuleBlockState, 
        ### COMPILE VS RUNTIME ###
        # n_prior_rules_arr, n_rules_per_grp_arr, n_grps_per_block_arr, all_rule_fns,
        rule_block,
        ### COMPILE VS RUNTIME ###
        obj_to_idxs, n_objs, jit):

    lvl, block_app_i, cancelled, restart, prev_again, win, rng, block_i = \
        rule_block_state.lvl, rule_block_state.block_app_i, rule_block_state.cancelled, \
        rule_block_state.restart, rule_block_state.again, rule_block_state.win, \
        rule_block_state.rng, rule_block_state.block_i

    block_app_i += 1
    block_applied = False
    win = False

    _loop_rule_grp = partial(
        loop_rule_grp, 
        ### COMPILE VS RUN-TIME ###
        # n_prior_rules_arr=n_prior_rules_arr, n_rules_per_grp_arr=n_rules_per_grp_arr, all_rule_fns=all_rule_fns,
        ### COMPILE VS RUN-TIME ###
        obj_to_idxs=obj_to_idxs, n_objs=n_objs, jit=jit)

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
        if not jit:
            if block_applied:
                print(f'Level state after rule block {block_i} application:\n{multihot_to_desc(lvl[0], objs_to_idxs=obj_to_idxs, n_objs=n_objs)}')

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
    )

    return rule_block_state

def loop_rule_block(
        carry,
        ### COMPILE VS RUN-TIME ###
        # all_rule_fns, n_prior_rules_arr, n_rules_per_grp_arr, n_grps_per_block_arr, blocks_are_looping_lst,
        rule_block, looping,
        ### COMPILE VS RUN-TIME ###
        obj_to_idxs, n_objs, jit):
    """This function is called on each rule block in sequence. It either applies the rule block once (if not a looping 
    block) or else applies a given rule block until it no longer has an
    effect or is otherwise interrupted."""

    lvl, applied, prev_again, prev_cancelled, prev_restart, prev_win, rng, block_i = carry

    # looping = blocks_are_looping_lst[block_i]

    _apply_rule_block = partial(
        apply_rule_block,
        ### COMPILE VS RUN-TIME ###
        # n_prior_rules_arr=n_prior_rules_arr, n_rules_per_grp_arr=n_rules_per_grp_arr, all_rule_fns=all_rule_fns,
        # n_grps_per_block_arr=n_grps_per_block_arr,
        rule_block=rule_block,
        ### COMPILE VS RUN-TIME ###
        obj_to_idxs=obj_to_idxs, n_objs=n_objs, jit=jit)

    block_applied = True
    block_app_i = 0

    if not jit:
        print(f'Level when applying block {block_i}:\n', multihot_to_desc(lvl[0], obj_to_idxs, n_objs))

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
    )
    if jit:
        def apply_block_loop():
            return jax.lax.while_loop(
                cond_fun=lambda x: x.applied & ~x.cancelled & ~x.restart,
                body_fun=_apply_rule_block,
                init_val=loop_block_state,
            )

        def apply_block():
            return _apply_rule_block(loop_block_state)
        
        loop_block_state = jax.lax.cond(
            looping,
            apply_block_loop,
            apply_block,
        )
    else:
        if looping:
            while loop_block_state.applied and not loop_block_state.cancelled and not loop_block_state.restart:
                loop_block_state = _apply_rule_block(loop_block_state)
        else:
            loop_block_state = _apply_rule_block(loop_block_state)
    lvl, block_applied, block_app_i, cancelled, restart, block_again, win, rng, block_i = \
        loop_block_state.lvl, loop_block_state.applied, loop_block_state.block_app_i, \
        loop_block_state.cancelled, loop_block_state.restart, loop_block_state.again, \
        loop_block_state.win, loop_block_state.rng, loop_block_state.block_i

    block_applied = block_app_i > 1
    # else:
    #     lvl, block_applied, block_app_i, cancelled, restart, block_again, win, rng, block_i = \
    #         _apply_rule_block(init_carry)
    if DEBUG:
        if jit:
            jax.debug.print('loop_rule_block: block {block_i} applied: {block_applied}. again: {block_again}', block_i=block_i, block_applied=block_applied, block_again=block_again)
        else:
            if block_applied:
                print(f'block {block_i} applied: True')
            else:
                print(f'block {block_i} applied: False')

    applied = applied | block_applied
    again = prev_again | block_again
    restart = prev_restart | restart
    cancelled = prev_cancelled | cancelled
    win = prev_win | win

    return lvl, applied, again, cancelled, restart, win, rng, block_i + 1


def apply_movement(rng, lvl, obj_to_idxs, coll_mat, n_objs, jit=True):
    coll_mat = jnp.array(coll_mat, dtype=bool)
    n_lvl_cells = lvl.shape[2] * lvl.shape[3]
    force_arr = lvl[0, n_objs:-1]
    # Mask out all forces corresponding to ACTION.
    force_mask = np.ones((force_arr.shape[0],), dtype=bool)
    force_mask[ACTION::N_FORCES] = 0
    force_arr = force_arr[force_mask]
    # Rearrange the array, since we want to apply force to the "first" objects spatially on the map.
    force_arr = rearrange(force_arr, "c h w -> h w c")
    # Get the first x,y,c coordinates where force is present.
    coords = jnp.argwhere(force_arr, size=n_lvl_cells, fill_value=-1)

    def attempt_move(carry):
        # NOTE: This depends on movement forces preceding any other forces (per object) in the channel dimension.
        lvl, _, _, i = carry
        x, y, c = coords[i]
        is_force_present = x != -1
        # Get the obj idx on which the force is applied.
        obj_idx = c // (N_FORCES - 1)
        # Determine where the object would move and whether such a move would be legal.
        forces_to_deltas = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        delta = forces_to_deltas[c % (N_FORCES - 1)]
        x_1, y_1 = x + delta[0], y + delta[1]
        would_collide = jnp.any(lvl[0, :n_objs, x_1, y_1] * coll_mat[obj_idx])
        out_of_bounds = (x_1 < 0) | (x_1 >= lvl.shape[2]) | (y_1 < 0) | (y_1 >= lvl.shape[3])
        can_move = is_force_present & ~would_collide & ~out_of_bounds
        # Now, in the new level, move the object in the direction of the force.
        new_lvl = lvl.at[0, obj_idx, x, y].set(0)
        new_lvl = new_lvl.at[0, obj_idx,  x_1, y_1].set(1)
        # And remove any forces that were applied to the object before it moved.
        new_lvl = jax.lax.dynamic_update_slice(
            new_lvl,
            jnp.zeros((1, N_FORCES, 1, 1), dtype=bool),
            (0, n_objs + (obj_idx * N_FORCES), x, y)
        )
        lvl = jax.lax.select(can_move, new_lvl, lvl)
        i += 1
        if DEBUG:
            jax.debug.print('      at position {xy}, the object {obj} moved to {new_xy}', xy=(x, y), obj=obj_idx, new_xy=(x_1, y_1))
            jax.debug.print('      would collide: {would_collide}, out of bounds: {out_of_bounds}, can_move: {can_move}',
                            would_collide=would_collide, out_of_bounds=out_of_bounds, can_move=can_move)
        return lvl, can_move, rng, i

    init_carry = (lvl, False, rng, 0)

    # Iterate through possible moves until we apply one, or run out of possible moves.
    if jit:
        lvl, can_move, rng, i = jax.lax.while_loop(
            lambda carry: (coords[carry[3], 0] != -1) & ~carry[1],
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


@flax.struct.dataclass
class PSState:
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
class PSParams:
    level: chex.Array

@flax.struct.dataclass
class PSObs:
    multihot_level: chex.Array
    flat_obs: Optional[chex.Array] = None


def get_names_to_alts(objects):
    names_to_alts = {}
    for obj_key, obj in objects.items():
        obj: PSObject
        if obj.alt_name is not None:
            names_to_alts[obj.alt_name] = obj_key
    return names_to_alts


class PSEnv:
    def __init__(self, tree: PSGameTree, jit: bool = True, level_i: int = 0, max_steps: int = np.inf,
                 debug: bool = False, print_score: bool = True):
        global DEBUG, PRINT_SCORE
        DEBUG, PRINT_SCORE = debug, print_score
        self.jit = jit
        self.title = tree.prelude.title
        self._has_randomness = False
        self.tree = tree
        self.levels = tree.levels
        self.level_i = level_i
        self.require_player_movement = tree.prelude.require_player_movement
        if DEBUG:
            print(f"Processing legend for {self.title}")
        obj_to_char, meta_objs, joint_tiles = process_legend(tree.legend)
        self.joint_tiles = joint_tiles
        names_to_alts = get_names_to_alts(tree.objects)
        alts_to_names = {v: k for k, v in names_to_alts.items()}
        self.meta_objs = meta_objs
        self.max_steps = max_steps

        # Add to the legend any objects to whose keys are specified in their object definition
        for obj_key, obj in tree.objects.items():
            obj_key = obj.legend_key
            if obj_key is not None:
                obj_to_char[obj.name] = obj_key
                if obj.alt_name is not None:
                    obj_to_char[obj.alt_name] = obj_key
        self.char_to_obj = char_to_obj = {v: k for k, v in obj_to_char.items()}

        if DEBUG:
            print(f"Expanding collision layers for {self.title}")
        self.collision_layers = collision_layers = expand_collision_layers(tree.collision_layers, meta_objs, char_to_obj)
        atomic_obj_names = [name for layer in collision_layers for name in layer]
        # atomic_obj_names = [name for name in tree.objects.keys()]
        # atomic_obj_names = [name for name in atomic_obj_names]
        self.atomic_obj_names = atomic_obj_names
        objs, self.objs_to_idxs, coll_masks = assign_vecs_to_objs(collision_layers, atomic_obj_names)
        for obj, sub_objs in meta_objs.items():
            # Meta-objects that are actually just alternate names.
            if DEBUG:
                print(f'sub_objs {sub_objs}')
            sub_objs = expand_meta_objs(sub_objs, meta_objs, char_to_obj)
            if len(sub_objs) == 1 and (obj not in self.objs_to_idxs):
                self.objs_to_idxs[obj] = self.objs_to_idxs[sub_objs[0]]
        self.n_objs = len(atomic_obj_names)
        self.coll_mat = np.einsum('ij,ik->jk', coll_masks, coll_masks, dtype=bool)
        if DEBUG:
            print(f"Generating tick function for {self.title}")
        self.tick_fn = self.gen_tick_fn()
        if DEBUG:
            print(f"Generating check win function for {self.title}")
        self.check_win = gen_check_win(tree.win_conditions, self.objs_to_idxs, meta_objs, self.char_to_obj, jit=self.jit)
        if 'player' in self.objs_to_idxs:
            self.player_idxs = [self.objs_to_idxs['player']]
        elif 'player' in meta_objs:
            player_objs = expand_meta_objs(['player'], meta_objs, char_to_obj)
            self.player_idxs = [self.objs_to_idxs[p] for p in player_objs]
        elif 'player' in joint_tiles: 
            sub_objs = joint_tiles['player']
            self.player_idxs = [self.objs_to_idxs[sub_obj] for sub_obj in sub_objs]
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
                    meta_objs[obj_key] = alts_to_names[obj_key]
                    obj_key = alts_to_names[obj_key]
                elif obj_key in names_to_alts:
                    meta_objs[obj_key] = names_to_alts[obj_key]
                    obj_key = names_to_alts[obj_key]
                else:
                    raise ValueError(f"Object {obj_key} not found in tree.objects")
                    # breakpoint()
            obj = tree.objects[obj_key]
            if obj.sprite is not None:
                if DEBUG:
                    print(f'rendering pixel sprite for {obj_key}')
                im = render_sprite(obj.colors, obj.sprite)

            else:
                # assert len(obj.colors) == 1
                if len(obj.colors) != 1:
                    raise ValueError(f"Object {obj_key} has more than one color, but no sprite.")
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
        self.chars_to_idxs = {obj_to_char[k]: v for k, v in self.objs_to_idxs.items() if k in obj_to_char}
        self.chars_to_idxs.update({k: v for k, v in self.objs_to_idxs.items() if len(k) == 1})

        # Generate vectors to detect joint objects
        for jo, subobjects in joint_tiles.items():
            vec = np.zeros(self.n_objs, dtype=bool)
            subobjects = expand_meta_objs(subobjects, meta_objs, char_to_obj)
            for so in subobjects:
                if DEBUG:
                    print(so)
                vec += self.obj_vecs[self.objs_to_idxs[so]]
            assert jo not in self.chars_to_idxs
            self.chars_to_idxs[jo] = self.obj_vecs.shape[0]
            self.obj_vecs = np.concatenate((self.obj_vecs, vec[None]), axis=0)

        if self.jit:
            self.step = jax.jit(self.step)
            self.step_env = jax.jit(self.step_env)
            self.reset = jax.jit(self.reset)
            self.apply_player_force = jax.jit(self.apply_player_force)
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
        sub_objs = background_sub_objs
        bg_obj = background_sub_objs[0]
        bg_idx = self.objs_to_idxs[bg_obj]
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

    def reset(self, rng, params: PSParams):
        lvl = params.level
        again = False
        _, _, init_heuristic = self.check_win(lvl)
        state = PSState(
            multihot_level=lvl,
            win=jnp.array(False),
            score=0,
            heuristic=np.iinfo(np.int32).min,
            restart=jnp.array(False),
            step_i=0,
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
    def apply_player_force(self, action, state: PSState):
        multihot_level = state.multihot_level
        # Add a dummy object at the front. Add one final channel to mark the player's effect.
        force_map = jnp.zeros((N_FORCES * (multihot_level.shape[0] + 1) + 1, *multihot_level.shape[1:]), dtype=bool)
        
        def place_force(force_map, action):
            # This is a map-shape array of the obj-indices corresponding to the player objects active on these respective cells.
            player_int_mask = (self.player_idxs[...,None,None] + 1) * multihot_level[self.player_idxs]

            # Turn the int mask into coords, by flattening it, and appending it with xy coords
            xy_coords = jnp.indices(force_map.shape[1:])
            xy_coords = xy_coords[:, None].repeat(len(self.player_idxs), axis=1)

            # This is a map-shaped array of the force-indices (and xy indices) that should be applied, given the player objects at these cells.
            player_force_mask = player_int_mask * N_FORCES + action

            player_force_mask = jnp.concatenate((player_force_mask[None], xy_coords), axis=0)
            player_coords = player_force_mask.reshape(3, -1).T
            force_map = force_map.at[tuple(player_coords.T)].set(1)

            # Similarly, activate the player_effect channel (the last channel) wherever this action is applied.
            player_effect_mask = (player_int_mask > 0) * (force_map.shape[0] - 1)
            player_effect_mask = jnp.concatenate((player_effect_mask[None], xy_coords), axis=0)
            player_effect_mask = player_effect_mask.reshape(3, -1).T
            force_map = force_map.at[tuple(player_effect_mask.T)].set(1)

            # force_map_sum = force_map.sum()
            # jax.debug.print('force_map: {force_map}', force_map=force_map)
            # jax.debug.print('force map sum: {force_map_sum}', force_map_sum=force_map_sum)
            return force_map

        # apply movement (<4) and/or action (if not noaction)
        should_apply_force = (action != -1) & ((action < 4) | (~self.tree.prelude.noaction))

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
        # remove the dummy object
        force_map = force_map[N_FORCES:]

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
    def step_env(self, rng, state: PSState, action, params: Optional[PSParams] = None):
        init_lvl = state.multihot_level.copy()
        lvl = self.apply_player_force(action, state)

        # Actually, just apply the rule function once
        cancelled = False
        restart = False
        final_lvl, tick_applied, turn_app_i, cancelled, restart, tick_win, rng = self.tick_fn(rng, lvl)

        accept_lvl_change = ((not self.require_player_movement) or 
                             player_has_moved(init_lvl, final_lvl, self.objs_to_idxs, self.meta_objs, self.char_to_obj)) & ~cancelled
        # if DEBUG:
        #     jax.debug.print('accept level change: {accept_lvl_change}', accept_lvl_change=accept_lvl_change) 

        final_lvl = jax.lax.select(
            accept_lvl_change,
            final_lvl,
            lvl,
        )
        multihot_level = final_lvl[:self.n_objs]
        win, score, heuristic = self.check_win(multihot_level)
        win = win | tick_win
        if PRINT_SCORE:
            jax.debug.print('heuristic: {heuristic}, score: {score}', heuristic=heuristic, score=score)

        # reward = (heuristic - state.init_heuristic) / jnp.abs(state.init_heuristic)
        reward = heuristic - state.prev_heuristic
        # reward += 10 if win else 0
        reward = jax.lax.select(win, reward + 1, reward)
        reward -= 0.01

        done = win | ((state.step_i + 1) >= self.max_steps)
        info = {}
        state = PSState(
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
        return multihot_level

    def gen_subrules_meta(self, rule: Rule, rule_name: str):
        has_right_pattern = len(rule.right_kernels) > 0

        def is_obj_forceless(obj_idx, m_cell):
            # note that `action` does not count as a force
            return jnp.sum(jax.lax.dynamic_slice(m_cell, (self.n_objs + (obj_idx * N_FORCES),), (4,))) == 0

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
            force_is_present = m_cell[self.n_objs + (obj_idx * N_FORCES) + force_idx] == 1
            active = obj_is_present & force_is_present
            is_detected = np.zeros(m_cell.shape, dtype=bool)
            is_detected[obj_idx] = 1
            is_detected[self.n_objs + (obj_idx * N_FORCES) + force_idx] = 1
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
        def detect_force_on_meta(m_cell: chex.Array, obj_idxs, force_idx):
            dummy_force_obj_vec = jnp.zeros(self.n_objs + self.n_objs * N_FORCES + 1, dtype=bool)

            def force_obj_vec_fn(obj_idx):
                force_obj_vec = dummy_force_obj_vec.at[obj_idx].set(1)
                force_obj_vec = force_obj_vec.at[self.n_objs + obj_idx * N_FORCES + force_idx].set(1)
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
            is_detected = is_detected.at[self.n_objs + (obj_idx * N_FORCES) + force_idx].set(1)
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
        def detect_moving_meta(m_cell, obj_idxs, vertical=False, horizontal=False):
            # TODO: vmap this?
            active_obj_idx = -1
            active_force_idx = -1
            detected = jnp.zeros(m_cell.shape, dtype=bool)

            for obj_idx in obj_idxs:

                obj_is_present = m_cell[obj_idx] == 1
                obj_forces = jax.lax.dynamic_slice(m_cell, (self.n_objs + (obj_idx * N_FORCES),), (4,))
                if vertical:
                    vertical_mask = np.array([0, 1, 0, 1], dtype=bool)
                    obj_forces = jnp.logical_and(obj_forces, vertical_mask)
                elif horizontal:
                    horizontal_mask = np.array([1, 0, 1, 0], dtype=bool)
                    obj_forces = jnp.logical_and(obj_forces, horizontal_mask)
                force_idx = jnp.argwhere(obj_forces, size=1, fill_value=-1)[0, 0]
                obj_active = obj_is_present & (force_idx != -1)

                active_detected = detected.at[obj_idx].set(1)
                active_detected = active_detected.at[self.n_objs + (obj_idx * N_FORCES) + force_idx].set(1)

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
            no, force, directional_force, stationary, action, moving, vertical, horizontal = \
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
                    moving = True
                elif obj == 'vertical':
                    vertical = True
                elif obj == 'horizontal':
                    horizontal = True
                else:
                    obj_names.append(obj)
                    sub_objs = expand_meta_objs([obj], self.meta_objs, self.char_to_obj)
                    obj_idxs = np.array([self.objs_to_idxs[so] for so in sub_objs])
                    obj_vec = np.zeros((self.n_objs + self.n_objs * N_FORCES + 1), dtype=bool)
                    obj_vec[obj_idxs] = 1
                    if obj in self.char_to_obj:
                        obj = self.char_to_obj[obj]
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
                            fns.append(partial(detect_stationary_meta, obj_idxs=obj_idxs))
                            stationary = False
                        elif moving:
                            fns.append(partial(detect_moving_meta, obj_idxs=obj_idxs))
                            moving = False
                        elif vertical:
                            fns.append(partial(detect_moving_meta, obj_idxs=obj_idxs, vertical=True))
                            vertical = False
                        elif horizontal:
                            fns.append(partial(detect_moving_meta, obj_idxs=obj_idxs, horizontal=True))
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
            coll_vec = coll_vec.at[obj_idx].set(0)
            # print('various shapes lol', m_cell.shape, n_objs, obj_idx, coll_vec.shape, coll_mat.shape)
            m_cell = m_cell.at[:self.n_objs].set(m_cell[:self.n_objs] * ~coll_vec)
            return m_cell
        
        if self.jit:
            remove_colliding_objs = jax.jit(remove_colliding_objs)
        else:
            remove_colliding_objs = remove_colliding_objs

        # @partial(jax.jit, static_argnums=(3))
        def project_obj(rng, m_cell, cell_i, cell_detect_out: CellFnReturn, kernel_detect_out: KernelFnReturn,
                        pattern_detect_out: PatternFnReturn, obj, random=False):
            """
            Project an object into a cell in the output pattern.
                m_cell: an n_channels-size vector of all object and per-object force activations and player effect at the
                    current cell
                cell_i: the index of the cell relative to its position in the kernel
                cell_detect_out: the output of the corrsponding cell detection function
            """
            detected_obj_idx = cell_detect_out.detected_obj_idxs[cell_i]
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
                obj_idx = disambiguate_meta(obj, detected_meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)
            if DEBUG:
                jax.debug.print('        projecting obj {obj}, disambiguated index: {obj_idx}', obj=obj, obj_idx=obj_idx)
            if not self.jit:
                if obj_idx == -1:
                    breakpoint()
            m_cell = m_cell.at[obj_idx].set(1)

            def transfer_force(m_cell, obj_idx, detected_obj_idx):
                # Reassign any forces belonging to the detected object to the new object
                # First, identify the forces of the detected object.
                detected_forces = jax.lax.dynamic_slice(
                    m_cell, (self.n_objs + detected_obj_idx * N_FORCES,), (N_FORCES,)
                )
                # Then remove them from the detected object.
                m_cell = jax.lax.dynamic_update_slice(
                    m_cell, jnp.zeros(N_FORCES, dtype=bool), (self.n_objs + detected_obj_idx * N_FORCES,)
                )
                # Then copy them to the new object.
                m_cell = jax.lax.dynamic_update_slice(
                    m_cell, detected_forces, (self.n_objs + obj_idx * N_FORCES,)
                )
                return m_cell

            m_cell = jax.lax.cond(
                detected_obj_idx != -1,
                transfer_force,
                lambda m_cell, _, __: m_cell,
                m_cell, obj_idx, detected_obj_idx
            )

            m_cell = remove_colliding_objs(m_cell, obj_idx, self.coll_mat)
            return rng, m_cell

        # @partial(jax.jit, static_argnums=(3))
        def project_no_obj(rng, m_cell, cell_i, cell_detect_out: CellFnReturn, kernel_detect_out: KernelFnReturn,
                        pattern_detect_out: PatternFnReturn, obj):
            meta_objs = cell_detect_out.detected_meta_objs
            kernel_meta_objs = kernel_detect_out.detected_meta_objs
            pattern_meta_objs = pattern_detect_out.detected_meta_objs
            obj_idx = disambiguate_meta(obj, meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)
            # Remove the object
            m_cell = m_cell.at[obj_idx].set(0)
            # Remove any existing forces from the object
            jax.lax.dynamic_update_slice(
                m_cell, jnp.zeros(N_FORCES, dtype=bool), (self.n_objs + obj_idx * N_FORCES,)
            )
            return rng, m_cell

        # @partial(jax.jit, static_argnums=(3))
        def project_no_meta(rng, m_cell: chex.Array, cell_i, cell_detect_out, kernel_detect_out, pattern_detect_out, obj: int):
            sub_objs = expand_meta_objs([obj], self.meta_objs, self.char_to_obj)
            obj_idxs = np.array([self.objs_to_idxs[so] for so in sub_objs])
            # TODO: vmap this
            for obj_idx in obj_idxs:
                m_cell = m_cell.at[obj_idx].set(0)
                # Remove any existing forces from the object
                jax.lax.dynamic_update_slice(
                    m_cell, jnp.zeros(N_FORCES, dtype=bool), (self.n_objs + obj_idx * N_FORCES,)
                )
            return rng, m_cell

        # @partial(jax.jit, static_argnums=(3, 4))
        def project_force_obj(rng, m_cell, cell_i, cell_detect_out: CellFnReturn, kernel_detect_out: KernelFnReturn,
                            pattern_detect_out: PatternFnReturn, obj, force_idx):
            meta_objs = cell_detect_out.detected_meta_objs
            kernel_meta_objs = kernel_detect_out.detected_meta_objs
            pattern_meta_objs = pattern_detect_out.detected_meta_objs
            obj_idx = disambiguate_meta(obj, meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)
            # Add the object
            m_cell = m_cell.at[obj_idx].set(1)
            if force_idx is None:
                # Generate random movement.
                force_idx = jax.random.randint(rng, (1,), 0, N_FORCES-1)[0]
                rng, _ = jax.random.split(rng)
            # Remove any existing forces from the object
            m_cell = jax.lax.dynamic_update_slice(
                m_cell, jnp.zeros(N_FORCES, dtype=bool), (self.n_objs + obj_idx * N_FORCES,)
            )
            # Also remove player action mask
            m_cell = m_cell.at[-1].set(0)

            # Place the new force
            # m_cell = m_cell.at[n_objs + (obj_idx * N_FORCES) + force_idx].set(1)

            # Remove any existing forces from the object and add the new one
            force_arr = jnp.zeros(N_FORCES, dtype=bool)
            force_arr = force_arr.at[force_idx].set(1)
            m_cell = jax.lax.dynamic_update_slice(
                m_cell, force_arr, (self.n_objs + obj_idx * N_FORCES,)
            )

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
        def project_moving_obj(rng, m_cell, cell_i, cell_detect_out: CellFnReturn, kernel_detect_out: KernelFnReturn,
                            pattern_detect_out: PatternFnReturn, obj):
            meta_objs = cell_detect_out.detected_meta_objs
            kernel_meta_objs = kernel_detect_out.detected_meta_objs
            pattern_meta_objs = pattern_detect_out.detected_meta_objs
            obj_idx = disambiguate_meta(obj, meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)

            # Look for detected force index in corresponding input cell, then kernel, then pattern.
            # This should never end up as -1.
            force_idx = jax.lax.select(
                cell_detect_out.detected_moving_idx != -1,
                cell_detect_out.detected_moving_idx,
                kernel_detect_out.detected_moving_idx,
            )
            force_idx = jax.lax.select(
                force_idx == -1,
                pattern_detect_out.detected_moving_idx,
                force_idx,
            )
                
            m_cell = m_cell.at[obj_idx].set(1)

            # Remove any existing forces from the object and add the new one
            force_arr = jnp.zeros(N_FORCES, dtype=bool)
            force_arr = force_arr.at[force_idx].set(1)
            # m_cell = m_cell.at[self.n_objs + (obj_idx * N_FORCES) + force_idx].set(1)
            m_cell = jax.lax.dynamic_update_slice(
                m_cell, force_arr, (self.n_objs + obj_idx * N_FORCES,)
            )

            if DEBUG:
                jax.debug.print('project_moving_obj, obj_idx: {obj_idx}, force_idx: {force_idx}',
                                obj_idx=obj_idx, force_idx=force_idx)
            m_cell = remove_colliding_objs(m_cell, obj_idx, self.coll_mat)
            return rng, m_cell

        def project_stationary_obj(rng, m_cell, cell_i, cell_detect_out: CellFnReturn, kernel_detect_out: KernelFnReturn,
                                pattern_detect_out: PatternFnReturn, obj):
            meta_objs = cell_detect_out.detected_meta_objs
            kernel_meta_objs = kernel_detect_out.detected_meta_objs
            pattern_meta_objs = pattern_detect_out.detected_meta_objs
            obj_idx = disambiguate_meta(obj, meta_objs, kernel_meta_objs, pattern_meta_objs, self.objs_to_idxs)
            m_cell = m_cell.at[obj_idx].set(1)
            m_cell = m_cell.at[self.n_objs + (obj_idx * N_FORCES): self.n_objs + ((obj_idx + 1) * N_FORCES)].set(0)
            m_cell = remove_colliding_objs(m_cell, obj_idx, self.coll_mat)
            return rng, m_cell

        def gen_cell_projection_fn(r_cell, right_force_idx):
            fns = []
            if r_cell is None:
                r_cell = []
            else:
                r_cell = r_cell.split(' ')
            no, force, moving, stationary, random, vertical, horizontal, random_dir = False, False, False, False, False, False, False, False
            for obj in r_cell:
                obj = obj.lower()
                if obj in self.char_to_obj:
                    obj = self.char_to_obj[obj]
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
                elif (obj in self.objs_to_idxs) or (obj in self.meta_objs):
                    if no:
                        if obj in self.objs_to_idxs:
                            fns.append(partial(project_no_obj, obj=obj))
                        elif obj in self.meta_objs:
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
                        fns.append(partial(project_force_obj, obj=obj, force_idx=None))
                        random_dir = False
                    else:
                        fns.append(partial(project_obj, obj=obj))
                else:
                    raise Exception(f'Invalid object `{obj}` in rule.')
            
            # @partial(jax.jit, static_argnums=())
            def project_cell(rng, m_cell, cell_i, cell_detect_out, kernel_detect_out, pattern_detect_out):
                m_cell = m_cell & ~cell_detect_out.detected
                assert len(m_cell.shape) == 1, f'Invalid cell shape {m_cell.shape}'
                for proj_fn in fns:
                    rng, m_cell = proj_fn(rng=rng, m_cell=m_cell, cell_i=cell_i, cell_detect_out=cell_detect_out,
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
                if has_right_pattern and rp is None:
                    rp = np.array([None] * len(lp))
                if has_right_pattern:
                    for i, r_cell in enumerate(rp):
                        if r_cell == '...':
                            assert is_line_detector, f"`...` not found in left pattern of rule {self.rule_name}"
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

                        if self.jit:
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
                                                    rule_name=self.rule_name, kernel_detected=kernel_detected, lp=lp),
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
                    return kernel_activations, cell_detect_outs, kernel_detect_out


                def project_kernel(rng, lvl, kernel_activations, 
                                cell_detect_outs: List[CellFnReturn],
                                kernel_detect_outs: List[KernelFnReturn], 
                                pattern_detect_out: List[PatternFnReturn]
                                ):
                    n_tiles = np.prod(lvl.shape[-2:])
                    # Ensure we always have some invalid coordinates so that the loop will break even when all tiles are active
                    if self.jit:
                        kernel_activ_xys = jnp.argwhere(kernel_activations == 1, size=n_tiles+1, fill_value=-1)
                    else:
                        kernel_activ_xys = np.argwhere(kernel_activations == 1)
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
                    # TODO: use a jax.lax.switch
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
                    if rule.command is None and not np.all([r is None for r in rps]):
                        print(rps)

                    if rule.command == 'cancel':
                        cancel = pattern_detected
                    elif rule.command == 'restart':
                        restart = pattern_detected
                if rule.command == 'again':
                    again = rule_applied
                    if DEBUG:
                        jax.debug.print('applying the {command} command: {rule_applied}', command=rule.command, rule_applied=rule_applied)
                elif rule.command == 'win':
                    win = True

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
            kern_tpls = gen_perp_par_subrules(l_kerns, r_kerns)
        else:
            kern_tpls = [(l_kerns, r_kerns)]

        # Replace any empty lists in lp and rp with a None
        for l_kerns, r_kerns in kern_tpls:
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

                rule_fns.append(gen_rotated_rule_fn(l_kerns_rot, r_kerns_rot, rot, rule.command))

        return rule_fns
            

    def gen_tick_fn(self):
        rule_blocks = []
        late_rule_grps = []
        for rule_block in self.tree.rules:

            # FIXME: what's with this unnecessary list?
            assert len(rule_block) == 1
            rule_block = rule_block[0]

            looping = rule_block.looping
            rule_grps = []
            for rule in rule_block.rules:
                sub_rule_fns = self.gen_subrules_meta(rule, rule_name=str(rule))
                if '+' in rule.prefixes:
                    rule_grps[-1].extend(sub_rule_fns)
                elif 'late' in rule.prefixes:
                    late_rule_grps.append(sub_rule_fns)
                else:
                    rule_grps.append(sub_rule_fns)
            rule_blocks.append((looping, rule_grps))

        _move_rule_fn = partial(apply_movement, obj_to_idxs=self.objs_to_idxs, coll_mat=self.coll_mat,
                                n_objs=self.n_objs, jit=self.jit)
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
                    print('\n' + multihot_to_desc(lvl[0], self.objs_to_idxs, self.n_objs))

            def apply_turn(carry):
                lvl, _, turn_app_i, cancelled, restart, turn_again, win, rng = carry
                turn_app_i += 1
                applied = False
                again = False

                # for block_i, (looping, rule_grps) in enumerate(rule_blocks):
                    # n_prior_rules
                
                _loop_rule_block = partial(
                    loop_rule_block,
                    ### COMPILE VS RUNTIME ###
                    # n_prior_rules_arr=n_prior_rules_arr, n_rules_per_grp_arr=n_rules_per_grp_arr, all_rule_fns=all_rule_fns,
                    # n_grps_per_block_arr=n_grps_per_block_arr, blocks_are_looping_lst=blocks_are_looping_lst,
                    ### COMPILE VS RUNTIME ###
                    obj_to_idxs=self.objs_to_idxs, n_objs=self.n_objs,
                    jit=self.jit,
                )

                ### COMPILE VS RUNTIME ###
                for block_i, (looping, rule_block) in enumerate(rule_blocks):
                    carry = (lvl, applied, again, cancelled, restart, win, rng, block_i)
                    lvl, applied, again, cancelled, restart, win, rng, block_i = _loop_rule_block(
                        carry=carry,
                        rule_block=rule_block,
                        looping=looping,
                    )

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

                return lvl, turn_applied, turn_app_i, cancelled, restart, turn_again, win, rng

            turn_applied = True
            turn_app_i = 0
            turn_again = True
            win = False

            init_carry = (lvl, turn_applied, turn_app_i, cancelled, restart, turn_again, win, rng)
            carry = apply_turn(init_carry)
            lvl, turn_applied, turn_app_i, cancelled, restart, turn_again, win, rng = carry

            # if not jit:
            #     print('\nLevel after applying rules:\n', multihot_to_desc(lvl[0], obj_to_idxs, n_objs))
            #     print('grp_applied:', grp_applied)
            #     print('cancelled:', cancelled)

            return lvl[0], turn_applied, turn_app_i, cancelled, restart, win, rng

        if self.jit:
            tick_fn = jax.jit(tick_fn)
        return tick_fn


def multihot_to_desc(multihot_level, objs_to_idxs, n_objs, show_background=False):
    """Converts a multihot array to a 2D list of descriptions.
    
    Args:
        multihot_level: A multihot array of shape [n_objects + n_forces, height, width].
        obj_to_idxs: Dictionary mapping object names to their indices.
    
    Returns:
        A 2D list where each cell contains a string describing all objects and forces present.
    """
    height, width = multihot_level.shape[1:]
    
    # Create a reverse mapping from indices to object names
    idxs_to_obj = {idx: obj for obj, idx in objs_to_idxs.items()}
    
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
                    if obj_name == 'background' and not show_background:
                        continue
                    obj_desc = obj_name
                    
                    # Check if there's a force applied to this object
                    force_names = ["left", "down", "right", "up", "action"]
                    forces = []
                    for f_idx, force_name in enumerate(force_names):
                        force_channel = n_objs + (obj_idx * N_FORCES) + f_idx
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