from functools import partial
from typing import Iterable, List

from einops import rearrange
import flax
import jax
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np

from jax_utils import stack_leaves
from ps_game import LegendEntry, PSGame, WinCondition

# TODO: double check these colors
color_hex_map = {
    "black": "#000000",
    "white": "#FFFFFF",
    "lightgray": "#D3D3D3",
    "lightgrey": "#D3D3D3",
    "gray": "#808080",
    "grey": "#808080",
    "darkgray": "#A9A9A9",
    "darkgrey": "#A9A9A9",
    "red": "#FF0000",
    "darkred": "#8B0000",
    "lightred": "#FF6666",
    "brown": "#A52A2A",
    "darkbrown": "#5C4033",
    "lightbrown": "#C4A484",
    "orange": "#FFA500",
    "yellow": "#FFFF00",
    "green": "#008000",
    "darkgreen": "#006400",
    "lightgreen": "#90EE90",
    "blue": "#0000FF",
    "lightblue": "#ADD8E6",
    "darkblue": "#00008B",
    "purple": "#800080",
    "pink": "#FFC0CB",
    "transparent": "#00000000"  # Transparent in RGBA format
}

# @partial(jax.jit, static_argnums=(0))
def disambiguate_meta(obj, meta_objs, obj_to_idxs):
    """In the right pattern of rules, we may have a meta-object (mapping to a corresponding meta-object in the 
    left pattern). This function uses the `meta_objs` dictionary returned by the detection function to project
    the correct object during rule application."""
    if obj in obj_to_idxs:
        return obj_to_idxs[obj]
    return meta_objs[obj]

def replace_bg_tiles(x):
    if x == '.':
        return 0
    else:
        return int(x) + 1

def render_solid_color(color):
    color = color.lower()
    if color in color_hex_map:
        c = color_hex_map[color]
    else:
        c = '#000000'
    c = hex_to_rgba(c, 255)
    im = np.zeros((5, 5, 4), dtype=np.uint8)
    im[:, :, :] = np.array(c)
    return im

def render_sprite(colors, sprite):
    sprite = np.vectorize(replace_bg_tiles)(sprite)
    colors = np.array(['transparent'] + colors)
    colors_vec = np.zeros((len(colors), 4), dtype=np.uint8)
    for i, c in enumerate(colors):
        c = c.lower()
        if c in color_hex_map:
            alpha = 255
            if c == 'transparent':
                alpha = 0
            c = color_hex_map[c]
        c = hex_to_rgba(c, alpha)
        colors_vec[i] = np.array(c)
    im = colors_vec[sprite]
    return im

def level_to_multihot(level):
    pass

def assign_vecs_to_objs(collision_layers):
    n_lyrs = len(collision_layers)
    n_objs = sum([len(lyr) for lyr in collision_layers])
    coll_masks = np.zeros((n_lyrs, n_objs), dtype=bool)
    objs_to_idxs = {}
    # vecs = np.eye(n_objs, dtype=np.uint8)
    # obj_vec_dict = {}
    j = 0
    for i, layer in enumerate(collision_layers):
        for obj in layer:
            # obj_vec_dict[obj] = vecs[i]
            objs_to_idxs[obj] = j
            coll_masks[i, j] = 1
            j += 1
    return objs_to_idxs, coll_masks

def hex_to_rgba(hex_code, alpha):
    """Converts a hex color code to RGB values."""

    hex_code = hex_code.lstrip('#')
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return (*rgb, alpha)

def process_legend(legend):
    char_legend = {}
    meta_tiles = {}
    conjoined_tiles = {}
    for k, v in legend.items():
        k = k.split(' ')[0]
        v: LegendEntry
        if v.operator is None:
            assert len(v.obj_names) == 1
            char_legend[v.obj_names[0]] = k.strip()
        elif v.operator == 'or':
            meta_tiles[k.strip()] = v.obj_names
        elif v.operator == 'and':
            conjoined_tiles[k.strip()] = v.obj_names
        else: raise Exception('Invalid LegendEntry operator.')
    return char_legend, meta_tiles, conjoined_tiles

def expand_collision_layers(collision_layers, meta_tiles):
    # Preprocess collision layers to replace joint objects with their sub-objects
    for i, l in enumerate(collision_layers):
        j = 0
        for o in l:
            if o in meta_tiles:
                subtiles = meta_tiles[o]
                l = l[:j] + subtiles + l[j+1:]
                collision_layers[i] = l
                # HACK: we could do this more efficiently
                expand_collision_layers(collision_layers, meta_tiles=meta_tiles)
                j += len(subtiles)
            else:
                j += 1
    return collision_layers

def expand_meta_tiles(tile_list, obj_to_idxs, meta_tiles):
    expanded_meta_tiles = []
    for mt in tile_list:
        if mt in obj_to_idxs:
            expanded_meta_tiles.append(mt)
        elif mt in tile_list:
            expanded_meta_tiles += expand_meta_tiles(meta_tiles[mt], obj_to_idxs, meta_tiles)
    return expanded_meta_tiles

def get_meta_channel(lvl, obj_idxs):
    return jnp.any(lvl[jnp.array(obj_idxs)], axis=0)

def gen_check_win(win_conditions: Iterable[WinCondition], obj_to_idxs, meta_tiles, jit=True):

    def check_all(lvl, src, trg):
        src_channel = get_meta_channel(lvl, src)
        trg_channel = get_meta_channel(lvl, trg)
        # There can be no source objects that do not overlap target objects
        return ~jnp.any(src_channel & ~trg_channel)

    def check_some(lvl, src, trg):
        src_channel = get_meta_channel(lvl, src)
        trg_channel = get_meta_channel(lvl, trg)
        return jnp.any(src_channel & trg_channel)

    def check_none(lvl, src):
        src_channel = get_meta_channel(lvl, src)
        return ~jnp.any(src_channel)

    funcs = []
    for win_condition in win_conditions:
        src, trg = win_condition.src_obj, win_condition.trg_obj
        if src in obj_to_idxs:
            src = [obj_to_idxs[src]]
        else:
            src_objs = disambiguate_meta(src, meta_tiles, obj_to_idxs)
            src = [obj_to_idxs[obj] for obj in src_objs]
        if trg is not None:
            if trg in obj_to_idxs:
                trg = [obj_to_idxs[trg]]
            else:
                trg_objs = disambiguate_meta(trg, meta_tiles, obj_to_idxs)
                trg = [obj_to_idxs[obj] for obj in trg_objs]
        if win_condition.quantifier == 'all':
            func = partial(check_all, src=src, trg=trg)
        elif win_condition.quantifier == 'some':
            func = partial(check_some, src=src, trg=trg)
        elif win_condition.quantifier == 'no':
            func = partial(check_none, src=src)
        else:
            breakpoint()
            raise Exception('Invalid quantifier.')
        funcs.append(func)

    def check_win(lvl):

        if len(funcs) == 0:
            return False
        
        def apply_win_condition_func(i, lvl):
            return jax.lax.switch(i, funcs, lvl)

        if jit:
            func_returns = jax.vmap(apply_win_condition_func, in_axes=(0, None))(jnp.arange(len(funcs)), lvl)
        else:
            func_returns = jnp.array([f(lvl) for f in funcs])
        return jnp.all(func_returns)

    return check_win


# def gen_subrule(rule, n_objs, obj_to_idxs, meta_tiles):
#     assert len(rule.prefixes) == 0
#     lp = rule.left_patterns
#     rp = rule.right_patterns
#     n_kernels = len(lp)
#     assert n_kernels == 1
#     assert len(rp) == n_kernels
#     lp = lp[0]
#     rp = rp[0]
#     n_cells = len(lp)
#     assert len(rp) == n_cells

#     lr_in = np.zeros((n_objs + (n_objs * 4), 1, n_cells), dtype=np.int8)
#     rr_in = np.zeros_like(lr_in)
#     ur_in = np.zeros((n_objs + (n_objs * 4), n_cells, 1), dtype=np.int8)
#     dr_in = np.zeros_like(ur_in)
#     lr_out = np.zeros_like(lr_in)
#     rr_out = np.zeros_like(rr_in)
#     ur_out = np.zeros_like(ur_in)
#     dr_out = np.zeros_like(dr_in)
#     for i, (l_cell, r_cell) in enumerate(zip(lp, rp)):
#         # Left cell
#         if len(l_cell) == 1:
#             l_cell = l_cell[0]
#             l_cell = l_cell.split(' ')
#         l_force = False
#         l_no = False
#         for obj in l_cell:
#             obj = obj.lower()
#             if obj in meta_tiles:
#                 return None
#             if obj in obj_to_idxs:
#                 fill_val = 1
#                 if l_no:
#                     fill_val = -1
#                 obj_i = obj_to_idxs[obj]
#                 lr_in[obj_i, 0, i] = fill_val
#                 rr_in[obj_i, 0, n_cells-1-i] = fill_val
#                 ur_in[obj_i, i, 0] = fill_val
#                 dr_in[obj_i, n_cells-1-i, 0] = fill_val
#                 if l_force:
#                     lr_in[n_objs + (obj_i * 4) + 2, 0, i] = 1
#                     rr_in[n_objs + (obj_i * 4) + 0, 0, n_cells-1-i] = 1
#                     ur_in[n_objs + (obj_i * 4) + 1, i, 0] = 1
#                     dr_in[n_objs + (obj_i * 4) + 3, n_cells-1-i, 0] = 1
#                 l_force, l_no = False, False
#             elif obj == '>':
#                 l_force = True
#             elif obj == 'no':
#                 l_no = True
#             else: raise Exception(f'Invalid object `{obj}` in rule.')
#         # Right cell
#         if len(r_cell) == 1:
#             r_cell = r_cell[0]
#             r_cell = r_cell.split(' ')
#         r_force = False
#         r_no = False
#         for obj in r_cell:
#             obj = obj.lower()
#             # NOTE: Should never have a meta-tile here if we didn't already see one on the left side (?)
#             if obj in obj_to_idxs:
#                 fill_val = 1
#                 if r_no:
#                     fill_val = -1
#                 obj_i = obj_to_idxs[obj]
#                 lr_out[obj_i, 0, i] = fill_val
#                 rr_out[obj_i, 0, n_cells-1-i] = fill_val
#                 ur_out[obj_i, i, 0] = fill_val
#                 dr_out[obj_i, n_cells-1-i, 0] = fill_val
#                 if r_force:
#                     lr_out[n_objs + (obj_i * 4) + 2, 0, i] = 1
#                     rr_out[n_objs + (obj_i * 4) + 0, 0, n_cells-1-i] = 1
#                     ur_out[n_objs + (obj_i * 4) + 1, i, 0] = 1
#                     dr_out[n_objs + (obj_i * 4) + 3, n_cells-1-i, 0] = 1
#                 r_force, r_no = False, False
#             elif obj == '>':
#                 r_force = True
#             elif obj == 'no':
#                 r_no = True
#             else: raise Exception('Invalid object in rule.')
#     lr_out = lr_out - np.clip(lr_in, 0, 1)
#     rr_out = rr_out - np.clip(rr_in, 0, 1)
#     ur_out = ur_out - np.clip(ur_in, 0, 1)
#     dr_out = dr_out - np.clip(dr_in, 0, 1)
#     lr_rule = np.stack((lr_in, lr_out), axis=0)
#     rr_rule = np.stack((rr_in, rr_out), axis=0)
#     ur_rule = np.stack((ur_in, ur_out), axis=0)
#     dr_rule = np.stack((dr_in, dr_out), axis=0)
#     # TODO: If horizontal/vertical etc. has been specified, filter out unnecessary rules here
#     rules = [lr_rule, rr_rule, ur_rule, dr_rule]
#     rule_names = [f"{rule}_{dir}" for dir in enumerate(['lr', 'ur', 'rr', 'dr'])]
#     rule_fns = [partial(apply_move_rule, move_rule=rule, rule_name=rule_name) 
#                     for rule, rule_name in zip(rules, rule_names)]
#     return rule_fns


@flax.struct.dataclass
class ObjFnReturn:
    active: bool = False
    # detected object/force indices
    force_idx: int = None
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
    meta_objs: dict


def gen_subrules_meta(rule, n_objs, obj_to_idxs, meta_tiles, rule_name, jit=True):
    idxs_to_objs = {v: k for k, v in obj_to_idxs.items()}

    def is_obj_forceless(obj_idx, m_cell):
        # return jnp.sum(m_cell[n_objs + (obj_idx * 4):n_objs + (obj_idx * 4) + 4]) == 0
        return jnp.sum(jax.lax.dynamic_slice(m_cell, (n_objs + (obj_idx * 4),), (4,))) == 0

    ### Functions for detecting regular atomic objects
    # @partial(jax.jit, static_argnums=(0,))
    def detect_obj_in_cell(obj_idx, m_cell):
        # active = m_cell[obj_idx] == 1 & is_obj_forceless(obj_idx, m_cell)
        active = m_cell[obj_idx] == 1

        # jax.lax.cond(
        #     active,
        #     lambda: jax.debug.print('detected obj_idx: {obj_idx}', obj_idx=obj_idx),
        #     lambda: None,
        # )
        return ObjFnReturn(active=active, obj_idx=obj_idx)

    # @partial(jax.jit, static_argnums=(0,))
    def detect_no_obj_in_cell(obj_idx, m_cell):
        active = m_cell[obj_idx] == 0
        return ObjFnReturn(active=active, obj_idx=obj_idx)

    # @partial(jax.jit, static_argnums=(0, 1))
    def detect_force_on_obj(obj_idx, force_idx, m_cell):
        obj_is_present = m_cell[obj_idx] == 1
        force_is_present = m_cell[n_objs + (obj_idx * 4) + force_idx] == 1
        # force_idx = np.argwhere(m_cell[n_objs + (obj_idx * 4):n_objs + (obj_idx * 4) + 4] == 1)
        # assert len(force_idx) <= 1
        active = obj_is_present & force_is_present
        obj_idx = jax.lax.select(
            active,
            obj_idx,
            -1,
        )
        # jax.lax.cond(
        #     active,
        #     lambda: jax.debug.print('detected force_idx {force_idx} on obj_idx: {obj_idx}', obj_idx=obj_idx, force_idx=force_idx),
        #     lambda: None,
        # )
        return ObjFnReturn(active=active, obj_idx=obj_idx)

    ### Functions for detecting meta-objects
    # @partial(jax.jit, static_argnums=())
    def detect_any_objs_in_cell(objs_vec, m_cell):
        """Given a multi-hot vector indicating a set of objects, return the index of the object contained in this cell."""
        m_cell_forceless_objs = jax.vmap(is_obj_forceless, in_axes = (0, None))(jnp.arange(n_objs), m_cell)
        m_cell_forceless = m_cell.at[:n_objs].set(m_cell_forceless_objs * m_cell[:n_objs])
        detected_vec_idx = jnp.argwhere(objs_vec * m_cell_forceless > 0, size=1, fill_value=-1)[0, 0]
        active = detected_vec_idx != -1
        obj_idx = jax.lax.select(
            active,
            detected_vec_idx,
            -1,
        )
        # jax.lax.cond(
        #     active,
        #     lambda: jax.debug.print('detected obj_idx: {obj_idx}', obj_idx=obj_idx),
        #     lambda: None,
        # )
        return ObjFnReturn(active=active, obj_idx=obj_idx)

    # @partial(jax.jit, static_argnums=())
    def detect_no_objs_in_cell(objs_vec, m_cell):
        active = ~jnp.any(objs_vec * m_cell)
        return ObjFnReturn(active=active)

    # @partial(jax.jit, static_argnums=(0, 1))
    def detect_force_on_meta(obj_idxs, force_idx, m_cell):
        force_obj_vecs = []
        for obj_idx in obj_idxs:
            force_obj_vec = np.zeros(n_objs + n_objs * 4, dtype=bool)
            force_obj_vec[obj_idx] = 1
            force_obj_vec[n_objs + obj_idx * 4 + force_idx] = 1
            force_obj_vecs.append(force_obj_vec)
        obj_activations = jnp.sum(jnp.array(force_obj_vecs) * m_cell[None], axis=1) 
        active = jnp.any(obj_activations == 2)

        # Right before the first cond, capture the intended value:
        captured_obj_idx = obj_idx  # This captures the current loop's obj_idx

        obj_idxs = jnp.array(obj_idxs)
        obj_idx = jax.lax.select(
            active,
            obj_idxs[jnp.argwhere(obj_activations == 2, size=1)[0][0]],
            -1,
        )
        captured_selected_obj_idx = obj_idx  # Capture after selection

        active = ((obj_idx != -1) & active)

        # Why does this print out so much, even when apparently non-active??
        # jax.lax.cond(
        #     active,
        #     lambda captulevel-{level_i}level-{level_i}__red_selected_obj_idx=captured_selected_obj_idx: jax.debug.print(
        #         "B) detected force_idx {force_idx} on meta-obj_idx: {obj_idx}",
        #         force_idx=force_idx, obj_idx=captured_selected_obj_idx),
        #     lambda: None
        # )
        return ObjFnReturn(active=active, obj_idx=obj_idx)

    dirs_to_force_idx = {
        'up': 3,
        'right': 2,
        'down': 1,
        'left': 0,
    }

    def gen_cell_detection_fn(l_cell, force_idx):
        """Produce a function to detect whether all objects/conditions in a cell (within a kernel, within the left pattern of a rule) are present.
        So for the rule `[> Player | Crate] -> [> Player | > Crate]`, this will return a function that detects, for the first cell in the left pattern,
        if the player is present, and has a force applied to it.
        """
        fns = []
        l_cell = l_cell.split(' ')
        no, force, directional_force = False, False, False
        obj_names = []
        for obj in l_cell:
            obj = obj.lower()
            if obj == 'no':
                no = True
            elif obj == '>':
                force = True
            elif obj in ['up', 'down', 'left', 'right']:
                directional_force = True
                dir_force_idx = dirs_to_force_idx[obj]
            else:
                obj_names.append(obj)
                if obj in obj_to_idxs:
                    obj_idx = obj_to_idxs[obj]
                    if no:
                        fns.append(partial(detect_no_obj_in_cell, obj_idx=obj_idx))
                        no = False
                    elif force:
                        fns.append(partial(detect_force_on_obj, obj_idx=obj_idx, force_idx=force_idx))
                        force = False
                    elif directional_force:
                        fns.append(partial(detect_force_on_obj, obj_idx=obj_idx, force_idx=dir_force_idx))
                        directional_force = False
                    else:
                        fns.append(partial(detect_obj_in_cell, obj_idx=obj_idx))
                elif obj in meta_tiles:
                    sub_objs = expand_meta_tiles([obj], obj_to_idxs, meta_tiles)
                    sub_obj_idxs = [obj_to_idxs[so] for so in sub_objs]
                    sub_objs_vec = np.zeros((n_objs + n_objs * 4), dtype=bool)
                    sub_objs_vec[sub_obj_idxs] = 1
                    if no:
                        print(l_cell)
                        fns.append(partial(detect_no_objs_in_cell, objs_vec=sub_objs_vec))
                        no = False
                    elif force:
                        fns.append(partial(detect_force_on_meta, obj_idxs=tuple(sub_obj_idxs), force_idx=force_idx))
                        force = False
                    elif directional_force:
                        fns.append(partial(detect_force_on_meta, obj_idxs=tuple(sub_obj_idxs), force_idx=dir_force_idx))
                        directional_force = False
                    else:
                        fns.append(partial(detect_any_objs_in_cell, objs_vec=sub_objs_vec))
                else:
                    raise Exception(f'Invalid object `{obj}` in rule.')
        
        # @partial(jax.jit, static_argnums=())
        def cell_detection_fn(m_cell):
            # TODO: can vmap this
            fn_outs: List[ObjFnReturn] = [fn(m_cell=m_cell) for fn in fns]
            activated = jnp.all(jnp.array([f.active for f in fn_outs]))
            detected = jnp.zeros(m_cell.shape, dtype=bool)
            force_idx = None
            for i, f in enumerate(fn_outs):
                # if f.obj_idx != -1:
                #     jax.debug.print('obj_idx: {obj_idx}', obj_idx=f.obj_idx)
                #     detected = detected.at[f.obj_idx].set(1)
                detected = jax.lax.cond(
                    f.obj_idx != -1,
                    lambda x, detected: detected.at[x].set(1),
                    lambda x, detected: detected,
                    f.obj_idx, detected
                )
                if f.force_idx is not None:
                    detected = detected.at[f.force_idx].set(1)
                    if force_idx is None:
                        force_idx = f.force_idx
            meta_objs = {k: fn_out.obj_idx for k, fn_out in zip(obj_names, fn_outs)}
            ret = CellFnReturn(
                detected=detected,
                force_idx=force_idx,
                meta_objs=meta_objs,
            )
            # jax.debug.print('detected: {detected}', detected=detected)
            return activated, ret

        return cell_detection_fn

    # @partial(jax.jit, static_argnums=(2))
    def project_obj(m_cell, detect_out, obj):
        meta_objs = detect_out.meta_objs
        obj_idx = disambiguate_meta(obj, meta_objs, obj_to_idxs)
        m_cell = m_cell.at[obj_idx].set(1)
        return m_cell

    # @partial(jax.jit, static_argnums=(2))
    def project_no_obj(m_cell, detect_out, obj):
        meta_objs = detect_out.meta_objs
        obj_idx = disambiguate_meta(obj, meta_objs, obj_to_idxs)
        m_cell[obj_idx] = 0
        return m_cell

    # @partial(jax.jit, static_argnums=(2))
    def project_force_on_obj(m_cell, detect_out, obj, force_idx):
        meta_objs = detect_out.meta_objs
        obj_idx = disambiguate_meta(obj, meta_objs, obj_to_idxs)
        m_cell = m_cell.at[obj_idx].set(1)
        m_cell = m_cell.at[n_objs + (obj_idx * 4) + force_idx].set(1)

        if jit:
            jax.debug.print('project_force_on_obj: {obj_idx}', obj_idx=obj_idx)

        else:
            # TODO: jax this
            obj_name = idxs_to_objs[int(obj_idx)]
            jax.debug.print('project_force_on_obj: {obj_name}', obj_name=obj_name)

        return m_cell

    def gen_cell_projection_fn(r_cell, force_idx):
        fns = []
        if r_cell is None:
            r_cell = []
        else:
            r_cell = r_cell.split(' ')
        no, force, directional_force = False, False, False
        for obj in r_cell:
            obj = obj.lower()
            if obj == 'no':
                no = True
            elif obj == '>':
                force = True
            elif obj in ['up', 'down', 'left', 'right']:
                directional_force = True
                dir_force_idx = dirs_to_force_idx[obj]
            elif (obj in obj_to_idxs) or (obj in meta_tiles):
                if no:
                    fns.append(partial(project_no_obj, obj=obj))
                elif force:
                    fns.append(partial(project_force_on_obj, obj=obj, force_idx=force_idx))
                elif directional_force:
                    fns.append(partial(project_force_on_obj, obj=obj, force_idx=dir_force_idx))
                else:
                    fns.append(partial(project_obj, obj=obj))
            else:
                raise Exception(f'Invalid object `{obj}` in rule.')
        
        # @partial(jax.jit, static_argnums=())
        def cell_projection_fn(m_cell, detect_out):
            m_cell = m_cell & ~detect_out.detected
            for proj_fn in fns:
                m_cell = proj_fn(m_cell=m_cell, detect_out=detect_out)
            removed_something = jnp.any(detect_out.detected)
            jax.lax.cond(
                removed_something,
                lambda: jax.debug.print('removing detected: {det}', det=detect_out.detected),
                lambda: None
            )
            jax.debug.print('removing detected: {det}', det=detect_out.detected)
            return m_cell

        return cell_projection_fn

    def gen_rotated_rule_fn(lp, rp, rot, r_command):
        lp = np.array(lp)
        # Here we map force directions to corresponding rule rotations
        # if rot == 1:
        #     breakpoint()
        rot_to_force = [1, 2, 3, 0]
        force_idx = rot_to_force[rot]
        is_horizontal = lp.shape[0] == 1
        is_vertical = lp.shape[1] == 1
        is_single = lp.shape[0] == 1 and lp.shape[1] == 1
        in_patch_shape = lp.shape
        has_rp = rp is not None
        # TODO: kernels. We assume just 1 here.
        if is_horizontal:
            lp = lp[0, :]
            if has_rp:
                rp = rp[0, :]
        elif is_vertical:
            lp = lp[:, 0]
            if has_rp:
                rp = rp[:, 0]
        elif is_single:
            lp = lp[0, 0]
            if has_rp: 
                rp = rp[0, 0]
        else:
            raise Exception('Invalid rule shape.')
        is_line_detector = False
        cell_detection_fns = []
        cell_projection_fns = []
        for i, l_cell in enumerate(lp):
            if l_cell == '...':
                is_line_detector = True
                cell_detection_fns.append('...')
            cell_detection_fns.append(gen_cell_detection_fn(l_cell, force_idx))
        if has_rp:
            for i, r_cell in enumerate(rp):
                if r_cell == '...':
                    assert is_line_detector, f"`...` not found in left pattern of rule {rule_name}"
                    cell_projection_fns.append('...')
                cell_projection_fns.append(gen_cell_projection_fn(r_cell, force_idx))

        # @jax.jit
        def rot_rule_fn(lvl):
            n_chan = lvl.shape[1]

            # @jax.jit
            def detect_cells(in_patch):
                cell_outs_patch = []
                patch_active = True
                for i, cell_fn in enumerate(cell_detection_fns):
                    in_patch = in_patch.reshape((n_chan, *in_patch_shape))
                    if is_vertical:
                        m_cell = in_patch[:, i, 0]
                    if is_horizontal:
                        m_cell = in_patch[:, 0, i]
                    if is_single:
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
                    patch_activations, detect_outs = jax.vmap(jax.vmap(detect_cells))(patches)

                else:
                    patch_activations = jnp.zeros((patches.shape[0], patches.shape[1]), dtype=bool)
                    # What's up with this???
                    detect_outs = [[None for _ in range(patches.shape[0])] for _ in range(patches.shape[1])]
                    for i in range(patches.shape[0]):
                        for j in range(patches.shape[1]):
                            patch = patches[i, j]
                            patch_active, cell_out = detect_cells(patch)
                            patch_activations = patch_activations.at[i, j].set(patch_active)
                            detect_outs[j][i] = cell_out
                    detect_outs = stack_leaves(stack_leaves(detect_outs))

            else:
                breakpoint()

            # eliminate all but one activation
            lp_detected = patch_activations.sum() > 0
            jax.lax.cond(
                # True,
                lp_detected,
                lambda: jax.debug.print('Rule {rule_name} applied: {lp_detected}', rule_name=rule_name, lp_detected=lp_detected),
                lambda: None,
            )
            cancelled = False

            if has_rp:

                def project_cells(lvl, patch_activations, detect_outs):
                    # print('NONZERO ACTIVATIONS')
                    # print(lp, rp, force_idx)
                    # print(jnp.argwhere(patch_activations, size=1))

                    # Mask out everything but the position of the "first" activation
                    first_a = jnp.argwhere(patch_activations == 1, size=1)[0]
                    patch_activations = jnp.zeros_like(patch_activations)
                    patch_activations = patch_activations.at[*first_a].set(1)
                    # detect_outs = detect_outs[first_a[0]][first_a[1]]
                    detect_outs = jax.tree_map(lambda x: x[first_a[0]][first_a[1]], detect_outs)

                    # Apply projection functions to the affected cells
                    out_cell_idxs = np.indices(in_patch_shape)
                    out_cell_idxs = rearrange(out_cell_idxs, "xy h w -> h w xy")
                    if is_vertical:
                        out_cell_idxs = out_cell_idxs[:, 0]
                    elif is_horizontal:
                        out_cell_idxs = out_cell_idxs[0, :]
                    elif is_single:
                        out_cell_idxs = out_cell_idxs[0, 0]

                    lvl = jnp.array(lvl)
                    # assert len(detect_outs) == len(out_cell_idxs) == len(cell_projection_fns):
                    if not len(detect_outs) == len(out_cell_idxs) == len(cell_projection_fns):
                        breakpoint()
                    for out_cell_idx, detect_out, cell_proj_fn in zip(out_cell_idxs, detect_outs, cell_projection_fns):
                        out_cell_idx += first_a
                        m_cell = lvl[0, :, *out_cell_idx]
                        m_cell = jnp.array(m_cell)
                        m_cell = cell_proj_fn(m_cell, detect_out)
                        lvl = lvl.at[0, :, *out_cell_idx].set(m_cell)
                    return lvl

                new_lvl = jax.lax.cond(
                    lp_detected,
                    project_cells,
                    lambda lvl, _, __: lvl,
                    lvl, patch_activations, detect_outs
                )
                rule_applied = jnp.any(new_lvl != lvl)
                lvl = new_lvl

            else:
                assert r_command is not None
                if r_command == 'cancel':
                    cancelled = lp_detected
                    rule_applied = False

            return lvl, rule_applied, cancelled
        
        if jit:
            rot_rule_fn = jax.jit(rot_rule_fn)

        return rot_rule_fn
    
    rule_fns = []
    print('RULE PATTERNS', rule.left_patterns, rule.right_patterns)

    has_right_pattern = rule.right_patterns is not None

    for i, lp in enumerate(rule.left_patterns):
        # Replace any empty lists in lp and rp with a None
        lp = [[None] if len(l) == 0 else [' '.join(l)] for l in lp]
        lp = np.array(lp)

        rp, rp_rot = None, None
        if has_right_pattern:
            rp = rule.right_patterns[i]
            rp = [[None] if len(l) == 0 else [' '.join(l)] for l in rp]
            rp = np.array(rp)

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
        elif len(lp) > 1:
            rots = [0, 1, 2, 3]
        else:
            rots = [0]
        if lp.shape[0] == 1 and lp.shape[1] == 1:
            rule_fns.append(gen_rotated_rule_fn(lp, rp, 0, rule.command))
        for rot in rots:
            # rotate the patterns
            lp_rot = np.rot90(lp, rot)

            if has_right_pattern:
                rp_rot = np.rot90(rp, rot)

            rule_fns.append(gen_rotated_rule_fn(lp_rot, rp_rot, rot, rule.command))

    return rule_fns
        
def player_has_moved(lvl_0, lvl_1, obj_to_idxs):
    player_idx = obj_to_idxs['player']
    player_moved = jnp.sum(lvl_0[player_idx] != lvl_1[player_idx]) > 0
    jax.debug.print('player moved: {player_moved}', player_moved=player_moved)
    return player_moved
        
def gen_rule_fn(obj_to_idxs, coll_mat, tree_rules, meta_tiles, jit=True):
    n_objs = len(obj_to_idxs)
    rule_grps = []
    late_rule_grps = []
    if len(tree_rules) == 0:
        pass
    elif len(tree_rules) == 1:
        # TODO: Extend to multiple rule blocks
        rule_blocks = tree_rules[0]
        for rule_block in rule_blocks:
            assert rule_block.looping == False
            for rule in rule_block.rules:
                # TODO: rule-block and loop logics
                sub_rule_fns = gen_subrules_meta(rule, n_objs, obj_to_idxs, meta_tiles, rule_name=str(rule), jit=jit)
                if not 'late' in rule.prefixes:
                    rule_grps.append(sub_rule_fns)
                else:
                    late_rule_grps.append(sub_rule_fns)

    rule_grps.append(gen_move_rules(obj_to_idxs, coll_mat, jit=jit))
    rule_grps += late_rule_grps

    def rule_fn(lvl):
        changed = False
        cancelled = False
        lvl = lvl[None]

        if not jit:
            print('\n' + multihot_to_desc(lvl[0], obj_to_idxs))

        for grp_i, rule_grp in enumerate(rule_grps):
            grp_applied = True

            def apply_rule_grp(carry):
                lvl, grp_applied, grp_app_i, cancelled = carry
                grp_app_i += 1
                grp_applied = False

                for rule_fn in rule_grp:

                    def apply_rule_fn(carry):
                        lvl, rule_applied, i, cancelled = carry
                        lvl, rule_applied, cancelled = rule_fn(lvl)
                        i += 1
                        return lvl, rule_applied, i, cancelled

                    rule_applied = True
                    rule_app_i = 0
                    lvl, rule_applied, rule_app_i, cancelled = jax.lax.while_loop(
                        cond_fun=lambda x: x[1] & ~x[3],
                        body_fun=lambda x: apply_rule_fn(x),
                        init_val=(lvl, rule_applied, rule_app_i, cancelled),
                    )
                    rule_applied = rule_app_i > 1
                    grp_applied = jnp.logical_or(grp_applied, rule_applied)
                jax.debug.print('group {grp_i} applied: {grp_applied}', grp_i=grp_i, grp_applied=grp_applied)
                # print('rule: {rule}', rule=tree_rules[0][0].rules[grp_i])
                if not jit:
                    if grp_applied:
                        print('\n' + multihot_to_desc(lvl[0], obj_to_idxs))
                # force_is_present = jnp.sum(lvl[0, n_objs:n_objs + (n_objs * 4)]) > 0
                # jax.debug.print('force is present: {force_is_present}', force_is_present=force_is_present)
                # jax.debug.print('lvl:\n{lvl}', lvl=lvl)
                return lvl, grp_applied, grp_app_i, cancelled

            grp_app_i = 0
            if jit:
                lvl, grp_applied, grp_app_i, cancelled = jax.lax.while_loop(
                    cond_fun=lambda x: x[1] & ~x[3],
                    body_fun=apply_rule_grp,
                    init_val=(lvl, grp_applied, grp_app_i, cancelled),
                )
            else:
                while grp_applied and not cancelled:
                    lvl, grp_applied, grp_app_i, cancelled = apply_rule_grp((lvl, grp_applied, grp_app_i, cancelled))

            grp_applied = grp_app_i > 1


        return lvl[0], grp_applied, cancelled

    if jit:
        rule_fn = jax.jit(rule_fn)
    return rule_fn

def gen_move_rules(obj_to_idxs, coll_mat, jit=True):
    n_objs = len(obj_to_idxs)
    rule_fns = []
    coll_mat = coll_mat.astype(np.int8)
    for obj, idx in obj_to_idxs.items():
        if obj == 'Background':
            continue
        coll_vector = coll_mat[idx]

        left_rule_in = np.zeros((n_objs + (n_objs * 4), 1, 2), dtype=np.int8)
        # object must be present in right cell
        left_rule_in[idx, 0, 1] = 1
        # leftward force
        left_rule_in[n_objs + (idx * 4) + 0, 0, 1] = 1
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
        up_rule_in = np.zeros((n_objs + (n_objs * 4), 2, 1), dtype=np.int8)
        # object must be present in lower cell
        up_rule_in[idx, 0] = 1
        # upward force
        up_rule_in[n_objs + (idx * 4) + 1, 0] = 1
        # absence of collidable tiles in the upper cell
        up_rule_in[:n_objs, 1, 0] = -coll_vector
        up_rule_out = np.zeros_like(up_rule_in)
        # object moves to upper cell
        up_rule_out[idx, 1] = 1
        # remove anything that was present in the input
        up_rule_out -= np.clip(up_rule_in, 0, 1)
        up_rule = np.stack((up_rule_in, up_rule_out), axis=0)

        right_rule_in = np.zeros((n_objs + (n_objs * 4), 1, 2), dtype=np.int8)
        # object must be present in left cell
        right_rule_in[idx, 0, 0] = 1
        # rightward force
        right_rule_in[n_objs + (idx * 4) + 2, 0, 0] = 1
        # absence of collidable tiles in the right cell
        right_rule_in[:n_objs, 0, 1] = -coll_vector
        right_rule_out = np.zeros_like(right_rule_in)
        # object moves to right cell
        right_rule_out[idx, 0, 1] = 1
        # remove anything that was present in the input
        right_rule_out -= np.clip(right_rule_in, 0, 1)
        right_rule = np.stack((right_rule_in, right_rule_out), axis=0)

        down_rule_in = np.zeros((n_objs + (n_objs * 4), 2, 1), dtype=np.int8)
        # object must be present in upper cell
        down_rule_in[idx, 1] = 1
        # downward force
        down_rule_in[n_objs + (idx * 4) + 3, 1] = 1
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
    jax.lax.cond(
        changed,
        lambda _: jax.debug.print('rule {rule_name} applied: {changed}', rule_name=rule_name, changed=changed),
        lambda _: None,
        None
    )    # if out.sum() != 0:
    #     print('lvl')
    #     print(lvl[0,2])
    #     print('out')
    #     print(out[0,2])
    #     breakpoint()
    lvl += out
    lvl = lvl.astype(bool)

    return lvl, changed, False

    

@flax.struct.dataclass
class PSState:
    multihot_level: np.ndarray
    win: bool

class PSEnv:
    def __init__(self, tree: PSGame, jit: bool = True):
        self.jit = jit
        self.title = tree.prelude.title
        self.levels = tree.levels
        self.require_player_movement = tree.prelude.require_player_movement
        obj_legend, meta_tiles, joint_tiles = process_legend(tree.legend)
        collision_layers = expand_collision_layers(tree.collision_layers, meta_tiles)
        self.obj_to_idxs, coll_masks = assign_vecs_to_objs(collision_layers)
        self.n_objs = len(self.obj_to_idxs)
        coll_mat = np.einsum('ij,ik->jk', coll_masks, coll_masks, dtype=bool)
        self.rule_fn = gen_rule_fn(self.obj_to_idxs, coll_mat, tree.rules, meta_tiles, jit=self.jit)
        self.check_win = gen_check_win(tree.win_conditions, self.obj_to_idxs, meta_tiles, jit=self.jit)
        if 'player' in self.obj_to_idxs:
            self.player_idxs = [self.obj_to_idxs['player']]
        else:
            player_objs = disambiguate_meta('player', meta_tiles, self.obj_to_idxs)
            self.player_idxs = [self.obj_to_idxs[p] for p in player_objs]
        self.player_idxs = np.array(self.player_idxs)
        print(f'player_idxs: {self.player_idxs}')
        sprite_stack = []
        for obj_name in self.obj_to_idxs:
            obj = tree.objects[obj_name]
            if obj.sprite is not None:
                im = render_sprite(obj.colors, obj.sprite)
            else:
                assert len(obj.colors) == 1
                im = render_solid_color(obj.colors[0])
            sprite_stack.append(im)
        self.sprite_stack = np.array(sprite_stack)
        n_objs = len(self.obj_to_idxs)
        char_legend = {v: k for k, v in obj_legend.items()}
        # Generate vectors to detect atomic objects
        self.obj_vecs = np.eye(n_objs, dtype=bool)
        joint_obj_vecs = []
        self.chars_to_idxs = {obj_legend[k]: v for k, v in self.obj_to_idxs.items() if k in obj_legend}

        for jo, subobjects in joint_tiles.items():
            vec = np.zeros(n_objs, dtype=bool)
            subobjects = expand_meta_tiles(subobjects, self.obj_to_idxs, meta_tiles)
            for so in subobjects:
                vec += self.obj_vecs[self.obj_to_idxs[so]]
            assert jo not in self.chars_to_idxs
            self.chars_to_idxs[jo] = len(self.chars_to_idxs)
            self.obj_vecs = np.concatenate((self.obj_vecs, vec[None]), axis=0)

        if self.jit:
            self.step = jax.jit(self._step)
            self.apply_player_force = jax.jit(self._apply_player_force)
        else:
            self.step = self._step
            self.apply_player_force = self._apply_player_force
        self.joint_tiles = joint_tiles

    def char_level_to_multihot(self, level):
        int_level = np.vectorize(lambda x: self.chars_to_idxs[x])(level)
        multihot_level = self.obj_vecs[int_level]
        bg_idx = self.obj_to_idxs['background']
        multihot_level = rearrange(multihot_level, "h w c -> c h w")
        multihot_level[bg_idx] = 1
        multihot_level = multihot_level.astype(bool)
        return multihot_level

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

    def reset(self, lvl_i):
        lvl = self.get_level(lvl_i)
        return PSState(
            multihot_level=lvl,
            win = jnp.array(False),
        )

    # @partial(jax.jit, static_argnums=(0))
    def _apply_player_force(self, action, state: PSState):
        multihot_level = state.multihot_level
        # add a dummy object at the front
        force_map = jnp.zeros((4 * (multihot_level.shape[0] + 1), *multihot_level.shape[1:]), dtype=bool)
        
        def place_force(force_map, action):
            action = jnp.array([0, 1, 2, 3])[action]
            player_int_mask = (self.player_idxs[...,None,None] + 1) * multihot_level[self.player_idxs]
            # turn the int mask into coords, by flattening it, and appending it with xy coords
            xy_coords = jnp.indices(force_map.shape[1:])
            xy_coords = xy_coords[:, None].repeat(len(self.player_idxs), axis=1)
            player_int_mask = player_int_mask * 4 + action
            player_int_mask = jnp.concatenate((player_int_mask[None], xy_coords), axis=0)
            player_coords = player_int_mask.reshape(3, -1).T
            force_map = force_map.at[tuple(player_coords.T)].set(1)
            force_map_sum = force_map.sum()
            # jax.debug.print('force_map: {force_map}', force_map=force_map)
            # jax.debug.print('force map sum: {force_map_sum}', force_map_sum=force_map_sum)
            return force_map

        force_map = jax.lax.cond(
            action < 4, 
            place_force,
            lambda force_map, action: force_map,
            force_map, action
        )
        force_map = force_map[4:]

        lvl = jnp.concatenate((multihot_level, force_map), axis=0)

        return lvl

    # @partial(jax.jit, static_argnums=(0))
    def _step(self, action, state: PSState):
        init_lvl = state.multihot_level.copy()
        lvl = self.apply_player_force(action, state)
        
        def cond_fun(loop_state):
            lvl, lvl_changed, n_apps, cancelled = loop_state
            return jax.numpy.logical_and(lvl_changed, n_apps < 100) & ~cancelled
            
        def body_fun(carry):
            lvl, lvl_changed_last, n_apps, cancelled = carry
            new_lvl, lvl_changed, cancelled = self.rule_fn(lvl)
            return (new_lvl, lvl_changed, n_apps + 1, cancelled)
        
        # Initial state for the while loop
        init_state = (lvl, True, 0, False)

        if self.jit:
            final_lvl, _, _, cancelled = jax.lax.while_loop(cond_fun, body_fun, init_state)
        
        else:
            while cond_fun(init_state):
                init_state = body_fun(init_state)
            final_lvl, _, _, cancelled = init_state

        
        final_lvl = jax.lax.select(
            ((not self.require_player_movement) or player_has_moved(init_lvl, final_lvl, self.obj_to_idxs)) & ~cancelled,
            final_lvl,
            lvl,
        )
        multihot_level = final_lvl[:self.n_objs]
        win = self.check_win(multihot_level)
        state = PSState(
            multihot_level=multihot_level,
            win=win,
        )
        return state

    def get_level(self, level_idx):
        level = self.levels[level_idx][0]
        # Convert the level to a multihot representation and render it
        multihot_level = self.char_level_to_multihot(level)
        return multihot_level

def multihot_to_desc(multihot_level, obj_to_idxs):
    """Converts a multihot array to a 2D list of descriptions.
    
    Args:
        multihot_level: A multihot array of shape [n_objects + n_forces, height, width].
        obj_to_idxs: Dictionary mapping object names to their indices.
    
    Returns:
        A 2D list where each cell contains a string describing all objects and forces present.
    """
    n_objs = len(obj_to_idxs)
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
                    force_names = ["left", "down", "right", "up"]
                    forces = []
                    for f_idx, force_name in enumerate(force_names):
                        force_channel = n_objs + (obj_idx * 4) + f_idx
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
