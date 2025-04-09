from dataclasses import dataclass
from functools import partial
import os
import pickle
import re
import time
from typing import List, Dict, Optional, Any, Set, Tuple
import copy
import random

import cv2
from einops import rearrange
import flax
import jax
import jax.numpy as jnp
from lark import Token, Transformer, Tree
import numpy as np
from PIL import Image

from parse_lark import TEST_GAMES
from ps_game import LegendEntry, PSGame, PSObject, Prelude, Rule, RuleBlock, WinCondition

class GenPSTree(Transformer):
    """
    Reduces the parse tree to a minimal functional version of the grammar.
    """
    def object_data(self, items):
        name_line = items[0]
        name = str(name_line.children[0].children[0]).lower()
        colors = []
        color_line = items[1]
        legend_key = str(name_line.children[1].children[0]) if len(name_line.children) > 1 else None
        for color in color_line.children:
            colors.append(str(color.children[0]))
        if len(items) < 3:
            sprite = None
        else:
            sprite = np.array([c for c in items[2].children])

        return PSObject(
            name=name,
            colors=colors,
            sprite=sprite,
        )

    def rule_content(self, items):
        return ' '.join(items)

    def cell_border(self, items):
        return '|'

    def rule_block_once(self, items):
        # One RuleBlock with possible nesting (right?)
        # assert len(items) == 1
        return RuleBlock(looping=False, rules=items)

    def rule_block_loop(self, items):
        return RuleBlock(looping=True, rules=items)

    def rule_part(self, items):
        cells = []
        cell = []
        for item in items:
            if item != '|':
                cell.append(item)
            else:
                cells.append(cell)
                cell = []
        cells.append(cell)
        return cells

    def legend_data(self, items):
        key = items[0].lower()
        # The first entry in this legend key's mapping is just an object name
        assert len(items[1].children) == 1
        obj_names = [str(items[1].children[0]).lower()]
        # Every subsequent item is preceded by an AND or OR legend operator.
        # They should all be the same
        operator = None
        for it in items[2:]:
            obj_name = str(it.children[1].children[0]).lower()
            obj_names.append(obj_name)
            new_op = str(it.children[0])
            if operator is not None:
                assert operator == new_op
            else:
                operator = new_op

        return LegendEntry(key=key, obj_names=obj_names, operator=operator)

    def prefix(self, items):
        out = str(items[0])
        if out.lower().startswith('sfx'):
            return None
        else:
            return out.lower()

    def rule_data(self, items):
        prefixes = []
        l = []
        for i, item in enumerate(items):
            if item == 'late':
                prefixes.append('late')
                continue
            if isinstance(item, Token) and item.type == 'RULE':
                breakpoint()
            if isinstance(item, Token) and item.type == 'THEN':
                r = items[i+1:]
                break
            l.append(item)
        l = [it for it in l if it is not None]
        r = [it for it in r if it is not None]
        rule = Rule(
            prefixes=prefixes,
            left_patterns = l,
            right_patterns = r,
        )
        return rule
    
    def return_items_lst(self, items):
        return items

    def objects_section(self, items: List[PSObject]):
        return {ik.name: ik for ik in items}

    def legend_section(self, items: List[LegendEntry]):
        return {it.key: it for it in items}
    
    def layer_data(self, items):
        obj_names = [str(it.children[0]).lower() for it in items]
        return obj_names

    def condition_data(self, items):
        quant = str(items[0])
        src_obj = str(items[1].children[0]).lower()
        trg_obj = None
        if len(items) > 2:
            trg_obj = str(items[3].children[0]).lower()
        return WinCondition(quantifier=quant, src_obj=src_obj, trg_obj=trg_obj)

    levels_section = collisionlayers_section = rule_block = rules_section \
        = winconditions_section = return_items_lst

    def level_data(self, items):
        return np.vectorize(lambda x: str(x).lower())(items)

    def ps_game(self, items):
        prelude_items = items[0].children
        title, author, homepage = None, None, None
        flickscreen = False
        verbose_logging = False
        require_player_movement = False
        for pi in prelude_items:
            pi_items = pi.children
            keyword = pi_items[0].lower()
            value = None
            if len(pi_items) > 1:
                value = str(pi_items[1])
            if keyword == 'title':
                title = value
            elif keyword == 'author':
                author = value
            elif keyword == 'homepage':
                homepage = value
            elif keyword == 'flickscreen':
                flickscreen = True
            elif keyword == 'verbose_logging':
                verbose_logging = value
            elif keyword == 'require_player_movement':
                require_player_movement = True
        # assert title is not None
        return PSGame(
            prelude=Prelude(
                title=title,
                author=author,
                homepage=homepage,
                flickscreen=flickscreen,
                verbose_logging=verbose_logging,
                require_player_movement=require_player_movement,
            ),
            objects = items[1],
            legend=items[2],
            collision_layers=items[3],
            rules=items[4],
            win_conditions=items[5],
            levels=items[6],
        )

data_dir = 'data'


