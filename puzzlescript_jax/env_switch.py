"""
Switch-based variant of PuzzleJaxEnv.

Instead of unrolling all rule functions via Python for-loops (causing JAX to
trace each one inline into a single enormous HLO graph), this version uses
``jax.lax.switch`` to dispatch rule functions at runtime.  JAX then traces each
rule function exactly *once* (as a branch of the switch), dramatically reducing
compile time for games with many rules, rotations, or meta-object expansions.

Usage:
    from puzzlejax.env_switch import PuzzleJaxEnvSwitch
    env = PuzzleJaxEnvSwitch(tree, jit=True, ...)
"""

from functools import partial
import logging
from typing import List

import chex
import jax
import jax.numpy as jnp
import numpy as np

from puzzlescript_jax.env import (
    DEBUG,
    MAX_LOOPS,
    InvalidObjectError,
    LoopRuleBlockState,
    LoopRuleGroupState,
    LoopRuleState,
    PuzzleJaxEnv,
    RuleBlockState,
    RuleGroupState,
    RuleState,
    _expand_or_meta_rules,
)
from puzzlescript_jax.env_utils import multihot_to_desc

logger = logging.getLogger(__name__)


class PuzzleJaxEnvSwitch(PuzzleJaxEnv):
    """A drop-in replacement for :class:`PuzzleJaxEnv` that uses
    ``jax.lax.switch`` for rule dispatch instead of inlined Python for-loops.

    Construction arguments are identical to :class:`PuzzleJaxEnv`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-generate tick_fn OUTSIDE JIT to avoid tracer leaks.
        # The parent's reset() would call gen_tick_fn inside JIT, creating
        # jnp arrays that leak from the trace scope via closures.
        lvl = self.get_level(self.level_i)
        self.tick_fn = self.gen_tick_fn(lvl.shape)

    def reset(self, rng, params):
        """Override reset to skip gen_tick_fn (pre-generated in __init__)."""
        from puzzlescript_jax.env import PJState, PJStateMultiLevel, PSObs, PRINT_SCORE
        requested_level_i = jnp.asarray(params.level_i, dtype=jnp.int32)
        if not self._is_multi_level:
            sampled_level_i = requested_level_i
            lvl = params.level
            level_height = None
            level_width = None
            valid_mask = None
        else:
            rng, level_rng = jax.random.split(rng)
            sampled_level_i = jax.lax.cond(
                requested_level_i < 0,
                lambda key: jax.random.randint(key, shape=(), minval=0, maxval=len(self._compiled_levels), dtype=jnp.int32),
                lambda _key: requested_level_i,
                level_rng,
            )
            lvl = jax.lax.cond(
                requested_level_i < 0,
                lambda idx: jax.lax.switch(
                    idx,
                    tuple(lambda _, level=level: level for level in self._compiled_levels),
                    None,
                ),
                lambda _idx: params.level,
                sampled_level_i,
            )
            level_height = self._level_heights[sampled_level_i]
            level_width = self._level_widths[sampled_level_i]
            valid_mask = self._make_valid_mask(level_height, level_width)
            lvl = jnp.where(valid_mask[None], lvl, False)
        # NOTE: tick_fn already generated in __init__, do NOT regenerate here.
        win, score, init_heuristic = self.check_win(lvl)
        if PRINT_SCORE:
            jax.debug.print(
                'heuristic: {heuristic}, score: {score}, win: {win}',
                heuristic=init_heuristic, score=score, win=win,
            )
        state = PJState(
            multihot_level=lvl,
            level_i=sampled_level_i,
            win=jnp.array(False),
            score=jnp.array(0, dtype=jnp.int32),
            heuristic=init_heuristic,
            restart=jnp.array(False),
            step_i=jnp.array(0, dtype=jnp.int32),
            init_heuristic=init_heuristic,
            prev_heuristic=init_heuristic,
            rng=rng,
            view_bounds=self._get_default_view_bounds(lvl.shape[1:]),
        )
        if self._is_multi_level:
            state = PJStateMultiLevel(
                **state,
                level_height=level_height,
                level_width=level_width,
                valid_mask=valid_mask,
            )
        if self.tree.prelude.run_rules_on_level_start:
            lvl = self.apply_player_force(-1, state)
            lvl, _, _, _, _, _, rng = self.tick_fn(rng, lvl)
            lvl = lvl[:self.n_objs]
            state = state.replace(multihot_level=lvl, rng=rng)
        state = state.replace(view_bounds=self._compute_view_bounds(state.multihot_level, state.view_bounds))
        obs = self.get_obs(state)
        return obs, state

    # ------------------------------------------------------------------
    # Atomic rule application – switch-based
    # ------------------------------------------------------------------

    def apply_rule_fn(
        self,
        loop_rule_state: LoopRuleState,
        all_rule_fns,
        n_prior_rules_arr,
    ):
        """Apply an atomic rule once, dispatching via ``jax.lax.switch``."""

        prev_loop_rule_state = loop_rule_state
        rule_i = loop_rule_state.rule_i
        grp_i = loop_rule_state.grp_i
        block_i = loop_rule_state.block_i
        rng, lvl = prev_loop_rule_state.rng, prev_loop_rule_state.lvl

        global_rule_idx = n_prior_rules_arr[block_i, grp_i] + rule_i

        if self.jit:
            rule_state: RuleState = jax.lax.switch(
                global_rule_idx, all_rule_fns, rng, lvl,
            )
        else:
            rule_state = all_rule_fns[global_rule_idx](rng, lvl)

        rule_had_effect = jnp.any(rule_state.lvl != lvl)
        applied = rule_had_effect & jnp.all(rng == rule_state.rng)
        again = rule_state.again | prev_loop_rule_state.again
        restart = rule_state.restart | prev_loop_rule_state.restart
        cancelled = rule_state.cancelled | prev_loop_rule_state.cancelled
        win = rule_state.win | prev_loop_rule_state.win

        if DEBUG:
            jax.debug.print(
                "      apply_rule_fn: rule {rule_i} had effect: {rule_had_effect}. again: {again}",
                rule_i=rule_i,
                rule_had_effect=rule_had_effect,
                again=again,
            )

        return LoopRuleState(
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

    # ------------------------------------------------------------------
    # Loop a single rule until convergence – switch-based
    # ------------------------------------------------------------------

    def loop_rule_fn(
        self,
        rule_group_state: RuleGroupState,
        all_rule_fns,
        n_prior_rules_arr,
    ):
        _apply = partial(
            self.apply_rule_fn,
            all_rule_fns=all_rule_fns,
            n_prior_rules_arr=n_prior_rules_arr,
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

        if self.jit:
            loop_rule_state = jax.lax.while_loop(
                cond_fun=lambda s: s.applied & ~s.cancelled & ~s.restart & (s.app_i < MAX_LOOPS),
                body_fun=_apply,
                init_val=loop_rule_state,
            )
        else:
            while (
                loop_rule_state.applied
                and not loop_rule_state.cancelled
                and not loop_rule_state.restart
            ):
                loop_rule_state = _apply(loop_rule_state)

        rule_applied = loop_rule_state.app_i > 1
        grp_applied = rule_applied | loop_rule_state.applied

        return RuleGroupState(
            lvl=loop_rule_state.lvl,
            applied=grp_applied,
            cancelled=rule_group_state.cancelled | loop_rule_state.cancelled,
            restart=rule_group_state.restart | loop_rule_state.restart,
            again=rule_group_state.again | loop_rule_state.again,
            win=rule_group_state.win | loop_rule_state.win,
            rng=loop_rule_state.rng,
            rule_i=rule_group_state.rule_i + 1,
            grp_i=rule_group_state.grp_i,
            block_i=rule_group_state.block_i,
        )

    # ------------------------------------------------------------------
    # Random group application – switch-based
    # ------------------------------------------------------------------

    def apply_random_rule_grp(
        self,
        rule_group_state: RuleGroupState,
        all_rule_fns,
        all_count_fns,
        all_apply_one_fns,
        n_prior_rules_arr,
        n_rules_per_grp_arr,
    ):
        """Apply exactly one random match across all rules in the group."""
        init_lvl = rule_group_state.lvl
        rng = rule_group_state.rng
        block_i = rule_group_state.block_i
        grp_i = rule_group_state.grp_i

        n_rules = n_rules_per_grp_arr[block_i, grp_i]
        prior = n_prior_rules_arr[block_i, grp_i]

        # Count matches per rule in the group via switch
        if self.jit:
            def count_one(idx):
                return jax.lax.switch(prior + idx, all_count_fns, init_lvl)

            max_rules = self._max_rules_per_grp
            totals = jax.vmap(count_one)(jnp.arange(max_rules))
            # Zero out entries beyond n_rules
            mask = jnp.arange(max_rules) < n_rules
            totals = jnp.where(mask, totals, 0)
        else:
            totals = jnp.array(
                [all_count_fns[prior + i](init_lvl) for i in range(int(n_rules))]
            )

        total_matches = jnp.sum(totals)

        def no_match():
            return RuleGroupState(
                lvl=init_lvl,
                applied=False,
                cancelled=rule_group_state.cancelled,
                restart=rule_group_state.restart,
                again=rule_group_state.again,
                win=rule_group_state.win,
                rng=rng,
                rule_i=rule_group_state.rule_i,
                grp_i=grp_i,
                block_i=block_i,
            )

        def do_apply():
            rng_select, rng_apply = jax.random.split(rng)
            r = jax.random.randint(rng_select, (1,), 0, total_matches)[0]
            cumsum = jnp.cumsum(totals)
            rule_idx_in_grp = jnp.argmax(r < cumsum)
            global_idx = prior + rule_idx_in_grp

            if self.jit:
                rule_state = jax.lax.switch(
                    global_idx, all_apply_one_fns, rng_apply, init_lvl,
                )
            else:
                rule_state = all_apply_one_fns[int(global_idx)](rng_apply, init_lvl)

            return RuleGroupState(
                lvl=rule_state.lvl,
                applied=jnp.any(rule_state.lvl != init_lvl),
                cancelled=rule_group_state.cancelled | rule_state.cancelled,
                restart=rule_group_state.restart | rule_state.restart,
                again=rule_group_state.again | rule_state.again,
                win=rule_group_state.win | rule_state.win,
                rng=rule_state.rng,
                rule_i=rule_group_state.rule_i,
                grp_i=grp_i,
                block_i=block_i,
            )

        return jax.lax.cond(total_matches > 0, do_apply, no_match)

    # ------------------------------------------------------------------
    # Apply a rule group (iterate over rules) – switch-based
    # ------------------------------------------------------------------

    def apply_rule_grp(
        self,
        loop_group_state: LoopRuleGroupState,
        all_rule_fns,
        all_count_fns,
        all_apply_one_fns,
        n_prior_rules_arr,
        n_rules_per_grp_arr,
        grps_are_random_arr,
    ):
        block_i = loop_group_state.block_i
        grp_i = loop_group_state.grp_i
        init_lvl = loop_group_state.lvl
        is_random = grps_are_random_arr[block_i, grp_i]

        _loop_rule_fn = partial(
            self.loop_rule_fn,
            all_rule_fns=all_rule_fns,
            n_prior_rules_arr=n_prior_rules_arr,
        )

        n_rules_in_grp = n_rules_per_grp_arr[block_i, grp_i]

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

        def _apply_random(rgs):
            return self.apply_random_rule_grp(
                rgs,
                all_rule_fns=all_rule_fns,
                all_count_fns=all_count_fns,
                all_apply_one_fns=all_apply_one_fns,
                n_prior_rules_arr=n_prior_rules_arr,
                n_rules_per_grp_arr=n_rules_per_grp_arr,
            )

        def _apply_normal(rgs):
            if self.jit:
                rgs = jax.lax.while_loop(
                    cond_fun=lambda s: (s.rule_i < n_rules_in_grp) & ~s.cancelled & ~s.restart,
                    body_fun=_loop_rule_fn,
                    init_val=rgs,
                )
            else:
                while (
                    rgs.rule_i < n_rules_in_grp
                    and not rgs.cancelled
                    and not rgs.restart
                ):
                    rgs = _loop_rule_fn(rgs)
            return rgs

        if self.jit:
            rule_group_state = jax.lax.cond(
                is_random,
                _apply_random,
                _apply_normal,
                rule_group_state,
            )
        else:
            if is_random:
                rule_group_state = _apply_random(rule_group_state)
            else:
                rule_group_state = _apply_normal(rule_group_state)

        lvl = rule_group_state.lvl

        loop_group_state = LoopRuleGroupState(
            lvl=lvl,
            applied=jnp.any(lvl != init_lvl),
            cancelled=rule_group_state.cancelled,
            restart=rule_group_state.restart,
            again=loop_group_state.again | rule_group_state.again,
            win=loop_group_state.win | rule_group_state.win,
            rng=rule_group_state.rng,
            app_i=loop_group_state.app_i + 1,
            grp_i=loop_group_state.grp_i,
            block_i=loop_group_state.block_i,
        )
        return loop_group_state

    # ------------------------------------------------------------------
    # Loop a rule group until convergence – switch-based
    # ------------------------------------------------------------------

    def loop_rule_grp(
        self,
        rule_block_state: RuleBlockState,
        all_rule_fns,
        all_count_fns,
        all_apply_one_fns,
        n_prior_rules_arr,
        n_rules_per_grp_arr,
        grps_are_random_arr,
    ):
        lvl = rule_block_state.lvl
        grp_applied_prev = rule_block_state.applied
        cancelled = rule_block_state.cancelled
        restart = rule_block_state.restart
        prev_again = rule_block_state.again
        rng = rule_block_state.rng
        win = rule_block_state.win
        grp_i = rule_block_state.grp_i
        block_i = rule_block_state.block_i

        is_random = grps_are_random_arr[block_i, grp_i]

        _apply_rule_grp = partial(
            self.apply_rule_grp,
            all_rule_fns=all_rule_fns,
            all_count_fns=all_count_fns,
            all_apply_one_fns=all_apply_one_fns,
            n_prior_rules_arr=n_prior_rules_arr,
            n_rules_per_grp_arr=n_rules_per_grp_arr,
            grps_are_random_arr=grps_are_random_arr,
        )

        loop_group_state = LoopRuleGroupState(
            lvl=lvl,
            applied=True,
            cancelled=cancelled,
            restart=restart,
            again=prev_again,
            win=win,
            rng=rng,
            app_i=0,
            grp_i=grp_i,
            block_i=block_i,
        )

        if self.jit:
            # Random groups run once; non-random groups loop to convergence
            loop_group_state = jax.lax.cond(
                is_random,
                lambda s: _apply_rule_grp(s),
                lambda s: jax.lax.while_loop(
                    cond_fun=lambda x: x.applied & ~x.cancelled & ~x.restart & (x.app_i < MAX_LOOPS),
                    body_fun=_apply_rule_grp,
                    init_val=s,
                ),
                loop_group_state,
            )
        else:
            if is_random:
                loop_group_state = _apply_rule_grp(loop_group_state)
            else:
                while (
                    loop_group_state.applied
                    and not loop_group_state.cancelled
                    and not loop_group_state.restart
                    and loop_group_state.app_i < MAX_LOOPS
                ):
                    loop_group_state = _apply_rule_grp(loop_group_state)

        grp_applied = jnp.logical_or(
            loop_group_state.app_i > 1,
            jnp.logical_and(is_random, loop_group_state.applied),
        )
        block_applied = grp_applied_prev | grp_applied

        return RuleBlockState(
            lvl=loop_group_state.lvl,
            applied=block_applied,
            cancelled=cancelled | loop_group_state.cancelled,
            restart=restart | loop_group_state.restart,
            again=prev_again | loop_group_state.again,
            win=loop_group_state.win | rule_block_state.win,
            rng=loop_group_state.rng,
            grp_i=grp_i + 1,
            block_i=block_i,
        )

    # ------------------------------------------------------------------
    # Apply a rule block (iterate over groups) – switch-based
    # ------------------------------------------------------------------

    def apply_rule_block(
        self,
        loop_rule_block_state: LoopRuleBlockState,
        all_rule_fns,
        all_count_fns,
        all_apply_one_fns,
        n_prior_rules_arr,
        n_rules_per_grp_arr,
        n_grps_per_block_arr,
        grps_are_random_arr,
    ):
        lvl = loop_rule_block_state.lvl
        block_app_i = loop_rule_block_state.block_app_i + 1
        cancelled = loop_rule_block_state.cancelled
        restart = loop_rule_block_state.restart
        prev_again = loop_rule_block_state.again
        win = loop_rule_block_state.win
        rng = loop_rule_block_state.rng
        block_i = loop_rule_block_state.block_i

        n_rule_grps = n_grps_per_block_arr[block_i]

        _loop_rule_grp = partial(
            self.loop_rule_grp,
            all_rule_fns=all_rule_fns,
            all_count_fns=all_count_fns,
            all_apply_one_fns=all_apply_one_fns,
            n_prior_rules_arr=n_prior_rules_arr,
            n_rules_per_grp_arr=n_rules_per_grp_arr,
            grps_are_random_arr=grps_are_random_arr,
        )

        rule_block_state = RuleBlockState(
            lvl=lvl,
            applied=False,
            cancelled=cancelled,
            restart=restart,
            again=prev_again,
            win=False,
            rng=rng,
            grp_i=0,
            block_i=block_i,
        )

        if self.jit:
            rule_block_state = jax.lax.while_loop(
                cond_fun=lambda x: (x.grp_i < n_rule_grps) & ~x.cancelled & ~x.restart,
                body_fun=_loop_rule_grp,
                init_val=rule_block_state,
            )
        else:
            while (
                rule_block_state.grp_i < n_rule_grps
                and not rule_block_state.cancelled
                and not rule_block_state.restart
            ):
                rule_block_state = _loop_rule_grp(rule_block_state)

        lvl = rule_block_state.lvl
        block_applied = rule_block_state.applied
        again = prev_again | rule_block_state.again

        return LoopRuleBlockState(
            lvl=lvl,
            applied=block_applied,
            block_app_i=block_app_i,
            cancelled=rule_block_state.cancelled,
            restart=rule_block_state.restart,
            again=again,
            win=rule_block_state.win,
            rng=rule_block_state.rng,
            block_i=block_i,
            app_i=loop_rule_block_state.app_i + 1,
        )

    # ------------------------------------------------------------------
    # Loop a rule block until convergence – switch-based
    # ------------------------------------------------------------------

    def loop_rule_block(
        self,
        carry,
        all_rule_fns,
        all_count_fns,
        all_apply_one_fns,
        n_prior_rules_arr,
        n_rules_per_grp_arr,
        n_grps_per_block_arr,
        blocks_are_looping_lst,
        grps_are_random_arr,
    ):
        lvl, applied, prev_again, prev_cancelled, prev_restart, prev_win, rng, block_i = carry
        looping = blocks_are_looping_lst[block_i]

        _apply_rule_block = partial(
            self.apply_rule_block,
            all_rule_fns=all_rule_fns,
            all_count_fns=all_count_fns,
            all_apply_one_fns=all_apply_one_fns,
            n_prior_rules_arr=n_prior_rules_arr,
            n_rules_per_grp_arr=n_rules_per_grp_arr,
            n_grps_per_block_arr=n_grps_per_block_arr,
            grps_are_random_arr=grps_are_random_arr,
        )

        loop_block_state = LoopRuleBlockState(
            lvl=lvl,
            applied=True,
            block_app_i=0,
            cancelled=prev_cancelled,
            restart=prev_restart,
            again=prev_again,
            win=prev_win,
            rng=rng,
            block_i=block_i,
            app_i=0,
        )

        if self.jit:
            loop_block_state = jax.lax.cond(
                looping,
                lambda s: jax.lax.while_loop(
                    cond_fun=lambda x: x.applied & ~x.cancelled & ~x.restart & (x.app_i < MAX_LOOPS),
                    body_fun=_apply_rule_block,
                    init_val=s,
                ),
                lambda s: _apply_rule_block(s),
                loop_block_state,
            )
        else:
            if looping:
                while (
                    loop_block_state.applied
                    and not loop_block_state.cancelled
                    and not loop_block_state.restart
                ):
                    loop_block_state = _apply_rule_block(loop_block_state)
            else:
                loop_block_state = _apply_rule_block(loop_block_state)

        block_applied = loop_block_state.block_app_i > 1
        applied = applied | block_applied
        again = prev_again | loop_block_state.again
        restart = prev_restart | loop_block_state.restart
        cancelled = prev_cancelled | loop_block_state.cancelled
        win = prev_win | loop_block_state.win

        return (
            loop_block_state.lvl,
            applied,
            again,
            cancelled,
            restart,
            win,
            loop_block_state.rng,
            block_i + 1,
        )

    # ------------------------------------------------------------------
    # gen_tick_fn – builds everything with switch-based dispatch
    # ------------------------------------------------------------------

    def gen_tick_fn(self, lvl_shape):
        rule_blocks = []
        late_rule_grps = []

        for rule_block in self.tree.rules:
            looping = rule_block.looping
            rule_grps = []
            last_subrule_fns_were_late = None

            for rule in rule_block.rules:
                try:
                    expanded_rules = _expand_or_meta_rules(
                        self.meta_objs,
                        rule,
                        properties_single_layer=self.properties_single_layer,
                    )
                    sub_rule_fns = []
                    for expanded_rule in expanded_rules:
                        sub_rule_fns.extend(
                            self.gen_subrules_meta(
                                expanded_rule,
                                rule_name=str(expanded_rule),
                                lvl_shape=lvl_shape,
                            )
                        )
                except InvalidObjectError as e:
                    print(e)
                    continue

                is_random_group = "random" in rule.prefixes
                if "+" in rule.prefixes:
                    if is_random_group:
                        logger.warn(
                            "Ignoring `random` on a `+`-prefixed rule; random is a rule-group modifier and must "
                            "appear on the first rule in the group."
                        )
                        is_random_group = False
                    if last_subrule_fns_were_late is None:
                        logger.warn(
                            "Initial rule has `+` prefix, but no rule precedes it, so ignoring `+` and adding "
                            "this rule as a new rule group."
                        )
                        rule_grps.append((sub_rule_fns, is_random_group))
                    if "late" in rule.prefixes:
                        if last_subrule_fns_were_late:
                            late_rule_grps[-1][0].extend(sub_rule_fns)
                        else:
                            logger.warn(
                                "Attempting to add `late` rule to a non-late rule. Ignoring `+` and creating a "
                                "new `late` rule group."
                            )
                            late_rule_grps.append((sub_rule_fns, is_random_group))
                            last_subrule_fns_were_late = True
                    else:
                        if last_subrule_fns_were_late:
                            logger.warn(
                                "Attempting to add `+` non-late rule to a late rule. Ignoring `+` and creating "
                                "a new non-late rule group."
                            )
                            rule_grps.append((sub_rule_fns, is_random_group))
                            last_subrule_fns_were_late = False
                        else:
                            rule_grps[-1][0].extend(sub_rule_fns)
                elif "late" in rule.prefixes:
                    late_rule_grps.append((sub_rule_fns, is_random_group))
                    last_subrule_fns_were_late = True
                else:
                    rule_grps.append((sub_rule_fns, is_random_group))
                    last_subrule_fns_were_late = False

            rule_blocks.append((looping, rule_grps))

        # Movement rule
        _move_rule_fn = partial(
            self.apply_movement,
            coll_mat=self.coll_mat,
            n_objs=self.n_objs,
            obj_force_masks=self.obj_force_masks,
            jit=self.jit,
        )
        rule_blocks.append((False, [([_move_rule_fn], False)]))
        # Late rules
        rule_blocks.append((False, late_rule_grps))

        # ----- Build flat function lists and index arrays -----
        all_rule_fns: List = []
        for _, rule_grps in rule_blocks:
            for rule_grp, _ in rule_grps:
                all_rule_fns.extend(rule_grp)

        # Build count_matches / apply_one lists for random rules.
        # For non-random rules we still need placeholder functions of the
        # correct signature so that the flat indices line up.
        all_count_fns: List = []
        all_apply_one_fns: List = []
        for _, rule_grps in rule_blocks:
            for rule_grp, is_random in rule_grps:
                for rule_fn in rule_grp:
                    if is_random and hasattr(rule_fn, "count_matches"):
                        all_count_fns.append(rule_fn.count_matches)
                        all_apply_one_fns.append(rule_fn.apply_one)
                    else:
                        # Placeholders – will never be called for non-random groups
                        all_count_fns.append(lambda lvl: jnp.int32(0))
                        all_apply_one_fns.append(all_rule_fns[0])  # dummy, same signature
                        

        max_n_grps = max(len(rg) for _, rg in rule_blocks) if rule_blocks else 1
        n_blocks = len(rule_blocks)

        n_prior_rules_arr = np.zeros((n_blocks, max_n_grps), dtype=np.int32)
        n_rules_per_grp_arr = np.zeros((n_blocks, max_n_grps), dtype=np.int32)
        n_grps_per_block_arr = np.zeros((n_blocks,), dtype=np.int32)
        grps_are_random_arr = np.zeros((n_blocks, max_n_grps), dtype=bool)
        n_rules_counted = 0

        for block_i, (_, rule_grps) in enumerate(rule_blocks):
            n_grps_per_block_arr[block_i] = len(rule_grps)
            for grp_i, (rule_grp, is_random) in enumerate(rule_grps):
                n_prior_rules_arr[block_i, grp_i] = n_rules_counted
                n_rules_per_grp_arr[block_i, grp_i] = len(rule_grp)
                grps_are_random_arr[block_i, grp_i] = is_random
                n_rules_counted += len(rule_grp)

        # Precompute static shape info as Python ints (must not be traced)
        self._max_rules_per_grp = int(n_rules_per_grp_arr.max())

        n_prior_rules_arr = jnp.array(n_prior_rules_arr)
        n_rules_per_grp_arr = jnp.array(n_rules_per_grp_arr)
        n_grps_per_block_arr = jnp.array(n_grps_per_block_arr)
        blocks_are_looping_lst = jnp.array([looping for looping, _ in rule_blocks])
        grps_are_random_arr = jnp.array(grps_are_random_arr)

        n_total_blocks = jnp.int32(n_blocks)

        # ----- The tick function -----
        _loop_rule_block = partial(
            self.loop_rule_block,
            all_rule_fns=all_rule_fns,
            all_count_fns=all_count_fns,
            all_apply_one_fns=all_apply_one_fns,
            n_prior_rules_arr=n_prior_rules_arr,
            n_rules_per_grp_arr=n_rules_per_grp_arr,
            n_grps_per_block_arr=n_grps_per_block_arr,
            blocks_are_looping_lst=blocks_are_looping_lst,
            grps_are_random_arr=grps_are_random_arr,
        )

        def tick_fn(rng, lvl):
            lvl_changed = False
            cancelled = False
            restart = False
            again = False
            lvl = lvl[None]

            if not self.jit:
                if DEBUG:
                    print(
                        "\n"
                        + multihot_to_desc(
                            lvl[0],
                            self.objs_to_idxs,
                            self.n_objs,
                            obj_idxs_to_force_idxs=self.obj_idxs_to_force_idxs,
                        )
                    )

            def apply_turn(carry):
                prev_carry = carry
                init_lvl, _, turn_app_i, cancelled, restart, turn_again, win, rng = carry
                lvl = init_lvl
                turn_app_i += 1
                applied = False

                # Use jax.lax.while_loop over all blocks
                block_i = jnp.int32(0)
                block_carry = (lvl, applied, False, cancelled, restart, win, rng, block_i)

                if self.jit:
                    block_carry = jax.lax.while_loop(
                        cond_fun=lambda x: ~x[3] & ~x[4] & (x[7] < n_total_blocks),
                        body_fun=_loop_rule_block,
                        init_val=block_carry,
                    )
                else:
                    while (
                        not block_carry[3]
                        and not block_carry[4]
                        and block_carry[7] < n_total_blocks
                    ):
                        block_carry = _loop_rule_block(block_carry)

                lvl, applied, block_again, cancelled, restart, win, rng, _ = block_carry

                turn_applied = jnp.any(lvl != init_lvl)
                win_turn, score, heuristic = self.check_win(lvl[0])
                win = win | win_turn
                lvl = lvl.at[:, self.n_objs :].set(0)

                new_carry = (lvl, turn_applied, turn_app_i, cancelled, restart, block_again, win, rng)
                prev_carry = (
                    prev_carry[0],
                    False,
                    prev_carry[2],
                    True,
                    prev_carry[4],
                    prev_carry[5],
                    prev_carry[6],
                    prev_carry[7],
                )
                return jax.lax.cond(
                    cancelled,
                    lambda _: prev_carry,
                    lambda _: new_carry,
                    operand=None,
                )

            turn_applied = True
            turn_app_i = 0
            turn_again = True
            win = False

            carry = (lvl, turn_applied, turn_app_i, cancelled, restart, turn_again, win, rng)
            if not self.jit:
                while turn_again and turn_applied and not win:
                    carry = apply_turn(carry)
                    lvl, turn_applied, turn_app_i, cancelled, restart, turn_again, win, rng = carry
            else:
                carry = jax.lax.while_loop(
                    cond_fun=lambda x: ~x[3] & ~x[4] & (x[5] & x[1]) & (~x[6]),
                    body_fun=apply_turn,
                    init_val=carry,
                )
            lvl, turn_applied, turn_app_i, cancelled, restart, turn_again, win, rng = carry

            return lvl[0], turn_applied, turn_app_i, cancelled, restart, win, rng

        if self.jit:
            tick_fn = jax.jit(tick_fn)
        return tick_fn
