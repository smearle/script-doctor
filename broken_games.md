# Broken Games

## test_easyenigme

(from Easy Enigma)
An object (expanded from a property) appears as `v` in the rule, and is thereby mistaken as a force, leading to an empty list of cell detection functions in this cell.

## test_pitman
(from `Pitman_MZ-700`)
The player spawns weirdly (wrongly?) in JavaScript. But not in the updated web version of PS. So I don't think this bug is on us. Looks like `run_rules_on_level_start` was not having the proper effect in the JS version, so the player was able to move in their first turn from on top of a floating boulder. Would be trivial to edit this level to make it solvable via an otherwise identical solution though at least.

## test_circuit

(from `Circuit_Breaker`)
Problem: In JAX, only one of the edges underneath the crate ends up having force applied. I think this is because `detect_any_objs_in_cell` always returns the first matching object. So after force is applied to the first edge, the second application of the rule does not apply force to the second edge.
So I guess this function instead needs to return the set of all matching objects, then when applying the rule, we iterate through each possible object, attempting to apply the rule to it? Yikes.

## Impasse

~~In level 3, in JAX, the orange blockers morph into the player. This is because when applying the line (`...`) rule, the `upper` is confused with the `player`. But why?~~
The above was fixed hackishly, but ignoring meta-objects recognized in cells that did not belong to recognized lines. But the issue persists when we recognize _multiple_ lines with different sub-objects of the same meta-object type.

What we really need to do is be much more careful about which detected meta-objects we use when projecting a given line kernel. Somewhere I think we are collapsing the `detected_meta_objs` dictionary entries (which have now been modified to be 2D arrays) to a single integer (I think with `max`). This should not be so.


## ponies_jumping_synchronously_fixed

Some problems with score/win condition, as yet undetermined.

## pusH

Problem: Score is unaligned on the last level. It's 0 in jax but a big number in JS, maybe because in the latter, since goal is absent, we imagine it is off the edge of the board or something? 

## GDD301_Game

In JS, the player is able to slide through their own trail when the trail is 2 tiles away from the player without triggering a level restart. This bug is not present in the latest online version of PuzzleScript. We'll need to update the nodejs version

## Pipe_Navigation

???

## test_michaelbay

From `Michael Bay's Legend of Zelda`. `cancel` triggers inside a hypothetical call to `again`, but it cancels the prior turn (the one in which `again` was triggered), when in fact it should only cancel the hypothetical current turn, resulting in `again` effectively not being applied (but the initial turn being applied).
Attempting solution: `accept_lvl_change` inside `step_env` no longer cares about the value of `cancelled` received from `tick_fn`, since inside this latter tick function, `cancelled` will already prevent the offending call to `apply_turn`, so we should be set in general.

# Games broken because of changes to PuzzleScript

## Dogs, dogs and additional dogs

The black square (bone?) does not appear in JAX nor in current PuzzleScript on web, but it does appear in nodejs.

# Fixed Games

## Sponge_Game

~~When the player slides over water and hits the sponge, they should stop (in JS) but in JAX, they also push the sponge.~~
Solution: the issue was that we weren't removing forces from the multihot level between repeated calls to `apply_turn`.

## Depth-First_Maze
~~Our maze is not truly random. Apparently this is because we apply all matching kernels at once. I guess we would instead need to iterate through all individual matches and apply them one by one (splitting the RNG in the process). Or wait, couldn't we just split the RNG n_matches many times then pass it to all the parallel operations?~~
Actually, we needed to apply just one random possible rule occurrence from among the `random` group. The maze is properly random now.
