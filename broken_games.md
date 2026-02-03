# Broken Games

## pusH

Problem: Score is unaligned on the last level. It's 0 in jax but a big number in JS, maybe because in the latter, since goal is absent, we imagine it is off the edge of the board or something? 

## test_circuit

(from Circuit_Breaker)
Problem: In JAX, only one of the edges underneath the crate ends up having force applied. I think this is because `detect_any_objs_in_cell` always returns the first matching object. So after force is applied to the first edge, the second application of the rule does not apply force to the second edge.

## ponies_jumping_synchronously_fixed

Some problems with score/win condition, as yet undetermined.

## Pipe_Navigation

???

# Fixed Games

## Sponge_Game

~~When the player slides over water and hits the sponge, they should stop (in JS) but in JAX, they also push the sponge.~~
Solution: the issue was that we weren't removing forces from the multihot level between repeated calls to `apply_turn`.
