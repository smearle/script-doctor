"""Quick comparison of compile times: standard env vs switch-based env."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
from timeit import default_timer as timer

from puzzlejax.utils import init_ps_lark_parser, get_tree_from_txt
from puzzlejax.env import PuzzleJaxEnv, PJParams
from puzzlejax.env_switch import PuzzleJaxEnvSwitch

GAME = "notsnake"

parser = init_ps_lark_parser()
tree, _, _ = get_tree_from_txt(parser, GAME, test_env_init=False)

results = {}

for label, EnvClass in [("standard", PuzzleJaxEnv), ("switch", PuzzleJaxEnvSwitch)]:
    print(f"\n{'='*60}")
    print(f"Testing {label} env ({EnvClass.__name__}) on {GAME}")
    print(f"{'='*60}")

    jax.clear_caches()

    env = EnvClass(tree, jit=True, level_i=0, max_steps=100, print_score=False, debug=False, vmap=True)
    lvl = env.get_level(0)
    params = PJParams(level=lvl)

    rng = jax.random.PRNGKey(42)
    reset_rng = jax.random.split(rng, 1)

    def _env_step(carry, unused):
        env_state, rng = carry
        rng, _rng = jax.random.split(rng)
        rand_act = jax.random.randint(_rng, (1,), 0, env.action_space.n)
        rng_step = jax.random.split(_rng, 1)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, rand_act, params)
        return (env_state, rng), None

    _step_jit = jax.jit(_env_step)

    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, params)
    carry = (env_state, rng)

    # 1st step: compile + execute
    t0 = timer()
    carry, _ = _step_jit(carry, None)
    carry[0].multihot_level.block_until_ready()
    compile_time = timer() - t0

    # 2nd step: execute only
    t0 = timer()
    carry, _ = _step_jit(carry, None)
    carry[0].multihot_level.block_until_ready()
    exec_time = timer() - t0

    est_compile = compile_time - exec_time
    print(f"  1st step (compile + exec): {compile_time:.2f}s")
    print(f"  2nd step (exec only):      {exec_time:.4f}s")
    print(f"  Estimated compile time:    {est_compile:.2f}s")

    results[label] = est_compile

print(f"\n{'='*60}")
print(f"SUMMARY for {GAME}")
print(f"{'='*60}")
for label, t in results.items():
    print(f"  {label:10s}: {t:.2f}s")
if "standard" in results and "switch" in results:
    ratio = results["standard"] / results["switch"]
    print(f"  speedup (standard/switch): {ratio:.2f}x")
