"""Experimental rule pruning for one immutable level, with a JS correctness gate.

This is not a general environment API: the experiment must not introduce objects
from outside the conservative reachable set by replacing levels or editing state.
It leaves the production engine unchanged and reports no extrapolated throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def word_mask(words):
    return sum((int(word) & 0xffffffff) << (32 * i) for i, word in enumerate(words))


def object_reachability(compiled, level_i=0):
    rules = [r for group in compiled['rules'] + compiled['lateRules'] for r in group]
    constraints = []
    for rule in rules:
        needed, produced, alternatives = 0, 0, []
        for row in rule['patterns']:
            for cell in row:
                if not isinstance(cell, dict):
                    continue
                needed |= word_mask(cell['objectsPresent'])
                alternatives.extend(word_mask(a) for a in cell['anyObjectsPresent'])
                replacement = cell.get('replacement')
                if replacement:
                    produced |= word_mask(replacement['objectsSet']) | word_mask(replacement['randomEntityMask'])
        constraints.append((needed, alternatives, produced))
    level = next(l for l in compiled['levels'] if l['type'] == 'level' and l['index'] == level_i)
    stride, reachable = compiled['STRIDE_OBJ'], 0
    for offset in range(0, len(level['objects']), stride):
        reachable |= word_mask(level['objects'][offset:offset + stride])
    initial = reachable
    while True:
        before = reachable
        for needed, alternatives, produced in constraints:
            if needed & ~reachable == 0 and all(a & reachable for a in alternatives):
                reachable |= produced
        if reachable == before:
            break
    possible = sum(needed & ~reachable == 0 and all(a & reachable for a in alternatives)
                   for needed, alternatives, _ in constraints)
    return {
        'initial_object_types': initial.bit_count(), 'reachable_object_types': reachable.bit_count(),
        'compiled_rules': len(rules), 'potentially_reachable_rules': possible,
        'reachable_names': sorted({name.lower() for i, name in enumerate(compiled['idDict']) if reachable & (1 << i)}),
    }


def specialized_env_class(reachable_names):
    from puzzlescript_jax.env import PuzzleJaxEnv
    reachable_names = frozenset(reachable_names)

    class FixedLevelReachableEnv(PuzzleJaxEnv):
        def gen_subrules_meta(self, rule, rule_name, lvl_shape):
            for row in rule.left_kernels:
                for cell in row:
                    words = [str(token).lower() for token in cell]
                    # Ignore cells with a negation; treating all their positive
                    # requirements as optional is conservative. Unknown property
                    # names are also kept. Only impossible concrete objects prune.
                    if 'no' in words:
                        continue
                    if any(word in self.objs_to_idxs and word not in reachable_names for word in words):
                        return []
            return super().gen_subrules_meta(rule, rule_name, lvl_shape)

        def _gen_rule_blocks(self, lvl_shape):
            # Preserve original group assembly (including '+' and random flags)
            # before dropping empty groups. Block loop boundaries stay intact.
            result = []
            for looping, groups in super()._gen_rule_blocks(lvl_shape):
                groups = [(fns, random) for fns, random in groups if fns]
                if groups:
                    result.append((looping, groups))
            return result

    return FixedLevelReachableEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', required=True)
    parser.add_argument('--compiled-json', type=Path, required=True)
    parser.add_argument('--batches', nargs='+', type=int, default=[16, 256, 4096])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import jax
    import numpy as np
    from backends import NodeJSPuzzleScriptBackend
    from puzzlejax.validate_actions import run_test_case
    from puzzlescript_jax.preprocessing import get_tree_from_txt
    from puzzlescript_jax.utils import init_ps_lark_parser
    from scripts.benchmarks.benchmark_cpp_scoring import compile_game
    from scripts.benchmarks.benchmark_paper_throughput import benchmark, save_result

    compiled_text = args.compiled_json.read_text().strip()
    # The closure and validation must use the same actual source/compiler.
    assert hashlib.sha256(compile_game(args.game).encode()).hexdigest() == hashlib.sha256(compiled_text.encode()).hexdigest()
    reachability = object_reachability(json.loads(compiled_text))
    cls = specialized_env_class(reachability['reachable_names'])
    root = Path(__file__).resolve().parents[2]
    result = {'game': args.game, 'level': 0, 'experiment': 'fixed-level reachability pruning',
              'scope': 'Immutable level-zero parameters and ordinary engine transitions only; no external state edits',
              'devices': [d.device_kind for d in jax.devices()], 'jax_version': jax.__version__,
              'compiled_sha256': hashlib.sha256(compiled_text.encode()).hexdigest(),
              'reachability': reachability, 'trials': 5, 'seed': 42, 'results': [],
              'source_hashes': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                                ['puzzlescript_jax/env.py', 'puzzlejax/validate_actions.py',
                                 'scripts/benchmarks/benchmark_paper_throughput.py', str(Path(__file__).relative_to(root))]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_result(args.output, result)
    actions = list(range(5)) * 8 + np.random.default_rng(42).integers(0, 5, 88).tolist()
    start = time.perf_counter()
    ok, message = run_test_case(args.game, 0, actions, NodeJSPuzzleScriptBackend(), init_ps_lark_parser(),
                                env_cls=cls, stop_on_win=True)
    result['validation'] = {'passed': ok, 'message': message, 'actions': actions,
                            'elapsed_s': time.perf_counter() - start}
    save_result(args.output, result)
    if not ok:
        raise AssertionError(message)
    tree, _, error = get_tree_from_txt(init_ps_lark_parser(), args.game, test_env_init=False)
    assert tree is not None, error
    env = cls(tree, level_i=0, print_score=False, max_steps=np.iinfo(np.int32).max)
    assert not env.has_randomness() and not env._is_multi_level
    blocks = env._gen_rule_blocks(env.get_level(0).shape[1:])
    result['pruned_jax_rule_functions'] = sum(len(fns) for _, groups in blocks for fns, _ in groups) - 1
    for batch in args.batches:
        result['pending_batch'] = batch
        save_result(args.output, result)
        row = benchmark(env, batch, max(5000 // batch, 100), 5, 42)
        result['results'].append(row)
        result.pop('pending_batch', None)
        save_result(args.output, result)
        print(json.dumps(row), flush=True)
    result['stop_reason'] = 'fixed_pilot_batches'
    save_result(args.output, result)


if __name__ == '__main__':
    main()
