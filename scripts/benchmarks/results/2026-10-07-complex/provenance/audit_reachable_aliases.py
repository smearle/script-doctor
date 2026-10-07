"""Check the final alias guard against the exact measured Crate pruning decisions."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from puzzlescript_jax.env import PuzzleJaxEnv
from puzzlescript_jax.preprocessing import get_tree_from_txt
from puzzlescript_jax.utils import init_ps_lark_parser

worktree=Path('/tmp/script-doctor-gpu-throughput')
paths=[worktree/'scripts/benchmarks/results/2026-10-07-complex/provenance/benchmark_reachable_rules_v2_measured.py',
       worktree/'scripts/benchmarks/benchmark_reachable_rules.py']
modules=[]
for i,path in enumerate(paths):
    spec=importlib.util.spec_from_file_location(f'pruning_variant_{i}',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);modules.append(mod)
compiled=json.loads(Path('results/metadata-original/compiled-games/Crate_Assembler.json').read_text())
reachable=modules[0].object_reachability(compiled)['reachable_names']
traces=[]
for mod in modules:
    decisions=[]
    cls=mod.specialized_env_class(reachable)
    class Probe(cls):
        def gen_subrules_meta(self,rule,rule_name,lvl_shape):
            result=super().gen_subrules_meta(rule,rule_name,lvl_shape)
            decisions.append([rule_name,str(rule.left_kernels),len(result)])
            return result
    tree,_,error=get_tree_from_txt(init_ps_lark_parser(),'Crate_Assembler',test_env_init=False)
    assert tree is not None,error
    env=Probe(tree,level_i=0,print_score=False,max_steps=100)
    blocks=env._gen_rule_blocks(env.get_level(0).shape[1:])
    traces.append({'decisions':decisions,'functions':sum(len(fns) for _,groups in blocks for fns,_ in groups)-1})
assert traces[0]==traces[1], 'The measured and final Crate pruning decisions differ'
# Exercise an alias and an unreachable object without building another game.
original=PuzzleJaxEnv.gen_subrules_meta
try:
    PuzzleJaxEnv.gen_subrules_meta=lambda *args:['kept']
    cls=modules[1].specialized_env_class({'player'})
    instance=object.__new__(cls);instance.objs_to_idxs={'player':0,'avatar':0,'absent':1}
    def apply(cell):
        return instance.gen_subrules_meta(SimpleNamespace(left_kernels=[[cell]]),'fixture',(1,1))
    assert apply(['> avatar'])==['kept']
    assert apply(['> absent'])==[]
    assert apply(['no absent'])==['kept']
    assert apply(['unknown_property'])==['kept']
finally:
    PuzzleJaxEnv.gen_subrules_meta=original
result={'game':'Crate_Assembler','functions':traces[0]['functions'],
        'identical_pruning_decisions':len(traces[0]['decisions']),
        'decisions_sha256':hashlib.sha256(json.dumps(traces[0]['decisions']).encode()).hexdigest(),
        'measured_sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        'final_sha256':hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        'alias_guard_checks':4,'result':'passed'}
Path('results/reachable-alias-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
