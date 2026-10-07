"""Native work distribution for the exact actions used by the chunking pilot.

These instrumented counts are diagnostics, not timings or a prediction of GPU
instruction counts. This pass does not compare native and JAX boards.
"""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from puzzlescript_jax.globals import JS_TO_JAX_ACTIONS
from scripts.benchmarks.profile_cpp_rules import COUNTERS

library=Path('/tmp/puzzlejax-complex-cpp-profile/profiled/puzzlescript_cpp/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so')
name='puzzlescript_cpp._puzzlescript_cpp';spec=importlib.util.spec_from_file_location(name,library)
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
batch,steps,seed=256,16,42
jax_to_js={int(jax):js for js,jax in enumerate(JS_TO_JAX_ACTIONS)}
actions=np.random.default_rng(seed).integers(0,5,(batch,steps),dtype=np.int32)
result={'batch':batch,'steps':steps,'seed':seed,'action_layout':'default_rng(seed).integers(0,5,(batch,steps),dtype=int32); matches chunking pilot',
        'scope':'Native engine work counts; negative/spatial matching work depends on state. Boards are not compared with JAX in this pass.',
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'games':[]}
for game in ['Sokoboros','Vacuum']:
    compiled=Path(f'results/metadata-original/compiled-games/{game}.json').read_text().strip()
    data=np.zeros((batch,steps,len(COUNTERS)),dtype=np.int64);again_caps=wins=0
    for i in range(batch):
        engine=module.Engine();assert engine.load_from_json(compiled);engine.load_level(0,'batch-profile')
        for t,action in enumerate(actions[i]):
            engine.clear_benchmark_counters();engine.process_input(jax_to_js[int(action)])
            again=0
            while engine.is_againing() and again<50:
                engine.process_input(-1);again+=1
            again_caps+=bool(engine.is_againing())
            data[i,t]=engine.benchmark_counters()
            if engine.is_winning() or engine.check_win():
                wins+=1;engine.load_level(0,'batch-profile')
    summary={}
    for index,key in enumerate(COUNTERS):
        totals=data[:,:,index].sum(axis=1);means=data[:,:,index].mean(axis=0)
        ratios=np.divide(data[:,:,index].max(axis=0),means,out=np.zeros_like(means),where=means>0)
        summary[key]={'env_total_percentiles':dict(zip(['min','p50','p90','p99','max'],np.percentile(totals,[0,50,90,99,100]).tolist())),
                      'mean_env_total':float(totals.mean()),'mean_per_step_max_over_mean':float(ratios.mean())}
    result['games'].append({'game':game,'compiled_sha256':hashlib.sha256(compiled.encode()).hexdigest(),
                            'again_cap_hits':again_caps,'wins':wins,'summary':summary,
                            'counter_order':COUNTERS,'counts_env_step_counter':data.tolist()})
    Path('results/batch-rule-work.json').write_text(json.dumps(result,indent=2)+'\n')
    print(game,json.dumps(summary['tuple_attempts']),flush=True)
