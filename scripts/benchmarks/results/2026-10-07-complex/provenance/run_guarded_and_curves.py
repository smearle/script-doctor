import json,os,subprocess,sys,time
from pathlib import Path
from scripts.benchmarks.complex_games import COMPLEX_GAMES
from scripts.benchmarks.benchmark_paper_throughput import BENCHMARK_GAMES
status=Path('results/guarded-wide/status.json')
while not status.exists() or len(json.loads(status.read_text()))<64:
 time.sleep(5)
rows=[r for p in status.parent.glob('*.json') if p.name!='status.json' for r in json.loads(p.read_text())['results']]
assert len(rows)==128 and all(r['exit_code']==0 for r in json.loads(status.read_text()))
assert min(r['speedup'] for r in rows)>.97, 'Investigate regression before promoting candidate'
lib='/tmp/puzzlejax-complex-cpp-guarded/guarded/puzzlescript_cpp/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so'
cmd=['taskset','-c','0-7',sys.executable,'-u','-m','scripts.benchmarks.benchmark_cpp_scoring','--baseline','/tmp/puzzlejax-cpp-experiments/optimized-fixed/puzzlescript_cpp/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so','--candidate',lib,'--games','atlas shrank','SwapBot','Beam_Islands','--batches','64','--threads','8','--steps','300','--trials','9','--seeds','42','1042','--compiled-dir','results/metadata-original/compiled-games','--output','results/guarded-original.json']
with Path('results/guarded-original.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
cmd=['taskset','-c','0-31',sys.executable,'-u','-m','scripts.benchmarks.run_complex_cpp_suite','--library',lib,'--compiled-dir','results/metadata-original/compiled-games','--output-dir','results/cpp-throughput','--games',*(COMPLEX_GAMES+BENCHMARK_GAMES)]
subprocess.run(cmd,check=True)
