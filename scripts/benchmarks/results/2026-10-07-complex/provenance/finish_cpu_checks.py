import json, os, signal, subprocess, sys, time
from pathlib import Path
while len(json.loads(Path('results/cpp-throughput/launch-status.json').read_text())['games']) < 32:
    time.sleep(10)
lib='/tmp/puzzlejax-complex-cpp-guarded/guarded/puzzlescript_cpp/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so'
base='/tmp/puzzlejax-cpp-experiments/optimized-fixed/puzzlescript_cpp/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so'
cmd=['taskset','-c','0',sys.executable,'-u','-m','scripts.benchmarks.benchmark_cpp_scoring','--baseline',base,'--candidate',lib,'--games','blocks','--batches','1','--threads','1','--steps','10000','--trials','21','--seeds','42','1042','--compiled-dir','results/metadata-original/compiled-games','--output','results/guarded-blocks-recheck.json']
with Path('results/guarded-blocks-recheck.log').open('w') as log:
    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
statuses=[]
for game in ['castlecloset','Vacuum']:
    cmd=['prlimit',f'--as={32*1024**3}','--','taskset','-c','0-31',sys.executable,'-u','-m','scripts.benchmarks.benchmark_cpp_throughput','--library',lib,'--compiled-dir','results/metadata-original/compiled-games','--output-dir','results/cpp-throughput','--games',game,'--max-threads','32','--resume','--extend','--max-batch','32768']
    start=time.monotonic()
    with Path(f'results/cpp-throughput/{game}-extended.log').open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=p.wait(timeout=1800);status='complete' if code==0 else 'error'
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);code=p.wait();status='timeout'
    statuses.append({'game':game,'status':status,'exit_code':code,'elapsed_s':time.monotonic()-start,'time_limit_s':1800})
    Path('results/cpp-throughput/extension-status.json').write_text(json.dumps(statuses,indent=2)+'\n')
    print(statuses[-1],flush=True)
env=dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/puzzlejax-mpl')
with Path('results/cached-reset-final-tests.log').open('w') as log:
    subprocess.run(['taskset','-c','0-3',sys.executable,'-m','pytest','-q','tests/test_cached_reset.py','tests/test_env_switch.py','tests/test_validation_wide_objects.py'],cwd='/tmp/puzzlejax-cache-final-tests',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('All final CPU checks complete',flush=True)
