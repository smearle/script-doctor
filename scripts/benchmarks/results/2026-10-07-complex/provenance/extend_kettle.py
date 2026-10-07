import json,os,signal,subprocess,sys,time
from pathlib import Path
lib='/tmp/puzzlejax-complex-cpp-guarded/guarded/puzzlescript_cpp/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so'
cmd=['prlimit',f'--as={32*1024**3}','--','taskset','-c','0-31',sys.executable,'-u','-m','scripts.benchmarks.benchmark_cpp_throughput','--library',lib,'--compiled-dir','results/metadata-original/compiled-games','--output-dir','results/cpp-throughput','--games','kettle','--max-threads','32','--resume','--extend','--max-batch','32768']
start=time.monotonic()
with Path('results/cpp-throughput/kettle-extended.log').open('w') as log:
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        code=p.wait(timeout=900);status='complete' if code==0 else 'error'
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);code=p.wait();status='timeout'
path=Path('results/cpp-throughput/extension-status.json');rows=json.loads(path.read_text())
row={'game':'kettle','status':status,'exit_code':code,'elapsed_s':time.monotonic()-start,'time_limit_s':900}
rows.append(row);path.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
