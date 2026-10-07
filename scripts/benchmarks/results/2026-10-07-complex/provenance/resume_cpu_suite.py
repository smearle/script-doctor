import json,os,signal,subprocess,sys,time
from pathlib import Path
status_path=Path('results/cpp-throughput/launch-status.json')
status=json.loads(status_path.read_text())
status['interruption']={'outer_session_exit_code':143,'partial_game':'Travelling_salesman','action':'Resume completed checkpoints; original driver and library unchanged'}
lib='/tmp/puzzlejax-complex-cpp-guarded/guarded/puzzlescript_cpp/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so'
for game in ['Travelling_salesman','Multi-word_Dictionary_Game','Microban','Magnet_Jack','HyperMaze']:
    cmd=['prlimit',f'--as={32*1024**3}','--','taskset','-c','0-31',sys.executable,'-u','-m','scripts.benchmarks.benchmark_cpp_throughput','--library',lib,'--compiled-dir','results/metadata-original/compiled-games','--output-dir','results/cpp-throughput','--games',game,'--max-threads','32']
    if Path(f'results/cpp-throughput/{game}.json').exists():cmd.append('--resume')
    start=time.monotonic()
    with Path(f'results/cpp-throughput/{game}-resumed.log').open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=p.wait(timeout=900);state='complete' if code==0 else 'error'
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);code=p.wait();state='timeout'
    row={'game':game,'status':state,'exit_code':code,'elapsed_s':time.monotonic()-start}
    status['games'].append(row);status_path.write_text(json.dumps(status,indent=2)+'\n');print(row,flush=True)
