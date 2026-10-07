"""Validated, coverage-aware access to the corrected paper ExIt experiment."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def completed_runs(experiment, require_full=False):
    """Never infer completion from an early solution or a partial checkpoint."""
    root=Path(experiment)
    manifest=read_json(root/'manifest.json')
    seen=set();runs=[];states=Counter();expected=defaultdict(set)
    for job in manifest['jobs']:
        cfg=job['config'];key=(cfg['game'],cfg['level_i'],cfg['max_nodes'])
        if key in seen:raise ValueError(f'Duplicate requested configuration: {key}')
        seen.add(key);expected[(cfg['game'],cfg['max_nodes'])].add(cfg['level_i'])
        dest=root/'runs'/job['id'];status_path=dest/'status.json'
        status=read_json(status_path) if status_path.exists() else {'state':'pending'}
        states[status['state']]+=1
        if status['state']!='complete':continue
        if status.get('config')!=cfg:raise ValueError(f'Completed config mismatch: {job["id"]}')
        path=dest/job['run_subdir']/'history.json'
        history=read_json(path)
        if len(history)!=cfg['n_iterations'] or [r['iteration'] for r in history]!=list(range(len(history))):
            raise ValueError(f'Incomplete or noncontiguous final history: {job["id"]}')
        if any(not isinstance(r['solved'],bool) for r in history):
            raise ValueError(f'Invalid solved flags: {job["id"]}')
        runs.append(dict(job=job,history=history,history_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    coverage=dict(requested=len(manifest['jobs']),completed=len(runs),states=dict(states),
                  full=bool(manifest['jobs']) and len(runs)==len(manifest['jobs']))
    if require_full and not coverage['full']:
        raise ValueError(f'Full ExIt results required before updating the paper: {coverage}')
    return manifest,runs,coverage


def full_results_bundle(experiment):
    root=Path(experiment).resolve()
    manifest,runs,coverage=completed_runs(root,require_full=True)
    groups=defaultdict(list)
    for run in runs:
        cfg=run['job']['config']
        if cfg['n_iterations']!=200 or cfg['seed']!=0 or cfg['max_nodes'] not in (100000,1000000):
            raise ValueError('Expected the pinned 200-iteration, seed-0 paper configurations')
        groups[(cfg['max_nodes'],cfg['game'])].append(run)
    results={}
    for (budget,game),group in sorted(groups.items()):
        label=f"ExIt · {'100k' if budget==100000 else '1M'} nodes"
        results.setdefault(label,{})[game]=dict(
            pct_solved=sum(any(r['solved'] for r in v['history']) for v in group)/len(group),
            n_levels=len(group),n_levels_requested=len(group),n_iters=200)
    return dict(schema_version=1,complete=True,coverage=coverage,experiment=str(root),
                heuristic_sign='corrected',base_commit=manifest.get('base_commit'),
                manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
                results=results,histories={v['job']['id']:v['history_sha256'] for v in runs})


def load_published_bundle(path):
    bundle=read_json(path)
    coverage=bundle.get('coverage',{})
    requested=coverage.get('requested',0)
    if (not bundle.get('complete') or not coverage.get('full') or not isinstance(requested,int)
        or requested<=0 or coverage.get('completed')!=requested):
        raise ValueError('Refusing to plot a partial corrected ExIt publication')
    if bundle.get('heuristic_sign')!='corrected':raise ValueError('Expected corrected ExIt results')
    if bundle.get('schema_version')!=1:raise ValueError('Unsupported ExIt publication schema')
    groups=bundle.get('results',{})
    count=0
    for group in groups.values():
        if not group:raise ValueError('Empty corrected ExIt result group')
        for stats in group.values():
            n=stats['n_levels']
            if (not isinstance(n,int) or n<=0 or n!=stats['n_levels_requested']
                or stats['n_iters']!=200 or not 0<=stats['pct_solved']<=1):
                raise ValueError('Invalid corrected ExIt result coverage or value')
            count+=n
    if count!=requested or len(bundle.get('histories',{}))!=requested:
        raise ValueError('Corrected ExIt result counts do not match the publication coverage')
    return bundle
