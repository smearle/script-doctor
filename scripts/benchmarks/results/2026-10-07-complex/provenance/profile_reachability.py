"""Conservative object-type reachability from the frozen level-zero inputs.

This is a structural experiment, not an enabled engine optimization. Negative
conditions, spatial relationships and movement are ignored. This can only add
possible rules/objects. Positive property alternatives are disjunctions.
"""
import hashlib, json
from pathlib import Path


def bits(words):
    return sum((int(w) & 0xffffffff) << (32*i) for i,w in enumerate(words))


def requirements(rule):
    needed=0; alternatives=[]; produced=0
    for row in rule['patterns']:
        for cell in row:
            if not isinstance(cell,dict):
                continue
            needed |= bits(cell['objectsPresent'])
            alternatives.extend(bits(a) for a in cell['anyObjectsPresent'])
            replacement=cell.get('replacement')
            if replacement:
                produced |= bits(replacement['objectsSet']) | bits(replacement['randomEntityMask'])
    return needed, alternatives, produced


results=[]
for path in sorted(Path('results/metadata-original/compiled-games').glob('*.json')):
    compiled=json.loads(path.read_text())
    rules=[r for group in compiled['rules']+compiled['lateRules'] for r in group]
    constraints=[requirements(r) for r in rules]
    level=next(l for l in compiled['levels'] if l['type']=='level' and l['index']==0)
    stride=compiled['STRIDE_OBJ']; reachable=0
    for offset in range(0,len(level['objects']),stride):
        reachable |= bits(level['objects'][offset:offset+stride])
    initial=reachable
    passes=0
    while True:
        before=reachable
        for needed,alternatives,produced in constraints:
            if needed & ~reachable == 0 and all(a & reachable for a in alternatives):
                reachable |= produced
        passes+=1
        if reachable==before:
            break
    possible=sum(needed & ~reachable == 0 and all(a & reachable for a in alternatives)
                 for needed,alternatives,_ in constraints)
    results.append({'game':path.stem,'compiled_sha256':hashlib.sha256(path.read_text().strip().encode()).hexdigest(),
                    'initial_object_types':initial.bit_count(),'reachable_object_types':reachable.bit_count(),
                    'compiled_rules':len(rules),'potentially_reachable_rules':possible,
                    'unreachable_rules':len(rules)-possible,'closure_passes':passes})
output={'scope':'Static conservative analysis of fixed level zero; no engine pruning or timing performed',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results}
Path('results/object-reachability.json').write_text(json.dumps(output,indent=2)+'\n')
for r in sorted(results,key=lambda r:r['unreachable_rules'],reverse=True)[:12]:
    print(r)
