"""Read-only interpretation of ALL recorded fits and persistent delayed gate."""
import argparse
import json
from pathlib import Path

def summarize(root):
    reports=[]
    for path in sorted(root.rglob('metrics.jsonl')):
        config_path=path.parent/'config.json'
        if not config_path.exists():continue
        config=json.loads(config_path.read_text())
        rows=[json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows:continue
        train99=next((r['step'] for r in rows if r['train_accuracy']>=.99),None)
        train100=next((r['step'] for r in rows if r['train_accuracy']==1),None)
        held99=next((r['step'] for r in rows if r['heldout_accuracy']>=.99),None)
        held100=next((r['step'] for r in rows if r['heldout_accuracy']==1),None)
        persist=any(all(r['heldout_accuracy']>=.99 for r in rows[i:i+3]) for i in range(len(rows)-2))
        delayed=(train99 is not None and held99 is not None and held99>train99 and persist
                 and any(r['step']>=train99 and r['step']<held99 and r['heldout_accuracy']<.9 for r in rows))
        last=rows[-1]
        reports.append(dict(path=str(path.parent),method=config['method'],population=config['population']
                     if config['method']=='dust' else 0,world_size=config.get('population_world_size',1),
                     initial_parameter_sha256=config['initial_parameter_sha256'],
                     split_sha256=config['split_sha256'],last_step=last['step'],
                     train_accuracy=last['train_accuracy'],heldout_accuracy=last['heldout_accuracy'],
                     train_loss=last['train_loss'],heldout_loss=last['heldout_loss'],
                     max_heldout_accuracy=max(r['heldout_accuracy'] for r in rows),
                     first_train99=train99,first_train100=train100,first_held99=held99,first_held100=held100,
                     heldout99_three_check_persistence=persist,delayed_transition_observed=delayed,
                     wall_seconds=last['wall_seconds'],seconds_per_update=last['update_seconds']/max(last['step'],1),
                     classification='DELAYED_GROKKING_OBSERVED' if delayed else
                       'MEMORIZATION_GENERALIZATION_PENDING' if train99 is not None else 'NO_MEMORIZATION_YET',
                     completion=json.loads((path.parent/'completion.json').read_text())
                       if (path.parent/'completion.json').exists() else None))
    return reports

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();value=summarize(a.root)
    if a.output:a.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))
