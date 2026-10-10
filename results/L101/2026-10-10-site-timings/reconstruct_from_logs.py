"""Read-only L101 timing reconstruction; Python 3.6 compatible for Daint."""
import glob
import json
import os
import re

sites = (1, 10, 20, 30)
wanted = {((state-1)*101 + site-1)*3 + channel
          for state in (1, 2, 3) for site in sites for channel in (1, 2, 3)}
root = '/ritom/scratch/cscs/cpi/aiida/4b/41/17de-7c29-4a5b-a1fe-78f438c854eb'
lineage = []
seen = set()
while root not in seen:
    seen.add(root)
    with open(os.path.join(root, 'segment.json')) as handle:
        config = json.load(handle)
    lineage.append((root, config))
    resume = config.get('resume_state')
    if not resume or not os.path.islink(os.path.join(root, resume)):
        break
    previous = os.path.realpath(os.path.join(root, resume))
    root = os.path.dirname(previous) if os.path.basename(previous) == 'state' else previous

attempts = []
for root, config in reversed(lineage):
    for manifest in sorted(glob.glob(os.path.join(root, 'task_manifest', 'worker_*.txt'))):
        with open(manifest) as handle:
            ids = [int(value) for value in handle.read().split()]
        if not wanted.intersection(ids):
            continue
        worker = os.path.basename(manifest)[:-4]
        logfile = os.path.join(root, 'state', 'workers', worker, 'stdout.txt')
        current = None
        records = []
        worker_seconds = None
        with open(logfile) as handle:
            for line in handle:
                start = re.search(r'starting task=(\d+) state=(\d+) site=(\d+) channel=(zz|pm|mp)', line)
                if start:
                    if current is not None:
                        records.append(current)
                    task, state, site = map(int, start.groups()[:3])
                    current = ({'task': task, 'state': state, 'site': site, 'channel': start.group(4),
                                'root': root, 'logfile': logfile, 'assigned_tasks': len(ids),
                                'start_n': 2, 'resumed': False, 'last_n': 2,
                                'last_checkpoint_n': None, 'recurrence_seconds': 0.0,
                                'completed': False, 'target_n': 1474}
                               if task in wanted else None)
                    continue
                done_worker = re.search(r'worker \d+ complete: .*seconds=([\d.eE+-]+)', line)
                if done_worker:
                    worker_seconds = float(done_worker.group(1))
                if current is None:
                    continue
                resume = re.search(r'resumed recurrence completed_n=(\d+)', line)
                if resume:
                    current['start_n'] = int(resume.group(1))
                    current['last_checkpoint_n'] = current['start_n']
                    current['resumed'] = True
                progress = re.search(r'progress completed_n=(\d+)/(\d+).*elapsed_seconds=([\d.eE+-]+)', line)
                if progress:
                    current['last_n'] = int(progress.group(1))
                    current['target_n'] = int(progress.group(2))
                    current['recurrence_seconds'] = float(progress.group(3))
                checkpoint = re.search(r'checkpointed recurrence completed_n=(\d+)/(\d+)', line)
                if checkpoint:
                    current['last_checkpoint_n'] = int(checkpoint.group(1))
                done = re.search(r'completed task=(\d+) seconds=([\d.eE+-]+)', line)
                if done and int(done.group(1)) == current['task']:
                    current['completed'] = True
                    current['final_task_seconds'] = float(done.group(2))
        if current is not None:
            records.append(current)
        for record in records:
            if len(ids) == 1 and worker_seconds is not None:
                record['active_seconds'] = worker_seconds
                record['timing_source'] = 'single-task worker completion timer'
            elif record['completed']:
                record['active_seconds'] = record['final_task_seconds']
                record['timing_source'] = 'completed task invocation timer'
            else:
                record['active_seconds'] = record['recurrence_seconds']
                record['timing_source'] = 'last logged recurrence elapsed; lower bound'
            attempts.append(record)

tasks = []
continuity_notes = []
for task in sorted(wanted):
    records = [record for record in attempts if record['task'] == task]
    if not records or records[0]['resumed'] or not records[-1]['completed']:
        raise RuntimeError('Incomplete log coverage for task {}'.format(task))
    if sum(record['completed'] for record in records) != 1:
        raise RuntimeError('Unexpected completion count for task {}'.format(task))
    if any(record['target_n'] != 1474 for record in records):
        raise RuntimeError('Unexpected recurrence target')
    if records[-1]['last_n'] != 1474:
        raise RuntimeError('Final task missing full progress trace')
    for previous, current in zip(records, records[1:]):
        if not current['resumed'] or not previous['last_checkpoint_n'] <= current['start_n'] <= previous['last_n']:
            continuity_notes.append({'task': task, 'previous_root': previous['root'],
                                     'previous_checkpoint_n': previous['last_checkpoint_n'],
                                     'previous_last_n': previous['last_n'],
                                     'next_root': current['root'], 'next_start_n': current['start_n'],
                                     'next_resumed': current['resumed']})
    tasks.append({'task': task, 'site': records[0]['site'], 'state': records[0]['state'],
                  'channel': records[0]['channel'], 'attempts': len(records),
                  'active_seconds': sum(r['active_seconds'] for r in records),
                  'recurrence_seconds': sum(r['recurrence_seconds'] for r in records),
                  'has_lower_bound_attempt': any('lower bound' in r['timing_source'] for r in records),
                  'final_task_seconds': records[-1]['final_task_seconds']})
print(json.dumps({'tasks': tasks, 'attempts': attempts, 'continuity_notes': continuity_notes,
                  'source_latest_segment_uuid': '4b4117de-7c29-4a5b-a1fe-78f438c854eb',
                  'method': 'Sum recorded active invocation times across checkpoint ancestry; interrupted attempts without a terminal timer use last logged recurrence elapsed as a lower bound. Queue and startup outside the worker timer are excluded; repeated work after rollback is included.'}, indent=2))
