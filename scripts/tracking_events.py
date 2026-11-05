#!/usr/bin/env python3
"""MOT17-10 val event extraction using installed TrackEval preprocessing and CLEAR matching.
Uses frozen CPU trace; emits a local, server-free viewer. No parameter search.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

SEQ = 'MOT17-10-SDP'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(data, np, assignment):
    """Keep CLEAR's early-continue behavior, threshold and previous-match preference verbatim in semantics."""
    n = data['num_gt_ids']
    prev = np.full(n, np.nan)
    previous_step = np.full(n, np.nan)
    last_frame = np.full(n, -1, dtype=int)
    segments = np.zeros(n, dtype=int)
    events, matches = [], []
    counts = Counter()
    for t, (gs, ts) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
        frame_matches = []
        matches.append(frame_matches)
        if not len(gs):
            counts['CLR_FP'] += len(ts)
            continue
        if not len(ts):
            counts['CLR_FN'] += len(gs)
            continue
        sim = data['similarity_scores'][t]
        score = 1000 * (ts[None, :] == previous_step[gs[:, None]]) + sim
        score[sim < 0.5 - np.finfo(float).eps] = 0
        rr, cc = assignment(-score)
        keep = score[rr, cc] > np.finfo(float).eps
        rr, cc = rr[keep], cc[keep]
        for r, c in zip(rr, cc):
            g, tid = int(gs[r]), int(ts[c])
            known = not np.isnan(prev[g])
            switch = known and int(prev[g]) != tid
            fragment = np.isnan(previous_step[g]) and segments[g] > 0
            gap = known and t - last_frame[g] > 1
            counts['IDSW'] += int(switch)
            counts['Frag'] += int(fragment)
            frame_matches.append({'gt_dense': g, 'tracker_dense': tid, 'iou': float(sim[r, c])})
            if switch or fragment or gap:
                events.append({'frame': t+1, 'gt_dense': g, 'old_tracker_dense': int(prev[g]),
                               'tracker_dense': tid, 'previous_match_frame': int(last_frame[g])+1,
                               'gap_frames': int(t-last_frame[g]-1), 'idsw': bool(switch),
                               'clear_fragment': bool(fragment), 'gap_recovery': bool(gap)})
            if np.isnan(previous_step[g]):
                segments[g] += 1
            prev[g], last_frame[g] = tid, t
        previous_step[:] = np.nan
        previous_step[gs[rr]] = ts[cc]
        counts['CLR_TP'] += len(rr)
        counts['CLR_FN'] += len(gs)-len(rr)
        counts['CLR_FP'] += len(ts)-len(rr)
    return events, matches, dict(counts)


def original_ids(raw, data, np):
    gtmap, possible = {}, {}
    for t in range(data['num_timesteps']):
        keep = (raw['gt_extras'][t]['zero_marked'] != 0) & (raw['gt_classes'][t] == 1)
        for dense, original in zip(data['gt_ids'][t], raw['gt_ids'][t][keep]):
            if int(dense) in gtmap and gtmap[int(dense)] != int(original):
                raise ValueError('GT ID mapping drift')
            gtmap[int(dense)] = int(original)
        for dense, box in zip(data['tracker_ids'][t], data['tracker_dets'][t]):
            ids = set(map(int, raw['tracker_ids'][t][np.all(raw['tracker_dets'][t] == box, axis=1)]))
            dense = int(dense)
            possible[dense] = possible.get(dense, ids) & ids
    if any(len(ids) != 1 for ids in possible.values()):
        raise ValueError('ambiguous original tracker ID mapping')
    return gtmap, {d: next(iter(ids)) for d, ids in possible.items()}


def xyxy(b):
    return [float(b[0]), float(b[1]), float(b[0]+b[2]), float(b[1]+b[3])]


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/area if area else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--legacy', type=Path, default=Path(os.environ.get('SACCADE_LEGACY_ROOT', '/home/ray/developer/ai/saccade')))
    args = p.parse_args()
    if args.output.exists():
        p.error('output must be new')
    mot = args.trace / f'{SEQ}.txt'
    if mot.read_bytes() != args.baseline.read_bytes():
        raise ValueError('traced MOT differs from frozen G1 baseline')
    sys.path.insert(0, str(args.legacy/'third_party/TrackEval'))
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from PIL import Image
    for name, alias in [('float',float),('int',int),('bool',bool)]:
        if name not in np.__dict__:
            setattr(np,name,alias)
    import trackeval
    out = args.output.resolve()
    custody_paths = [Path(__file__).resolve(), ROOT/'tools/tracking_event_trace.cpp',
                     ROOT/'tools/tracking_event_viewer.html',args.trace/'trace.jsonl',mot,
                     args.baseline,args.trace/'tracker-config.txt',
                     args.legacy/f'datasets/MOT17/train/{SEQ}/gt/gt.txt',
                     args.legacy/f'datasets/MOT17/train/{SEQ}/seqinfo.ini']
    custody_paths += list((args.legacy/'third_party/TrackEval/trackeval').rglob('*.py'))
    frozen = {str(x.resolve()):sha(x) for x in custody_paths}
    out.mkdir(parents=True)
    (out/'trackers/v1').mkdir(parents=True)
    shutil.copyfile(mot, out/f'trackers/v1/{SEQ}.txt')
    dataset = trackeval.datasets.MotChallenge2DBox({
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
        'GT_FOLDER': str(args.legacy/'datasets/MOT17/train'), 'SKIP_SPLIT_FOL': True,
        'SEQ_INFO': {SEQ: None}, 'TRACKERS_FOLDER': str(out/'trackers'),
        'TRACKER_SUB_FOLDER': '', 'TRACKERS_TO_EVAL': ['v1'], 'BENCHMARK': 'MOT17',
        'SPLIT_TO_EVAL': 'train', 'PRINT_CONFIG': False})
    raw = dataset.get_raw_seq_data('v1', SEQ)
    data = dataset.get_preprocessed_seq_data(raw, 'pedestrian')
    events, matches, counts = extract(data,np,linear_sum_assignment)
    official = trackeval.metrics.CLEAR({'PRINT_CONFIG':False}).eval_sequence(data)
    checked = {k: {'extracted': counts.get(k,0), 'official': int(official[k])}
               for k in ('IDSW','Frag','CLR_TP','CLR_FN','CLR_FP')}
    if any(v['extracted'] != v['official'] for v in checked.values()):
        raise ValueError(f'CLEAR event mismatch: {checked}')
    if counts['IDSW'] != 56 or counts['Frag'] != 150:
        raise ValueError('baseline differs from frozen val summary')
    gm, tm = original_ids(raw,data,np)
    # Also retain previously matched GTs whose final annotated span never recovers.
    # These tails are observations, not extra CLEAR Frag or IDSW counts.
    for g in range(data['num_gt_ids']):
        observed = [(t,m) for t,ms in enumerate(matches) for m in ms if m['gt_dense']==g]
        if not observed:
            continue
        last, last_match = observed[-1]
        tail = [t for t in range(last+1,data['num_timesteps']) if g in data['gt_ids'][t]]
        if tail:
            events.append({'frame':tail[0]+1,'gt_dense':g,
                           'old_tracker_dense':last_match['tracker_dense'],
                           'tracker_dense':last_match['tracker_dense'],
                           'previous_match_frame':last+1,'gap_frames':len(tail),
                           'idsw':False,'clear_fragment':False,'gap_recovery':False,
                           'unrecovered_tail':True,'tail_end_frame':tail[-1]+1})
    events.sort(key=lambda e:(e['frame'],e['gt_dense']))
    traces = [json.loads(line) for line in (args.trace/'trace.jsonl').read_text().splitlines()]
    if len(traces) != data['num_timesteps'] or [x['frame'] for x in traces] != list(range(1,len(traces)+1)):
        raise ValueError('trace frame coverage mismatch')
    mot_frames = [[] for _ in traces]
    for row in csv.reader(mot.open()):
        f, tid = int(row[0]), int(row[1])
        mot_frames[f-1].append({'id':tid,'box':xyxy(list(map(float,row[2:6]))),'interpolated':float(row[6]) < 0})
    output_map = {}
    for t, trace in enumerate(traces):
        for r in trace['packet_rows']:
            output_map.setdefault(r['output_id'],len(output_map)+1)
            r['mot_id'] = output_map[r['output_id']]
        observed = {r['mot_id']:r for r in trace['packet_rows']}
        final_observed = {r['id']:r for r in mot_frames[t] if not r['interpolated']}
        if observed.keys() != final_observed.keys():
            raise ValueError('packet output-ID to MOT-ID mapping mismatch')
        for mid, row in observed.items():
            if max(abs(a-b) for a,b in zip(row['box'],final_observed[mid]['box'])) > .0011:
                raise ValueError('packet to MOT box mismatch')
        for group in ('before','predicted','after'):
            for r in trace[group]:
                r['mot_id'] = output_map.get(r['output_id'])
        trace['gt'] = [{'id':gm[int(g)], 'box':xyxy(b)} for g,b in zip(data['gt_ids'][t],data['gt_dets'][t])]
        trace['mot'] = mot_frames[t]
        trace['matches'] = [{'gt':gm[m['gt_dense']], 'mot':tm[m['tracker_dense']], 'iou':m['iou']} for m in matches[t]]
    for index, e in enumerate(events):
        e.update(index=index+1, gt=gm[e['gt_dense']], old_mot=tm[e['old_tracker_dense']], mot=tm[e['tracker_dense']])
        start, end = e['previous_match_frame']+1, e.get('tail_end_frame',e['frame'])
        evidence = []
        for f in range(start,end+1):
            tr = traces[f-1]
            g = next((x for x in tr['gt'] if x['id']==e['gt']),None)
            if g is None:
                evidence.append({'frame':f,'gt_present':False})
                continue
            best_raw = max((iou(g['box'],d['box']) for d in tr['candidates']),default=0)
            best_batch = max((iou(g['box'],d['box']) for d in tr['batch']),default=0)
            evidence.append({'frame':f,'gt_present':True,'best_candidate_iou':best_raw,'best_batch_iou':best_batch})
        e['evidence'] = evidence
        # Observable screening labels, not causal verdicts. Distinct from metric definitions.
        if any(not x['gt_present'] for x in evidence):
            label='gt_absence_in_interval'
        elif any(x['best_candidate_iou'] < .5 for x in evidence):
            label='candidate_coverage_gap'
        elif any(x['best_batch_iou'] < .5 for x in evidence):
            label='batch_coverage_gap'
        else:
            label='association_or_output_review'
        e['screening'] = label
    (out/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    with (out/'events.csv').open('w',newline='') as f:
        fields=['index','frame','gt','old_mot','mot','previous_match_frame','gap_frames','idsw','clear_fragment','gap_recovery','unrecovered_tail','tail_end_frame','screening']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(events)
    (out/'images').mkdir()
    image_hashes={}
    for t in range(len(traces)):
        src=args.legacy/f'datasets/MOT17/train/{SEQ}/img1/{t+1:06}.jpg'
        image_hashes[str(src)]=sha(src)
        with Image.open(src) as im:
            im.thumbnail((960,960)); im.save(out/f'images/{t+1:06}.jpg',quality=78)
    payload={'events':events,'frames':traces,'checks':checked}
    template=(ROOT/'tools/tracking_event_viewer.html').read_text()
    (out/'index.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,separators=(',',':')).replace('</','<\\/')))
    if frozen != {str(x.resolve()):sha(x) for x in custody_paths}:
        raise ValueError('source or inputs changed during extraction')
    summary={'sequence':SEQ,'frames':len(traces),'events':len(events),'checks':checked,
             'gap_recoveries':sum(e['gap_recovery'] for e in events),
             'unrecovered_tails':sum(e.get('unrecovered_tail',False) for e in events),
             'idsw_screening':dict(Counter(e['screening'] for e in events if e['idsw'])),
             'all_screening':dict(Counter(e['screening'] for e in events)),
             'baseline_mot_bytes_equal':True,'packet_to_mot_mapping_checked':True,'formal_acceptance':False,
             'scope':'CPU fixed G1 replay, validation only; screening labels are not causal attribution',
             'inputs_sha256':frozen,'inputs_unchanged':True,'images_sha256':image_hashes}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if 'sha256' not in k},indent=2))


if __name__=='__main__':
    main()
