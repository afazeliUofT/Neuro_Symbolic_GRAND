#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--output', required=True, help='Output directory, e.g. outputs/hybrid_bp_nsg_v14_pusch_cdl_full')
args = ap.parse_args()
root = Path(args.output)
print('=== V14 DATASET SANITY PROBE ===')
print('output:', root)
files = sorted((root/'datasets').glob('**/*.npz'))
print('npz_files:', len(files))
if not files:
    raise SystemExit('No npz files found')
by_split = defaultdict(list)
for p in files:
    split = 'train' if '/train/' in str(p) else 'val' if '/val/' in str(p) else 'unknown'
    by_split[split].append(p)
for split, plist in sorted(by_split.items()):
    n=0; tw=Counter(); std=exp=res=stdn=expn=resn=0; candn=anypos=top1=postot=validtot=0; snr=Counter(); prof=Counter()
    for p in plist:
        with np.load(p, allow_pickle=False) as z:
            nn = int(z['target_weight'].shape[0]); n += nn
            for x in z['target_weight'].reshape(-1): tw[int(x)] += 1
            std += int(z['standard_reachable'].sum()); stdn += int(z['standard_reachable'].size)
            exp += int(z['expanded_reachable'].sum()); expn += int(z['expanded_reachable'].size)
            res += int(z['rescueable'].sum()); resn += int(z['rescueable'].size)
            lab = z['candidate_labels'].astype(bool); val = z['candidate_valid'].astype(bool); pos = lab & val
            candn += int(pos.shape[0]); anypos += int(pos.any(axis=1).sum()); top1 += int(pos[:,0].sum())
            postot += int(pos.sum()); validtot += int(val.sum())
            for x in z['snr_db'].reshape(-1): snr[round(float(x),3)] += 1
            for x in z['profile_id'].reshape(-1): prof[int(x)] += 1
    print('\n--- split', split, '---')
    print('samples:', n)
    print('target_weight_hist_top:', tw.most_common(30))
    weights=[]
    for k,c in tw.items(): weights.extend([k]*c)
    weights=sorted(weights)
    if weights:
        for q in [0,.25,.5,.75,.9,.95,.99,1.0]:
            print(f'target_weight_q{q:.2f}={weights[min(len(weights)-1,int(q*(len(weights)-1)))]}')
    print('standard_reachable_rate:', std/max(1,stdn), 'positives:', std, 'n:', stdn)
    print('expanded_reachable_rate:', exp/max(1,expn), 'positives:', exp, 'n:', expn)
    print('rescueable_rate:', res/max(1,resn), 'positives:', res, 'n:', resn)
    print('candidate_any_positive_rate:', anypos/max(1,candn), 'positives:', anypos, 'n:', candn)
    print('candidate_top1_positive_rate:', top1/max(1,candn), 'positives:', top1, 'n:', candn)
    print('candidate_pos_total:', postot, 'candidate_valid_total:', validtot, 'avg_valid_candidates:', validtot/max(1,candn))
    print('snr_hist:', snr.most_common())
    print('profile_hist:', prof.most_common())
    if weights and max(weights) > 32:
        raise SystemExit('ERROR: target_weight exceeds 32; do not train')
    if anypos == 0:
        raise SystemExit('ERROR: no positive candidates; do not train')
print('\n[DONE] Dataset sanity passed.')
