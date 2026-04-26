#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from hybrid_bp_nsg.config import load_config
from hybrid_bp_nsg.code import build_code
ap=argparse.ArgumentParser(); ap.add_argument('config')
args=ap.parse_args()
cfg=load_config(args.config)
code=build_code(cfg)
print(json.dumps({
  'config': args.config,
  'output_dir': cfg['project']['output_dir'],
  'target_basis': cfg['rescue'].get('target_basis'),
  'max_expanded_weight': cfg['rescue'].get('max_expanded_weight'),
  'filter_by_target_weight': cfg['data'].get('filter_by_target_weight'),
  'require_candidate_positive': cfg['data'].get('require_candidate_positive'),
  'candidate_bank_inject_oracle_positive': cfg['rescue'].get('candidate_bank_inject_oracle_positive'),
  'code': code.code_summary(),
}, indent=2, default=str))
