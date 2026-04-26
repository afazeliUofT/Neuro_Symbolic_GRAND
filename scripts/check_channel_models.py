#!/usr/bin/env python3
from __future__ import annotations
import argparse
from hybrid_bp_nsg.config import load_config
from hybrid_bp_nsg.code import build_code
from hybrid_bp_nsg.channels import channel_diagnostics
ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--profiles', nargs='*', default=None)
args=ap.parse_args()
cfg=load_config(args.config)
code=build_code(cfg)
profiles=args.profiles or cfg.get('data',{}).get('profiles',['AWGN'])
for line in channel_diagnostics(code,cfg,profiles): print(line)
