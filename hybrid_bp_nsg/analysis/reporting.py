from __future__ import annotations

from pathlib import Path
import gzip
import pandas as pd

from ..plotting.twc_plots import build_twc_plots
from ..utils.io import ensure_dir


def build_reports(cfg, output_dir: Path, logger, tail_summary_path: str | None = None) -> None:
    eval_root = output_dir / 'evaluation'
    summary_path = eval_root / 'evaluation_summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary_df = pd.read_csv(summary_path)
    with gzip.open(eval_root / 'all_raw_records.csv.gz', 'rt', encoding='utf-8') as f:
        raw_df = pd.read_csv(f)
    tail_summary_df = pd.read_csv(tail_summary_path) if tail_summary_path and Path(tail_summary_path).exists() else None
    reports_root = ensure_dir(output_dir / 'reports')
    repo_export = ensure_dir(output_dir / 'repo_export')
    manifest = build_twc_plots(
        summary_df,
        raw_df,
        output_dir,
        paper_width=float(cfg['plots']['paper_width_in']),
        paper_height=float(cfg['plots']['paper_height_in']),
        tail_summary_df=tail_summary_df,
        primary_baseline=str(cfg.get('report', {}).get('primary_baseline', 'bp_nms_20')),
        min_errors_to_plot=int(cfg['eval'].get('min_frame_errors_to_report', 30)),
    )
    if len(manifest) != 10:
        raise RuntimeError(f"Expected 10 TWC figures, found {len(manifest)}")
    twc_dir = output_dir / 'TWC_plots'
    required = [twc_dir / 'manifest.csv', twc_dir / 'README.md']
    required += [twc_dir / f"{fig}.png" for fig in manifest['figure']]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    hyb = summary_df[summary_df.decoder=='hybrid_bp_nsg']
    comp_rows = []
    for dec in sorted(summary_df.decoder.unique()):
        if dec == 'hybrid_bp_nsg':
            continue
        comp = hyb.merge(summary_df[summary_df.decoder==dec], on=['profile','snr_db'], suffixes=('_hyb', '_base'))
        comp['bler_gain'] = comp['bler_base'] - comp['bler_hyb']
        comp['latency_delta_ms'] = comp['avg_latency_ms_hyb'] - comp['avg_latency_ms_base']
        comp['query_delta'] = comp['avg_queries_hyb'] - comp['avg_queries_base']
        comp_rows.append({
            'baseline': dec,
            'mean_bler_gain': comp['bler_gain'].mean(),
            'mean_latency_delta_ms': comp['latency_delta_ms'].mean(),
            'mean_query_delta': comp['query_delta'].mean(),
            'wins_bler_points': int((comp['bler_gain'] > 0).sum()),
            'ties_bler_points': int((comp['bler_gain'] == 0).sum()),
            'losses_bler_points': int((comp['bler_gain'] < 0).sum()),
            'wins_latency_points': int((comp['latency_delta_ms'] < 0).sum()),
        })
    pairwise = pd.DataFrame(comp_rows).sort_values('mean_bler_gain', ascending=False)
    pairwise.to_csv(reports_root / 'hybrid_vs_all_pairwise_summary.csv', index=False)
    pairwise.to_csv(repo_export / 'hybrid_vs_all_pairwise_summary.csv', index=False)
    summary_df.to_csv(repo_export / 'overview_by_profile_snr.csv', index=False)
    primary = str(cfg.get('report', {}).get('primary_baseline', 'bp_nms_20'))
    primary_map = summary_df[summary_df.decoder==primary][['profile','snr_db','bler']].rename(columns={'bler':'primary_bler'})
    merged = summary_df.merge(primary_map, on=['profile','snr_db'])
    nontrivial = merged[merged.primary_bler < float(cfg['eval']['saturated_bler_threshold'])]
    nontrivial.to_csv(repo_export / 'overview_nontrivial_by_profile_snr.csv', index=False)
    action_contrib = raw_df[raw_df.decoder=='hybrid_bp_nsg'].groupby('action', as_index=False).agg(
        count=('action','size'),
        avg_block_error=('block_error','mean'),
        avg_latency_ms=('latency_ms','mean'),
        avg_queries=('queries','mean'),
    )
    total = max(1, len(raw_df[raw_df.decoder=='hybrid_bp_nsg']))
    action_contrib['count_fraction'] = action_contrib['count'] / total
    action_contrib.to_csv(repo_export / 'hybrid_action_contribution_summary.csv', index=False)
    hybrid_raw = raw_df[raw_df.decoder=='hybrid_bp_nsg']
    post = hybrid_raw[hybrid_raw.action.str.contains('rescue', na=False)].copy()
    if not post.empty:
        post_summary = post.groupby('action', as_index=False).agg(
            count=('action','size'),
            avg_latency_ms=('latency_ms','mean'),
            avg_queries=('queries','mean'),
            bler=('block_error','mean'),
        )
        post_summary.to_csv(repo_export / 'postsearch_outcome_summary.csv', index=False)
    else:
        pd.DataFrame(columns=['action','count','avg_latency_ms','avg_queries','bler']).to_csv(repo_export / 'postsearch_outcome_summary.csv', index=False)
    md = []
    md.append('# Hybrid BP + AI-guided GRAND Rescue Report')
    md.append('')
    primary_baseline = str(cfg.get('report', {}).get('primary_baseline', 'bp_nms_20'))
    best_vs_bp = pairwise[pairwise.baseline==primary_baseline]
    if not best_vs_bp.empty:
        r = best_vs_bp.iloc[0]
        md.append(f"Hybrid vs {primary_baseline}: mean BLER gain {r['mean_bler_gain']:.4f}, mean latency delta {r['mean_latency_delta_ms']:.4f} ms, BLER wins on {int(r['wins_bler_points'])} operating points.")
    if not hyb.empty:
        primary_point_map = primary_map.rename(columns={'primary_bler':'bler_ref'})
        comp = hyb.merge(primary_point_map, on=['profile','snr_db'])
        comp['gain'] = comp['bler_ref'] - comp['bler']
        if not comp.empty:
            best_point = comp.sort_values('gain', ascending=False).iloc[0]
            md.append(f"Strongest BLER gain vs {primary_baseline} at profile {best_point['profile']}, {best_point['snr_db']:.1f} dB: {best_point['gain']:.4f}.")
    md.append('')
    rescue_used = int(hybrid_raw['rescue_used'].sum())
    rescued = int(((hybrid_raw['action']=='rescue_success_direct') | (hybrid_raw['action']=='rescue_success_micro')).sum())
    md.append(f'Rescue invoked on {rescue_used} packets and recovered {rescued} of them.')
    md.append('')
    md.append(f"Monte Carlo stopping: each operating point runs until selected decoders reach {int(cfg['eval'].get('target_frame_errors', 200))} frame errors or the sample cap; points with fewer than {int(cfg['eval'].get('min_frame_errors_to_report', 30))} frame errors are omitted from BLER plots.")
    md.append('')
    md.append('TWC_plots/ contains 10 publication-style figures with matching CSV files and a folder-level README that embeds the PNGs for direct GitHub viewing.')
    report_text = '\n'.join(md)
    (reports_root / 'report.md').write_text(report_text, encoding='utf-8')
    (repo_export / 'report.md').write_text(report_text, encoding='utf-8')
