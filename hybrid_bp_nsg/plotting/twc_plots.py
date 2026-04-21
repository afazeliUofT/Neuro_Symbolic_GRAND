from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_fig(fig, path_base: Path, df: Optional[pd.DataFrame] = None) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path_base.with_suffix('.png'), dpi=160, bbox_inches='tight')
    if df is not None:
        df.to_csv(path_base.with_suffix('.csv'), index=False)
    plt.close(fig)


def _mask_bler(cur: pd.DataFrame, min_errors_to_plot: int) -> pd.Series:
    if 'frame_errors' in cur.columns:
        vals = cur['bler'].where(cur['frame_errors'] >= min_errors_to_plot, np.nan)
    elif 'plot_eligible' in cur.columns:
        vals = cur['bler'].where(cur['plot_eligible'] > 0, np.nan)
    else:
        vals = cur['bler']
    return vals.clip(lower=1e-7)


def _aggregate_for_plot(sub: pd.DataFrame) -> pd.DataFrame:
    grp = sub.groupby(['decoder','snr_db'], as_index=False).agg(
        samples=('samples','sum'),
        frame_errors=('frame_errors','sum'),
        avg_latency_ms=('avg_latency_ms','mean'),
        avg_queries=('avg_queries','mean'),
    )
    grp['bler'] = grp['frame_errors'] / np.maximum(grp['samples'], 1)
    return grp



def _write_gallery_readme(twc_dir: Path, manifest: pd.DataFrame) -> None:
    lines = [
        '# TWC plots',
        '',
        'GitHub does not render a gallery automatically for folders. This README embeds the generated figure PNGs so they are visible directly in the folder view.',
        '',
        'Each figure is available as **PNG**, **PDF**, and **CSV**.',
        '',
    ]
    for row in manifest.itertuples(index=False):
        lines.append(f"## {row.figure}: {row.title}")
        lines.append(f"[PNG]({row.png}) · [PDF]({row.pdf}) · [CSV]({row.csv})")
        lines.append('')
        lines.append(f"![{row.figure}]({row.png})")
        lines.append('')
    (twc_dir / 'README.md').write_text('\n'.join(lines), encoding='utf-8')

def build_twc_plots(summary_df: pd.DataFrame, raw_df: pd.DataFrame, output_dir: Path, paper_width: float = 7.2, paper_height: float = 4.8,
                    tail_summary_df: Optional[pd.DataFrame] = None, primary_baseline: str = 'bp_nms_20',
                    min_errors_to_plot: int = 30) -> pd.DataFrame:
    twc_dir = output_dir / 'TWC_plots'
    twc_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    def record(name: str, title: str):
        manifest_rows.append({'figure': name, 'title': title, 'pdf': f'{name}.pdf', 'png': f'{name}.png', 'csv': f'{name}.csv'})

    key_decoders = ['bp_nms_20', 'bp_nms_50', 'bp_wbf_post', 'bp_orb_rescue', 'bp_cdf_rescue', 'bp_segmented_rescue', 'hybrid_bp_nsg']
    ldpc_decoders = ['bp_10', 'bp_20', 'bp_50', 'bp_nms_20', 'bp_nms_50', 'hybrid_bp_nsg']
    rescue_decoders = ['bp_orb_rescue', 'bp_cdf_rescue', 'bp_segmented_rescue', 'hybrid_bp_nsg']
    profiles = sorted(summary_df['profile'].unique())

    fig, axes = plt.subplots(1, len(profiles), figsize=(paper_width * 1.4, paper_height), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, profiles):
        sub = summary_df[(summary_df.profile == p) & (summary_df.decoder.isin(key_decoders))]
        for dec in key_decoders:
            cur = sub[sub.decoder == dec].sort_values('snr_db')
            if cur.empty:
                continue
            ax.plot(cur.snr_db, _mask_bler(cur, min_errors_to_plot), marker='o', label=dec)
        ax.set_yscale('log')
        ax.set_title(f'Profile {p}')
        ax.set_xlabel('SNR (dB)')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('BLER')
    axes[-1].legend(fontsize=8)
    name='fig01_bler_profiles_log'
    _save_fig(fig, twc_dir / name, summary_df[summary_df.decoder.isin(key_decoders)])
    record(name, 'BLER vs SNR across profiles')

    fig, ax = plt.subplots(figsize=(paper_width, paper_height))
    sub = _aggregate_for_plot(summary_df[summary_df.decoder.isin(ldpc_decoders)])
    for dec in ldpc_decoders:
        cur = sub[sub.decoder == dec].sort_values('snr_db')
        if cur.empty:
            continue
        ax.plot(cur.snr_db, _mask_bler(cur, min_errors_to_plot), marker='o', label=dec)
    ax.set_yscale('log'); ax.grid(True, which='both', alpha=0.3); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Mean BLER')
    ax.legend(fontsize=8)
    name='fig02_bler_ldpc_log'
    _save_fig(fig, twc_dir / name, sub)
    record(name, 'BLER of BP-family baselines and hybrid')

    fig, ax = plt.subplots(figsize=(paper_width, paper_height))
    sub = _aggregate_for_plot(summary_df[summary_df.decoder.isin(rescue_decoders)])
    for dec in rescue_decoders:
        cur = sub[sub.decoder == dec].sort_values('snr_db')
        if cur.empty:
            continue
        ax.plot(cur.snr_db, _mask_bler(cur, min_errors_to_plot), marker='o', label=dec)
    ax.set_yscale('log'); ax.grid(True, which='both', alpha=0.3); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Mean BLER')
    ax.legend(fontsize=8)
    name='fig03_bler_grand_family_log'
    _save_fig(fig, twc_dir / name, sub)
    record(name, 'BLER of rescue-stage GRAND variants')

    fig, axes = plt.subplots(1, len(profiles), figsize=(paper_width * 1.4, paper_height), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, profiles):
        sub = summary_df[(summary_df.profile == p) & (summary_df.decoder.isin(key_decoders))]
        for dec in key_decoders:
            cur = sub[sub.decoder == dec].sort_values('snr_db')
            if cur.empty:
                continue
            ax.plot(cur.snr_db, cur.avg_latency_ms, marker='o', label=dec)
        ax.set_title(f'Profile {p}')
        ax.set_xlabel('SNR (dB)')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Average latency (ms)')
    axes[-1].legend(fontsize=8)
    name='fig04_avg_latency_profiles'
    _save_fig(fig, twc_dir / name, summary_df[summary_df.decoder.isin(key_decoders)])
    record(name, 'Average latency vs SNR across profiles')

    fig, axes = plt.subplots(1, len(profiles), figsize=(paper_width * 1.4, paper_height), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, profiles):
        sub = summary_df[(summary_df.profile == p) & (summary_df.decoder.isin(key_decoders))]
        for dec in key_decoders:
            cur = sub[sub.decoder == dec].sort_values('snr_db')
            if cur.empty:
                continue
            ax.plot(cur.snr_db, cur.p95_latency_ms, marker='o', label=dec)
        ax.set_title(f'Profile {p}')
        ax.set_xlabel('SNR (dB)')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('p95 latency (ms)')
    axes[-1].legend(fontsize=8)
    name='fig05_p95_latency_profiles'
    _save_fig(fig, twc_dir / name, summary_df[summary_df.decoder.isin(key_decoders)])
    record(name, 'p95 latency vs SNR across profiles')

    fig, ax = plt.subplots(figsize=(paper_width, paper_height))
    sub = _aggregate_for_plot(summary_df[summary_df.decoder.isin(rescue_decoders)])
    for dec in rescue_decoders:
        cur = sub[sub.decoder == dec].sort_values('snr_db')
        ax.plot(cur.snr_db, cur.avg_queries, marker='o', label=dec)
    ax.grid(True, alpha=0.3); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Average rescue queries')
    ax.legend(fontsize=8)
    name='fig06_avg_queries_rescue_family'
    _save_fig(fig, twc_dir / name, sub)
    record(name, 'Average rescue queries across GRAND variants')

    fig, ax = plt.subplots(figsize=(paper_width, paper_height))
    sub = _aggregate_for_plot(summary_df[summary_df.decoder.isin(key_decoders)]).copy()
    grouped = sub.groupby('decoder', as_index=False).agg({'bler':'mean', 'avg_latency_ms':'mean'})
    for _, row in grouped.iterrows():
        ax.scatter(row['avg_latency_ms'], max(row['bler'], 1e-7), label=row['decoder'])
        ax.annotate(row['decoder'], (row['avg_latency_ms'], max(row['bler'], 1e-7)))
    ax.set_yscale('log'); ax.grid(True, which='both', alpha=0.3); ax.set_xlabel('Average latency (ms)'); ax.set_ylabel('Mean BLER')
    name='fig07_bler_latency_tradeoff'
    _save_fig(fig, twc_dir / name, grouped)
    record(name, 'Mean BLER-latency tradeoff')

    base = summary_df[summary_df.decoder==primary_baseline][['profile','snr_db','bler','avg_latency_ms']].rename(columns={'bler':'ref_bler','avg_latency_ms':'ref_latency'})
    hyb = summary_df[summary_df.decoder=='hybrid_bp_nsg'][['profile','snr_db','bler','avg_latency_ms']].rename(columns={'bler':'hyb_bler','avg_latency_ms':'hyb_latency'})
    gain = base.merge(hyb, on=['profile','snr_db'])
    gain['bler_gain'] = gain['ref_bler'] - gain['hyb_bler']
    gain['latency_delta_ms'] = gain['hyb_latency'] - gain['ref_latency']
    fig, ax = plt.subplots(figsize=(paper_width, paper_height))
    for p in profiles:
        cur = gain[gain.profile==p]
        ax.plot(cur.snr_db, cur.bler_gain, marker='o', label=f'{p}: BLER gain')
    ax.axhline(0.0, color='k', linewidth=0.8)
    ax.grid(True, alpha=0.3); ax.set_xlabel('SNR (dB)'); ax.set_ylabel(f'Absolute BLER improvement vs {primary_baseline}')
    ax.legend(fontsize=8)
    name='fig08_hybrid_gain_vs_mainbp'
    _save_fig(fig, twc_dir / name, gain)
    record(name, f'Hybrid BLER gain vs {primary_baseline}')

    action_df = raw_df[raw_df.decoder=='hybrid_bp_nsg'].groupby(['profile','snr_db','action'], as_index=False).size()
    pivot = action_df.pivot_table(index=['profile','snr_db'], columns='action', values='size', fill_value=0).reset_index()
    count_cols = [c for c in pivot.columns if c not in {'profile','snr_db'}]
    pivot[count_cols] = pivot[count_cols].div(pivot[count_cols].sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(paper_width, paper_height))
    mean_action = pivot[count_cols].mean(axis=0).sort_values(ascending=False)
    ax.bar(mean_action.index, mean_action.values)
    ax.set_ylabel('Average fraction'); ax.set_xlabel('Hybrid action'); ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    name='fig09_action_breakdown'
    _save_fig(fig, twc_dir / name, action_df)
    record(name, 'Hybrid action breakdown')

    if tail_summary_df is None:
        tail_summary_df = summary_df[summary_df.snr_db >= 8]
    fig, ax = plt.subplots(figsize=(paper_width, paper_height))
    sub = _aggregate_for_plot(tail_summary_df[tail_summary_df.decoder.isin(key_decoders)])
    for dec in key_decoders:
        cur = sub[sub.decoder==dec].sort_values('snr_db')
        if cur.empty:
            continue
        ax.plot(cur.snr_db, _mask_bler(cur, min_errors_to_plot), marker='o', label=dec)
    ax.set_yscale('log'); ax.grid(True, which='both', alpha=0.3); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Mean BLER')
    ax.legend(fontsize=8)
    name='fig10_tail_high_snr_log'
    _save_fig(fig, twc_dir / name, sub)
    record(name, 'High-SNR tail BLER')

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(twc_dir / 'manifest.csv', index=False)
    _write_gallery_readme(twc_dir, manifest)
    return manifest
