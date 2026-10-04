"""Render the complete dated exploratory results; no raw EEG or new analysis."""
import csv
import hashlib
import json
import os
from pathlib import Path

import argparse
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT/'outputs/exploratory')
args = parser.parse_args()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = ROOT / 'results/exploratory_2026-09-20'
MAN = args.output
for folder in ['figures', 'tables', 'sections']:
    (MAN/folder).mkdir(parents=True, exist_ok=True)
s = json.loads((OUT / 'summary_results.json').read_text())
o = s['order_control']
cases = s['simulation_scenarios']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
    'axes.titlesize': 12.5, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
blue, orange = '#23567A', '#B86124'
crosswalk = []

def fmt(location, key, value, digits=4):
    text = f'{value:.{digits}f}'
    crosswalk.append(dict(location=location, key=key, raw_value=value,
                          decimal_places=digits, displayed=text))
    return text

def save(fig, stem):
    for suffix in ['pdf', 'png']:
        fig.savefig(MAN / 'figures' / f'{stem}.{suffix}', dpi=600,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)

with np.load(OUT / 'order_permutations.npz', allow_pickle=False) as data:
    delta = data['delta_z']
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1),
                         gridspec_kw={'width_ratios': [1, 1.15]}, layout='constrained')
axes[0].hist(delta, bins=30, color=blue, edgecolor='white', linewidth=.5)
axes[0].axvline(0, color='0.35', linestyle='--', linewidth=1)
axes[0].set(xlabel=r'Contrast $\Delta z$', ylabel='Randomizations',
            title='a  Randomized cell order')
observed = s['observed_complete_support']['delta_z']
lo, hi = o['central_95_range']
axes[1].errorbar(o['mean_delta_z'], 0,
    xerr=[[o['mean_delta_z']-lo], [hi-o['mean_delta_z']]],
    fmt='o', color=blue, capsize=4, markersize=6)
axes[1].plot(observed, 1, 'D', color=orange, markersize=6)
axes[1].axvline(0, color='0.35', linestyle='--', linewidth=1)
axes[1].set(yticks=[0, 1], yticklabels=['Reordered', 'Original order'], ylim=(-.5, 1.6),
            xlabel=r'Contrast $\Delta z$', title='b  Contrast comparison')
axes[1].text(observed, 1.2, f'{observed:.4f}', ha='center', color=orange)
axes[1].text(o['mean_delta_z'], .2, f"{o['mean_delta_z']:.4f}", ha='center', color=blue,
             bbox=dict(facecolor='white', edgecolor='none', pad=1))
axes[1].margins(x=.16)
save(fig, 'figure_s2_temporal_order_control')

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.45), sharey=True, layout='constrained')
plot_records = []
for ax, tau, panel in zip(axes, [.5, 1.0], ['a', 'b']):
    for gamma, color, label, offset in [(0, blue, 'Stationary (no drift)', -.012),
                                       (2, orange, 'Random slopes (SD = 2)', .012)]:
        rows = [r for r in cases if r['between_sd'] == tau and r['slope_sd'] == gamma]
        means = np.array([r['mean_delta_z'] for r in rows])
        bounds = np.array([r['central_95_range'] for r in rows]).T
        ax.errorbar(np.array([r['phi'] for r in rows])+offset, means,
                    yerr=np.vstack([means-bounds[0], bounds[1]-means]),
                    color=color, fmt='o-', capsize=4, label=label, linewidth=1.5)
        plot_records.extend(rows)
    ax.axhline(0, color='0.4', linestyle='--', linewidth=.8)
    ax.set(xticks=[0, .5, .9], xlabel=r'Serial-dependence coefficient $\phi$',
           title=rf'{panel}  Between-participant SD $\tau={tau:.1f}$')
    ax.set_xlim(-.07, .98)
axes[0].set_ylabel(r'Contrast $\Delta z$')
axes[0].legend(loc='upper left', frameon=False, fontsize=12.5)
save(fig, 'figure_s3_stationary_and_drift')

lines = [r'\begin{table}[!htbp]', r'\centering\small',
    r'\caption{All exploratory cell-band-power simulation scenarios. Each row summarizes 500 cohorts of 516 participants.}',
    r'\label{tab:s3-simulation}', r'\setlength{\tabcolsep}{3pt}',
    r'\begin{tabular}{rrrrrrcr}', r'\toprule',
    r'$\phi$ & $\gamma$ & \shortstack{Contiguous\\sampling\\$\bar\rho$} & \shortstack{Ten-stratum\\distributed\\sampling $\bar\rho$} & $\overline{\Delta z}$ & SD & Central 95\% range & MCSE \\',
    r'\midrule']
for tau in [.5, 1.0]:
    lines.append(r'\multicolumn{8}{l}{Between-participant SD $\tau='+f'{tau:.1f}'+r'$} \\')
    for r in [c for c in cases if c['between_sd'] == tau]:
        loc = f"Table S3 scenario {r['scenario_id']}"
        fields = [f"{r['phi']:.1f}", f"{r['slope_sd']:.0f}"]
        fields += [fmt(loc, k, r[k]) for k in ['mean_rho_C', 'mean_rho_F10', 'mean_delta_z', 'sample_sd_delta_z']]
        fields += ['['+fmt(loc, 'range_low', r['central_95_range'][0])+', '+fmt(loc, 'range_high', r['central_95_range'][1])+']',
                   fmt(loc, 'mcse_mean_delta_z', r['mcse_mean_delta_z'], 6)]
        lines.append(' & '.join(fields)+r' \\')
    if tau == .5: lines.append(r'\midrule')
lines += [r'\bottomrule', r'\end{tabular}',
    r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize',
    r'$\phi$: stationary autoregressive coefficient; $\gamma$: SD of participant-specific linear slopes. Strategy correlations are back-transformed Fisher means within each cohort, then averaged across cohorts. SD and MCSE refer to the cohort-level contrast; MCSE is SD$/\sqrt{500}$. Ranges are 2.5th--97.5th percentiles across generated cohorts, not confidence intervals. All values are dimensionless.',
    r'\end{minipage}', r'\end{table}', '']
(MAN / 'tables/table_s3_exploratory_simulation.tex').write_text('\n'.join(lines), encoding='utf-8')

loc = 'Supplementary temporal-order results'
text = ('The original-order complete-support contrast was $\\Delta z='+fmt(loc, 'observed_delta_z', observed)+
        '$. Across 1000 randomizations, the mean contrast was '+fmt(loc, 'mean_delta_z', o['mean_delta_z'])+
        ', with SD '+fmt(loc, 'sample_sd_delta_z', o['sample_sd_delta_z'])+
        ', a central 95\\% range of ['+fmt(loc, 'range_low', lo)+', '+fmt(loc, 'range_high', hi)+
        '], and Monte Carlo SE of the mean '+fmt(loc, 'mcse_mean_delta_z', o['mcse_mean_delta_z'], 6)+
        '. Mean back-transformed reproducibility across randomizations was '+fmt(loc, 'mean_rho_C', o['mean_rho_C'])+
        ' for contiguous sampling and '+fmt(loc, 'mean_rho_F10', o['mean_rho_F10'])+
        ' for ten-stratum distributed sampling (Figure~\\ref{fig:s2-order}). These values describe the fixed retained cell spectra under randomized ordering.\n')
(MAN / 'sections/exploratory_order_summary.tex').write_text(text, encoding='utf-8')

with (MAN / 'numerical_display.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(crosswalk[0]))
    writer.writeheader(); writer.writerows(crosswalk)
(MAN / 'figure_data.json').write_text(json.dumps(dict(order_control=o,
    observed_delta_z=observed, simulation=plot_records), indent=2)+'\n', encoding='utf-8')
print('Generated Figures S2-S3, Table S3 and numerical source data in', MAN)
