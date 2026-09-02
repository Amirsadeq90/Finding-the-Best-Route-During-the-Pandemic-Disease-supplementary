"""
visualize_sensitivity.py
Generates the two-panel sensitivity analysis figure for the manuscript:
    Panel A - Monte Carlo route risk (median + 90% interval) per route
    Panel B - OAT tornado chart (mean effect of each parameter, averaged
              across all 11 routes)

Run oat_sensitivity.py and monte_carlo_sensitivity.py first (or run this
script directly -- it will call them itself if the CSVs are missing).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lipp_model import ROUTES
from oat_sensitivity import run_oat, summarize_oat
from monte_carlo_sensitivity import run_monte_carlo
from rank_position_analysis import compute_rank_positions


def load_or_compute():
    if os.path.exists('montecarlo_summary.csv'):
        mc_df = pd.read_csv('montecarlo_summary.csv')
    else:
        mc_df, _, _ = run_monte_carlo(n_draws=5000, seed=0)
        mc_df.to_csv('model/output/montecarlo_summary.csv', index=False)

    rank_df = compute_rank_positions(n_draws=5000, seed=0)

    if os.path.exists('oat_summary.csv'):
        oat_summary = pd.read_csv('oat_summary.csv')
    else:
        oat_df = run_oat()
        oat_summary = summarize_oat(oat_df)
        oat_summary.to_csv('model/output/oat_summary.csv', index=False)

    return mc_df.sort_values('Baseline %'), oat_summary, rank_df


def make_figure(mc_df, oat_df, rank_df, out_path='model/output/lipp_sensitivity_analysis.jpg'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))

    # ---- Panel A: Monte Carlo uncertainty per route ----
    ax = axes[0]
    y = np.arange(len(mc_df))
    ax.errorbar(
        mc_df['MC Median %'], y,
        xerr=[mc_df['MC Median %'] - mc_df['MC 5th pct %'],
              mc_df['MC 95th pct %'] - mc_df['MC Median %']],
        fmt='o', color='#2166ac', ecolor='#92c5de',
        elinewidth=4, capsize=3, markersize=6
    )
    ax.scatter(mc_df['Baseline %'], y, color='#b2182b', marker='|', s=200,
               label='Point estimate (baseline)', zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(mc_df['Route'])
    ax.set_xlabel('Infection risk (%)')
    ax.set_title('A. Route risk under joint parameter uncertainty\n'
                  '(median, 90% interval, N=5000 Monte Carlo draws)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # ---- Panel B: OAT tornado chart ----
    # Uses RELATIVE % change (not absolute percentage points). Absolute pp
    # change is dominated by high-baseline routes (e.g. R6/R11 car routes,
    # ~17% baseline) simply because they start from a larger number, which
    # would misrepresent parameter importance. Relative % change normalizes
    # for each route's own baseline risk, giving a fairer cross-route measure
    # of sensitivity -- and it is directly comparable to the OAT perturbation
    # sizes themselves (e.g. "+/-20% inhalation rate -> X% change in risk").
    ax2 = axes[1]
    deltas = oat_df['Mean % change'].values
    labels = oat_df['Scenario'].values
    order = np.argsort(np.abs(deltas))
    labels_sorted = labels[order]
    vals_sorted = deltas[order]
    colors = ['#b2182b' if v > 0 else '#2166ac' for v in vals_sorted]

    ax2.barh(range(len(vals_sorted)), vals_sorted, color=colors)
    ax2.set_yticks(range(len(vals_sorted)))
    ax2.set_yticklabels(labels_sorted, fontsize=9)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_xlabel('Mean relative change in route risk (%, avg. over 11 routes)')
    ax2.set_title('B. One-at-a-time sensitivity\n(relative % change, average across all 11 routes)')
    ax2.grid(axis='x', alpha=0.3)


    # ---- Panel C: Monte Carlo rank routes ----
    ax3 = axes[2]
    n_routes, n_positions = rank_df.shape
    routes = rank_df.index.tolist()
    positions = rank_df.columns.tolist()
    ordinal = {1: 'st', 2: 'nd', 3: 'rd'}
    pos_labels = [f"{p}{ordinal.get(int(p), 'th') if int(p) not in (11, 12, 13) else 'th'} choice"
                  for p in positions]

    cmap = plt.get_cmap('tab20', n_routes)
    colors = {r: cmap(i) for i, r in enumerate(routes)}

    y = np.arange(n_positions)
    left = np.zeros(n_positions)

    for r in routes:
        vals = rank_df.loc[r].values.astype(float)
        bars = ax3.barh(y, vals, left=left, color=colors[r], edgecolor='white',
                        linewidth=0.5, label=r)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= 5.0:
                ax3.text(l + v / 2, yi, f"{r}\n{v:.0f}%", ha='center', va='center',
                        fontsize=7.5, color='black')
        left += vals

    ax3.set_yticks(y)
    ax3.set_yticklabels(pos_labels)
    ax3.invert_yaxis()  # 1st choice at top
    ax3.set_xlim(0, 100)
    ax3.set_xlabel('Share of Monte Carlo draws (%)')
    ax3.set_title('Route rank-position distribution across 5,000 Monte Carlo draws\n'
                  '(which route is safest / 2nd-safest / ... under joint parameter uncertainty)')
    ax3.legend(title='Route', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax3.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    mc_df, oat_df, rank_df = load_or_compute()
    make_figure(mc_df, oat_df, rank_df)
