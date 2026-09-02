"""
visualize_ablation.py
Two-panel figure for the ablation study:
    Panel A - route risk (%) under the full model vs. each ablation,
              one line per model variant, routes on the x-axis (sorted
              by full-model baseline risk) -- shows where and how much
              conclusions shift when a component is removed.
    Panel B - Spearman rank correlation of each ablation's route ranking
              against the full model's ranking -- a single-number
              summary of how "necessary" each component is.
"""

import numpy as np
import matplotlib.pyplot as plt
from ablation_study import run_ablation_study, summarize_ablation


def make_figure(out_path='model/output/ablation_study.jpg'):
    wide_df = run_ablation_study()
    summary_df = summarize_ablation(wide_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Panel A: route risk under each variant ----
    ax = axes[0]
    colors = {
        'Full model': '#1a1a1a',
        'A1: No geometry (flat r)': '#d73027',
        'A2: No mode-specific coef.': '#fc8d59',
        'A3: No activity dependence': '#fee090',
        'A4: Additive composition': '#4575b4',
    }
    styles = {
        'Full model': dict(linewidth=3, linestyle='-', marker='o', zorder=5),
        'A1: No geometry (flat r)': dict(linewidth=1.8, linestyle='--', marker='s'),
        'A2: No mode-specific coef.': dict(linewidth=1.8, linestyle='--', marker='^'),
        'A3: No activity dependence': dict(linewidth=1.8, linestyle='--', marker='d'),
        'A4: Additive composition': dict(linewidth=1.8, linestyle=':', marker='x'),
    }

    x = np.arange(len(wide_df))
    for col in wide_df.columns:
        ax.plot(x, wide_df[col], color=colors[col], label=col, **styles[col])

    ax.set_xticks(x)
    ax.set_xticklabels(wide_df.index, fontsize=9)
    ax.set_ylabel('Route infection risk (%)')
    ax.set_xlabel('Route (sorted by full-model baseline risk)')
    ax.set_title('A. Route risk: full model vs. each ablation')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    # ---- Panel B: Spearman rank correlation per ablation ----
    ax2 = axes[1]
    abl = summary_df.sort_values('Spearman rank corr. vs. full model')
    bar_colors = ['#d73027' if v < 0.7 else ('#fc8d59' if v < 0.95 else '#1a9850')
                  for v in abl['Spearman rank corr. vs. full model']]
    ax2.barh(abl['Ablation'], abl['Spearman rank corr. vs. full model'], color=bar_colors)
    ax2.axvline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Spearman rank correlation vs. full model ranking')
    ax2.set_title('B. How much each ablation disrupts\nroute rankings (1.0 = no change)')
    ax2.set_xlim(-1, 1.1)
    ax2.grid(axis='x', alpha=0.3)

    for i, (val, changed) in enumerate(zip(abl['Spearman rank corr. vs. full model'],
                                            abl['Best route changed?'])):
        label = 'best route CHANGED' if changed else 'best route unchanged'
        ax2.text(val + 0.03 if val >= 0 else val - 0.03, i, label,
                 va='center', ha='left' if val >= 0 else 'right', fontsize=7.5, style='italic')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    make_figure()
