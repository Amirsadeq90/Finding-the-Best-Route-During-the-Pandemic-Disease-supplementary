"""
rank_position_analysis.py
Rank-position stability analysis for the LIPP Monte Carlo sensitivity run.

For every Monte Carlo draw (same joint-uncertainty design as
monte_carlo_sensitivity.py), routes are ranked 1st (safest / lowest risk)
through 11th (riskiest) by their simulated infection risk. This script
tallies, for each rank position, what fraction of the 5000 draws placed
each route there -- e.g. "1st choice: R1 50%, R6 30%, R11 20%" -- and
renders it as a stacked horizontal bar chart.

Outputs:
    rank_position_summary.csv     -- routes x rank-position % table
    rank_position_distribution.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lipp_model import ROUTES, RHO_BASE, route_risk
from monte_carlo_sensitivity import PARAM_RANGES


def compute_rank_positions(routes=ROUTES, n_draws=5000, seed=0,
                            param_ranges=PARAM_RANGES):
    """
    Runs the same joint Monte Carlo draws as monte_carlo_sensitivity.py and,
    for each draw, records which route lands in each rank position
    (position 0 = safest ... position 10 = riskiest).

    Returns
    -------
    pct_table : DataFrame, index=route, columns=rank position (1..N),
                values = percentage of draws (0-100) the route occupied
                that rank.
    """
    rng = np.random.default_rng(seed)
    route_names = list(routes.keys())
    n_routes = len(route_names)

    # counts[route][position] = number of draws where `route` finished
    # in that rank position
    counts = {r: np.zeros(n_routes, dtype=int) for r in route_names}

    for _ in range(n_draws):
        rho = RHO_BASE * rng.uniform(*param_ranges['rho_range'])
        E_scale = rng.uniform(*param_ranges['E_scale_range'])
        r_scale = rng.uniform(*param_ranges['r_scale_range'])
        cap_scale = rng.uniform(*param_ranges['cap_scale_range'])

        risks = {
            r: route_risk(routes[r], rho=rho, E_scale=E_scale,
                           r_scale=r_scale, cap_scale=cap_scale)
            for r in route_names
        }
        # sort ascending by risk: index 0 = safest = rank 1
        ordered = sorted(risks, key=risks.get)
        for position, r in enumerate(ordered):
            counts[r][position] += 1

    pct = {r: counts[r] / n_draws * 100 for r in route_names}
    pct_table = pd.DataFrame(pct).T
    pct_table.columns = [f'{i + 1}' for i in range(n_routes)]
    pct_table.index.name = 'Route'

    # Order routes by their modal (most frequent) rank position, so the
    # legend / stacking order roughly follows typical route safety order
    modal_rank = pct_table.idxmax(axis=1).astype(int)
    pct_table = pct_table.loc[modal_rank.sort_values().index]

    return pct_table


def plot_rank_positions(pct_table, out_path='model/output/rank_position_distribution.jpg',
                         min_label_pct=5.0):
    """
    Stacked horizontal bar chart: one bar per rank position (1st..Nth
    choice), segments = % of Monte Carlo draws each route occupied that
    position.
    """
    n_routes, n_positions = pct_table.shape
    routes = pct_table.index.tolist()
    positions = pct_table.columns.tolist()
    print('roooooo',routes)
    print('possssssssssss',positions)

    ordinal = {1: 'st', 2: 'nd', 3: 'rd'}
    pos_labels = [f"{p}{ordinal.get(int(p), 'th') if int(p) not in (11, 12, 13) else 'th'} choice"
                  for p in positions]

    cmap = plt.get_cmap('tab20', n_routes)
    colors = {r: cmap(i) for i, r in enumerate(routes)}

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(n_positions)
    left = np.zeros(n_positions)

    for r in routes:
        vals = pct_table.loc[r].values.astype(float)
        bars = ax.barh(y, vals, left=left, color=colors[r], edgecolor='white',
                        linewidth=0.5, label=r)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= min_label_pct:
                ax.text(l + v / 2, yi, f"{r}\n{v:.0f}%", ha='center', va='center',
                        fontsize=7.5, color='black')
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(pos_labels)
    ax.invert_yaxis()  # 1st choice at top
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of Monte Carlo draws (%)')
    ax.set_title('Route rank-position distribution across 5,000 Monte Carlo draws\n'
                  '(which route is safest / 2nd-safest / ... under joint parameter uncertainty)')
    ax.legend(title='Route', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    pct_table = compute_rank_positions(n_draws=5000, seed=0)
    pct_table.to_csv('model/output/rank_position_summary.csv')

    print("Rank-position distribution (%, rows=route, cols=rank position):")
    print(pct_table.round(1).to_string())

    plot_rank_positions(pct_table)
