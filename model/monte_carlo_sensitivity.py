"""
monte_carlo_sensitivity.py
Joint Monte Carlo sensitivity / uncertainty analysis for the LIPP model.

Draws all four uncertain parameters simultaneously (prevalence, inhalation
rate, mean interpersonal distance, occupancy) from independent uniform
distributions and recomputes every route's infection risk for each draw.
Reports:
    - median risk and 90% simulation interval per route
    - probability each route is the single safest option
    - Spearman rank correlation between each draw's route ranking and the
      baseline ranking (route-ranking stability under uncertainty)

Outputs:
    montecarlo_summary.csv
    montecarlo_raw_draws.csv   (optional, full N x routes matrix)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lipp_model import ROUTES, RHO_BASE, route_risk

# Uncertainty ranges (uniform distributions), consistent with the OAT ranges
PARAM_RANGES = dict(
    rho_range=(0.70, 1.30),        # multiplicative range on RHO_BASE
    E_scale_range=(0.80, 1.20),
    r_scale_range=(0.85, 1.15),
    cap_scale_range=(0.75, 1.25),
)


def run_monte_carlo(routes=ROUTES, n_draws=5000, seed=0, param_ranges=PARAM_RANGES):
    rng = np.random.default_rng(seed)
    route_names = list(routes.keys())

    baseline = {r: route_risk(routes[r]) * 100 for r in route_names}
    baseline_rank = pd.Series(baseline).rank()

    draws = {r: np.empty(n_draws) for r in route_names}
    rank_corr = np.empty(n_draws)
    safest_counts = {r: 0 for r in route_names}

    for i in range(n_draws):
        rho = RHO_BASE * rng.uniform(*param_ranges['rho_range'])
        E_scale = rng.uniform(*param_ranges['E_scale_range'])
        r_scale = rng.uniform(*param_ranges['r_scale_range'])
        cap_scale = rng.uniform(*param_ranges['cap_scale_range'])

        risks = {
            r: route_risk(routes[r], rho=rho, E_scale=E_scale,
                           r_scale=r_scale, cap_scale=cap_scale) * 100
            for r in route_names
        }
        for r in route_names:
            draws[r][i] = risks[r]

        ranks = pd.Series(risks).rank()
        rank_corr[i], _ = spearmanr(baseline_rank, ranks)

        safest_route = min(risks, key=risks.get)
        safest_counts[safest_route] += 1

    summary_rows = []
    for r in route_names:
        arr = draws[r]
        summary_rows.append({
            'Route': r,
            'Baseline %': baseline[r],
            'MC Median %': np.median(arr),
            'MC 5th pct %': np.percentile(arr, 5),
            'MC 95th pct %': np.percentile(arr, 95),
            'P(safest route) %': safest_counts[r] / n_draws * 100,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values('Baseline %').reset_index(drop=True)

    rank_stats = {
        'mean_spearman': float(np.mean(rank_corr)),
        'p05_spearman': float(np.percentile(rank_corr, 5)),
        'p95_spearman': float(np.percentile(rank_corr, 95)),
    }

    draws_df = pd.DataFrame(draws)
    return summary_df, rank_stats, draws_df


if __name__ == '__main__':
    summary_df, rank_stats, draws_df = run_monte_carlo(n_draws=5000, seed=0)

    summary_df.to_csv('model/output/montecarlo_summary.csv', index=False)
    draws_df.to_csv('model/output/montecarlo_raw_draws.csv', index=False)

    print("Monte Carlo route risk summary:")
    print(summary_df.round(3).to_string(index=False))
    print()
    print(f"Mean Spearman rank correlation vs. baseline: {rank_stats['mean_spearman']:.3f}")
    print(f"5th-95th percentile: {rank_stats['p05_spearman']:.3f} - {rank_stats['p95_spearman']:.3f}")
