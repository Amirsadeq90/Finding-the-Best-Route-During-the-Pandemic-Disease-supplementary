"""
ablation_study.py
Ablation study for the LIPP model, tested on the 11 empirical Tehran
routes (Table 4) only.

Each ablation replaces ONE structural component of the full model with a
deliberately simplified/naive version, holding everything else fixed, to
test whether that component is actually contributing to the model's
conclusions or is unnecessary complexity.

Ablations:
    A1 - Geometry ablation: replace Eq. (20)'s mode-specific r_mean with
         a single flat distance (the mean r_mean across all five modes),
         removing the vehicle-geometry differentiation.
    A2 - Coefficient ablation: replace the five mode-specific (a_i, b_i)
         pairs (Table 1) with a single universal (a, b) averaged across
         modes, removing ventilation/mode differentiation.
    A3 - Activity ablation: replace the activity-dependent inhalation
         rate (moderate for walking, light for transit) with a single
         constant E averaged across the two categories.
    A4 - Composition ablation: replace the multiplicative route-risk
         formula (Theorem 1, Eq. 1) with a naive additive sum of segment
         risks, removing the independent-segments probability logic.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lipp_model import ROUTES, COEF, GEOM, RHO_BASE, E_WALK, E_TRANSIT

# ---------------------------------------------------------------------
# Ablated parameter sets, derived from the full model's own values
# (so ablations are principled averages, not arbitrary picks)
# ---------------------------------------------------------------------
FLAT_R_MEAN = float(np.mean([g['r_mean'] for g in GEOM.values()]))          # ~3.99 m
UNIVERSAL_A = float(np.mean([COEF[m]['a'] for m in COEF]))                  # avg of 5 a_i (car's a=0)
UNIVERSAL_B = float(np.mean([COEF[m]['b'] for m in COEF]))                  # avg of 5 b_i
UNIVERSAL_E = float(np.mean([E_WALK, E_TRANSIT]))                           # avg of 1740 & 780 = 1260


def segment_risk_ablation(mode, t_hours, rho=RHO_BASE,
                           ablate_geometry=False, ablate_coefficients=False,
                           ablate_activity=False):
    """Segment-level infection risk, with optional ablations applied."""
    if mode == 'transfer':
        return 0.0

    a, b = (UNIVERSAL_A, UNIVERSAL_B) if ablate_coefficients else (COEF[mode]['a'], COEF[mode]['b'])
    r = FLAT_R_MEAN if ablate_geometry else GEOM[mode]['r_mean']
    cap = GEOM[mode]['capacity']
    n = rho * cap

    if ablate_activity:
        E = UNIVERSAL_E
    else:
        E = E_WALK if mode == 'pedestrian' else E_TRANSIT

    exponent = (a * E + b) * n * t_hours / (r ** 2)
    return 1.0 - np.exp(exponent)


def route_risk_ablation(segments, rho=RHO_BASE, ablate_geometry=False,
                         ablate_coefficients=False, ablate_activity=False,
                         ablate_composition=False):
    """
    Route-level infection risk, with optional ablations applied.
    ablate_composition=True replaces the multiplicative combination
    (Theorem 1) with a naive additive sum of segment risks.
    """
    seg_risks = [
        segment_risk_ablation(mode, t, rho=rho,
                               ablate_geometry=ablate_geometry,
                               ablate_coefficients=ablate_coefficients,
                               ablate_activity=ablate_activity)
        for mode, t in segments
    ]
    if ablate_composition:
        return min(1.0, sum(seg_risks))       # naive additive sum, capped at 1
    else:
        survive = 1.0
        for p in seg_risks:
            survive *= (1 - p)
        return 1.0 - survive


# ---------------------------------------------------------------------
# Model variants: full model + one ablation at a time
# ---------------------------------------------------------------------
VARIANTS = {
    'Full model':                dict(),
    'A1: No geometry (flat r)':  dict(ablate_geometry=True),
    'A2: No mode-specific coef.': dict(ablate_coefficients=True),
    'A3: No activity dependence': dict(ablate_activity=True),
    'A4: Additive composition':  dict(ablate_composition=True),
}


def run_ablation_study(routes=ROUTES, variants=VARIANTS):
    rows = []
    for variant_name, kwargs in variants.items():
        for route_id, segs in routes.items():
            risk_pct = route_risk_ablation(segs, **kwargs) * 100
            rows.append({'Variant': variant_name, 'Route': route_id, 'Risk %': risk_pct})
    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot(index='Route', columns='Variant', values='Risk %')
    # keep a sensible route order (sorted by full-model baseline risk)
    wide_df = wide_df.loc[wide_df['Full model'].sort_values().index]
    return wide_df


def summarize_ablation(wide_df, baseline_col='Full model'):
    baseline = wide_df[baseline_col]
    baseline_rank = baseline.rank()
    baseline_best = baseline.idxmin()

    rows = []
    for col in wide_df.columns:
        if col == baseline_col:
            continue
        vals = wide_df[col]
        deltas = vals - baseline
        pct_change = deltas / baseline * 100
        rank_corr, _ = spearmanr(baseline_rank, vals.rank())
        best_route_here = vals.idxmin()
        rows.append({
            'Ablation': col,
            'Mean |delta| (pp)': deltas.abs().mean(),
            'Max |delta| (pp)': deltas.abs().max(),
            'Mean |% change|': pct_change.abs().mean(),
            'Max |% change|': pct_change.abs().max(),
            'Spearman rank corr. vs. full model': rank_corr,
            'Best route (full model)': baseline_best,
            'Best route (this ablation)': best_route_here,
            'Best route changed?': best_route_here != baseline_best,
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print(f"Flat r_mean used in A1: {FLAT_R_MEAN:.3f} m "
          f"(mean of pedestrian/subway/brt/bus/car r_mean)")
    print(f"Universal (a,b) used in A2: a={UNIVERSAL_A:.6f}, b={UNIVERSAL_B:.6f}")
    print(f"Universal E used in A3: {UNIVERSAL_E:.1f} L/hr")
    print()

    wide_df = run_ablation_study()
    print("Route risk (%) under each model variant:")
    print(wide_df.round(3).to_string())
    print()

    summary_df = summarize_ablation(wide_df)
    print("Ablation impact summary (vs. full model):")
    print(summary_df.round(3).to_string(index=False))

    wide_df.to_csv('model/output/ablation_route_risks.csv')
    summary_df.to_csv('model/output/ablation_summary.csv', index=False)
