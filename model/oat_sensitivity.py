"""
oat_sensitivity.py
One-at-a-time (OAT) sensitivity analysis for the LIPP model.

Perturbs one parameter at a time (prevalence, inhalation rate, mean
interpersonal distance, occupancy/capacity) while holding the others at
baseline, and reports the resulting change in route-level infection risk
for all 11 Tehran case-study routes.

Outputs:
    oat_sensitivity_full.csv  -- per-route risk under every scenario
    oat_summary.csv           -- mean/max effect of each parameter, averaged
                                  across all 11 routes
"""

import pandas as pd
from lipp_model import ROUTES, route_risk, RHO_BASE

# Perturbation scenarios: (label -> kwargs passed to route_risk)
SCENARIOS = {
    'Prevalence -30%':            dict(rho=RHO_BASE * 0.70),
    'Prevalence +30%':            dict(rho=RHO_BASE * 1.30),
    'Inhalation rate -20%':       dict(E_scale=0.80),
    'Inhalation rate +20%':       dict(E_scale=1.20),
    'Interpersonal dist. -15%':   dict(r_scale=0.85),
    'Interpersonal dist. +15%':   dict(r_scale=1.15),
    'Occupancy -25%':             dict(cap_scale=0.75),
    'Occupancy +25% (rush-hour)': dict(cap_scale=1.25),
}


def run_oat(routes=ROUTES, scenarios=SCENARIOS):
    rows = []
    for route_id, segs in routes.items():
        baseline_pct = route_risk(segs) * 100
        row = {'Route': route_id, 'Baseline %': baseline_pct}
        for name, kwargs in scenarios.items():
            perturbed_pct = route_risk(segs, **kwargs) * 100
            row[name] = perturbed_pct
            row[f'{name} (delta pp)'] = perturbed_pct - baseline_pct
        rows.append(row)
    return pd.DataFrame(rows).set_index('Route')


def summarize_oat(oat_df, scenarios=SCENARIOS):
    summary_rows = []
    for name in scenarios:
        delta_col = f'{name} (delta pp)'
        deltas = oat_df[delta_col]
        pct_change = deltas / oat_df['Baseline %'] * 100
        summary_rows.append({
            'Scenario': name,
            'Mean delta risk (pp)': deltas.mean(),
            'Max |delta risk| (pp)': deltas.abs().max(),
            'Mean % change': pct_change.mean(),
            'Max |% change|': pct_change.abs().max(),
        })
    return pd.DataFrame(summary_rows)


if __name__ == '__main__':
    oat_df = run_oat()
    summary_df = summarize_oat(oat_df)

    oat_df.to_csv('model/output/oat_sensitivity_full.csv')
    summary_df.to_csv('model/output/oat_summary.csv', index=False)

    print("Per-route OAT results (first few columns):")
    print(oat_df.iloc[:, :4].round(2).to_string())
    print()
    print("OAT summary (averaged across all 11 routes):")
    print(summary_df.round(2).to_string(index=False))
