# Finding the Best Route During the Pandemic Disease — Supplementary Materials

This repository contains supplementary code, data, and figures related to the "Finding the Best Route During the Pandemic Disease" research project, which develops the **LIPP (Least Infection Probability Path)** model for computing segment- and route-level infection risk across different transportation modes (pedestrian, subway, BRT, bus, car) and identifying the safest route between two points during a pandemic.

**Data availability note:** A preliminary version of this work is available as a preprint. Some of the results in this repository (e.g., the sensitivity, uncertainty, and ablation analyses in `model/`) were produced for a subsequent, revised manuscript that is currently under review at a peer-reviewed journal; this repository is provided to reviewers as supplementary material for that submission. Full citation details will be added once the review process is complete.

## Repository structure

```
.
├── infection_risk.ipynb        # Main analysis notebook: risk computation and
│                                  route comparison for the Tehran case study
├── CovidRiskPerMeter.xlsx       # Underlying per-meter infection risk data
├── Eng/                         # Figures (English-labeled) used in the manuscript
│   ├── CovidRiskPerMeter.jpg
│   ├── InfectionRiskComparisonAcrossRoutes.jpg
│   ├── Infection_Spread_Simulation.jpg
│   ├── SideWalk_Length_Improved.jpg
│   └── TransmissionRiskByRouteAndTransportType.jpg
└── model/                       # Standalone Python implementation of the LIPP
    │                             model and the robustness analyses (sensitivity,
    │                             ablation, uncertainty)
    ├── lipp_model.py             # Core risk model (Eqs. 17–18) and the 11
    │                             empirical Tehran routes (Table 4)
    ├── oat_sensitivity.py        # One-at-a-time (OAT) sensitivity analysis
    ├── monte_carlo_sensitivity.py# Joint Monte Carlo uncertainty analysis
    ├── rank_position_analysis.py # Route-ranking stability under uncertainty
    ├── ablation_study.py         # Ablation study (A1–A4) on model components
    ├── visualize_sensitivity.py  # Figure generation for the sensitivity results
    ├── visualize_ablation.py     # Figure generation for the ablation results
    └── output/                   # Generated CSVs and figures from the scripts above
```

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `python-bidi` (for rendering right-to-left text labels in the notebook)

Install with:

```bash
pip install numpy pandas scipy matplotlib python-bidi
```

## Reproducing the results

**Main case study (Tehran routes, notebook):**

```bash
jupyter notebook infection_risk.ipynb
```

**Sensitivity, uncertainty, and ablation analyses:**

```bash
cd model
python oat_sensitivity.py            # -> output/oat_sensitivity_full.csv, output/oat_summary.csv
python monte_carlo_sensitivity.py    # -> output/montecarlo_summary.csv, output/montecarlo_raw_draws.csv
python rank_position_analysis.py     # -> output/rank_position_summary.csv, output/rank_position_distribution.jpg
python ablation_study.py             # -> output/ablation_route_risks.csv, output/ablation_summary.csv
python visualize_sensitivity.py      # -> output/ combined sensitivity figure
python visualize_ablation.py         # -> output/ablation_study.jpg
```

All scripts can also be run directly; each will regenerate its required inputs if the corresponding CSVs are missing.

## Contents summary

- **`lipp_model.py`** implements the transmission-risk equations from Table 1 and Eqs. (17)–(18) of the manuscript for five transport modes (pedestrian, subway, BRT, bus, car).
- **Sensitivity analysis** (`oat_sensitivity.py`, `monte_carlo_sensitivity.py`, `rank_position_analysis.py`) tests how robust the route rankings are to uncertainty in prevalence, inhalation rate, interpersonal distance, and occupancy.
- **Ablation study** (`ablation_study.py`) tests whether each structural component of the model (mode-specific geometry, mode-specific coefficients, activity-dependent inhalation rate, multiplicative route-risk composition) is necessary, by replacing it with a simplified alternative and measuring the effect on results.

## Citation

Citation details will be added once the manuscript is published. In the meantime, please contact the authors if you wish to reference this code or the associated results.
