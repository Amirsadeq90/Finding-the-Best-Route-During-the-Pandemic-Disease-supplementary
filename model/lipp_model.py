"""
lipp_model.py
Core LIPP (Least Infection Probability Path) risk model.

Implements the segment- and route-level infection risk equations from
Mirgalooyebayat & Didehvar (2025), "Finding the Best Route During the
Pandemic Disease" (arXiv:2510.02396), Eqs. (17)-(18) and the "Risk
Computation" formulas (Routing Algorithm Implementation section).
"""

import math
import numpy as np

# ---------------------------------------------------------------------
# Table 1: environment-specific transmission coefficients (a_i, b_i)
# ---------------------------------------------------------------------
COEF = {
    'pedestrian': dict(a=-0.001439, b=0.714554),
    'subway':     dict(a=-0.001801, b=0.824029),
    'brt':        dict(a=-0.001491, b=0.388753),
    'bus':        dict(a=-0.002203, b=0.565659),
    'car':        dict(a=0.0,       b=-2.729480),   # Table 1: a_car is undefined ("-")
}


def av_dist(w, h):
    """
    Mean interpersonal distance for a w x h rectangular microenvironment
    (Eq. 20, Burgstaller & Pillichshammer 2009 / Mathai et al. 1999).
    """
    d = math.sqrt(w ** 2 + h ** 2)
    return 1 / 15 * (
        math.pow(w, 3) / math.pow(h, 2) + math.pow(h, 3) / math.pow(w, 2)
        + d * (3 - math.pow(w, 2) / math.pow(h, 2) - math.pow(h, 2) / math.pow(w, 2))
        + 5 / 2 * (h ** 2 / w * math.log((w + d) / h, 10)
                   + w ** 2 / h * math.log((h + d) / w, 10))
    )


# ---------------------------------------------------------------------
# Table 2: microenvironment geometry / capacity
# r_mean is computed directly from Eq. 20 (av_dist) using each
# microenvironment's (width, length) in metres, rather than the rounded
# constants in Table 2. Values reproduce Table 2 to within rounding
# (e.g. pedestrian: 7.367 here vs. 7.37 reported).
# ---------------------------------------------------------------------
GEOM = {
    'pedestrian': dict(r_mean=av_dist(4, 30.3), capacity=40),
    'subway':     dict(r_mean=av_dist(2.6, 19.52), capacity=180),
    'brt':        dict(r_mean=av_dist(2.55, 17.9), capacity=150),
    'bus':        dict(r_mean=av_dist(2.55, 12), capacity=80),
    'car':        dict(r_mean=av_dist(1.2, 1.5), capacity=4),
}

# Activity-level inhalation rates (Table 3)
E_WALK = 1740.0      # moderate activity, used for pedestrian segments
E_TRANSIT = 780.0    # light activity, used for subway/BRT/bus/car segments

# Baseline prevalence (Sept 2021 Tehran case study, rho = active cases / population)
RHO_BASE = 0.007682


def segment_risk(mode, t_hours, rho=RHO_BASE, E_override=None,
                  r_override=None, cap_override=None):
    """
    Infection probability for a single microenvironment segment (Eq. 17-18,
    validated sign convention -- see module docstring).

    Parameters
    ----------
    mode : str
        One of 'pedestrian', 'subway', 'brt', 'bus', 'car'.
    t_hours : float
        Exposure duration in hours.
    rho : float
        Disease prevalence (active cases / population).
    E_override, r_override, cap_override : float, optional
        Override the default inhalation rate, mean interpersonal distance,
        or vehicle/pathway capacity for this call (used by sensitivity
        analyses to perturb individual parameters).
    """
    if mode == 'transfer':
        # Interchange/transfer time (Eq. 20 does not define a microenvironment
        # for transfers; treated as zero-risk time cost only, per the paper's
        # "5-minute transfer risk" note, which is not otherwise quantified).
        return 0.0

    a, b = COEF[mode]['a'], COEF[mode]['b']
    r = r_override if r_override is not None else GEOM[mode]['r_mean']
    cap = cap_override if cap_override is not None else GEOM[mode]['capacity']
    n = rho * cap
    E_default = E_WALK if mode == 'pedestrian' else E_TRANSIT
    E = E_override if E_override is not None else E_default

    exponent = (a * E + b) * n * t_hours / (r ** 2)
    return 1.0 - np.exp(exponent)


def route_risk(segments, rho=RHO_BASE, E_scale=1.0, r_scale=1.0, cap_scale=1.0):
    """
    Composite route infection probability (Eq. 1 / Theorem 1):
        P_route = 1 - prod_i (1 - P_i)

    Parameters
    ----------
    segments : list of (mode, t_hours) tuples describing the route.
    rho : float
        Prevalence to use for every segment in this route.
    E_scale, r_scale, cap_scale : float
        Multiplicative scaling factors applied uniformly to the default
        inhalation rate, interpersonal distance, and capacity of every
        segment in the route. Used for one-at-a-time and Monte Carlo
        sensitivity analysis.
    """
    survive_prob = 1.0
    for mode, t in segments:
        if mode == 'transfer':
            continue  # zero-risk time cost, no GEOM/COEF entry to scale
        E_default = E_WALK if mode == 'pedestrian' else E_TRANSIT
        r = GEOM[mode]['r_mean'] * r_scale
        cap = GEOM[mode]['capacity'] * cap_scale
        p = segment_risk(mode, t, rho=rho,
                          E_override=E_default * E_scale,
                          r_override=r, cap_override=cap)
        survive_prob *= (1.0 - p)
    return 1.0 - survive_prob


def minutes(m):
    return m / 60.0


WALK_SPEED_M_PER_MIN = 5000.0 / 60.0   # 5 km/h
STOP_SPACING_MIN = 3.0                  # average station spacing
TRANSFER_MIN = 5.0                      # interchange time


def walk_min(meters):
    """Convert a walking distance in metres to time in minutes at 5 km/h."""
    return meters / WALK_SPEED_M_PER_MIN


def stops_min(n_stops):
    """Convert a number of stops to time in minutes at 3 min/stop."""
    return n_stops * STOP_SPACING_MIN


# ---------------------------------------------------------------------
# Tehran case-study routes (Table 4), reconstructed segment-by-segment
# from the exact distances/stop counts reported for each route. Walking
# segments use 5 km/h; transit segments use 3 min/stop; R10's explicit
# interchange uses the 5-minute transfer time. R9's segments are given
# directly in minutes in the source table.
# ---------------------------------------------------------------------
ROUTES = {
    # R1 -- Neshan -- Pedestrian only -- 8.2 km, 96 min, 1 segment
    'R1': [('pedestrian', minutes(96))],

    # R2 -- Neshan -- Walk(126m) + City Bus(18 stops) + Walk(1080m) -- 3 segments
    'R2': [('pedestrian', minutes(walk_min(126))),
           ('bus', minutes(stops_min(18))),
           ('pedestrian', minutes(walk_min(1080)))],

    # R3 -- Neshan -- Walk(461m) + City Bus(17 stops) + Walk(1490m) -- 3 segments
    'R3': [('pedestrian', minutes(walk_min(461))),
           ('bus', minutes(stops_min(17))),
           ('pedestrian', minutes(walk_min(1490)))],

    # R4 -- Neshan -- Walk(190m) + BRT(2 stops) + Walk(618m) + Subway(6 stops) + Walk(1020m) -- 5 segments
    'R4': [('pedestrian', minutes(walk_min(190))),
           ('brt', minutes(stops_min(2))),
           ('pedestrian', minutes(walk_min(618))),
           ('subway', minutes(stops_min(6))),
           ('pedestrian', minutes(walk_min(1020)))],

    # R5 -- Neshan -- Walk(190m) + City Bus(2 stops) + Walk(105m) + BRT(9 stops) + Walk(1090m) -- 5 segments
    'R5': [('pedestrian', minutes(walk_min(190))),
           ('bus', minutes(stops_min(2))),
           ('pedestrian', minutes(walk_min(105))),
           ('brt', minutes(stops_min(9))),
           ('pedestrian', minutes(walk_min(1090)))],

    # R6 -- Neshan -- Car only -- 7.3 km, 28 min, 1 segment
    'R6': [('car', minutes(28))],

    # R7 -- Balad -- Pedestrian only -- 8.4 km, 105 min, 1 segment
    'R7': [('pedestrian', minutes(105))],

    # R8 -- Balad -- Walk(100m) + City Bus(18 stops) + Walk(1000m) -- 3 segments
    'R8': [('pedestrian', minutes(walk_min(100))),
           ('bus', minutes(stops_min(18))),
           ('pedestrian', minutes(walk_min(1000)))],

    # R9 -- Balad -- Walk(7min) + Car(8min) + Walk(4min) + Car(7min) + Walk(4min) -- 5 segments
    'R9': [('pedestrian', minutes(7)),
           ('car', minutes(8)),
           ('pedestrian', minutes(4)),
           ('car', minutes(7)),
           ('pedestrian', minutes(4))],

    # R10 -- Balad -- Walk(1100m) + Subway(3 stops) + Transfer + Subway(3 stops) + Walk(1300m) -- 5 segments
    'R10': [('pedestrian', minutes(walk_min(1100))),
            ('subway', minutes(stops_min(3))),
            ('transfer', minutes(TRANSFER_MIN)),
            ('subway', minutes(stops_min(3))),
            ('pedestrian', minutes(walk_min(1300)))],

    # R11 -- Balad -- Car only -- 7.2 km, 27 min, 1 segment
    'R11': [('car', minutes(27))],
}

ROUTE_META = {
    'R1':  dict(source='Neshan', distance_km=8.2, duration_min=96, segments=1),
    'R2':  dict(source='Neshan', distance_km=7.5, duration_min=58, segments=3),
    'R3':  dict(source='Neshan', distance_km=7.8, duration_min=62, segments=3),
    'R4':  dict(source='Neshan', distance_km=6.8, duration_min=52, segments=5),
    'R5':  dict(source='Neshan', distance_km=7.1, duration_min=55, segments=5),
    'R6':  dict(source='Neshan', distance_km=7.3, duration_min=28, segments=1),
    'R7':  dict(source='Balad',  distance_km=8.4, duration_min=105, segments=1),
    'R8':  dict(source='Balad',  distance_km=7.6, duration_min=59, segments=3),
    'R9':  dict(source='Balad',  distance_km=7.4, duration_min=32, segments=5),
    'R10': dict(source='Balad',  distance_km=6.9, duration_min=48, segments=5),
    'R11': dict(source='Balad',  distance_km=7.2, duration_min=27, segments=1),
}


if __name__ == '__main__':
    print(f"{'Route':6}{'Risk %':>10}")
    for r, segs in ROUTES.items():
        p = route_risk(segs) * 100
        print(f"{r:6}{p:10.2f}")
