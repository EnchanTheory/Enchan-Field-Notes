#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan core model utilities (v0.3.0)

This module defines the Enchan "Quadrature Closure" derived from the 
Topological Defect Lagrangian in Enchan Theory v0.3.0.

Theoretical Basis
-----------------
In Enchan v0.3.0, the "Inception World" hypothesis and the logarithmic defect solution 
(S ~ ln r) lead to an effective total acceleration g_pred given by:

    g_pred = sqrt( g_bar^2 + a0 * g_bar )

Where:
- g_bar: Baryonic Newtonian acceleration (Anchor source)
- a0: Critical acceleration scale derived from vacuum stiffness (sigma_vac) 
      and defect core energy (V0).

This closure connects the Newtonian regime (g_bar >> a0) with the 
Geometric Dark Matter regime (g_bar << a0), reproducing the MONDian phenomenology 
from a purely geometric origin (topological defects in the time-dilation field).

Constants
---------
All accelerations are in SI (m/s^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
MSUN_KG = 1.98847e30        # kg
KMS_TO_MS = 1000.0          # (km/s) -> (m/s)


def g_pred_quadrature(g_bar: np.ndarray, a0: float) -> np.ndarray:
    """
    Enchan quadrature law (v0.3.0 verified): 
    g_pred = sqrt(g_bar^2 + a0*g_bar).
    
    This form is theoretically derived from the 'Anchor Condition' 
    of topological defects in the Enchan field (Chapter 6 of Field Notes v0.3.0).
    It represents the geometric sum of the Newtonian acceleration and 
    the stress energy of the vacuum defect.
    """
    gb = np.asarray(g_bar, dtype=float)
    a0 = float(a0)
    return np.sqrt(gb * gb + a0 * gb)


def a0_from_btfr(Vf_kms: np.ndarray, Mb_msun: np.ndarray) -> np.ndarray:
    """
    Calculate the implied acceleration scale a0 per galaxy from BTFR data.
    
    Theory (v0.3.0) predicts: 
        v^4 = G * Mb * a0
    
    This function inverts the relation to find the observed a0 for each galaxy,
    verifying the universality of the defect core parameter.
    
    Inputs:
      Vf_kms: flat velocity in km/s
      Mb_msun: baryonic mass in solar masses

    Output:
      a0 in m/s^2 (array)
    """
    Vf = np.asarray(Vf_kms, dtype=float) * KMS_TO_MS
    Mb = np.asarray(Mb_msun, dtype=float) * MSUN_KG
    good = (Vf > 0) & (Mb > 0) & np.isfinite(Vf) & np.isfinite(Mb)
    a0 = np.full_like(Vf, np.nan, dtype=float)
    a0[good] = (Vf[good] ** 4) / (G_SI * Mb[good])
    return a0


def robust_percentiles(x: np.ndarray, q: Tuple[float, float, float] = (16.0, 50.0, 84.0)) -> Tuple[float, float, float]:
    """Return (p16, p50, p84) ignoring NaNs."""
    xx = np.asarray(x, dtype=float)
    xx = xx[np.isfinite(xx)]
    if xx.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = np.percentile(xx, list(q))
    return float(p[0]), float(p[1]), float(p[2])

def mu_function(x: np.ndarray) -> np.ndarray:
    """
    Enchan Transition Function mu(x).
    
    Relates the total acceleration to the baryonic acceleration:
        g_bar = mu(x) * g_tot
    where x = g_tot / a0.

    This implementation uses the rationalized form:
        mu(x) = 2x / (sqrt(1 + 4x^2) + 1)
    
    This form is mathematically equivalent to (sqrt(1+4x^2)-1)/2x but
    is numerically stable for small x (x << 1), avoiding catastrophic
    cancellation in the Deep MOND regime.
    """
    return 2 * x / (np.sqrt(1 + 4 * x**2) + 1)