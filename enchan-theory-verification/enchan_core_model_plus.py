#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan Core Model Plus (v0.4.2 Fix)

This module implements the galaxy-calibrated *Phi-screening (pinning)* mechanism only.
It intentionally excludes any high-acceleration (g-screening) safeguard terms.

Key features
------------
1) Phi-screening only:
     a0_eff = a0_free * S_phi(|Phi|)
     S_phi(|Phi|) = [1 + (|Phi|/Phi_c)^n]^{-1}

2) Fail-fast behavior:
   - returns NaN when required inputs are missing or non-finite (when screening is enabled)
   - does not attempt extrapolation in the "local radius" proxy

3) Dual proxy support:
   - calculate_phi_at_radius: strict local proxy at a target radius (no extrapolation)
   - calculate_phi_median_proxy: robust median proxy (NaN-tolerant)

IMPORTANT
---------
- Do NOT pass g-screening parameters here.
  If g_bar/g_c/m are provided, this module raises.
- High-acceleration suppression is implemented separately in:
    enchan_core_model_g_screening.py
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

# =========================================================
# Constants & Defaults
# =========================================================

KPC_TO_M = 3.08567758e19

# Golden parameters (example defaults; may be overridden by callers)
DEFAULT_PHI_C = 40000.0  # (km/s)^2
DEFAULT_N = 1.0


# =========================================================
# 1) Proxy calculations (stateless, NumPy only)
# =========================================================

def calculate_phi_at_radius(
    r_kpc: np.ndarray,
    g_bar: np.ndarray,
    target_r_kpc: float,
) -> float:
    """
    Strict local potential-depth proxy at a specific radius:

        |Phi|(r) ~ V_bar(r)^2  with  V_bar^2 = g_bar(r) * r

    Inputs
    ------
    r_kpc : radius array [kpc]
    g_bar : baryonic acceleration array [m/s^2]
    target_r_kpc : target radius [kpc]

    Returns
    -------
    phi_proxy : float
        Proxy in (km/s)^2, or NaN if invalid / out-of-bounds.
        No extrapolation is performed.
    """
    r = np.asarray(r_kpc, dtype=float)
    g = np.asarray(g_bar, dtype=float)

    if r.size == 0 or r.shape != g.shape:
        return float("nan")

    # Fail-fast on non-finite arrays
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(g))):
        return float("nan")

    # Sort for interpolation
    sort_idx = np.argsort(r)
    r_sorted = r[sort_idx]
    g_sorted = g[sort_idx]

    # V^2 = g * r  (m^2/s^2); convert to (km/s)^2 by /1e6
    v2_profile = (g_sorted * (r_sorted * KPC_TO_M)) / 1.0e6

    # No extrapolation
    if target_r_kpc < r_sorted[0] or target_r_kpc > r_sorted[-1]:
        return float("nan")

    return float(np.interp(target_r_kpc, r_sorted, v2_profile))


def calculate_phi_median_proxy(
    r_kpc: np.ndarray,
    g_bar: np.ndarray,
    inner_cut_kpc: Optional[float] = None,
) -> float:
    """
    Robust galaxy-wide summary proxy (median) of |Phi| ~ V_bar^2.

    - NaN-tolerant: filters invalid points before taking median.
    - Optional inner_cut_kpc: uses only radii >= inner_cut_kpc.

    Returns
    -------
    phi_median : float
        Median proxy in (km/s)^2, or NaN if no valid points.
    """
    r = np.asarray(r_kpc, dtype=float)
    g = np.asarray(g_bar, dtype=float)

    if r.size == 0 or r.shape != g.shape:
        return float("nan")

    if inner_cut_kpc is not None:
        mask = r >= float(inner_cut_kpc)
        r = r[mask]
        g = g[mask]

    if r.size == 0:
        return float("nan")

    valid = np.isfinite(r) & np.isfinite(g)
    r = r[valid]
    g = g[valid]
    if r.size == 0:
        return float("nan")

    v2 = (g * (r * KPC_TO_M)) / 1.0e6
    return float(np.median(v2))


# =========================================================
# 2) Phi-screening (pinning) only
# =========================================================

def get_phi_screening_factor(
    phi_val: Union[float, np.ndarray],
    phi_c: float = DEFAULT_PHI_C,
    n: float = DEFAULT_N,
) -> Union[float, np.ndarray]:
    """
    Phi-screening (pinning) factor:

        S_phi(|Phi|) = [1 + (|Phi|/Phi_c)^n]^{-1}

    Conventions
    -----------
    - If phi_c is None or phi_c <= 0:
        screening is disabled and S_phi = 1 (phi_val is not validated).
    - If screening is enabled (phi_c > 0):
        phi_val must be provided and finite; otherwise returns NaN (fail-fast).

    Returns
    -------
    S_phi : float or np.ndarray
        Dimensionless screening factor in (0, 1].
    """
    if n is None or float(n) <= 0:
        raise ValueError(f"Phi-screening index 'n' must be > 0, got {n}")

    # Explicit disable: no screening
    if phi_c is None or float(phi_c) <= 0:
        if isinstance(phi_val, np.ndarray):
            return np.ones_like(phi_val, dtype=float)
        return 1.0

    if phi_val is None:
        return float("nan")

    if not np.all(np.isfinite(phi_val)):
        return float("nan")

    return 1.0 / (1.0 + (np.abs(phi_val) / float(phi_c)) ** float(n))


def apply_screening(
    a0_free: Union[float, np.ndarray],
    phi_val: Optional[Union[float, np.ndarray]] = None,
    phi_c: float = DEFAULT_PHI_C,
    n: float = DEFAULT_N,
    # Guard rails: reject g-screening args here
    g_bar: Optional[Union[float, np.ndarray]] = None,
    g_c: Optional[float] = None,
    m: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Apply Phi-screening to an input acceleration scale:

        a0_eff = a0_free * S_phi(|Phi|)

    Notes
    -----
    - This module implements Phi-screening only.
    - If phi_c is None or phi_c <= 0, screening is disabled and:
        a0_eff = a0_free   (phi_val not required)
    - If screening is enabled (phi_c > 0), phi_val must be provided and finite;
      otherwise returns NaN (fail-fast).
    """
    # Prevent accidental mixing with g-screening
    if g_bar is not None or g_c is not None or m is not None:
        raise ValueError(
            "g-screening arguments detected (g_bar/g_c/m). "
            "Use enchan_core_model_g_screening.py for high-acceleration suppression."
        )

    if not np.all(np.isfinite(a0_free)):
        return float("nan")

    # Explicit disable: no screening, phi_val not required
    if phi_c is None or float(phi_c) <= 0:
        return a0_free

    if phi_val is None:
        return float("nan")

    s_phi = get_phi_screening_factor(phi_val=phi_val, phi_c=phi_c, n=n)
    return a0_free * s_phi
