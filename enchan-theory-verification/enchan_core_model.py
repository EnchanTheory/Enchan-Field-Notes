#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan core model utilities (v0.4.5)

This module provides:
- Shared physical constants (SI / unit conversions)
- The baseline "quadrature closure" used as a reproducibility benchmark:
      g_pred = sqrt(g_bar^2 + a0 * g_bar)
- BTFR-implied a0 helper:
      a0 = Vf^4 / (G Mb)
- Numerically stable transition function mu(x) (and optional nu(y))

Scope / status
--------------
These utilities are used by the public-data reproducibility scripts.
They are *not* a claim of a unique fundamental derivation. They implement the
benchmark mapping and helper diagnostics used throughout the repository.

Units
-----
- Accelerations: m/s^2
- Velocities: km/s in tables, converted to m/s internally
- Masses: Msun in tables, converted to kg internally
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------
# Constants (CODATA/IAU standard values where applicable)
# ---------------------------------------------------------------------
G_SI: float = 6.67430e-11       # m^3 kg^-1 s^-2
MSUN_KG: float = 1.98847e30     # kg
KMS_TO_MS: float = 1000.0       # (km/s) -> (m/s)


# ---------------------------------------------------------------------
# Benchmark closure (RAR-style mapping used in reproducibility scripts)
# ---------------------------------------------------------------------
def g_pred_quadrature(g_bar: np.ndarray, a0: float) -> np.ndarray:
    """
    Enchan quadrature closure (benchmark mapping):
        g_pred = sqrt(g_bar^2 + a0*g_bar)

    Notes:
    - This is used as a compact empirical closure in the reproducibility package.
    - For safety and numerical stability, negative/invalid g_bar are clipped to 0.
    - a0 must be finite and >= 0; otherwise output will be NaN.

    Parameters
    ----------
    g_bar : array-like
        Baryonic Newtonian acceleration (m/s^2).
    a0 : float
        Acceleration scale (m/s^2).

    Returns
    -------
    np.ndarray
        Predicted total acceleration (m/s^2).
    """
    gb = np.asarray(g_bar, dtype=float)
    a0f = float(a0)

    out = np.full_like(gb, np.nan, dtype=float)

    if not np.isfinite(a0f) or a0f < 0:
        return out

    gb = np.where(np.isfinite(gb), gb, np.nan)
    gb = np.maximum(gb, 0.0)

    # sqrt(gb^2 + a0*gb) is safe for gb>=0 and a0>=0
    out = np.sqrt(gb * gb + a0f * gb)
    return out


# ---------------------------------------------------------------------
# BTFR-implied a0 (one galaxy -> one a0)
# ---------------------------------------------------------------------
def a0_from_btfr(Vf_kms: np.ndarray, Mb_msun: np.ndarray) -> np.ndarray:
    """
    Compute BTFR-implied acceleration scale per galaxy:

        a0 = (Vf^4) / (G * Mb)

    with Vf in km/s and Mb in Msun (converted internally to SI).

    Parameters
    ----------
    Vf_kms : array-like
        Flat/outer velocity in km/s.
    Mb_msun : array-like
        Baryonic mass in solar masses.

    Returns
    -------
    np.ndarray
        a0 in m/s^2; NaN where inputs are invalid.
    """
    Vf = np.asarray(Vf_kms, dtype=float) * KMS_TO_MS
    Mb = np.asarray(Mb_msun, dtype=float) * MSUN_KG

    a0 = np.full_like(Vf, np.nan, dtype=float)
    good = np.isfinite(Vf) & np.isfinite(Mb) & (Vf > 0) & (Mb > 0)
    a0[good] = (Vf[good] ** 4) / (G_SI * Mb[good])
    return a0


# ---------------------------------------------------------------------
# Robust percentiles helper
# ---------------------------------------------------------------------
def robust_percentiles(
    x: np.ndarray,
    q: Tuple[float, float, float] = (16.0, 50.0, 84.0)
) -> Tuple[float, float, float]:
    """Return (p16, p50, p84) ignoring NaNs/infs."""
    xx = np.asarray(x, dtype=float)
    xx = xx[np.isfinite(xx)]
    if xx.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = np.percentile(xx, list(q))
    return float(p[0]), float(p[1]), float(p[2])


# ---------------------------------------------------------------------
# Transition functions (numerically stable)
# ---------------------------------------------------------------------
def mu_function(x: np.ndarray) -> np.ndarray:
    """
    Transition function mu(x), written in a numerically stable form.

    Interpretation (benchmark compatibility):
        g_bar = mu(x) * g_tot
        x = g_tot / a0

    Stable rationalized form:
        mu(x) = 2x / (sqrt(1 + 4x^2) + 1)

    This avoids catastrophic cancellation for x << 1.

    Parameters
    ----------
    x : array-like
        Dimensionless ratio g_tot/a0.

    Returns
    -------
    np.ndarray
        mu(x) in [0,1] for x>=0; NaN where invalid.
    """
    xx = np.asarray(x, dtype=float)
    out = np.full_like(xx, np.nan, dtype=float)
    good = np.isfinite(xx) & (xx >= 0)
    if not np.any(good):
        return out
    xg = xx[good]
    out[good] = (2.0 * xg) / (np.sqrt(1.0 + 4.0 * xg**2) + 1.0)
    return out


def nu_function(y: np.ndarray) -> np.ndarray:
    """
    Companion function nu(y) often used in RAR-style formulations:
        g_tot = g_bar * nu(y)
        y = g_bar / a0

    For the quadrature closure used here:
        g_tot = sqrt(g_bar^2 + a0*g_bar) = g_bar * sqrt(1 + 1/y)
    so:
        nu(y) = sqrt(1 + 1/y)

    Parameters
    ----------
    y : array-like
        Dimensionless ratio g_bar/a0.

    Returns
    -------
    np.ndarray
        nu(y); NaN where invalid; for y<=0 returns NaN.
    """
    yy = np.asarray(y, dtype=float)
    out = np.full_like(yy, np.nan, dtype=float)
    good = np.isfinite(yy) & (yy > 0)
    if not np.any(good):
        return out
    yg = yy[good]
    out[good] = np.sqrt(1.0 + 1.0 / yg)
    return out
