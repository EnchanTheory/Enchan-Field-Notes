#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan Solar System Delta-Acceleration Diagnostic (v0.4.2 Fix)

Purpose
-------
Quantify the *anomalous* (non-Newtonian) sunward acceleration predicted by the
Enchan-style benchmark mapping (MOND-like closure used for reproducibility):

    g_enchan = sqrt(g_N^2 + a0_eff(r) * g_N)

This script supports the v0.4.2 "pinning-only in the main note" policy:

- Phi-screening / "pinning" (galaxy-calibrated regime extension):
      a0_eff(r) = a0_free * S_phi(|Phi|)

- Optional high-acceleration suppression (Solar-System extrapolation safeguard):
      a0_eff(r) = a0_phi(r) * S_g(g_N)

By default, high-acceleration suppression is OFF (g_c=None). Enable it explicitly via --g_c.

Important units
---------------
- Potential-depth proxy is implemented as |Phi| ~ V_circ^2 in units of (km/s)^2
- Therefore --phi_c must also be in (km/s)^2
- a0, g_N, delta_g are in (m/s^2)

Usage
-----
High-acc suppression ON:
    python enchan_solar_system_delta_accel.py --g_c 6e-11 --m 1.0

High-acc suppression OFF (default):
    python enchan_solar_system_delta_accel.py

Disable Phi-screening explicitly (pinning OFF):
    python enchan_solar_system_delta_accel.py --phi_c 0

Outputs
-------
- CSV with radius, Newtonian acceleration, boost, delta_g, phantom mass proxy, and screening factors
- Diagnostic plots (boost, delta_g) if matplotlib is available

Notes
-----
- This is a diagnostic, not a full ephemeris fit.
- "phantom mass" is the point-mass equivalent that would reproduce delta_g at radius r:
      M_ph(r) = delta_g(r) * r^2 / G
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
AU_M = 1.495978707e11
G_SI = 6.67430e-11
M_EARTH = 5.9722e24

# Solar gravitational parameter (m^3/s^2)
GM_SUN = 1.32712440018e20


# ---------------------------------------------------------------------
# Core imports (friendly error if modules are missing / path is wrong)
# ---------------------------------------------------------------------
try:
    # Phi-screening ("pinning") only
    from enchan_core_model_plus import apply_screening as apply_phi_screening
    # Optional high-acceleration suppression (Solar-System safeguard)
    from enchan_core_model_g_screening import apply_g_screening
except ImportError as e:
    raise SystemExit(
        "ERROR: Missing Enchan core modules.\n"
        "Run this script from the enchan-theory-verification/ directory, or fix PYTHONPATH.\n"
        f"Detail: {e}"
    )


@dataclass(frozen=True)
class BodyPoint:
    name: str
    r_au: float


DEFAULT_POINTS: List[BodyPoint] = [
    BodyPoint("Mercury", 0.387),
    BodyPoint("Earth", 1.0),
    BodyPoint("Jupiter", 5.204),
    BodyPoint("Pluto", 39.48),
    BodyPoint("Sedna", 506.0),
    BodyPoint("Planet 9?", 700.0),
]


def enchan_boost(g_newton: np.ndarray, a0_eff: np.ndarray) -> np.ndarray:
    """boost = g_enchan / g_N = sqrt(1 + a0_eff/g_N)."""
    g_newton = np.maximum(g_newton, 1e-30)
    return np.sqrt(1.0 + (a0_eff / g_newton))


def compute_profile(
    r_au: np.ndarray,
    a0_free: float,
    phi_c: float,
    n: float,
    g_c: Optional[float],
    m: float,
) -> pd.DataFrame:
    r_m = r_au * AU_M

    # Newtonian acceleration (m/s^2)
    g_N = GM_SUN / np.maximum(r_m, 1e-30) ** 2

    # Potential-depth proxy using circular speed squared:
    # v_circ^2 = GM/r in (m^2/s^2) -> (km/s)^2 by /1e6
    v_circ_sq_kms2 = (GM_SUN / np.maximum(r_m, 1e-30)) / 1.0e6

    # --------------------------------------------------------------
    # Phi-screening ("pinning") in the acceleration scale
    #
    # apply_phi_screening() in v0.4.2 Fix supports:
    #   - if phi_c <= 0 : screening disabled, returns a0_free (phi_val not required)
    #   - else          : phi_val must be finite
    #
    # Use positional args for broad compatibility.
    # --------------------------------------------------------------
    a0_phi = apply_phi_screening(a0_free, v_circ_sq_kms2, phi_c, n)
    a0_phi = np.asarray(a0_phi, dtype=float)

    # Screening factor (relative to a0_free)
    # If phi_c<=0, a0_phi==a0_free and S_phi==1.
    S_phi = a0_phi / float(a0_free)

    # --------------------------------------------------------------
    # Optional high-acceleration suppression (Solar-System safeguard)
    # --------------------------------------------------------------
    if g_c is not None and g_c > 0:
        a0_eff = apply_g_screening(a0_phi, g_N, g_c, m)
        a0_eff = np.asarray(a0_eff, dtype=float)
        S_g = a0_eff / np.maximum(a0_phi, 1e-300)
    else:
        a0_eff = a0_phi
        S_g = np.ones_like(g_N, dtype=float)

    S_total = a0_eff / float(a0_free)

    boost = enchan_boost(g_N, a0_eff)
    g_en = boost * g_N
    delta_g = g_en - g_N  # m/s^2

    # Point-mass equivalent ("phantom") in Earth masses:
    M_ph_kg = delta_g * (r_m ** 2) / G_SI
    M_ph_Me = M_ph_kg / M_EARTH

    return pd.DataFrame({
        "r_AU": r_au,
        "r_m": r_m,
        "g_newton": g_N,
        "v_circ_sq_kms2": v_circ_sq_kms2,
        "S_phi": S_phi,
        "S_g": S_g,
        "S_total": S_total,
        "a0_free_ms2": float(a0_free),
        "a0_phi_ms2": a0_phi,
        "a0_eff_ms2": a0_eff,
        "boost": boost,
        "g_enchan": g_en,
        "delta_g": delta_g,
        "phantom_Me": M_ph_Me,
    })


def sample_points(df: pd.DataFrame, points: List[BodyPoint]) -> pd.DataFrame:
    """Nearest-neighbor sample at the named radii."""
    r = df["r_AU"].values
    rows = []
    for p in points:
        j = int(np.argmin(np.abs(r - p.r_au)))
        rows.append({
            "Body": p.name,
            "r(AU)": float(df.loc[j, "r_AU"]),
            "boost": float(df.loc[j, "boost"]),
            "Δg (m/s²)": float(df.loc[j, "delta_g"]),
            "phantom (M⊕)": float(df.loc[j, "phantom_Me"]),
            "S_phi": float(df.loc[j, "S_phi"]),
            "S_g": float(df.loc[j, "S_g"]),
            "S_total": float(df.loc[j, "S_total"]),
        })
    return pd.DataFrame(rows)


def plot_boost(df: pd.DataFrame, pts: List[BodyPoint], out_png: str) -> None:
    if not HAS_PLT:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(df["r_AU"].values, df["boost"].values)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("r (AU)")
    plt.ylabel("boost = g_enchan / g_N")
    plt.title("Enchan boost profile (diagnostic)")
    for p in pts:
        plt.axvline(p.r_au, linestyle="--", linewidth=1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_delta_g(df: pd.DataFrame, pts: List[BodyPoint], out_png: str) -> None:
    if not HAS_PLT:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(df["r_AU"].values, np.abs(df["delta_g"].values))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("r (AU)")
    plt.ylabel("|Δg| (m/s²)")
    plt.title("Enchan |Δg| profile (diagnostic)")
    for p in pts:
        plt.axvline(p.r_au, linestyle="--", linewidth=1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Enchan Solar System delta-acceleration profile (v0.4.2 Fix)")
    ap.add_argument("--a0", type=float, default=1.2e-10, help="Baseline a0_free (m/s^2)")
    ap.add_argument("--phi_c", type=float, default=40000.0,
                    help="Phi-screening threshold Phi_c in (km/s)^2. Set <=0 to disable Phi-screening.")
    ap.add_argument("--n", type=float, default=1.0, help="Phi-screening index n>0")
    ap.add_argument("--g_c", type=float, default=None,
                    help="Optional high-acceleration suppression scale g_c in (m/s^2). Omit to disable.")
    ap.add_argument("--m", type=float, default=1.0, help="High-acceleration suppression index m>0")
    ap.add_argument("--rmin", type=float, default=0.1, help="Min radius (AU)")
    ap.add_argument("--rmax", type=float, default=1.0e4, help="Max radius (AU)")
    ap.add_argument("--npts", type=int, default=2000, help="Number of sample points (log-spaced)")
    ap.add_argument("--out", type=str, default="enchan_solar_system_delta_accel", help="Output prefix")
    ap.add_argument("--no_plot", action="store_true", help="Disable plots even if matplotlib is available")
    args = ap.parse_args()

    # Input sanity checks (strict: avoid silent garbage)
    if args.a0 <= 0:
        raise SystemExit("ERROR: --a0 must be > 0")
    if args.n <= 0:
        raise SystemExit("ERROR: --n must be > 0")
    if args.g_c is not None and args.g_c <= 0:
        raise SystemExit("ERROR: --g_c must be > 0 if provided")
    if args.m <= 0:
        raise SystemExit("ERROR: --m must be > 0")
    if args.rmin <= 0 or args.rmax <= 0 or args.rmax <= args.rmin:
        raise SystemExit("ERROR: require 0 < --rmin < --rmax")
    if args.npts < 50:
        raise SystemExit("ERROR: --npts too small (>=50 recommended)")

    r_au = np.logspace(math.log10(args.rmin), math.log10(args.rmax), int(args.npts))

    df = compute_profile(
        r_au=r_au,
        a0_free=float(args.a0),
        phi_c=float(args.phi_c),
        n=float(args.n),
        g_c=args.g_c if args.g_c is None else float(args.g_c),
        m=float(args.m),
    )

    out_csv = f"{args.out}.csv"
    out_boost_png = f"{args.out}_boost.png"
    out_dg_png = f"{args.out}_delta_g.png"

    df.to_csv(out_csv, index=False)

    print("\n--- Enchan Solar System Delta-Acceleration (v0.4.2 Fix) ---")
    pinning_state = "OFF" if args.phi_c <= 0 else "ON"
    gsup_state = "OFF" if args.g_c is None else "ON"

    if gsup_state == "OFF":
        print(
            f"Params: a0={args.a0:.3e} m/s^2 | Phi-screening={pinning_state} "
            f"(Phi_c={args.phi_c:.1f} (km/s)^2, n={args.n:.2f}) | high-acc suppr={gsup_state}"
        )
    else:
        print(
            f"Params: a0={args.a0:.3e} m/s^2 | Phi-screening={pinning_state} "
            f"(Phi_c={args.phi_c:.1f} (km/s)^2, n={args.n:.2f}) | "
            f"high-acc suppr={gsup_state} (g_c={args.g_c:.3e} m/s^2, m={args.m:.2f})"
        )

    tab = sample_points(df, DEFAULT_POINTS)
    print(tab[["Body", "r(AU)", "boost", "Δg (m/s²)", "phantom (M⊕)", "S_phi", "S_g", "S_total"]].to_string(index=False))

    if (not args.no_plot) and HAS_PLT:
        plot_boost(df, DEFAULT_POINTS, out_boost_png)
        plot_delta_g(df, DEFAULT_POINTS, out_dg_png)
        print(f"\nSaved: {out_boost_png}")
        print(f"Saved: {out_dg_png}")
    elif not HAS_PLT and (not args.no_plot):
        print("NOTE: matplotlib not available; skipping plots.")
    else:
        print("Plots disabled (--no_plot).")

    print(f"\nSaved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
