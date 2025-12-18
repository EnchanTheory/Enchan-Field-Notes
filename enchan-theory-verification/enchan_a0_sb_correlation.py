#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan a0 vs Surface Brightness Correlation (v0.4.2 Final)

Exploratory test of the "Surface Density Anchor" ansatz (Chapter 6).
This script checks for correlations between:
1. a0 derived from BTFR (a0 ~ Vf^4 / Mb)
2. Surface Brightness proxy from Rotmod (SB_disk at inner radii)

Methodology & Robustness:
- Primary Proxy: Median of the innermost 3 positive SB_disk points.
- Robustness Checks:
  1. Trimmed Correlation: Excludes top/bottom 5% of outliers to check stability.
  2. Proxy Sensitivity: Checks if using innermost 5 points changes the result.
- Significance: Uses SciPy if available, otherwise runs a Permutation Test (N=10,000)
  with (k+1)/(n+1) correction to avoid p=0.

Pre-registered Interpretation Scenarios:
  P1 (Minimal Anchor): Weak positive correlation. (Supports Enchan ansatz)
  P2 (Self-Regulation): Near-zero correlation. (Implies cancellations)
  P3 (Proxy Limit): Unstable/noisy correlation. (Proxy/data quality dominance)

Usage:
  python enchan_a0_sb_correlation.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
import time
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import SciPy, else use manual permutation test
try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Ensure local imports work
sys.path.append(str(Path(__file__).parent))

try:
    from enchan_core_model import G_SI, MSUN_KG, KMS_TO_MS
    from enchan_btfr_reproduce_enchan import parse_mrt_fixedwidth, extract_btfr
except ImportError:
    print("Error: Helper modules (enchan_core_model, etc.) not found.")
    sys.exit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def norm_name(s: str) -> str:
    s = str(s).upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    m = re.match(r"^([A-Z]+)(\d+)$", s)
    if m:
        prefix, num = m.group(1), m.group(2)
        s = f"{prefix}{int(num)}"
    return s

# --- Statistical Helpers ---

def permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 10000) -> float:
    """
    Computes a two-sided p-value for Pearson correlation using permutation.
    Robust and dependency-free.
    Uses (count + 1) / (n_perm + 1) to avoid p=0.
    """
    if len(x) < 3: return float("nan")
    
    # Observed correlation
    r_obs = np.corrcoef(x, y)[0, 1]
    
    # Pre-calculate centered arrays
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    ss_x = np.sum(x_c**2)
    ss_y = np.sum(y_c**2)
    denom = np.sqrt(ss_x * ss_y)
    
    if denom == 0: return float("nan")
    
    count_extreme = 0
    y_perm = y_c.copy()
    
    # Use fixed seed for reproducibility
    rng = np.random.default_rng(42)
    
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        r_perm = np.sum(x_c * y_perm) / denom
        if abs(r_perm) >= abs(r_obs):
            count_extreme += 1
            
    # Conservative p-value estimate
    return (count_extreme + 1) / (n_perm + 1)

def get_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Calculate Pearson/Spearman and p-values (SciPy or Permutation)."""
    stats = {}
    if len(x) < 3:
        return {"r": np.nan, "p": np.nan, "rho": np.nan, "rho_p": np.nan}

    # Pearson
    if HAS_SCIPY:
        r, p = pearsonr(x, y)
        stats["r"] = r
        stats["p"] = p
    else:
        stats["r"] = np.corrcoef(x, y)[0, 1]
        stats["p"] = permutation_test(x, y)

    # Spearman
    if HAS_SCIPY:
        rho, rho_p = spearmanr(x, y)
        stats["rho"] = rho
        stats["rho_p"] = rho_p
    else:
        # Manual Spearman (rank correlation)
        rx = pd.Series(x).rank()
        ry = pd.Series(y).rank()
        stats["rho"] = np.corrcoef(rx, ry)[0, 1]
        stats["rho_p"] = np.nan # Permutation for Spearman is expensive, skip

    return stats

# --- Data Extraction ---

def get_sb_proxy(zip_path: Path, n_points: int = 3) -> pd.DataFrame:
    """
    Extracts SB proxy.
    n_points: Number of innermost points to take median of.
    """
    rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.endswith("_rotmod.dat"): continue
            gal_name_raw = name.replace("_rotmod.dat", "")
            
            try:
                raw = z.read(name).decode("utf-8", errors="ignore")
                data_lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
                if not data_lines: continue
                
                df_gal = pd.read_csv(StringIO("\n".join(data_lines)), sep=r"\s+", header=None,
                                     names=["r", "Vobs", "eV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"],
                                     engine="python")
                
                df_gal["r"] = pd.to_numeric(df_gal["r"], errors="coerce")
                df_gal["SBdisk"] = pd.to_numeric(df_gal["SBdisk"], errors="coerce")
                df_gal = df_gal.dropna(subset=["r", "SBdisk"]).sort_values("r")
                
                # Filter strictly positive SB
                sb_vals = df_gal.loc[df_gal["SBdisk"] > 0, "SBdisk"].to_numpy(dtype=float)
                
                if len(sb_vals) > 0:
                    take_n = min(len(sb_vals), n_points)
                    sb_proxy = float(np.median(sb_vals[:take_n]))
                    rows.append({
                        "name_norm": norm_name(gal_name_raw),
                        "SB_proxy": sb_proxy
                    })
            except Exception:
                continue
    
    # Ensure uniqueness
    return pd.DataFrame(rows).drop_duplicates(subset=["name_norm"])

# --- Main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrt", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--outdir", default="Enchan_Correlation_Test_v0_3_2")
    args = ap.parse_args()

    mrt_path, zip_path = Path(args.mrt), Path(args.zip)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("--- Enchan v0.3.2 Correlation Test ---")

    # 1. Load Data
    print("Loading BTFR...")
    df_raw = parse_mrt_fixedwidth(mrt_path)
    df_btfr = extract_btfr(df_raw)
    
    # Calculate a0
    Vf_ms = np.power(10.0, df_btfr["logVf"].values) * KMS_TO_MS
    Mb_kg = np.power(10.0, df_btfr["logMb"].values) * MSUN_KG
    df_btfr["a0_si"] = (Vf_ms**4) / (G_SI * Mb_kg)
    df_btfr["name_norm"] = df_btfr["name"].apply(norm_name)

    # 2. Load SB Proxies (Primary: 3-point, Sensitivity: 5-point)
    print("Loading SB Proxies (3-point and 5-point)...")
    df_sb3 = get_sb_proxy(zip_path, n_points=3).rename(columns={"SB_proxy": "SB_3pt"})
    df_sb5 = get_sb_proxy(zip_path, n_points=5).rename(columns={"SB_proxy": "SB_5pt"})

    # Merge
    merged = pd.merge(df_btfr, df_sb3, on="name_norm", how="left")
    merged = pd.merge(merged, df_sb5, on="name_norm", how="left")
    
    n_total_btfr = len(df_btfr)
    n_matched = merged["SB_3pt"].notna().sum()
    
    # 3. Drop Analysis
    df_work = merged.copy()
    
    # Count drops
    n_nan_sb = df_work["SB_3pt"].isna().sum() # unmatched or no-data
    
    # Subset to matched
    df_work = df_work.dropna(subset=["SB_3pt"])
    
    n_nan_a0 = df_work["a0_si"].isna().sum()
    df_work = df_work.dropna(subset=["a0_si"])
    
    n_nonpos_a0 = (df_work["a0_si"] <= 0).sum()
    n_nonpos_sb = (df_work["SB_3pt"] <= 0).sum()
    
    valid = df_work[(df_work["a0_si"] > 0) & (df_work["SB_3pt"] > 0)].copy()
    n_valid = len(valid)
    
    print(f"Total BTFR: {n_total_btfr}")
    print(f"Matched SB: {n_matched} (Dropped {n_total_btfr - n_matched} unmatched)")
    print(f"Valid Data: {n_valid}")
    print(f"  [Drops] NaN a0: {n_nan_a0}, Non-pos a0: {n_nonpos_a0}, Non-pos SB: {n_nonpos_sb}")

    # 4. Correlation Analysis
    results = {}
    
    # A. Primary (All Valid, 3pt)
    x = np.log10(valid["SB_3pt"].values)
    y = np.log10(valid["a0_si"].values)
    st = get_stats(x, y)
    results["primary_r"] = st["r"]
    results["primary_p"] = st["p"]
    results["primary_rho"] = st.get("rho", np.nan)
    
    # B. Robustness: Trimmed (Drop top/bottom 5% of a0)
    valid_sorted = valid.sort_values("a0_si")
    trim_n = int(n_valid * 0.05)
    if n_valid > 20 and trim_n > 0:
        valid_trimmed = valid_sorted.iloc[trim_n : -trim_n]
        xt = np.log10(valid_trimmed["SB_3pt"].values)
        yt = np.log10(valid_trimmed["a0_si"].values)
        st_t = get_stats(xt, yt)
        results["trimmed_r"] = st_t["r"]
        results["trimmed_n"] = len(valid_trimmed)
    else:
        results["trimmed_r"] = np.nan
        results["trimmed_n"] = 0
        
    # C. Robustness: Sensitivity (5-point proxy)
    valid5 = merged[(merged["a0_si"] > 0) & (merged["SB_5pt"] > 0)].copy()
    if len(valid5) > 5:
        x5 = np.log10(valid5["SB_5pt"].values)
        y5 = np.log10(valid5["a0_si"].values)
        st_5 = get_stats(x5, y5)
        results["sens_5pt_r"] = st_5["r"]
    else:
        results["sens_5pt_r"] = np.nan

    print("-" * 40)
    print(f"Primary Correlation (r): {results['primary_r']:.3f} (p={results['primary_p']:.1e})")
    print(f"Trimmed (5%) Correlation: {results['trimmed_r']:.3f} (N={results['trimmed_n']})")
    print(f"Sensitivity (5-pt Median): {results['sens_5pt_r']:.3f}")
    print("-" * 40)

    # 5. Outputs
    # CSV Data
    out_cols = ["name", "name_norm", "logMb", "logVf", "a0_si", "SB_3pt", "SB_5pt"]
    valid[out_cols].to_csv(outdir / "correlation_data_v0_3_2.csv", index=False)
    
    # Summary
    summary = {
        "mrt_sha256": sha256_file(mrt_path),
        "zip_sha256": sha256_file(zip_path),
        "n_total_btfr": n_total_btfr,
        "n_valid": n_valid,
        # Expanded Drop Logs
        "n_matched_sb3": n_matched,
        "n_dropped_unmatched_sb3": n_total_btfr - n_matched,
        "n_dropped_nan_sb3": n_nan_sb,
        "n_dropped_nan_a0": n_nan_a0,
        "n_dropped_nonpos_a0": n_nonpos_a0,
        "n_dropped_nonpos_sb": n_nonpos_sb,
        # Stats
        "primary_pearson_r": results["primary_r"],
        "primary_p_value": results["primary_p"],
        "p_value_method": "scipy" if HAS_SCIPY else "permutation_10k_plus1",
        "primary_spearman_rho": results["primary_rho"],
        "robust_trimmed_r": results["trimmed_r"],
        "robust_sensitivity_5pt_r": results["sens_5pt_r"],
        # Softer Interpretation
        "interpretation": "Positive r is consistent with P1 (minimal anchor) under this SB proxy; does not establish causality."
    }
    pd.DataFrame([summary]).to_csv(outdir / "correlation_summary.csv", index=False)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.6, edgecolors="k", label=f"Galaxies (N={n_valid})")
    
    if n_valid > 1:
        m, c = np.polyfit(x, y, 1)
        fit_x = np.linspace(min(x), max(x), 100)
        plt.plot(fit_x, m*fit_x + c, "k--", alpha=0.7, label=f"Fit (slope={m:.2f})")
    
    plt.xlabel(r"$\log_{10}(\mathrm{SB}_{\mathrm{proxy, 3pt}})$")
    plt.ylabel(r"$\log_{10}(a_0)$")
    plt.title(f"a0 vs Surface Brightness (v0.3.2)\nr={results['primary_r']:.3f}, Trimmed={results['trimmed_r']:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "fig_correlation_v0_3_2.png", dpi=150)
    plt.close()

    print(f"Done. Results in {outdir}")

if __name__ == "__main__":
    main()