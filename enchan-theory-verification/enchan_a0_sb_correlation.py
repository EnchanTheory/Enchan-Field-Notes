#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan a0 vs Surface Brightness Correlation (v0.3.1)

Exploratory test of the "Surface Density Anchor" ansatz (Chapter 6).
This script checks for correlations between:
1. a0 derived from BTFR (a0 ~ Vf^4 / Mb)
2. Surface Brightness proxy from Rotmod (SB_disk at inner radii)

Methodology & Caveats:
- Proxy: SB_proxy is the MEDIAN of the innermost 3 positive SB_disk points.
  This represents SBdisk at the innermost reliable radii (not necessarily the true central value),
  serving as a robustness measure against resolution effects and outliers.
- Physical Interpretation: We use SB_disk (Luminosity Surface Density) as an
  observational proxy for Sigma_b (Baryonic Mass Surface Density).
  Variations in Mass-to-Light ratio (M/L) are NOT corrected here.

Pre-registered Interpretation Scenarios:
  P1 (Minimal Model): If eta_S is constant and SB tracks Sigma_b,
      a weak positive correlation is expected.
  P2 (Self-Regulation): If eta_S inversely covaries with Sigma_b,
      the correlation may be zero or negligible.
  P3 (Proxy Limit): If SB_disk is a poor proxy for Sigma_b (e.g. variable M/L),
      the correlation may be unstable or noisy.

Usage:
  python enchan_a0_sb_correlation.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import SciPy for p-values, but fall back to manual calc if missing
try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Ensure local imports work if running from root or directory
sys.path.append(str(Path(__file__).parent))

# Import reusable logic
try:
    from enchan_core_model import G_SI, MSUN_KG, KMS_TO_MS
    from enchan_btfr_reproduce_enchan import parse_mrt_fixedwidth, extract_btfr
except ImportError:
    print("Error: Helper modules (enchan_core_model, enchan_btfr_reproduce_enchan) not found.")
    print("Please run this script from the enchan-theory-verification directory.")
    sys.exit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_name(s: str) -> str:
    """
    Normalize galaxy names for robust matching.
    1. Uppercase, remove ALL NON-ALPHANUMERIC characters (spaces, -, _, ., etc.).
    2. Normalize numeric parts to remove leading zeros (e.g., NGC0289 -> NGC289).
    """
    s = str(s).upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    
    # Match pattern like "NGC" + "0289" -> "NGC" + "289"
    m = re.match(r"^([A-Z]+)(\d+)$", s)
    if m:
        prefix, num = m.group(1), m.group(2)
        s = f"{prefix}{int(num)}"
    
    return s


# Manual correlation implementations for SciPy-less environments
def pearson_corr_manual(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    
    mx = np.mean(x)
    my = np.mean(y)
    xm = x - mx
    ym = y - my
    
    num = np.sum(xm * ym)
    den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    
    if den == 0:
        return float("nan")
    return float(num / den)

def spearman_corr_manual(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman is Pearson of ranks
    # pandas rank is robust and doesn't require scipy
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return pearson_corr_manual(rx, ry)


def get_sb_proxy(zip_path: Path) -> pd.DataFrame:
    """
    Extracts a Surface Brightness proxy for each galaxy from Rotmod_LTG.zip.
    Method: MEDIAN of the innermost 3 valid points.
    """
    rows = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.endswith("_rotmod.dat"):
                continue
            
            gal_name_raw = name.replace("_rotmod.dat", "")
            
            raw = z.read(name).decode("utf-8", errors="ignore")
            data_lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            if not data_lines:
                continue
            
            try:
                # Columns: r, Vobs, eV, Vgas, Vdisk, Vbul, SBdisk, SBbul
                df_gal = pd.read_csv(
                    StringIO("\n".join(data_lines)),
                    sep=r"\s+",
                    header=None,
                    names=["r", "Vobs", "eV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"],
                    engine="python"
                )
                
                # Robust numeric conversion
                df_gal["r"] = pd.to_numeric(df_gal["r"], errors="coerce")
                df_gal["SBdisk"] = pd.to_numeric(df_gal["SBdisk"], errors="coerce")
                
                # Drop invalid rows and sort
                df_gal = df_gal.dropna(subset=["r", "SBdisk"]).sort_values("r")
                
                # Filter strictly positive SB
                sb_vals = df_gal.loc[df_gal["SBdisk"] > 0, "SBdisk"].to_numpy(dtype=float)
                
                if len(sb_vals) > 0:
                    # Take up to 3 innermost points
                    take_n = min(len(sb_vals), 3)
                    inner_vals = sb_vals[:take_n]
                    sb_proxy = float(np.median(inner_vals))
                    
                    rows.append({
                        "name_raw": gal_name_raw,
                        "name_norm": norm_name(gal_name_raw),
                        "SB_proxy": sb_proxy
                    })
            except Exception:
                continue

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrt", required=True, help="Path to BTFR_Lelli2019.mrt")
    ap.add_argument("--zip", required=True, help="Path to Rotmod_LTG.zip")
    ap.add_argument("--outdir", default="Enchan_Correlation_Test_v0_3_1", help="Output directory")
    args = ap.parse_args()

    mrt_path = Path(args.mrt)
    zip_path = Path(args.zip)
    if not mrt_path.exists() or not zip_path.exists():
        raise FileNotFoundError("Input files not found.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load a0 from BTFR
    print("Loading BTFR a0 data...")
    df_raw = parse_mrt_fixedwidth(mrt_path)
    df_btfr = extract_btfr(df_raw)
    
    # Calculate a0 per galaxy: a0 = Vf^4 / (G * Mb)
    Vf_ms = np.power(10.0, df_btfr["logVf"].values) * KMS_TO_MS
    Mb_kg = np.power(10.0, df_btfr["logMb"].values) * MSUN_KG
    df_btfr["a0_si"] = (Vf_ms**4) / (G_SI * Mb_kg)
    
    # Normalize names
    df_btfr["name_norm"] = df_btfr["name"].apply(norm_name)
    
    # 2. Load SB proxy from Rotmod
    print("Loading SB proxy from Rotmod (Method: Median of innermost 3)...")
    df_sb = get_sb_proxy(zip_path)

    # 3. Merge
    merged = pd.merge(df_btfr, df_sb, on="name_norm", how="inner", suffixes=("", "_sb"))
    
    # Counts for reporting
    n_btfr = len(df_btfr)
    n_rotmod = len(df_sb)
    n_matched = len(merged)
    
    print(f"Loaded: BTFR={n_btfr}, Rotmod={n_rotmod}")
    print(f"Matched: {n_matched} galaxies available for correlation test.")

    # 4. Analysis
    # Strict cleaning for log scale (remove infs, nans, non-positives)
    valid = merged.copy()
    valid = valid.replace([np.inf, -np.inf], np.nan)
    valid = valid.dropna(subset=["a0_si", "SB_proxy"])
    valid = valid[(valid["a0_si"] > 0) & (valid["SB_proxy"] > 0)].copy()
    
    n_valid = len(valid)
    
    # Only calculate stats if sufficient data
    if n_valid > 5:
        x = np.log10(valid["SB_proxy"].values)  # log(SB [Lsun/pc^2])
        y = np.log10(valid["a0_si"].values)     # log(a0 [m/s^2])
        
        if HAS_SCIPY:
            pearson_r, p_val = pearsonr(x, y)
            spearman_r, s_p_val = spearmanr(x, y)
            stats_str = (
                f"N={n_valid}\n"
                f"Pearson r={pearson_r:.3f} (p={p_val:.1e})\n"
                f"Spearman r={spearman_r:.3f} (p={s_p_val:.1e})"
            )
        else:
            # Fallback to manual implementation
            pearson_r = pearson_corr_manual(x, y)
            spearman_r = spearman_corr_manual(x, y)
            p_val, s_p_val = float("nan"), float("nan")
            stats_str = (
                f"N={n_valid}\n"
                f"Pearson r={pearson_r:.3f}\n"
                f"Spearman r={spearman_r:.3f}\n"
                f"(No SciPy, p-values skipped)"
            )
    else:
        pearson_r, spearman_r = float("nan"), float("nan")
        p_val, s_p_val = float("nan"), float("nan")
        stats_str = f"N={n_valid} (Insufficient data)"

    print("-" * 40)
    print("Correlation Results:")
    print(stats_str)
    print("-" * 40)

    # Save Data
    valid.rename(columns={"name": "name_btfr"}, inplace=True)
    out_cols = ["name_btfr", "name_norm", "logMb", "logVf", "a0_si", "SB_proxy"]
    valid[out_cols].to_csv(outdir / "a0_sb_correlation_data.csv", index=False)

    # Summary File with Hashes and Counts
    summary = pd.DataFrame([{
        "mrt_file": mrt_path.name,
        "mrt_sha256": sha256_file(mrt_path),
        "zip_file": zip_path.name,
        "zip_sha256": sha256_file(zip_path),
        "n_btfr": n_btfr,
        "n_rotmod_extracted": n_rotmod,
        "n_matched": n_matched,
        "n_valid_points": n_valid,
        "sb_proxy_method": "median_innermost_3_positive",
        "pearson_r": pearson_r,
        "pearson_p_value": p_val,
        "spearman_r": spearman_r,
        "spearman_p_value": s_p_val,
        "note": "Exploratory check of Surface-Density Anchor. Sign/strength depends on proxy validity and eta_S behavior."
    }])
    summary.to_csv(outdir / "correlation_summary.csv", index=False)

    # 5. Plot (Neutral style)
    if n_valid > 0:
        plt.figure(figsize=(8, 6))
        x = np.log10(valid["SB_proxy"].values)
        y = np.log10(valid["a0_si"].values)
        
        plt.scatter(x, y, alpha=0.6, edgecolors="k", label="SPARC Galaxies")
        
        # Simple linear fit for visualization
        if n_valid > 1:
            try:
                m, c = np.polyfit(x, y, 1)
                fit_x = np.linspace(x.min(), x.max(), 100)
                plt.plot(fit_x, m*fit_x + c, "--", color="black", alpha=0.7, label=f"Fit slope={m:.2f}")
            except Exception:
                pass
        
        plt.xlabel(r"$\log_{10}(\mathrm{SB}_{\mathrm{proxy}} \; [L_\odot/\mathrm{pc}^2])$")
        plt.ylabel(r"$\log_{10}(a_0 \; [\mathrm{m/s}^2])$")
        plt.title("Exploratory Test: Acceleration Scale vs Surface Brightness")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add stats box
        plt.text(0.05, 0.95, stats_str, transform=plt.gca().transAxes,
                 verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(outdir / "fig_a0_sb_correlation.png", dpi=150)
        plt.close()
    
    print(f"Outputs written to: {outdir.resolve()}")

if __name__ == "__main__":
    main()