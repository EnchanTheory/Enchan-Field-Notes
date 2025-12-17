#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan C1 Pinning Test (Strict Nested CV Edition) - v0.4.0 Compatible

Goal
----
Verify the Pinning Mechanism using strict Nested Cross-Validation.
Compatible with enchan_core_model_plus v0.4.0 Gold Master.

Usage:
  python enchan_c1_pinning_test_strict.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip --inner_cut 1.0
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

# ---------------------------------------------------------
# Import Enchan Modules
# ---------------------------------------------------------
# Add root directory to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from enchan_core_model import KMS_TO_MS, G_SI, MSUN_KG
    from enchan_btfr_reproduce_enchan import parse_mrt_fixedwidth, extract_btfr
    from enchan_a0_sb_correlation import get_sb_proxy, norm_name
    from enchan_variable_a0_prediction import load_rar_data_by_galaxy
    
    # [NEW] Import from v0.4.0 Gold Master (Correct Function Name)
    from enchan_core_model_plus import (
        calculate_phi_median_proxy,  # Renamed in v0.4.0
        apply_screening
    )
    
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import Enchan modules: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------

def calculate_screened_rms(gal_rar, a0_free_val, phi_val, phi_c, n, inner_cut_kpc):
    """Calculate RMS with Pinning (Screening)."""
    df = gal_rar[gal_rar["r_kpc"] >= inner_cut_kpc]
    if len(df) < 3:
        return np.nan
        
    gb = df["g_bar"].values
    go = df["g_obs"].values
    
    # Apply Screening
    a0_eff = apply_screening(a0_free_val, phi_val=phi_val, phi_c=phi_c, n=n)
    
    g_pred = np.sqrt(gb**2 + a0_eff * gb)
    g_pred = np.maximum(g_pred, 1e-15)
    go = np.maximum(go, 1e-15)
    
    resid = np.log10(go) - np.log10(g_pred)
    return np.sqrt(np.mean(resid**2))

# ---------------------------------------------------------
# 2. Strict Nested CV Logic
# ---------------------------------------------------------

def get_param_grid():
    """Search grid for (Phi_c, n)."""
    phi_c_grid = [10000, 22500, 40000, 62500, 90000, 160000] 
    n_grid = [1.0, 2.0, 4.0]
    return list(product(phi_c_grid, n_grid))

def evaluate_fold_set(df_subset, rar_map, params, inner_cut_kpc, slope, intercept, a0_med_train):
    """
    Evaluate mean dRMS using PRE-CALCULATED slope/intercept.
    """
    res_q1 = []
    res_q4 = []
    phi_c, n = params
    
    for _, row in df_subset.iterrows():
        quartile = row["quartile"]
        if quartile not in ["Q1", "Q4"]: continue
            
        gname = row["name_norm"]
        sb_val = row["SB_proxy"]
        phi_val = row["phi_proxy"]
        
        # Robust check for valid phi
        if not np.isfinite(phi_val): continue
        
        # Predict using passed (trained) model
        a0_free = 10**(intercept + slope * np.log10(sb_val))
        gal_rar = rar_map[gname]
        
        # Model A vs B
        rms_A = calculate_screened_rms(gal_rar, a0_med_train, 0, 1e9, 1.0, inner_cut_kpc)
        rms_B = calculate_screened_rms(gal_rar, a0_free, phi_val, phi_c, n, inner_cut_kpc)
        
        if not np.isnan(rms_A) and not np.isnan(rms_B):
            d_rms = rms_A - rms_B
            if quartile == "Q1": res_q1.append(d_rms)
            if quartile == "Q4": res_q4.append(d_rms)
        
    mean_q1 = np.mean(res_q1) if res_q1 else 0.0
    mean_q4 = np.mean(res_q4) if res_q4 else 0.0
    return mean_q1, mean_q4

def run_nested_cv(master_df, rar_map, inner_cut_kpc, k_outer=5):
    """Perform Strict Nested CV."""
    rng = np.random.default_rng(42)
    indices = np.arange(len(master_df))
    rng.shuffle(indices)
    folds = np.array_split(indices, k_outer)
    
    results = []
    best_params_log = []
    param_grid = get_param_grid()
    
    print(f"Starting Strict Nested CV (Outer={k_outer})...")
    
    for i in range(k_outer):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k_outer) if j != i])
        
        train_df = master_df.iloc[train_idx].copy()
        test_df = master_df.iloc[test_idx].copy()
        
        # --- INNER LOOP (Strict Optimization) ---
        best_score = -9999.0
        best_p = param_grid[0]
        
        n_tr = len(train_df)
        inner_tr_idx = int(n_tr * 0.7)
        tr_inner = train_df.iloc[:inner_tr_idx]
        val_inner = train_df.iloc[inner_tr_idx:]
        
        # 1. Fit on Inner_Train
        lx_in = np.log10(tr_inner["SB_proxy"].values)
        ly_in = np.log10(tr_inner["a0_btfr"].values)
        if len(lx_in) > 5:
            slope_in, intercept_in = np.polyfit(lx_in, ly_in, 1)
        else:
            slope_in, intercept_in = 0.0, np.log10(1.2e-10)
        a0_med_in = np.median(tr_inner["a0_btfr"].values)
        
        # 2. Evaluate on Inner_Val
        for p in param_grid:
            d_q1, d_q4 = evaluate_fold_set(
                val_inner, rar_map, p, inner_cut_kpc, 
                slope_in, intercept_in, a0_med_in
            )
            # Objective
            score = d_q4 - 2.0 * max(0.0, -d_q1)
            if score > best_score:
                best_score = score
                best_p = p
        
        best_params_log.append(best_p)
        
        # --- OUTER TEST (Final Evaluation) ---
        # 1. Fit on Full Train
        lx = np.log10(train_df["SB_proxy"].values)
        ly = np.log10(train_df["a0_btfr"].values)
        if len(lx) > 5:
            slope, intercept = np.polyfit(lx, ly, 1)
        else:
            slope, intercept = 0.0, np.log10(1.2e-10)
        a0_med_train = np.median(train_df["a0_btfr"].values)
        
        # 2. Test on untouched Test set
        phi_c_opt, n_opt = best_p
        
        # Logging individual results
        for _, row in test_df.iterrows():
            if row["quartile"] not in ["Q1", "Q4"]: continue
            gname = row["name_norm"]
            sb_val = row["SB_proxy"]
            phi_val = row["phi_proxy"]
            
            if not np.isfinite(phi_val): continue
            
            a0_free = 10**(intercept + slope * np.log10(sb_val))
            gal_rar = rar_map[gname]
            
            rms_A = calculate_screened_rms(gal_rar, a0_med_train, 0, 1e9, 1.0, inner_cut_kpc)
            rms_B = calculate_screened_rms(gal_rar, a0_free, phi_val, phi_c_opt, n_opt, inner_cut_kpc)
            
            if not np.isnan(rms_A) and not np.isnan(rms_B):
                results.append({
                    "fold": i,
                    "quartile": row["quartile"],
                    "dRMS": rms_A - rms_B
                })

    return pd.DataFrame(results), best_params_log

# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrt", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--inner_cut", type=float, default=1.0)
    args = parser.parse_args()
    
    print(f"--- Enchan C1 Pinning Test (Strict Mode / v0.4.0) ---")
    
    # Load Data
    mrt_path = Path(args.mrt)
    zip_path = Path(args.zip)
    
    df_raw = parse_mrt_fixedwidth(mrt_path)
    df_btfr = extract_btfr(df_raw)
    Vf_ms = np.power(10.0, df_btfr["logVf"].values) * KMS_TO_MS
    Mb_kg = np.power(10.0, df_btfr["logMb"].values) * MSUN_KG
    a0_obs_arr = (Vf_ms**4) / (G_SI * Mb_kg)
    
    btfr_lookup = pd.DataFrame({
        "name_norm": df_btfr["name"].apply(norm_name),
        "a0_btfr": a0_obs_arr
    }).dropna().groupby("name_norm", as_index=False)["a0_btfr"].median()
    
    df_sb = get_sb_proxy(zip_path, n_points=3)
    if df_sb["name_norm"].duplicated().any():
        df_sb = df_sb.groupby("name_norm", as_index=False)["SB_proxy"].median()
        
    gal_data_tuple = load_rar_data_by_galaxy(zip_path, Yd=0.5, Yb=0.7, max_bulge_frac=0.5, quantile_thr=0.95)
    rar_data_map = gal_data_tuple[0]
    
    # Compute Potential Proxy using v0.4.0 Gold Master
    print("Calculating potential depths (|Phi| ~ V_bar^2) using calculate_phi_median_proxy...")
    phi_data = []
    for gname, df_rar in rar_data_map.items():
        # Pass numpy arrays explicitly for portability
        phi = calculate_phi_median_proxy(
            r_kpc=df_rar["r_kpc"].values,
            g_bar=df_rar["g_bar"].values,
            inner_cut_kpc=args.inner_cut
        )
        phi_data.append({"name_norm": gname, "phi_proxy": phi})
    df_phi = pd.DataFrame(phi_data)
    
    master_df = pd.merge(btfr_lookup, df_sb, on="name_norm", how="inner")
    master_df = pd.merge(master_df, df_phi, on="name_norm", how="inner")
    
    valid_names = set(rar_data_map.keys())
    master_df = master_df[master_df["name_norm"].isin(valid_names)].copy()
    master_df = master_df.dropna(subset=["a0_btfr", "SB_proxy", "phi_proxy"])
    master_df = master_df[(master_df["a0_btfr"] > 0) & (master_df["SB_proxy"] > 0)].reset_index(drop=True)
    master_df = master_df.drop_duplicates(subset=["name_norm"]).reset_index(drop=True)
    
    master_df["log_sb"] = np.log10(master_df["SB_proxy"])
    master_df["quartile"] = pd.qcut(master_df["log_sb"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    
    print(f"Dataset: {len(master_df)} galaxies.")
    
    # Run Strict CV
    res_df, best_params = run_nested_cv(master_df, rar_data_map, args.inner_cut)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS (Strict Mode / v0.4.0)")
    print("="*60)
    
    q1_res = res_df[res_df["quartile"] == "Q1"]
    q4_res = res_df[res_df["quartile"] == "Q4"]
    
    print(f"Low-SB  (Q1): dRMS = {q1_res['dRMS'].mean():+.5f} | Win Rate = {(q1_res['dRMS']>0).mean():.1%}")
    print(f"High-SB (Q4): dRMS = {q4_res['dRMS'].mean():+.5f} | Win Rate = {(q4_res['dRMS']>0).mean():.1%}")
    print("-" * 60)
    
    print("Optimized Parameters (Phi_c, n) in each fold:")
    for i, p in enumerate(best_params):
        print(f" Fold {i}: Phi_c={p[0]}, n={p[1]}")
    print("="*60)

if __name__ == "__main__":
    main()