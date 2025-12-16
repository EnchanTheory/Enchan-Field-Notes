#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan Variable a0 Prediction Test (Prediction C1) - Disk-Dominated Validation (v0.3.3 Diamond)

Hypothesis:
  The Anchor Hypothesis (a0 depends on surface density) is best tested in 
  disk-dominated galaxies where SB_disk is a valid proxy for the baryonic potential.

Methodology (Refined):
  1. Filter: Baryonic Bulge Dominance Check (Robust).
     - Calculate bulge fraction of baryonic force potential (proportional to V^2).
     - Metric: f_bul(r) = (Yb*Vbul^2) / (Vgas^2 + Yd*Vdisk^2 + Yb*Vbul^2)
     - Filter: Keep galaxy only if quantile(f_bul, q) < bulge_threshold.
       (Using quantile creates robustness against single-point noise spikes)
  2. K-Fold CV (k=5): Train a0-SB relation on train set, evaluate on test set.
  3. Metric: Mean/Median Galaxy RMS (Global and per-Fold).
  4. Logging: Comprehensive tracking of inputs, collisions, filter stats, and FULL TRACEABILITY.
  5. Analysis: Automatic breakdown of performance by Surface Brightness quartiles (Q1-Q4).

Usage:
  python enchan_variable_a0_prediction.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip --max_bulge_frac 0.5 --bulge_quantile 0.95
  
  * Sensitivity Check: Sweep --bulge_quantile (0.90, 0.95, 0.99) and --max_bulge_frac
    to verify if the signal is robust against filter strictness.
"""

import argparse
import hashlib
import sys
import zipfile
from io import StringIO
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
sys.path.append(str(Path(__file__).parent))
try:
    from enchan_core_model import KMS_TO_MS, G_SI, MSUN_KG
    from enchan_btfr_reproduce_enchan import parse_mrt_fixedwidth, extract_btfr
    from enchan_a0_sb_correlation import get_sb_proxy, norm_name
except ImportError:
    print("Error: Helper modules (enchan_core_model, etc.) not found.")
    sys.exit(1)

KPC_TO_M = 3.0856775814913673e19

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_rar_data_by_galaxy(
    zip_path: Path, 
    Yd: float, 
    Yb: float, 
    max_bulge_frac: float,
    quantile_thr: float
) -> Tuple[Dict[str, pd.DataFrame], int, List[dict], List[dict]]:
    """
    Load RAR data, filtering out galaxies where the bulge contributes significantly.
    
    Returns:
        (gal_data, n_files_parsed, dropped_list, collision_list)
    """
    gal_data = {}
    gal_sources = {} 
    n_files_parsed = 0
    
    # Traceability Lists
    dropped_list = []   # Detailed log of dropped galaxies
    collision_list = [] # Detailed log of name collisions
    
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.endswith("_rotmod.dat"):
                continue
            gal_name_raw = name.replace("_rotmod.dat", "")
            n_name = norm_name(gal_name_raw)
            
            raw = z.read(name).decode("utf-8", errors="ignore")
            data_lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            if not data_lines:
                continue
            
            try:
                df_gal = pd.read_csv(
                    StringIO("\n".join(data_lines)),
                    sep=r"\s+",
                    header=None,
                    names=["r_kpc", "Vobs", "eV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"],
                    engine="python"
                )
            except Exception:
                continue
            
            n_files_parsed += 1
                
            # Physics conversion
            r_m = df_gal["r_kpc"].to_numpy(dtype=float) * KPC_TO_M
            Vobs = df_gal["Vobs"].to_numpy(dtype=float) * KMS_TO_MS
            Vgas = df_gal["Vgas"].to_numpy(dtype=float) * KMS_TO_MS
            Vdisk = df_gal["Vdisk"].to_numpy(dtype=float) * KMS_TO_MS
            Vbul = df_gal["Vbul"].to_numpy(dtype=float) * KMS_TO_MS
            
            # --- Baryonic Bulge Fraction Filter ---
            F_gas = Vgas**2
            F_disk = Yd * Vdisk**2
            F_bul = Yb * Vbul**2
            F_tot_bar = F_gas + F_disk + F_bul
            
            with np.errstate(divide='ignore', invalid='ignore'):
                f_bul = F_bul / F_tot_bar
            
            valid_check = np.isfinite(f_bul) & (F_tot_bar > 0)
            
            if np.sum(valid_check) >= 3: 
                metric_val = np.quantile(f_bul[valid_check], quantile_thr)
                
                if metric_val >= max_bulge_frac:
                    dropped_list.append({
                        "galaxy": n_name,
                        "galaxy_raw": gal_name_raw,
                        "source_file": name,
                        "metric_val": metric_val,
                        "threshold": max_bulge_frac,
                        "quantile_q": quantile_thr
                    })
                    continue # Skip
            else:
                continue
            # --------------------------------------
            
            # Compute g_obs, g_bar
            with np.errstate(divide='ignore', invalid='ignore'):
                g_obs = (Vobs**2) / r_m
                g_bar = F_tot_bar / r_m
            
            valid = np.isfinite(g_obs) & np.isfinite(g_bar) & (g_obs > 0) & (g_bar > 0)
            
            if np.sum(valid) < 3: 
                continue

            df_out = pd.DataFrame({
                "g_obs": g_obs[valid],
                "g_bar": g_bar[valid],
                "r_kpc": df_gal.loc[valid, "r_kpc"]
            })
            
            # Collision Logic: Keep Largest
            if n_name in gal_data:
                len_old = len(gal_data[n_name])
                len_new = len(df_out)
                old_src = gal_sources.get(n_name, "unknown")
                
                # Tie-breaking: If lengths are equal, keep OLD for stability.
                if len_new <= len_old:
                    collision_list.append({
                        "galaxy": n_name,
                        "source_file_kept": old_src,
                        "source_file_dropped": name,
                        "action": "kept_old",
                        "len_old": len_old,
                        "len_new": len_new
                    })
                    continue
                else:
                    collision_list.append({
                        "galaxy": n_name,
                        "source_file_kept": name,
                        "source_file_replaced": old_src,
                        "action": "replaced_new",
                        "len_old": len_old,
                        "len_new": len_new
                    })
                    # Fall through to overwrite
            
            gal_data[n_name] = df_out
            gal_sources[n_name] = name 
            
    return gal_data, n_files_parsed, dropped_list, collision_list

def calculate_galaxy_rms(df: pd.DataFrame, a0_val: float) -> float:
    gb = df["g_bar"].values
    go = df["g_obs"].values
    g_pred = np.sqrt(gb**2 + a0_val * gb)
    resid = np.log10(go) - np.log10(g_pred)
    return np.sqrt(np.mean(resid**2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrt", required=True, help="BTFR MRT file")
    ap.add_argument("--zip", required=True, help="Rotmod ZIP file")
    ap.add_argument("--outdir", default="Enchan_Prediction_C1_DiskOnly_v0_3")
    ap.add_argument("--folds", type=int, default=5, help="Number of K-Fold splits")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    ap.add_argument("--Yd", type=float, default=0.50, help="Disk mass-to-light ratio")
    ap.add_argument("--Yb", type=float, default=0.70, help="Bulge mass-to-light ratio")
    
    ap.add_argument("--max_bulge_frac", type=float, default=0.50, 
                    help=("Filter threshold: q(f_bul) < threshold. "
                          "Metric: (Yb*Vbul^2) / (Vgas^2 + Yd*Vdisk^2 + Yb*Vbul^2)."))
    ap.add_argument("--bulge_quantile", type=float, default=0.95,
                    help="Quantile for bulge filter (default 0.95). Use 0.99 for stricter, 0.90 for looser.")
    
    args = ap.parse_args()
    
    # Validation
    if not (0.0 < args.max_bulge_frac < 1.0):
        raise ValueError("--max_bulge_frac must be in (0, 1).")
    if not (0.0 < args.bulge_quantile < 1.0):
        raise ValueError("--bulge_quantile must be in (0, 1).")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    q_val = args.bulge_quantile * 100
    q_label_str = f"{q_val:.1f}" if q_val < 99.9 else f"{q_val:.4g}"
    
    print("--- Enchan Differential Prediction C1: Disk-Dominated Validation (Diamond) ---")
    print(f"Protocol: {args.folds}-Fold Cross Validation")
    print(f"Filter:   q{q_label_str}%(f_bul) < {args.max_bulge_frac}")
    print(f"Params:   Yd={args.Yd}, Yb={args.Yb}")
    
    # 1. Prepare Master Data
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
    }).dropna()
    
    if btfr_lookup["name_norm"].duplicated().any():
        btfr_lookup = btfr_lookup.groupby("name_norm", as_index=False)["a0_btfr"].median()
    n_btfr_unique = len(btfr_lookup)

    df_sb = get_sb_proxy(zip_path, n_points=3)
    if df_sb["name_norm"].duplicated().any():
        df_sb = df_sb.groupby("name_norm", as_index=False)["SB_proxy"].median()
    n_sb_unique = len(df_sb)

    # Load RAR Data
    gal_data_tuple = load_rar_data_by_galaxy(
        zip_path, 
        Yd=args.Yd, 
        Yb=args.Yb, 
        max_bulge_frac=args.max_bulge_frac,
        quantile_thr=args.bulge_quantile
    )
    rar_data_map, n_files_parsed, dropped_list, collision_list = gal_data_tuple
    
    # Export Detailed Diagnostic CSVs
    if dropped_list:
        pd.DataFrame(dropped_list).to_csv(outdir / "diagnostic_dropped_bulge.csv", index=False)
    if collision_list:
        pd.DataFrame(collision_list).to_csv(outdir / "diagnostic_collisions.csv", index=False)
    
    print(f"  [Filter] Parsed {n_files_parsed} files.")
    if collision_list:
        n_kept = sum(1 for c in collision_list if c["action"] == "kept_old")
        n_repl = sum(1 for c in collision_list if c["action"] == "replaced_new")
        print(f"  [Notice] Collisions: Kept Old={n_kept}, Replaced New={n_repl}.")
    
    print(f"  [Filter] Dropped {len(dropped_list)} due to bulge dominance.")
    
    # Merge
    master_df = pd.merge(btfr_lookup, df_sb, on="name_norm", how="inner")
    n_intersection_btfr_sb = len(master_df)
    
    valid_names = set(rar_data_map.keys())
    master_df = master_df[master_df["name_norm"].isin(valid_names)].copy()
    master_df = master_df[(master_df["a0_btfr"] > 0) & (master_df["SB_proxy"] > 0)].reset_index(drop=True)
    master_df = master_df.drop_duplicates(subset=["name_norm"]).reset_index(drop=True)
    
    n_gal = len(master_df)
    print(f"Total valid Disk-Dominated galaxies: {n_gal}")
    
    if n_gal < args.folds:
        print("Error: Not enough galaxies.")
        return

    # 2. K-Fold CV
    rng = np.random.default_rng(args.seed)
    indices = np.arange(n_gal)
    rng.shuffle(indices)
    folds = np.array_split(indices, args.folds)
    
    results_detail = []
    summary_metrics = []
    
    for k in range(args.folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(args.folds) if j != k])
        
        train_df = master_df.iloc[train_idx]
        test_df = master_df.iloc[test_idx]
        
        # Train
        lx = np.log10(train_df["SB_proxy"].values)
        ly = np.log10(train_df["a0_btfr"].values)
        slope, intercept = np.polyfit(lx, ly, 1)
        a0_median_train = np.median(train_df["a0_btfr"].values)
        
        # Test
        fold_res_A = []
        fold_res_B = []
        
        for _, row in test_df.iterrows():
            gname = row["name_norm"]
            sb_val = row["SB_proxy"]
            a0_true = row["a0_btfr"]
            
            # Predict
            a0_pred_B = 10**(intercept + slope * np.log10(sb_val))
            
            # Evaluate
            gal_rar = rar_data_map[gname]
            rms_A = calculate_galaxy_rms(gal_rar, a0_median_train)
            rms_B = calculate_galaxy_rms(gal_rar, a0_pred_B)
            
            fold_res_A.append(rms_A)
            fold_res_B.append(rms_B)
            
            results_detail.append({
                "fold": k+1,
                "galaxy": gname,
                "SB_proxy": sb_val,
                "a0_btfr_obs": a0_true,
                "a0_pred_var": a0_pred_B,
                "a0_fixed_train": a0_median_train,
                "rms_modelA_fixed": rms_A,
                "rms_modelB_variable": rms_B,
                "delta_rms": rms_A - rms_B
            })
            
        mean_rms_A = np.mean(fold_res_A)
        mean_rms_B = np.mean(fold_res_B)
        median_rms_A = np.median(fold_res_A)
        median_rms_B = np.median(fold_res_B)
        
        summary_metrics.append({
            "fold": k+1,
            "slope": slope,
            "intercept": intercept,
            "mean_rms_A": mean_rms_A,
            "mean_rms_B": mean_rms_B,
            "median_rms_A": median_rms_A,
            "median_rms_B": median_rms_B,
            "improvement_mean": mean_rms_A - mean_rms_B,
            "improvement_median": median_rms_A - median_rms_B
        })
        
        print(f"  Fold {k+1}: Slope={slope:.3f} | "
              f"MeanRMS Fix={mean_rms_A:.4f}/Var={mean_rms_B:.4f} | "
              f"MedRMS Fix={median_rms_A:.4f}/Var={median_rms_B:.4f}")

    # 3. Summary
    res_df = pd.DataFrame(results_detail)
    global_mean_A = res_df["rms_modelA_fixed"].mean()
    global_mean_B = res_df["rms_modelB_variable"].mean()
    global_median_A = res_df["rms_modelA_fixed"].median()
    global_median_B = res_df["rms_modelB_variable"].median()
    
    wins = (res_df["rms_modelB_variable"] < res_df["rms_modelA_fixed"]).sum()
    win_rate = wins / len(res_df) * 100.0
    
    print("="*60)
    print("FINAL RESULTS (Disk-Dominated Only)")
    print(f"Filter: q{q_label_str}%(f_bul) < {args.max_bulge_frac}")
    print(f"Galaxies Tested: {len(res_df)}")
    print(f"Model A (Fixed a0):     Mean={global_mean_A:.4f}, Median={global_median_A:.4f} dex")
    print(f"Model B (Variable a0):  Mean={global_mean_B:.4f}, Median={global_median_B:.4f} dex")
    print(f"Net Mean Improvement:   {global_mean_A - global_mean_B:+.4f} dex")
    print(f"Win Rate:               {wins}/{len(res_df)} ({win_rate:.1f}%)")
    
    # --- [Feature] Automatic Quartile Analysis ---
    print("-" * 60)
    print(">>> BREAKDOWN by Surface Brightness (SB_proxy) Quartiles")
    
    try:
        res_df["log_sb"] = np.log10(res_df["SB_proxy"])
        res_df["sb_quartile"] = pd.qcut(res_df["log_sb"], 4, labels=["Q1 (Low SB)", "Q2", "Q3", "Q4 (High SB)"])
        
        # Win Rate per Quartile
        # [Fix] Silence FutureWarning by observing data (observed=False retains current behavior safely)
        grp = res_df.groupby("sb_quartile", observed=False)
        
        q_stats = grp["delta_rms"].apply(lambda x: (x > 0).mean() * 100)
        q_counts = grp["delta_rms"].count()
        q_means = grp["delta_rms"].mean()
        
        print(f"{'Quartile':<15} | {'Count':<5} | {'Win Rate':<10} | {'Mean Improv. (dex)':<20}")
        print("-" * 60)
        for cat in ["Q1 (Low SB)", "Q2", "Q3", "Q4 (High SB)"]:
            wr = q_stats[cat]
            cnt = q_counts[cat]
            mn = q_means[cat]
            mark = "★" if wr >= 60 else ""
            print(f"{cat:<15} | {cnt:<5} | {wr:5.1f}% {mark:<3} | {mn:+.4f}")
            
    except Exception as e:
        print(f"Warning: Could not compute quartile stats ({e})")
    
    print("="*60)
    
    # Save
    res_df.to_csv(outdir / "prediction_c1_disk_details.csv", index=False)
    pd.DataFrame(summary_metrics).to_csv(outdir / "prediction_c1_fold_summary.csv", index=False)
    
    # Plot Scatter
    plt.figure(figsize=(8,6))
    plt.scatter(np.log10(res_df["SB_proxy"]), res_df["delta_rms"], alpha=0.6, c='b')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel(r"$\log_{10}(\mathrm{SB}_{\mathrm{proxy}})$")
    plt.ylabel("Improvement (Fixed - Var) [dex]")
    plt.title(f"Disk-Dominated (q{q_label_str}% < {args.max_bulge_frac})\nWin Rate: {win_rate:.1f}%")
    plt.tight_layout()
    plt.savefig(outdir / "fig_c1_disk_improvement.png")
    plt.close()
    
    # Run Summary
    pd.DataFrame([{
        "mrt_file": mrt_path.name,
        "mrt_sha256": sha256_file(mrt_path),
        "zip_file": zip_path.name,
        "zip_sha256": sha256_file(zip_path),
        "Yd": args.Yd,
        "Yb": args.Yb,
        "folds": args.folds,
        "seed": args.seed,
        "filter_metric": f"q{q_label_str}%( (Yb*Vbul^2) / (Vgas^2 + Yd*Vdisk^2 + Yb*Vbul^2) )",
        "max_bulge_frac": args.max_bulge_frac,
        "bulge_quantile": args.bulge_quantile,
        "n_files_parsed": n_files_parsed,
        "n_btfr_unique": n_btfr_unique,
        "n_sb_unique": n_sb_unique,
        "n_dropped_bulge": len(dropped_list),
        "n_collisions_kept": sum(1 for c in collision_list if c["action"] == "kept_old"),
        "n_collisions_replaced": sum(1 for c in collision_list if c["action"] == "replaced_new"),
        "n_rar_galaxies_passed_filter": len(rar_data_map),
        "n_intersection_btfr_sb": n_intersection_btfr_sb,
        "n_galaxies_tested_final": len(res_df),
        "global_mean_rms_fixed": global_mean_A,
        "global_mean_rms_variable": global_mean_B,
        "global_median_rms_fixed": global_median_A,
        "global_median_rms_variable": global_median_B,
        "win_rate_percent": win_rate
    }]).to_csv(outdir / "run_summary.csv", index=False)
    
    print(f"Done. Outputs in {outdir}")

if __name__ == "__main__":
    main()