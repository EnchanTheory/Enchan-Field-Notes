#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan Variable a0 Prediction Test (Prediction C1) - Disk-Dominated Validation (v0.4.2 Fix)

Hypothesis
----------
Anchor Hypothesis: a0 depends on a baryonic surface-density proxy.
This is tested in disk-dominated galaxies where SB_disk is a valid proxy.

Method
------
1) Disk-dominated filter (robust bulge dominance check):
   f_bul(r) = (Yb*Vbul^2) / (Vgas^2 + Yd*Vdisk^2 + Yb*Vbul^2)
   Keep galaxy only if quantile(f_bul, q) < max_bulge_frac.
2) K-fold CV:
   Train log10(a0_btfr) vs log10(SB_proxy) on train set,
   evaluate per-galaxy RMS on test set.
3) Compare:
   Model A: fixed a0 = median(train a0_btfr)
   Model B: variable a0 predicted from SB proxy
4) Traceability:
   logs hashes, dropped galaxies, name collisions, and run summary.
5) Automatic breakdown by SB quartiles.

Inputs
------
- BTFR_Lelli2019.mrt (SPARC BTFR table)
- Rotmod_LTG.zip     (SPARC rotation-curve decomposition archive)

Usage (minimum)
---------------
python enchan_variable_a0_prediction.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip

Recommended (explicit filter + M/L)
-----------------------------------
python enchan_variable_a0_prediction.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip \
  --Yd 0.50 --Yb 0.70 --max_bulge_frac 0.50 --bulge_quantile 0.95
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from io import StringIO
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# matplotlib is optional
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# Local imports
sys.path.append(str(Path(__file__).parent))
try:
    from enchan_core_model import KMS_TO_MS, G_SI, MSUN_KG
    from enchan_btfr_reproduce_enchan import parse_mrt_fixedwidth, extract_btfr
    from enchan_a0_sb_correlation import get_sb_proxy, norm_name
except ImportError as e:
    raise SystemExit(
        "ERROR: Helper modules not found.\n"
        "Run from enchan-theory-verification/ or fix PYTHONPATH.\n"
        f"Detail: {e}"
    )

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
    Load RAR data from Rotmod_LTG.zip and apply a disk-dominance filter.

    Returns:
      (gal_data_map, n_files_parsed, dropped_list, collision_list)

    gal_data_map[gname] is a DataFrame with columns:
      - g_obs (m/s^2)
      - g_bar (m/s^2)
      - r_kpc
    """
    gal_data: Dict[str, pd.DataFrame] = {}
    gal_sources: Dict[str, str] = {}
    n_files_parsed = 0

    dropped_list: List[dict] = []
    collision_list: List[dict] = []

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

            # --- Disk-dominance (bulge fraction) filter ---
            F_gas = Vgas**2
            F_disk = Yd * Vdisk**2
            F_bul = Yb * Vbul**2
            F_tot_bar = F_gas + F_disk + F_bul

            with np.errstate(divide="ignore", invalid="ignore"):
                f_bul = F_bul / F_tot_bar

            valid_check = np.isfinite(f_bul) & np.isfinite(F_tot_bar) & (F_tot_bar > 0)

            if np.sum(valid_check) >= 3:
                metric_val = float(np.quantile(f_bul[valid_check], quantile_thr))
                if metric_val >= max_bulge_frac:
                    dropped_list.append({
                        "galaxy": n_name,
                        "galaxy_raw": gal_name_raw,
                        "source_file": name,
                        "metric_val": metric_val,
                        "threshold": max_bulge_frac,
                        "quantile_q": quantile_thr
                    })
                    continue
            else:
                continue
            # ---------------------------------------------

            # Compute g_obs, g_bar
            with np.errstate(divide="ignore", invalid="ignore"):
                g_obs = (Vobs**2) / r_m
                g_bar = F_tot_bar / r_m

            valid = np.isfinite(g_obs) & np.isfinite(g_bar) & (g_obs > 0) & (g_bar > 0)
            if np.sum(valid) < 3:
                continue

            df_out = pd.DataFrame({
                "g_obs": g_obs[valid],
                "g_bar": g_bar[valid],
                "r_kpc": df_gal.loc[valid, "r_kpc"].to_numpy(dtype=float),
            })

            # Collision policy: keep the longer record; ties keep old for stability
            if n_name in gal_data:
                len_old = len(gal_data[n_name])
                len_new = len(df_out)
                old_src = gal_sources.get(n_name, "unknown")

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

            gal_data[n_name] = df_out
            gal_sources[n_name] = name

    return gal_data, n_files_parsed, dropped_list, collision_list


def calculate_galaxy_rms(df: pd.DataFrame, a0_val: float) -> float:
    gb = df["g_bar"].to_numpy(dtype=float)
    go = df["g_obs"].to_numpy(dtype=float)
    a0_val = float(a0_val)

    gb = np.maximum(gb, 1e-30)
    go = np.maximum(go, 1e-30)

    g_pred = np.sqrt(gb**2 + a0_val * gb)
    g_pred = np.maximum(g_pred, 1e-30)

    resid = np.log10(go) - np.log10(g_pred)
    return float(np.sqrt(np.mean(resid**2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrt", required=True, help="BTFR MRT file (e.g., BTFR_Lelli2019.mrt)")
    ap.add_argument("--zip", required=True, help="Rotmod ZIP file (e.g., Rotmod_LTG.zip)")
    ap.add_argument("--outdir", default="Enchan_Prediction_C1_DiskOnly_v0_4_2", help="Output directory")
    ap.add_argument("--folds", type=int, default=5, help="Number of K-Fold splits")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    ap.add_argument("--Yd", type=float, default=0.50, help="Disk mass-to-light ratio")
    ap.add_argument("--Yb", type=float, default=0.70, help="Bulge mass-to-light ratio")
    ap.add_argument("--max_bulge_frac", type=float, default=0.50,
                    help=("Disk-dominated filter: quantile(f_bul, q) < threshold. "
                          "f_bul=(Yb*Vbul^2)/(Vgas^2 + Yd*Vdisk^2 + Yb*Vbul^2)."))
    ap.add_argument("--bulge_quantile", type=float, default=0.95,
                    help="Quantile q for bulge filter (default 0.95).")
    ap.add_argument("--no_plot", action="store_true", help="Disable plots even if matplotlib is available")
    args = ap.parse_args()

    # Validation
    if not (args.folds >= 2):
        raise SystemExit("ERROR: --folds must be >= 2")
    if not (0.0 < args.max_bulge_frac < 1.0):
        raise SystemExit("ERROR: --max_bulge_frac must be in (0,1)")
    if not (0.0 < args.bulge_quantile < 1.0):
        raise SystemExit("ERROR: --bulge_quantile must be in (0,1)")
    if args.Yd < 0 or args.Yb < 0:
        raise SystemExit("ERROR: --Yd and --Yb must be >= 0")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    q_val = args.bulge_quantile * 100
    q_label_str = f"{q_val:.1f}" if q_val < 99.9 else f"{q_val:.4g}"

    print("--- Enchan Differential Prediction C1: Disk-Dominated Validation (v0.4.2) ---")
    print(f"Protocol: {args.folds}-Fold Cross Validation (seed={args.seed})")
    print(f"Filter:   q{q_label_str}%(f_bul) < {args.max_bulge_frac}")
    print(f"Params:   Yd={args.Yd}, Yb={args.Yb}")

    # 1) Load BTFR
    mrt_path = Path(args.mrt)
    zip_path = Path(args.zip)
    if not mrt_path.exists():
        raise SystemExit(f"ERROR: Missing mrt file: {mrt_path}")
    if not zip_path.exists():
        raise SystemExit(f"ERROR: Missing zip file: {zip_path}")

    df_raw = parse_mrt_fixedwidth(mrt_path)
    df_btfr = extract_btfr(df_raw)

    Vf_ms = np.power(10.0, df_btfr["logVf"].values) * KMS_TO_MS
    Mb_kg = np.power(10.0, df_btfr["logMb"].values) * MSUN_KG

    with np.errstate(divide="ignore", invalid="ignore"):
        a0_obs_arr = (Vf_ms**4) / (G_SI * Mb_kg)

    btfr_lookup = pd.DataFrame({
        "name_norm": df_btfr["name"].apply(norm_name),
        "a0_btfr": a0_obs_arr
    })
    btfr_lookup = btfr_lookup.replace([np.inf, -np.inf], np.nan).dropna()
    btfr_lookup = btfr_lookup[btfr_lookup["a0_btfr"] > 0].copy()

    if btfr_lookup["name_norm"].duplicated().any():
        btfr_lookup = btfr_lookup.groupby("name_norm", as_index=False)["a0_btfr"].median()
    n_btfr_unique = len(btfr_lookup)

    # 2) Load SB proxy
    df_sb = get_sb_proxy(zip_path, n_points=3)
    df_sb = df_sb.replace([np.inf, -np.inf], np.nan).dropna()
    df_sb = df_sb[df_sb["SB_proxy"] > 0].copy()

    if df_sb["name_norm"].duplicated().any():
        df_sb = df_sb.groupby("name_norm", as_index=False)["SB_proxy"].median()
    n_sb_unique = len(df_sb)

    # 3) Load RAR data with disk-dominated filter
    rar_data_map, n_files_parsed, dropped_list, collision_list = load_rar_data_by_galaxy(
        zip_path,
        Yd=args.Yd,
        Yb=args.Yb,
        max_bulge_frac=args.max_bulge_frac,
        quantile_thr=args.bulge_quantile
    )

    # Diagnostics
    if dropped_list:
        pd.DataFrame(dropped_list).to_csv(outdir / "diagnostic_dropped_bulge.csv", index=False)
    if collision_list:
        pd.DataFrame(collision_list).to_csv(outdir / "diagnostic_collisions.csv", index=False)

    print(f"  [RAR] Parsed {n_files_parsed} files.")
    print(f"  [RAR] Dropped {len(dropped_list)} due to bulge dominance.")
    if collision_list:
        n_kept = sum(1 for c in collision_list if c["action"] == "kept_old")
        n_repl = sum(1 for c in collision_list if c["action"] == "replaced_new")
        print(f"  [RAR] Collisions: kept_old={n_kept}, replaced_new={n_repl}")

    # 4) Merge into master list
    master_df = pd.merge(btfr_lookup, df_sb, on="name_norm", how="inner")
    n_intersection_btfr_sb = len(master_df)

    valid_names = set(rar_data_map.keys())
    master_df = master_df[master_df["name_norm"].isin(valid_names)].copy()
    master_df = master_df.dropna(subset=["a0_btfr", "SB_proxy"])
    master_df = master_df[(master_df["a0_btfr"] > 0) & (master_df["SB_proxy"] > 0)].copy()
    master_df = master_df.drop_duplicates(subset=["name_norm"]).reset_index(drop=True)

    n_gal = len(master_df)
    print(f"  [BTFR] unique galaxies: {n_btfr_unique}")
    print(f"  [SB]   unique galaxies: {n_sb_unique}")
    print(f"  [Join] BTFR∩SB: {n_intersection_btfr_sb}")
    print(f"Total valid Disk-Dominated galaxies: {n_gal}")

    if n_gal < args.folds:
        raise SystemExit("ERROR: Not enough galaxies after filtering for requested folds.")

    # 5) K-fold CV
    rng = np.random.default_rng(args.seed)
    indices = np.arange(n_gal)
    rng.shuffle(indices)
    folds = np.array_split(indices, args.folds)

    results_detail: List[dict] = []
    summary_metrics: List[dict] = []

    for k in range(args.folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(args.folds) if j != k])

        train_df = master_df.iloc[train_idx]
        test_df = master_df.iloc[test_idx]

        # Train log10(a0) ~ intercept + slope*log10(SB)
        lx = np.log10(train_df["SB_proxy"].values)
        ly = np.log10(train_df["a0_btfr"].values)

        if len(lx) < 5:
            slope, intercept = 0.0, np.log10(np.median(train_df["a0_btfr"].values))
        else:
            slope, intercept = np.polyfit(lx, ly, 1)

        a0_median_train = float(np.median(train_df["a0_btfr"].values))

        fold_res_A: List[float] = []
        fold_res_B: List[float] = []

        for _, row in test_df.iterrows():
            gname = row["name_norm"]
            sb_val = float(row["SB_proxy"])
            a0_true = float(row["a0_btfr"])

            a0_pred_B = float(10 ** (intercept + slope * np.log10(sb_val)))

            gal_rar = rar_data_map[gname]
            rms_A = calculate_galaxy_rms(gal_rar, a0_median_train)
            rms_B = calculate_galaxy_rms(gal_rar, a0_pred_B)

            fold_res_A.append(rms_A)
            fold_res_B.append(rms_B)

            results_detail.append({
                "fold": k + 1,
                "galaxy": gname,
                "SB_proxy": sb_val,
                "a0_btfr_obs": a0_true,
                "a0_pred_var": a0_pred_B,
                "a0_fixed_train": a0_median_train,
                "rms_modelA_fixed": rms_A,
                "rms_modelB_variable": rms_B,
                "delta_rms": rms_A - rms_B
            })

        mean_rms_A = float(np.mean(fold_res_A))
        mean_rms_B = float(np.mean(fold_res_B))
        median_rms_A = float(np.median(fold_res_A))
        median_rms_B = float(np.median(fold_res_B))

        summary_metrics.append({
            "fold": k + 1,
            "slope": float(slope),
            "intercept": float(intercept),
            "mean_rms_A": mean_rms_A,
            "mean_rms_B": mean_rms_B,
            "median_rms_A": median_rms_A,
            "median_rms_B": median_rms_B,
            "improvement_mean": mean_rms_A - mean_rms_B,
            "improvement_median": median_rms_A - median_rms_B
        })

        print(
            f"  Fold {k+1}: slope={slope:.3f} | "
            f"MeanRMS Fix={mean_rms_A:.4f}/Var={mean_rms_B:.4f} | "
            f"MedRMS Fix={median_rms_A:.4f}/Var={median_rms_B:.4f}"
        )

    # 6) Global summary
    res_df = pd.DataFrame(results_detail)
    global_mean_A = float(res_df["rms_modelA_fixed"].mean())
    global_mean_B = float(res_df["rms_modelB_variable"].mean())
    global_median_A = float(res_df["rms_modelA_fixed"].median())
    global_median_B = float(res_df["rms_modelB_variable"].median())

    wins = int((res_df["rms_modelB_variable"] < res_df["rms_modelA_fixed"]).sum())
    win_rate = 100.0 * wins / max(1, len(res_df))

    print("=" * 60)
    print("FINAL RESULTS (Disk-Dominated Only)")
    print(f"Filter: q{q_label_str}%(f_bul) < {args.max_bulge_frac}")
    print(f"Galaxies Tested: {len(res_df)}")
    print(f"Model A (Fixed a0):     Mean={global_mean_A:.4f}, Median={global_median_A:.4f} dex")
    print(f"Model B (Variable a0):  Mean={global_mean_B:.4f}, Median={global_median_B:.4f} dex")
    print(f"Net Mean Improvement:   {global_mean_A - global_mean_B:+.4f} dex")
    print(f"Win Rate:               {wins}/{len(res_df)} ({win_rate:.1f}%)")

    # 7) Quartile breakdown
    print("-" * 60)
    print(">>> BREAKDOWN by Surface Brightness (SB_proxy) Quartiles")
    try:
        res_df["log_sb"] = np.log10(res_df["SB_proxy"].astype(float))
        res_df["sb_quartile"] = pd.qcut(res_df["log_sb"], 4, labels=["Q1 (Low SB)", "Q2", "Q3", "Q4 (High SB)"])
        grp = res_df.groupby("sb_quartile", observed=False)

        q_win = grp["delta_rms"].apply(lambda x: float((x > 0).mean() * 100.0))
        q_cnt = grp["delta_rms"].count()
        q_mean = grp["delta_rms"].mean()

        print(f"{'Quartile':<15} | {'Count':<5} | {'Win Rate':<10} | {'Mean Improv. (dex)':<20}")
        print("-" * 60)
        for cat in ["Q1 (Low SB)", "Q2", "Q3", "Q4 (High SB)"]:
            wr = float(q_win[cat])
            cnt = int(q_cnt[cat])
            mn = float(q_mean[cat])
            mark = "★" if wr >= 60 else ""
            print(f"{cat:<15} | {cnt:<5} | {wr:5.1f}% {mark:<3} | {mn:+.4f}")
    except Exception as e:
        print(f"Warning: Could not compute quartile stats ({e})")

    print("=" * 60)

    # 8) Save outputs
    res_df.to_csv(outdir / "prediction_c1_disk_details.csv", index=False)
    pd.DataFrame(summary_metrics).to_csv(outdir / "prediction_c1_fold_summary.csv", index=False)

    # Plot (optional)
    if (not args.no_plot) and HAS_PLT:
        plt.figure(figsize=(8, 6))
        plt.scatter(np.log10(res_df["SB_proxy"].values), res_df["delta_rms"].values, alpha=0.6)
        plt.axhline(0, color="k", linestyle="--")
        plt.xlabel(r"$\log_{10}(\mathrm{SB}_{\mathrm{proxy}})$")
        plt.ylabel("Improvement (Fixed - Var) [dex]")
        plt.title(f"Disk-Dominated (q{q_label_str}% < {args.max_bulge_frac})\nWin Rate: {win_rate:.1f}%")
        plt.tight_layout()
        plt.savefig(outdir / "fig_c1_disk_improvement.png", dpi=150)
        plt.close()
    elif (not args.no_plot) and (not HAS_PLT):
        print("NOTE: matplotlib not available; skipping plots.")
    else:
        print("Plots disabled (--no_plot).")

    # Run summary (hash + config)
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
