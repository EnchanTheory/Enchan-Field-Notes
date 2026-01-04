#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan C1 Pinning Test (Strict Nested CV, leakage-guarded) - v0.4.3

What this script does
---------------------
A strict, nested cross-validation test of the Phi-screening ("pinning") mechanism.

Model components
----------------
1) Free a0 model (trained ONLY on training data inside each fold):
      log10(a0_free) = intercept + slope * log10(SB_proxy)

2) Phi-screening (pinning) with global parameters (phi_c, n) optimized ONLY in
   the inner CV loop:
      a0_eff = a0_free * S_phi(|Phi|)
      S_phi(|Phi|) = [1 + (|Phi|/Phi_c)^n]^{-1}

3) RAR-style benchmark closure:
      g_pred = sqrt(g_bar^2 + a0_eff * g_bar)

Baselines (for honest comparison)
---------------------------------
A) Constant a0 baseline: a0 = median(a0_btfr) from training set.
B) SB-only baseline:     a0 = a0_free (no screening).
C) SB+Pinning model:     a0 = a0_free screened by Phi (phi_c, n).

Primary outcome
---------------
We evaluate per-galaxy RMS in log space:
    RMS = sqrt(mean( (log10(g_obs) - log10(g_pred))^2 ))
and compare improvements:
    dRMS = RMS_baseline - RMS_model
Positive dRMS => model improves fit.

Strictness / anti-leakage rules
-------------------------------
- Outer test fold is NEVER used for:
    * fitting SB->a0_free
    * choosing (phi_c, n)
    * choosing any thresholds
- Inner CV optimizes (phi_c, n) by average objective over inner folds.
- Splits are randomized but reproducible (seeded).

Inputs
------
--mrt : BTFR_Lelli2019.mrt
--zip : Rotmod_LTG.zip
--inner_cut : inner radius cut in kpc for RAR points (default 1.0)

Notes
-----
- Requires local helper modules:
    enchan_btfr_reproduce_enchan, enchan_a0_sb_correlation,
    enchan_variable_a0_prediction, enchan_core_model_plus, enchan_core_model
- Uses calculate_phi_median_proxy from enchan_core_model_plus
- Uses apply_screening from enchan_core_model_plus (Phi-screening only)

Usage
-----
python enchan_c1_pinning_test_strict.py \
  --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip --inner_cut 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from itertools import product
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Imports (project-local)
# ---------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from enchan_core_model import KMS_TO_MS, G_SI, MSUN_KG
    from enchan_btfr_reproduce_enchan import parse_mrt_fixedwidth, extract_btfr
    from enchan_a0_sb_correlation import get_sb_proxy, norm_name
    from enchan_variable_a0_prediction import load_rar_data_by_galaxy
    from enchan_core_model_plus import calculate_phi_median_proxy, apply_screening
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import Enchan modules: {e}")
    sys.exit(1)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SEED = 42

# Search grid for (Phi_c, n) in units of (km/s)^2 for Phi_c
PHI_C_GRID = [10000, 22500, 40000, 62500, 90000, 160000]
N_GRID = [1.0, 2.0, 4.0]


# ---------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------
def safe_log10(x: np.ndarray, floor: float = 1e-15) -> np.ndarray:
    """Safe log10 with a configurable positive floor."""
    xx = np.asarray(x, dtype=float)
    return np.log10(np.maximum(xx, float(floor)))


def calc_rms_log_residual(df_rar: pd.DataFrame, a0_eff: float, inner_cut_kpc: float) -> float:
    """
    Per-galaxy RMS in log space using the benchmark closure:
        g_pred = sqrt(g_bar^2 + a0_eff * g_bar)
    """
    df = df_rar[df_rar["r_kpc"] >= float(inner_cut_kpc)]
    if len(df) < 3:
        return float("nan")

    gb = df["g_bar"].to_numpy(dtype=float)
    go = df["g_obs"].to_numpy(dtype=float)

    # Defensive: require finite
    m = np.isfinite(gb) & np.isfinite(go)
    gb = gb[m]
    go = go[m]
    if gb.size < 3:
        return float("nan")

    # Require non-negative baryonic acceleration
    gb = np.maximum(gb, 0.0)

    # Enforce a0_eff finite and non-negative
    if not np.isfinite(a0_eff) or a0_eff < 0:
        return float("nan")

    g_pred = np.sqrt(gb * gb + a0_eff * gb)

    resid = safe_log10(go) - safe_log10(g_pred)
    return float(np.sqrt(np.mean(resid * resid)))


def fit_sb_to_a0(train_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Fit: log10(a0_btfr) = intercept + slope * log10(SB_proxy)
    Returns (slope, intercept).
    Falls back to a gentle default if insufficient data.
    """
    x = np.log10(train_df["SB_proxy"].to_numpy(dtype=float))
    y = np.log10(train_df["a0_btfr"].to_numpy(dtype=float))

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size >= 6 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)

    # Fallback: MOND-ish constant scale
    return 0.0, float(np.log10(1.2e-10))


def predict_a0_free(sb_val: float, slope: float, intercept: float) -> float:
    if not np.isfinite(sb_val) or sb_val <= 0:
        return float("nan")
    return float(10.0 ** (intercept + slope * np.log10(sb_val)))


def get_param_grid() -> List[Tuple[float, float]]:
    return list(product(PHI_C_GRID, N_GRID))


# ---------------------------------------------------------
# CV splitting (stratified by quartile label)
# ---------------------------------------------------------
def stratified_kfold_indices(
    y_strata: np.ndarray,
    n_splits: int,
    rng: np.random.Generator
) -> List[np.ndarray]:
    """
    Produce stratified folds (indices arrays), without scikit-learn.
    y_strata: array of discrete labels (e.g., 'Q1','Q2','Q3','Q4')
    """
    y = np.asarray(y_strata)
    unique = pd.unique(y)
    folds: List[List[int]] = [[] for _ in range(n_splits)]

    for lab in unique:
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, n_splits)
        for k in range(n_splits):
            folds[k].extend(parts[k].tolist())

    return [np.array(sorted(f), dtype=int) for f in folds]


# ---------------------------------------------------------
# Evaluation for a fold set
# ---------------------------------------------------------
def eval_models_on_rows(
    df_rows: pd.DataFrame,
    rar_map: Dict[str, pd.DataFrame],
    inner_cut_kpc: float,
    slope: float,
    intercept: float,
    a0_const_train: float,
    phi_c: Optional[float],
    n: Optional[float],
) -> pd.DataFrame:
    """
    Evaluate baselines and model on the provided rows (galaxies).
    Returns per-galaxy results with RMS and dRMS values.
    """
    out = []
    for _, row in df_rows.iterrows():
        gname = row["name_norm"]
        sb_val = float(row["SB_proxy"])
        phi_val = float(row["phi_proxy"])

        if gname not in rar_map:
            continue

        df_rar = rar_map[gname]

        # Baseline A: constant a0 (no screening)
        rms_A = calc_rms_log_residual(df_rar, a0_const_train, inner_cut_kpc)

        # Baseline B: SB-only a0_free (no screening)
        a0_free = predict_a0_free(sb_val, slope, intercept)
        rms_SB = calc_rms_log_residual(df_rar, a0_free, inner_cut_kpc)

        # Model C: SB + pinning (if phi_c provided)
        if phi_c is None or float(phi_c) <= 0:
            # screening disabled explicitly
            a0_eff = a0_free
        else:
            # screening enabled; require finite phi
            if not np.isfinite(phi_val):
                a0_eff = float("nan")
            else:
                a0_eff = apply_screening(a0_free, phi_val, float(phi_c), float(n))

        rms_PIN = calc_rms_log_residual(df_rar, float(a0_eff), inner_cut_kpc)

        if not (np.isfinite(rms_A) and np.isfinite(rms_SB) and np.isfinite(rms_PIN)):
            continue

        out.append({
            "name_norm": gname,
            "quartile": row["quartile"],
            "RMS_const": rms_A,
            "RMS_sb": rms_SB,
            "RMS_pin": rms_PIN,
            "dRMS_sb_vs_const": rms_A - rms_SB,
            "dRMS_pin_vs_const": rms_A - rms_PIN,
            "dRMS_pin_vs_sb": rms_SB - rms_PIN,
        })

    return pd.DataFrame(out)


# ---------------------------------------------------------
# Nested CV
# ---------------------------------------------------------
def run_nested_cv(
    master_df: pd.DataFrame,
    rar_map: Dict[str, pd.DataFrame],
    inner_cut_kpc: float,
    k_outer: int = 5,
    k_inner: int = 4,
) -> Tuple[pd.DataFrame, List[Tuple[float, float]]]:
    """
    Strict nested CV:
    - Outer folds: evaluate final generalization
    - Inner folds: choose (phi_c, n) by average objective on inner validation folds
    Objective: maximize improvement in Q4 while avoiding harm in Q1,
               using dRMS_pin_vs_sb as the improvement metric (pinning vs SB-only).
    """
    rng = np.random.default_rng(SEED)
    param_grid = get_param_grid()

    # Outer folds stratified by quartile
    outer_folds = stratified_kfold_indices(master_df["quartile"].to_numpy(), k_outer, rng)

    all_test_rows = []
    best_params_log: List[Tuple[float, float]] = []

    print(f"Starting Nested CV: outer={k_outer}, inner={k_inner}, grid={len(param_grid)}")

    for fold_i in range(k_outer):
        test_idx = outer_folds[fold_i]
        train_idx = np.setdiff1d(np.arange(len(master_df)), test_idx)

        train_df = master_df.iloc[train_idx].reset_index(drop=True)
        test_df = master_df.iloc[test_idx].reset_index(drop=True)

        # Inner folds (stratified) on training set ONLY
        inner_rng = np.random.default_rng(SEED + 100 + fold_i)
        inner_folds = stratified_kfold_indices(train_df["quartile"].to_numpy(), k_inner, inner_rng)

        # Choose best (phi_c, n) via inner CV
        best_score = -np.inf
        best_p = (PHI_C_GRID[0], N_GRID[0])

        for (phi_c, n) in param_grid:
            fold_scores = []

            for inner_k in range(k_inner):
                val_idx = inner_folds[inner_k]
                tr_idx = np.setdiff1d(np.arange(len(train_df)), val_idx)

                tr_inner = train_df.iloc[tr_idx]
                val_inner = train_df.iloc[val_idx]

                # Fit SB->a0 on tr_inner only
                slope, intercept = fit_sb_to_a0(tr_inner)
                a0_const = float(np.median(tr_inner["a0_btfr"].to_numpy(dtype=float)))

                # Evaluate on val_inner
                res_val = eval_models_on_rows(
                    df_rows=val_inner,
                    rar_map=rar_map,
                    inner_cut_kpc=inner_cut_kpc,
                    slope=slope,
                    intercept=intercept,
                    a0_const_train=a0_const,
                    phi_c=phi_c,
                    n=n,
                )
                if res_val.empty:
                    fold_scores.append(np.nan)
                    continue

                # Metric: pinning improvement over SB-only, split by quartile
                q1 = res_val[res_val["quartile"] == "Q1"]["dRMS_pin_vs_sb"].to_numpy(dtype=float)
                q4 = res_val[res_val["quartile"] == "Q4"]["dRMS_pin_vs_sb"].to_numpy(dtype=float)

                # Require both present to avoid "empty -> 0" bias
                if q1.size == 0 or q4.size == 0:
                    fold_scores.append(np.nan)
                    continue

                mean_q1 = float(np.mean(q1))
                mean_q4 = float(np.mean(q4))

                # Objective: prefer improving high-SB (Q4) while penalizing harm in low-SB (Q1)
                # If Q1 improvement is negative (harm), penalize it.
                score = mean_q4 - 2.0 * max(0.0, -mean_q1)
                fold_scores.append(score)

            fold_scores = np.asarray(fold_scores, dtype=float)
            if not np.any(np.isfinite(fold_scores)):
                continue

            score_avg = float(np.nanmean(fold_scores))
            if np.isfinite(score_avg) and score_avg > best_score:
                best_score = score_avg
                best_p = (phi_c, n)

        best_params_log.append(best_p)
        phi_c_opt, n_opt = best_p

        # Outer evaluation on untouched test set using models fit on full train_df
        slope_full, intercept_full = fit_sb_to_a0(train_df)
        a0_const_full = float(np.median(train_df["a0_btfr"].to_numpy(dtype=float)))

        res_test = eval_models_on_rows(
            df_rows=test_df,
            rar_map=rar_map,
            inner_cut_kpc=inner_cut_kpc,
            slope=slope_full,
            intercept=intercept_full,
            a0_const_train=a0_const_full,
            phi_c=phi_c_opt,
            n=n_opt,
        )
        if res_test.empty:
            continue

        res_test["fold"] = fold_i
        res_test["phi_c_opt"] = phi_c_opt
        res_test["n_opt"] = n_opt
        all_test_rows.append(res_test)

        print(f"[Outer fold {fold_i}] best(phi_c,n)=({phi_c_opt},{n_opt})  inner_score={best_score:.5f}  "
              f"test_n={len(res_test)}")

    if not all_test_rows:
        return pd.DataFrame(), best_params_log

    return pd.concat(all_test_rows, ignore_index=True), best_params_log


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrt", required=True, type=str)
    ap.add_argument("--zip", required=True, type=str)
    ap.add_argument("--inner_cut", type=float, default=1.0)
    ap.add_argument("--k_outer", type=int, default=5)
    ap.add_argument("--k_inner", type=int, default=4)
    args = ap.parse_args()

    print(f"--- Enchan C1 Pinning Test (Strict Nested CV) v0.4.3 | seed={SEED} ---")

    mrt_path = Path(args.mrt)
    zip_path = Path(args.zip)

    # -----------------------------
    # Load BTFR and compute a0_btfr
    # -----------------------------
    df_raw = parse_mrt_fixedwidth(mrt_path)
    df_btfr = extract_btfr(df_raw)

    Vf_ms = np.power(10.0, df_btfr["logVf"].to_numpy(dtype=float)) * KMS_TO_MS
    Mb_kg = np.power(10.0, df_btfr["logMb"].to_numpy(dtype=float)) * MSUN_KG
    a0_btfr = (Vf_ms ** 4) / (G_SI * Mb_kg)

    btfr_lookup = pd.DataFrame({
        "name_norm": df_btfr["name"].apply(norm_name),
        "a0_btfr": a0_btfr
    })
    btfr_lookup = btfr_lookup.dropna(subset=["name_norm", "a0_btfr"])
    btfr_lookup = btfr_lookup.groupby("name_norm", as_index=False)["a0_btfr"].median()

    # -----------------------------
    # Load SB proxy (median of innermost 3 positive SB points)
    # -----------------------------
    df_sb = get_sb_proxy(zip_path, n_points=3)  # columns: name_norm, SB_proxy
    if df_sb["name_norm"].duplicated().any():
        df_sb = df_sb.groupby("name_norm", as_index=False)["SB_proxy"].median()

    # -----------------------------
    # Load RAR-by-galaxy map
    # -----------------------------
    tmp = load_rar_data_by_galaxy(
        zip_path,
        Yd=0.5,
        Yb=0.7,
        max_bulge_frac=0.5,
        quantile_thr=0.95
    )
    rar_map = tmp if isinstance(tmp, dict) else tmp[0]

    # -----------------------------
    # Compute Phi proxy per galaxy (median of V_bar^2 ~ g_bar*r)
    # -----------------------------
    phi_rows = []
    for gname, df_rar in rar_map.items():
        phi = calculate_phi_median_proxy(
            r_kpc=df_rar["r_kpc"].to_numpy(dtype=float),
            g_bar=df_rar["g_bar"].to_numpy(dtype=float),
            inner_cut_kpc=float(args.inner_cut)
        )
        phi_rows.append({"name_norm": gname, "phi_proxy": phi})
    df_phi = pd.DataFrame(phi_rows)

    # -----------------------------
    # Merge master table (galaxy-level)
    # -----------------------------
    master = pd.merge(btfr_lookup, df_sb, on="name_norm", how="inner")
    master = pd.merge(master, df_phi, on="name_norm", how="inner")

    # Keep only galaxies that exist in rar_map
    valid_names = set(rar_map.keys())
    master = master[master["name_norm"].isin(valid_names)].copy()

    # Clean
    master = master.dropna(subset=["a0_btfr", "SB_proxy", "phi_proxy"])
    master = master[(master["a0_btfr"] > 0) & (master["SB_proxy"] > 0)].copy()
    master = master.drop_duplicates(subset=["name_norm"]).reset_index(drop=True)

    if len(master) < 12:
        print(f"CRITICAL: Too few galaxies after merging/cleaning: N={len(master)}")
        sys.exit(2)

    # Quartiles by log SB (for stratification and reporting)
    master["log_sb"] = np.log10(master["SB_proxy"].to_numpy(dtype=float))
    master["quartile"] = pd.qcut(master["log_sb"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    print(f"Dataset ready: N={len(master)} galaxies | inner_cut={args.inner_cut} kpc")
    print(master["quartile"].value_counts().sort_index().to_string())

    # -----------------------------
    # Run strict nested CV
    # -----------------------------
    res_df, best_params = run_nested_cv(
        master_df=master,
        rar_map=rar_map,
        inner_cut_kpc=float(args.inner_cut),
        k_outer=int(args.k_outer),
        k_inner=int(args.k_inner),
    )

    if res_df.empty:
        print("No valid results (all folds empty). Check data coverage / inner_cut.")
        sys.exit(3)

    # -----------------------------
    # Summaries
    # -----------------------------
    print("\n" + "=" * 70)
    print("RESULTS (Outer-test only; leakage-guarded)")
    print("=" * 70)

    def summarize(label: str, col: str) -> None:
        q1 = res_df[res_df["quartile"] == "Q1"][col].to_numpy(dtype=float)
        q4 = res_df[res_df["quartile"] == "Q4"][col].to_numpy(dtype=float)

        def stats(arr: np.ndarray) -> Tuple[float, float, float]:
            if arr.size == 0:
                return (np.nan, np.nan, np.nan)
            return (float(np.mean(arr)), float(np.median(arr)), float(np.mean(arr > 0)))

        m1, med1, win1 = stats(q1)
        m4, med4, win4 = stats(q4)

        print(f"{label}")
        print(f"  Q1: mean={m1:+.5f}  median={med1:+.5f}  win_rate={win1:.1%}  (n={q1.size})")
        print(f"  Q4: mean={m4:+.5f}  median={med4:+.5f}  win_rate={win4:.1%}  (n={q4.size})")

    summarize("SB-only improvement vs CONST (dRMS_sb_vs_const)", "dRMS_sb_vs_const")
    print("-" * 70)
    summarize("PIN improvement vs CONST (dRMS_pin_vs_const)", "dRMS_pin_vs_const")
    print("-" * 70)
    summarize("PIN improvement vs SB-only (dRMS_pin_vs_sb)", "dRMS_pin_vs_sb")

    print("-" * 70)
    print("Optimized (phi_c, n) per outer fold:")
    for i, p in enumerate(best_params):
        print(f"  Fold {i}: phi_c={p[0]} (km/s)^2, n={p[1]}")
    print("=" * 70)

    # Optional: save outputs
    out_csv = Path("enchan_c1_pinning_strict_results_v0_4_2.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved per-galaxy outer-test results to: {out_csv}")


if __name__ == "__main__":
    main()
