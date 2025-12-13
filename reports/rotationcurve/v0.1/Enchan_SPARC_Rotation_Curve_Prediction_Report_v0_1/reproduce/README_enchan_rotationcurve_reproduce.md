# Enchan SPARC rotation-curve prediction reproducibility

> **Status (baseline / non-Enchan):**  
> **Not derived from Enchan equations. Baseline only.**  
> This package validates a published empirical mapping (RAR/MDAR-style) on public SPARC mass-model data.  
> It is kept here as a reproducible benchmark and regression test for future Enchan-derived models.

This package reproduces the **fixed-parameter rotation-curve prediction test** from SPARC `Rotmod_LTG` mass-model files.

## What it does
From each galaxy's `*_rotmod.dat` table:

- computes observed acceleration `g_obs(r)=Vobs^2/r`
- computes baryonic proxy `g_bar(r)=(Vgas^2 + Yd*Vdisk^2 + Yb*Vbul^2)/r`
- maps `g_bar -> g_pred` using the one-parameter empirical curve used in RAR/MDAR literature
- predicts rotation speed `V_pred(r)=sqrt(g_pred*r)`
- outputs CSV tables + diagnostic figures (and optionally a minimal TeX stub)
**Note:** The `g_bar -> g_pred` mapping used here is an empirical baseline and is **not** derived from Enchan equations.

This is a **reproducibility/validation tool**, not a per-galaxy fit.

## Input (local; do not commit)
You need the public SPARC file:

- `Rotmod_LTG.zip` (contains `*_rotmod.dat` per galaxy)

Keep the ZIP local. The script prints and stores the file **sha256** to make runs comparable.

## Install
```bash
python -m pip install -r requirements_rotationcurve.txt
```

## Run
```bash
python enchan_rotationcurve_reproduce.py --zip Rotmod_LTG.zip
# also generate a minimal TeX stub:
python enchan_rotationcurve_reproduce.py --zip Rotmod_LTG.zip --make-tex
```

## Outputs
Written to `Enchan_SPARC_Rotation_Curve_Prediction_Report_v0_1/` by default:

- `sparc_vpred_points_Yd0.60_Yb0.70_a0fixed.csv`
- `sparc_vpred_galaxy_summary_Yd0.60_Yb0.70_a0fixed.csv`
- `sparc_vpred_global_summary_Yd0.60_Yb0.70_a0fixed.csv`
- `fig_vpred_vs_vobs.png`
- `fig_resid_logg_hist.png`
- `fig_galaxy_rms_hist.png`
- `enchan_sparc_rotationcurve_prediction_report_v0p1.tex` (only if `--make-tex`)

## Dependencies
- numpy
- pandas
- matplotlib
