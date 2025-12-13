# Enchan RAR reproduce

> **Status (baseline / non-Enchan):**  
> **Not derived from Enchan equations. Baseline only.**  
> This package validates a published empirical mapping (RAR/MDAR-style) on public SPARC mass-model data.  
> It is kept here as a reproducible benchmark and regression test for future Enchan-derived models.

This script reproduces a minimal SPARC RAR benchmark from the public `Rotmod_LTG.zip` archive.

> Note: `Rotmod_LTG.zip` is **not** committed to this repository.
> Download it from the SPARC site and place it next to the script (or pass a path via `--zip`).

## Input
- `Rotmod_LTG.zip` (contains 175 `*_rotmod.dat` files)

## Run
```bash
python enchan_rar_reproduce.py --zip Rotmod_LTG.zip
```

## Outputs
A directory is created (default: `Enchan_RAR_Test_Report_v0_1/`) containing:
- `sparc_rar_points_processed.csv`
- `rar_Ydisk_scan_results.csv`
- `sparc_rar_galaxy_medians_Yd0p60_Yb0p70.csv` (numbers use `p` instead of `.`, extension stays `.csv`)
- figures:
  - `fig_rar_points.png`
  - `fig_resid_sb_binned.png`
  - `fig_scan_rho_sb.png`
  - `fig_scan_rms.png`
  - `fig_rar_galaxy.png`
- TeX report:
  - `enchan_rar_test_report_v0p1.tex`

## Notes
- The fit is deterministic (grid search in log10(a0)).
- SB-clean subset is defined as `SBdisk > 0`.
