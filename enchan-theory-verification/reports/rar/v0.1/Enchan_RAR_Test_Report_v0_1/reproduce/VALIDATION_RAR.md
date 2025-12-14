# RAR validation checklist (v0.2.4 gate)

This file defines what "verified" means for the RAR artifact.

## 0) Policy (data handling)
- `Rotmod_LTG.zip` is not committed.
- Record the input hash (sha256) in the run log and keep the zip locally.

## 1) Input integrity
Run the script once and record the printed hash:
- Rotmod_LTG.zip sha256: ______________________________

If you want a strict, pinned reproduction, also record:
- download date:
- source page / version notes (if any):

## 2) One-command reproduction
Run:
```bash
python enchan_rar_reproduce.py --zip Rotmod_LTG.zip
```

Confirm the script prints:
- galaxies count
- points count (all)
- points count (SBdisk > 0)
- baseline (Yd=0.50, Yb=0.70): a0 and RMS
- recommended (Yd=0.60, Yb=0.70): a0 and RMS
- scan best Ydisk (min |rho|)

## 3) Artifact completeness
Confirm these files exist in `Enchan_RAR_Test_Report_v0_1/`:
- CSV:
  - `sparc_rar_points_processed.csv`
  - `rar_Ydisk_scan_results.csv`
  - `sparc_rar_galaxy_medians_*.csv`
- PNG:
  - `fig_rar_points.png`
  - `fig_resid_sb_binned.png`
  - `fig_scan_rho_sb.png`
  - `fig_scan_rms.png`
  - `fig_rar_galaxy.png`
- TeX:
  - `enchan_rar_test_report_v0p1.tex`

## 4) Acceptance criteria (pinned input hash)
If (and only if) the input zip hash matches:
- sha256 = 0a80cc90714828cc28b7dd57923576714d209f2490328c087c4a4ad607faf588

then the following should match (within small rounding tolerance):
- galaxies: 175
- points (all): 3391
- points (SBdisk > 0): 3111
- baseline (Yd=0.50, Yb=0.70): a0 ≈ 1.392e-10 m/s^2, RMS ≈ 0.213 dex
- recommended (Yd=0.60, Yb=0.70): a0 ≈ 1.119e-10 m/s^2, RMS ≈ 0.212 dex
- scan best (min |rho|): Ydisk ≈ 0.60

Recommended tolerances (for log-space comparison):
- |log10(a0) - log10(expected)| <= 0.01
- |RMS - expected| <= 0.005 dex

If the zip hash differs:
- treat the run as a *new pinned input*;
- update this section with the new sha256 and the new printed counts/fit numbers before release.
