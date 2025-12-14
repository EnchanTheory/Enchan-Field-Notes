# VALIDATION: SPARC rotation-curve prediction (fixed-parameter)

This file records minimal checks to accept the rotation-curve prediction artifact for release.

## 1) Input provenance
- Input ZIP: `Rotmod_LTG.zip`
- Record sha256 printed by the script and written to:
  - `sparc_vpred_global_summary_*.csv`

## 2) Reproduction command
```bash
python enchan_rotationcurve_reproduce.py --zip Rotmod_LTG.zip
```

## 3) Required outputs
- `sparc_vpred_points_*.csv`
- `sparc_vpred_galaxy_summary_*.csv`
- `sparc_vpred_global_summary_*.csv`
- `fig_vpred_vs_vobs.png`
- `fig_resid_logg_hist.png`
- `fig_galaxy_rms_hist.png`

## 4) Acceptance criteria (practical)
- Script runs end-to-end with no manual editing.
- `galaxies` and `points` are non-zero and plausible for Rotmod_LTG.
- Global metrics should be close to the TeX report benchmark for the same fixed parameters.
