# Enchan Theory Verification (public-data reproducibility)

This folder contains **reproducible, public-data benchmarks** used in *Enchan Field Notes v0.3.x*.

## What this is (and what it is not)

- **This is** a set of scripts that reproduce three well-known SPARC-based empirical regularities and related diagnostics.
- **This is** intended for *external checkability*: same inputs → same outputs (hash-logged).
- **This is not** a proof of Enchan Theory, and it does not rule out particle dark matter.
- **Important:** Several steps use an **effective closure** (“Enchan quadrature law”) as a working hypothesis:
  \[
  g_{\rm tot} = \sqrt{g_{\rm bar}^2 + a_0\,g_{\rm bar}}
  \]
  In v0.3.x, the **field-theoretic derivation of this closure and of \(a_0\)** is the *theory-side goal*; these scripts provide **targets + regression tests**.

---

## The three observational targets

These scripts focus on three “repeatable patterns” seen across many galaxies:

1. **RAR / MDAR** — at many radii, the observed acceleration is tightly linked to the baryonic one.
2. **BTFR** — the baryonic mass of a galaxy is tightly linked to its outer/flat rotation speed.
3. **Rotation-curve shapes** — using baryonic components + a fixed rule, you can predict nontrivial curve shapes across a large sample (stress test).

---

## Inputs (not committed)

You need the following **public SPARC files** locally:

- `Rotmod_LTG.zip` (mass-model rotation-curve decomposition files)
- `BTFR_Lelli2019.mrt` (SPARC BTFR table; CDS-style fixed-width)

Source: SPARC website  
https://astroweb.case.edu/SPARC/

**Repository policy:** large upstream datasets are **not committed**.  
Each script records the **SHA256** of your local input file(s) so results are comparable.

---

## Requirements

- Python 3.10+ recommended
- `numpy`, `pandas`, `matplotlib`
- `scipy` is **optional** (only used for p-values in correlation tests)

Install minimal dependencies:

```bash
python -m pip install numpy pandas matplotlib
```

Optional:

```bash
python -m pip install scipy
```

---

## Scripts in this folder

### 1) BTFR benchmark (+ implied (a_0) distribution)

Computes BTFR points, a simple log–log fit, and the implied per-galaxy:
[
a_{0,\rm BTFR} = \frac{V_f^4}{G,M_b}.
]

```bash
python enchan_btfr_reproduce_enchan.py --mrt BTFR_Lelli2019.mrt
# optional quality cut (if elogMb exists)
python enchan_btfr_reproduce_enchan.py --mrt BTFR_Lelli2019.mrt --max_elogMb 0.10
```

Outputs (default): `Enchan_BTFR_Test_Report_v0_1/`
Key outputs include CSV tables and figures, plus `btfr_a0_summary.csv`.

---

### 2) RAR benchmark using the Enchan effective closure

Uses SPARC `Rotmod_LTG.zip` to compute (g_{\rm bar}(r)) and compares to observed accelerations,
using the **Enchan quadrature law** as the mapping.

Requires an (a_0) value (recommended: BTFR median from step 1).

```bash
python enchan_rar_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.5e-10
# optional: override mass-to-light ratios
python enchan_rar_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.5e-10 --Yd 0.60 --Yb 0.70
```

Outputs (default): `Enchan_RAR_Test_Report_v0_1/`

---

### 3) Rotation-curve prediction (shape stress test) using the same closure

Predicts (V_{\rm pred}(r)) from baryonic components and the same closure, with **fixed global parameters**
(no per-galaxy tuning).

Requires an (a_0) value (recommended: BTFR median from step 1).

```bash
python enchan_rotationcurve_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.5e-10
# optional: override mass-to-light ratios
python enchan_rotationcurve_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.5e-10 --Yd 0.60 --Yb 0.70
```

Outputs (default): `Enchan_SPARC_Rotation_Curve_Prediction_Report_v0_1/`

---

### 4) Exploration: (a_0) vs surface-brightness proxy (v0.3.1)

This is an **exploratory** test of the Chapter 6 “surface-density anchor” idea.

* (a_0) is derived from BTFR: (a_0 \sim V_f^4/(G M_b))
* Surface brightness proxy is derived from `Rotmod_LTG.zip`:
  **SB_proxy = median of the innermost 3 positive `SBdisk` points** (robust to outliers)

Caveat: `SBdisk` is a *luminosity* surface density. It is used as a proxy for baryonic surface density
without correcting for M/L variations (adds scatter).

```bash
python enchan_a0_sb_correlation.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip
```

Outputs (default): `Enchan_Correlation_Test_v0_3_1/`

* `a0_sb_correlation_data.csv`
* `correlation_summary.csv`
* `fig_a0_sb_correlation.png`

**Interpretation is pre-registered in the script docstring** (P1 / P2 / P3 scenarios):

* P1: weak positive correlation (minimal anchor picture)
* P2: near-zero correlation (self-regulation / cancellation)
* P3: unstable/noisy (proxy limitation dominates)

---

## Recommended run order (quickstart)

1. Put `BTFR_Lelli2019.mrt` and `Rotmod_LTG.zip` next to these scripts.
2. Run BTFR and read the **median (a_0)** from the output CSV.
3. Use that (a_0) when running the RAR and rotation-curve scripts.
4. Run the correlation script to explore the Chapter 6 anchor ansatz.

---

## Notes for external reviewers

* All scripts log **input SHA256** for reproducibility.
* Name matching across datasets uses a **normalized galaxy-name key** (removes separators and leading zeros).
* These benchmarks are intentionally minimal and transparent:
  the goal is **checkability**, not maximum model complexity.