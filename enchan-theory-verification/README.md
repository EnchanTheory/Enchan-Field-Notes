# Enchan Theory Verification (public-data benchmarks)

This folder contains a small set of reproducible, public-data benchmarks referenced by
**Enchan Field Notes v0.4.3**.

## What this is (and what it is not)

* This is a collection of scripts to reproduce several widely discussed **galaxy-scale empirical regularities**
  using public observational products and standard SPARC-style mass-model definitions.
* This is intended for external checkability: given the same local inputs, the scripts generate the same outputs
  (input file hashes are recorded where applicable).
* This is **not** a proof of a unique theory and does not rule out particle dark matter.

Some scripts use a compact empirical mapping as a **baseline benchmark interface**.
This baseline is kept only as a regression target and should not be read as a completed derivation.

## Benchmarks covered

The scripts in this folder evaluate three benchmark targets commonly used in the literature:

1. **RAR / MDAR** (multi-point relation across radii)
2. **BTFR** (one galaxy = one point)
3. **Rotation-curve shape stress test** (fixed-rule prediction without per-galaxy tuning)

## Inputs (not committed)

Upstream public datasets are not committed to this repository.

You need the following public SPARC products locally:

* `Rotmod_LTG.zip` (mass-model rotation-curve decomposition files)
* `BTFR_Lelli2019.mrt` (SPARC BTFR table; CDS-style fixed-width)

Source (SPARC):
[https://astroweb.case.edu/SPARC/](https://astroweb.case.edu/SPARC/)

Each run records hashes of local input files so results can be compared across environments.

## Requirements

* Python 3.x
* `numpy`, `pandas`, `matplotlib`
* `scipy` is optional (used only for some statistics)

Install minimal dependencies:

```bash
python -m pip install numpy pandas matplotlib
```

Optional:

```bash
python -m pip install scipy
```

## How to run (minimum)

Run commands from **this directory** (`enchan-theory-verification/`) and place the SPARC input files next to the scripts
(or pass explicit paths).

### 1) BTFR benchmark (one point per galaxy)

```bash
python enchan_btfr_reproduce_enchan.py --mrt BTFR_Lelli2019.mrt
```

### 2) RAR benchmark (multi-point relation)

```bash
python enchan_rar_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.2e-10
```

### 3) Rotation-curve prediction stress test (shape)

```bash
python enchan_rotationcurve_reproduce_enchan.py --zip Rotmod_LTG.zip --a0 1.2e-10
```

### 4) C1 (disk-dominated) variable-a0 prediction test

```bash
python enchan_variable_a0_prediction.py --mrt BTFR_Lelli2019.mrt --zip Rotmod_LTG.zip --max_bulge_frac 0.5 --bulge_quantile 0.95
```

If you are unsure about options or output locations, run:

```bash
python <script_name>.py --help
```

and read the docstring at the top of the file.

## Outputs

Scripts write outputs to local result directories (tables and figures). These outputs are intended for:

* sanity checks (does it reproduce the known benchmark trend?)
* regression comparisons (same inputs → comparable outputs)

## License

See the repository root `LICENSE`.