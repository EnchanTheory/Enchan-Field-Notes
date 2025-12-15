# Enchan Field Notes (Theoretical Framework)

**Current version:** v0.3.1  
**Status:** speculative research note + public-data reproducibility package

Enchan Field Notes is a collection of theoretical memoranda exploring whether several
well-known galaxy-scale regularities can be described by an effective field picture
(topological defects in a dimensionless “time-dilation” field `S`).

This repository includes:
- a LaTeX/PDF note (`main.pdf`, `sections/`)
- a small Python reproducibility package based on public SPARC products (`enchan-theory-verification/`)

This work is not peer-reviewed. It does not claim to falsify particle dark matter.
In v0.3.x, several scripts use an **effective closure** as a benchmark mapping (see below);
a field-theoretic derivation is a theory-side goal rather than an established result.

---

## What is reproduced here (public-data benchmarks)

The reproducibility package focuses on three SPARC-based empirical regularities:

1. **RAR / MDAR**: across many radii and galaxies, observed acceleration tightly tracks baryonic acceleration.
2. **BTFR**: baryonic mass tightly correlates with outer/flat rotation velocity.
3. **Rotation-curve shape stress test**: with baryonic components + a single fixed rule, one can generate nontrivial curve shapes across a large sample (no per-galaxy tuning).

These are treated as **externally checkable targets** and regression tests.

---

## The benchmark closure used in v0.3.x scripts

Several scripts use an effective one-parameter mapping as a working hypothesis:

$$
g_{\mathrm{tot}} = \sqrt{g_{\mathrm{bar}}^2 + a_0\,g_{\mathrm{bar}}}
$$

This repository reproduces the above benchmarks using standard SPARC-style definitions and
reports scatter/diagnostics. This should be read as **reproducibility of the benchmark mapping**,
not as a completed derivation from a unique fundamental theory.

---

## What’s new in v0.3.1

- **Theory-side refinement (Chapter 6):** the note formalizes a “surface-density anchor” ansatz:
  the apparent acceleration scale `a0` is treated as an emergent scale linked (directly or indirectly)
  to a baryonic surface-density proxy at an anchor radius.
- **New exploratory analysis:** `enchan_a0_sb_correlation.py` checks whether a BTFR-derived `a0`
  shows any trend with a surface-brightness proxy extracted from SPARC rotmod files.

Important caveat: SPARC `SBdisk` is a **luminosity** surface density; no mass-to-light correction is applied.
Any correlation (or lack thereof) can reflect proxy limitations, selection effects, or self-regulation mechanisms.
Interpretation scenarios are pre-registered in the script docstring.

---

## Repository structure

- `main.pdf`: compiled note (Enchan Field Notes)
- `sections/`: LaTeX sources
- `enchan-theory-verification/`: reproducibility package (scripts, outputs, hashes)

---

## Verification (quick start)

The reproducibility package has its own README with detailed instructions:

- Go to `enchan-theory-verification/`
- Download SPARC public products:
  - `Rotmod_LTG.zip`
  - `BTFR_Lelli2019.mrt`
  from the SPARC website: http://astroweb.case.edu/SPARC/
- Run the scripts as described in `enchan-theory-verification/README.md`

---

## Disclaimer

This repository contains theoretical and cosmological descriptions.
Technological implementations, specific device architectures (e.g., “Enchan-001” referenced in the text),
and industrial applications are out of scope and are not disclosed here.

---

## Citation

If you wish to cite this work:

> Kobayashi, M. (2025). *Enchan Field Notes: The Inception World and Topological Defects*. v0.3.1. GitHub Repository.

---

## License

This repository uses a hybrid licensing scheme:

- **Source code / verification scripts (`enchan-theory-verification/`):**
  MIT License (see `LICENSE`)
- **Textual content of Enchan Field Notes (`main.tex`, PDF, and `sections/`):**
  Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You are free to share and adapt the text and PDF for **non-commercial** purposes,
as long as you provide proper attribution and indicate if changes were made.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17928705.svg)](https://doi.org/10.5281/zenodo.17928705)

---

*Note: This is a living document and subject to change without notice.*
