---
title: 'Enchan-Field-Notes: A Reproducible Python Framework for Testing Effective Field Descriptions of Galaxy Dynamics'
tags:
  - Python
  - astronomy
  - galactic dynamics
  - dark matter
  - effective field theory
  - SPARC
authors:
  - name: Mitsuhiro Kobayashi
    orcid: 0009-0008-0355-2704
    affiliation: 1
affiliations:
 - name: Enchan Project, Independent Researcher, Tokyo, Japan
   index: 1
date: 04 January 2026
bibliography: paper.bib
---

# Summary

The "missing mass" problem in galaxies—typically attributed to dark matter—remains one of the most significant open questions in astrophysics. Observational regularities such as the Radial Acceleration Relation (RAR) [@McGaugh:2016] and the Baryonic Tully-Fisher Relation (BTFR) [@Lelli:2019] suggest a tight coupling between baryonic mass distribution and total gravitational acceleration.

`Enchan-Field-Notes` is a Python software package designed to test effective field descriptions of these regularities against high-precision observational data. Specifically, it provides a deterministic verification pipeline for the **Enchan Field** framework [@Kobayashi:2025_theory], a scalar-field approach that models galactic acceleration scales as geometric defects anchored by baryonic matter.

Crucially, this software does not aim to prove a specific theory, but to provide a **transparent, reproducible, and rigorous testing ground** for effective field theories using the SPARC (Spitzer Photometry and Accurate Rotation Curves) database [@Lelli:2016].

# Statement of Need

In the field of galactic dynamics, comparing theoretical models to observational data often involves complex data cleaning, mass-to-light ratio assumptions, and diverse fitting procedures. This complexity can lead to reproducibility issues, where results depend heavily on unstated filtering choices or stochastic fitting parameters.

`Enchan-Field-Notes` addresses this need by providing:

1.  **Standardized Parsing:** Robust parsers for SPARC's `.mrt` and rotation curve files, handling CDS-style fixed-width formats and legacy data quirks (`enchan_btfr_reproduce_enchan.py`).
2.  **Deterministic Logic:** A specialized solver for the "quadrature closure" equation used in the Enchan framework, ensuring that predicted rotation curves are generated deterministically from baryonic inputs without stochastic tuning.
3.  **Reproducibility by Design:** The pipeline computes and logs SHA-256 hashes of all input data files, ensuring that benchmark results are strictly tied to specific versions of the observational datasets.
4.  **Rigorous Validation:** Implementation of strict Nested Cross-Validation (CV) protocols (`enchan_c1_pinning_test_strict.py`) to test environmental dependence hypotheses (Phi-screening) without data leakage.

This tool enables researchers to independently verify claims regarding the reproduction of the RAR and BTFR without requiring particle dark matter, using a fully open-source stack (`numpy`, `pandas`, `scipy`).

# Mathematics and Implementation

The core module (`enchan_core_model.py`) implements an effective acceleration mapping. Unlike standard MOND interpolations, the software primarily tests a quadrature closure derived from the field's asymptotic behavior:

$$g_{\text{tot}} = \sqrt{g_{\text{bar}}^2 + a_{\text{eff}} \, g_{\text{bar}}}$$

where $g_{\text{bar}}$ is the Newtonian acceleration from baryons, and $a_{\text{eff}}$ is an effective acceleration scale.

To handle environmental dependencies found in data, the package implements a "Phi-screening" mechanism (`enchan_core_model_plus.py`). The effective scale $a_{\text{eff}}$ is modulated by the depth of the gravitational potential $|\Phi|$:

$$a_{\text{eff}} = a_{0,\text{free}} \left[ 1 + \left( \frac{|\Phi|}{\Phi_c} \right)^n \right]^{-1}$$

The software solves these equations across thousands of radial points for over 175 galaxies, calculating residuals and statistical metrics (RMS, Pearson/Spearman correlations) to quantify the model's fidelity to observations [@Kobayashi:2025_proof].

# Acknowledgements

We acknowledge the foundational work of the SPARC team (Stacy McGaugh, Federico Lelli, James Schombert) for providing the high-quality observational data that makes this verification possible.

# References