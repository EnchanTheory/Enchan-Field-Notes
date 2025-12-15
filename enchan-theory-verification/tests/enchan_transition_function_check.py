#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enchan Transition Function Consistency Check (v0.3.3) - Fixed

Purpose:
  To numerically verify that the candidate transition function mu(x)
  is mathematically consistent with the benchmark quadrature closure.

  * FIX (v0.3.3): Implemented rationalized form of mu(x) to prevent 
    catastrophic cancellation at small x (Deep MOND regime).

Validation Target:
  Closure: g_tot = sqrt(g_bar^2 + a0 * g_bar)
  Candidate mu: mu(x) = 2x / (sqrt(1 + 4x^2) + 1)  <-- Numerically stable form
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def enchan_mu_function(x):
    """
    The candidate transition function derived in Chapter 6.
    x = g_tot / a0
    
    NOTE: Using the rationalized form:
      mu(x) = (sqrt(1+4x^2) - 1) / 2x
            = 2x / (sqrt(1+4x^2) + 1)
    This avoids precision loss (catastrophic cancellation) when x << 1.
    """
    # No need for x_safe hack anymore; this form is stable even at x -> 0
    return 2 * x / (np.sqrt(1 + 4 * x**2) + 1)

def enchan_quadrature_closure(g_bar, a0):
    """
    The benchmark closure used in simulation scripts.
    """
    return np.sqrt(g_bar**2 + a0 * g_bar)

def main():
    outdir = Path("Enchan_Transition_Check_v0_3_3")
    outdir.mkdir(exist_ok=True)
    
    print("--- Transition Function Consistency Check (Fixed) ---")
    
    # 1. Generate synthetic data covering all regimes
    # From deep MOND (1e-15) to strong gravity (1e2)
    g_bar_input = np.logspace(-15, 2, 1000)
    a0 = 1.2e-10
    
    # 2. Compute g_tot using the BENCHMARK CLOSURE
    g_tot_closure = enchan_quadrature_closure(g_bar_input, a0)
    
    # 3. Compute the reverse mapping using the STABLE CANDIDATE MU FUNCTION
    x = g_tot_closure / a0
    mu_val = enchan_mu_function(x)
    
    # Reconstruct g_bar from the field equation ansatz: g_bar = mu * g_tot
    g_bar_reconstructed = mu_val * g_tot_closure
    
    # 4. Check Residuals
    # Relative error
    rel_error = np.abs(g_bar_reconstructed - g_bar_input) / g_bar_input
    max_err = np.max(rel_error)
    
    print(f"Tested N={len(g_bar_input)} points.")
    print(f"Regime: g_bar = {g_bar_input.min():.1e} ... {g_bar_input.max():.1e}")
    print(f"Max Relative Error: {max_err:.3e}")
    
    # Check against a strict tolerance (Machine Epsilon level)
    # Previously failed at ~1e-12. Now it should be around ~1e-16.
    if max_err < 1e-14:
        print(">> PASS: The candidate mu-function is consistently stable.")
    else:
        print(">> FAIL: Inconsistency detected.")

    # 5. Visualization for the report
    plt.figure(figsize=(8, 5))
    x_plot = np.logspace(-2, 2, 200)
    mu_plot = enchan_mu_function(x_plot)
    
    # Label indicates the Definition, code uses the stable implementation
    plt.plot(x_plot, mu_plot, label=r"$\mu(x) = \frac{2x}{\sqrt{1+4x^2}+1}$ (Stable form)")
    plt.plot(x_plot, x_plot, "--", color="gray", alpha=0.5, label=r"Deep Limit ($\mu \to x$)")
    plt.axhline(1.0, ls=":", color="gray", alpha=0.5, label=r"Newtonian Limit ($\mu \to 1$)")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$x = g_{\mathrm{tot}} / a_0$")
    plt.ylabel(r"$\mu(x)$")
    plt.title("Enchan Transition Function Behavior (Numerically Stable)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "fig_mu_function_fixed.png", dpi=100)
    print(f"Plot saved to {outdir}")

if __name__ == "__main__":
    main()