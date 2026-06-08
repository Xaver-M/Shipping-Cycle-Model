"""
model/validation.py
====================
Empirical validation of the container shipping cycle model via
comparison of simulated freight rate dynamics against the
Shanghai Containerized Freight Index (SCFI) historical series.

Validation approach
-------------------
Rather than fitting the model to historical data (which would
overfit a single realisation of a stochastic process), we use a
*method of moments* comparison: the model is validated if the
simulated distribution of key statistics matches the empirical
distribution. This is the standard approach for calibrated
macroeconomic simulation models.

Moments compared:
    1. Mean freight rate level (normalised to LR average = 1.0)
    2. Standard deviation of annual rate changes (volatility)
    3. Autocorrelation (lag-1) of freight rates (persistence)
    4. Excess kurtosis (fat tails / extreme event frequency)
    5. Fraction of time spent in 'boom' (rate > 1.3x LR)
    6. Fraction of time spent in 'bust' (rate < 0.7x LR)
    7. Mean duration of boom episodes
    8. Mean duration of bust episodes

Empirical data source
---------------------
SCFI (Shanghai Containerized Freight Index) annual averages
2009-2024. Source: UNCTAD RMT 2025, Figure III.2.
Values are index-normalised to LR average = 1.0 using the
2010-2019 mean as the baseline (pre-COVID "normal" period).

SCFI annual averages (2009-2024, from UNCTAD RMT 2025):
    2009:  ~700   (post-GFC trough)
    2010: ~1200   (recovery)
    2011: ~1000
    2012:  ~850
    2013:  ~900
    2014:  ~950
    2015:  ~750
    2016:  ~700
    2017:  ~850
    2018:  ~900
    2019:  ~850
    2020: ~1050   (COVID disruption, logistics crunch)
    2021: ~3600   (COVID super-cycle peak)
    2022: ~2500   (still elevated)
    2023: ~1000   (normalisation)
    2024: ~2496   (Red Sea crisis, +149% YoY per UNCTAD RMT 2025)
    Source: UNCTAD RMT 2025, ch. III, p. 73-74.

Note: The 2021-2022 and 2024 values are outliers driven by
exogenous supply-chain shocks (COVID logistics crunch, Red Sea
rerouting). These are captured in the model as positive Type B
shocks. The 2010-2019 average (~875 SCFI points) is used as the
LR equilibrium normalisation baseline.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from demand import DemandParameters, DemandProcess
from supply import SupplyParameters, SupplyProcess
from market import MarketParameters, ShippingMarket


# ---------------------------------------------------------------------------
# Empirical SCFI data
# ---------------------------------------------------------------------------

# Annual SCFI averages 2009-2024, source: UNCTAD RMT 2025, p. 73-74
# Normalised: LR baseline = mean of 2010-2019 ≈ 875 points → 1.0
_SCFI_RAW = np.array([
    700,   # 2009
    1200,  # 2010
    1000,  # 2011
    850,   # 2012
    900,   # 2013
    950,   # 2014
    750,   # 2015
    700,   # 2016
    850,   # 2017
    900,   # 2018
    850,   # 2019
    1050,  # 2020
    3600,  # 2021
    2500,  # 2022
    1000,  # 2023
    2496,  # 2024
], dtype=float)
_SCFI_YEARS = np.arange(2009, 2025)
_SCFI_LR_BASELINE = _SCFI_RAW[1:11].mean()  # 2010-2019 mean
SCFI_NORMALISED = _SCFI_RAW / _SCFI_LR_BASELINE


# ---------------------------------------------------------------------------
# Moments extractor
# ---------------------------------------------------------------------------

def compute_moments(rate_series: np.ndarray, boom_threshold: float = 1.3,
                    bust_threshold: float = 0.7) -> dict:
    """
    Compute the eight validation moments from a freight rate series.

    Parameters
    ----------
    rate_series    : normalised freight rate series (LR avg = 1.0)
    boom_threshold : rate level above which a period is classified as 'boom'
    bust_threshold : rate level below which a period is classified as 'bust'

    Returns
    -------
    dict of moment values.
    """
    r    = np.asarray(rate_series)
    n    = len(r)
    diff = np.diff(r)

    # 1. Mean (should be ~1.0 by construction for normalised series)
    mean_rate = r.mean()

    # 2. Volatility (std of annual changes)
    volatility = diff.std() if len(diff) > 1 else 0.0

    # 3. Autocorrelation lag-1
    if n > 2:
        r_dm = r - r.mean()
        autocorr = float(np.corrcoef(r_dm[:-1], r_dm[1:])[0, 1])
    else:
        autocorr = 0.0

    # 4. Excess kurtosis (fat tails)
    if n > 3 and r.std() > 0:
        from scipy.stats import kurtosis
        excess_kurt = float(kurtosis(r, fisher=True))
    else:
        excess_kurt = 0.0

    # 5-6. Fraction in boom / bust
    frac_boom = float((r > boom_threshold).mean())
    frac_bust = float((r < bust_threshold).mean())

    # 7-8. Mean duration of boom / bust episodes
    def mean_episode_duration(binary_series: np.ndarray) -> float:
        """Average length of consecutive True-runs."""
        durations = []
        count = 0
        for val in binary_series:
            if val:
                count += 1
            elif count > 0:
                durations.append(count)
                count = 0
        if count > 0:
            durations.append(count)
        return float(np.mean(durations)) if durations else 0.0

    boom_duration = mean_episode_duration(r > boom_threshold)
    bust_duration = mean_episode_duration(r < bust_threshold)

    return {
        'mean_rate'     : mean_rate,
        'volatility'    : volatility,
        'autocorrelation': autocorr,
        'excess_kurtosis': excess_kurt,
        'frac_boom'     : frac_boom,
        'frac_bust'     : frac_bust,
        'boom_duration' : boom_duration,
        'bust_duration' : bust_duration,
    }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ModelValidator:
    """
    Validates the shipping cycle model against empirical SCFI moments.

    Workflow:
      1. Compute empirical moments from SCFI 2009-2024.
      2. Run N Monte Carlo simulations of the model.
      3. For each run, compute the same moments.
      4. Compare: for each moment, check whether the empirical value
         falls within the model's 5th-95th percentile range.
      5. Report a moment-by-moment validation table.

    A model 'passes' validation if at least 6/8 moments have empirical
    values within the simulated distribution's 5-95th percentile band.
    """

    def __init__(
        self,
        market: ShippingMarket,
        n_validation_runs: int = 500,
        seed: Optional[int] = 42,
    ):
        self.market       = market
        self.n_runs       = n_validation_runs
        self.seed         = seed

        # Empirical moments from SCFI
        self.empirical_moments = compute_moments(SCFI_NORMALISED)

    def run_validation(self) -> dict:
        """
        Run validation and return comparison table.

        Returns
        -------
        dict with:
          'empirical'       : dict of empirical moments
          'simulated_mean'  : dict of mean simulated moments
          'simulated_p5'    : dict of 5th percentile
          'simulated_p95'   : dict of 95th percentile
          'pass_fail'       : dict of bool (True = empirical within sim band)
          'n_passed'        : int (out of 8)
          'moment_series'   : dict of arrays (one value per MC run)
        """
        rng            = np.random.default_rng(self.seed)
        moment_keys    = list(self.empirical_moments.keys())
        moment_series  = {k: [] for k in moment_keys}

        for _ in range(self.n_runs):
            seed   = int(rng.integers(0, 10_000_000))
            result = self.market.run(seed=seed)
            moms   = compute_moments(result['rates'])
            for k in moment_keys:
                moment_series[k].append(moms[k])

        # Convert to arrays
        for k in moment_keys:
            moment_series[k] = np.array(moment_series[k])

        simulated_mean = {k: moment_series[k].mean()              for k in moment_keys}
        simulated_p5   = {k: np.percentile(moment_series[k],  5)  for k in moment_keys}
        simulated_p95  = {k: np.percentile(moment_series[k], 95)  for k in moment_keys}

        pass_fail = {
            k: bool(
                simulated_p5[k] <= self.empirical_moments[k] <= simulated_p95[k]
            )
            for k in moment_keys
        }
        n_passed = sum(pass_fail.values())

        return {
            'empirical'      : self.empirical_moments,
            'simulated_mean' : simulated_mean,
            'simulated_p5'   : simulated_p5,
            'simulated_p95'  : simulated_p95,
            'pass_fail'      : pass_fail,
            'n_passed'       : n_passed,
            'moment_series'  : moment_series,
        }

    def print_report(self, result: dict) -> None:
        """Print a formatted validation report."""
        em   = result['empirical']
        sm   = result['simulated_mean']
        p5   = result['simulated_p5']
        p95  = result['simulated_p95']
        pf   = result['pass_fail']

        print("=" * 72)
        print("MODEL VALIDATION REPORT — SCFI 2009-2024 vs. Simulated Moments")
        print(f"Simulation runs: {self.n_runs}")
        print("=" * 72)
        fmt = "{:<22} {:>10} {:>10} {:>10} {:>10} {:>6}"
        print(fmt.format("Moment", "Empirical", "Sim Mean", "Sim P5", "Sim P95", "Pass"))
        print("-" * 72)
        for k in em:
            label = k.replace('_', ' ').title()
            tick  = "✓" if pf[k] else "✗"
            print(fmt.format(
                label[:22],
                f"{em[k]:.4f}",
                f"{sm[k]:.4f}",
                f"{p5[k]:.4f}",
                f"{p95[k]:.4f}",
                tick,
            ))
        print("=" * 72)
        print(f"Moments passed: {result['n_passed']}/8  "
              f"({'PASS' if result['n_passed'] >= 6 else 'FAIL'})")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("Initialising model...")
    market = ShippingMarket(
        demand_params=DemandParameters(seed=42),
        supply_params=SupplyParameters(seed=42),
        market_params=MarketParameters(n_periods=50, seed=42),
    )

    validator = ModelValidator(market, n_validation_runs=300, seed=42)

    print("Running validation (300 MC runs)...")
    result = validator.run_validation()
    validator.print_report(result)

    # --- Visualisation ---
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: SCFI vs. simulated rate fan chart
    ax1 = fig.add_subplot(gs[0, :])
    # Monte Carlo fan
    rate_paths = []
    rng_val = np.random.default_rng(99)
    for _ in range(100):
        seed_run = int(rng_val.integers(0, 10_000_000))
        r        = market.run(seed=seed_run)['rates'][:16]  # 16 periods = 2009-2024
        rate_paths.append(r)
    rate_paths = np.array(rate_paths)
    years_sim = np.arange(2009, 2025)
    ax1.fill_between(years_sim,
                     np.percentile(rate_paths, 5, axis=0),
                     np.percentile(rate_paths, 95, axis=0),
                     alpha=0.25, color='steelblue', label='Model 5-95th pct')
    ax1.fill_between(years_sim,
                     np.percentile(rate_paths, 25, axis=0),
                     np.percentile(rate_paths, 75, axis=0),
                     alpha=0.40, color='steelblue', label='Model 25-75th pct')
    ax1.plot(years_sim, np.median(rate_paths, axis=0),
             color='steelblue', lw=2, label='Model median')
    ax1.plot(years_sim, SCFI_NORMALISED, color='firebrick', lw=2.5,
             marker='o', markersize=5, label='SCFI 2009-2024 (empirical)')
    ax1.axhline(1.0, color='black', lw=0.8, ls=':', alpha=0.5, label='LR average')
    ax1.set_ylabel('Rate index (LR avg = 1.0)')
    ax1.set_title('Model vs. SCFI: Rate Fan Chart (100 MC runs)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Volatility distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(result['moment_series']['volatility'], bins=30,
             color='steelblue', alpha=0.7, edgecolor='white', density=True)
    ax2.axvline(result['empirical']['volatility'], color='firebrick', lw=2,
                label=f"SCFI empirical: {result['empirical']['volatility']:.3f}")
    ax2.axvline(result['simulated_p5']['volatility'],  color='gray', lw=1.5, ls='--')
    ax2.axvline(result['simulated_p95']['volatility'], color='gray', lw=1.5, ls='--',
                label='5-95th pct')
    ax2.set_xlabel('Rate Volatility (σ of annual changes)')
    ax2.set_ylabel('Density')
    ax2.set_title('Validation: Volatility')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Autocorrelation distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(result['moment_series']['autocorrelation'], bins=30,
             color='steelblue', alpha=0.7, edgecolor='white', density=True)
    ax3.axvline(result['empirical']['autocorrelation'], color='firebrick', lw=2,
                label=f"SCFI empirical: {result['empirical']['autocorrelation']:.3f}")
    ax3.axvline(result['simulated_p5']['autocorrelation'],  color='gray', lw=1.5, ls='--')
    ax3.axvline(result['simulated_p95']['autocorrelation'], color='gray', lw=1.5, ls='--',
                label='5-95th pct')
    ax3.set_xlabel('Autocorrelation (lag-1)')
    ax3.set_ylabel('Density')
    ax3.set_title('Validation: Rate Persistence')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Model Validation — Simulated Moments vs. SCFI 2009-2024',
                 fontsize=12, fontweight='bold')
    import os
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/validation_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()
