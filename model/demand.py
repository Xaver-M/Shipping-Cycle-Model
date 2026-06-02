"""
model/demand.py
===============
Stochastic demand process for the container shipping cycle model.

Theoretical basis
-----------------
Container shipping demand is derived demand — it depends entirely on
global trade volumes, which in turn depend on GDP growth. This module
implements a two-layer demand process:

  Layer 1 — Long-run trend:
      Trade grows as a function of GDP, with an elasticity that is itself
      time-varying. Calibrated on Constantinescu, Mattoo & Ruta (2020),
      who estimate the long-run trade-to-GDP elasticity at approximately
      2.2 for 1986–2000, declining to ~1.3 post-2012.

  Layer 2 — Short-run cycle:
      Around the trend, trade oscillates with the business cycle. An
      AR(1) process captures the autocorrelation of demand shocks. The
      inventory/JIT amplification documented by Stopford (2009, p. 122)
      is modelled as an overshoot multiplier on GDP fluctuations.

  Exogenous shocks:
      Geopolitical disruptions (Red Sea, Suez blockages) are modelled as
      discrete events that compress *effective* demand (ton-miles) without
      altering nominal trade volumes. Drawn from a Poisson process with
      calibrated frequency and severity.

Key references
--------------
- Constantinescu, Mattoo & Ruta (2020): long-run elasticity estimates,
  pp. 121–124 and 134–138.
- Stopford (2009): inventory cycle amplification, p. 122.
- Luo, Fan & Liu (2009): demand enters supply equation exogenously,
  pp. 512–514.

Units
-----
Demand is expressed in million TEU per year throughout.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class DemandParameters:
    """
    All calibrated parameters for the demand process.

    Defaults are calibrated on empirical data from the sources listed above.
    Override any parameter to run sensitivity analyses.
    """

    # --- Long-run trend ----------------------------------------------------
    gdp_growth_mean: float = 0.026
    """Mean annual real GDP growth rate (global). World Bank average 2000-2023."""

    gdp_growth_std: float = 0.018
    """Standard deviation of annual GDP growth. Captures business cycle volatility."""

    trade_elasticity_high: float = 2.2
    """Trade-to-GDP elasticity, high-globalisation regime (1986-2000).
    Source: Constantinescu et al. (2020), p. 124."""

    trade_elasticity_low: float = 1.3
    """Trade-to-GDP elasticity, post-2012 slowdown regime.
    Source: Constantinescu et al. (2020), p. 124."""

    elasticity_transition_year: int = 30
    """Simulation period (0-indexed) at which elasticity shifts from high to low.
    Default: period 30 ≈ year 2012 if simulation starts at 1982."""

    # --- Short-run cycle ---------------------------------------------------
    ar1_coefficient: float = 0.45
    """AR(1) autocorrelation of demand shocks around trend.
    Captures the persistence of inventory cycles."""

    demand_shock_std: float = 0.035
    """Standard deviation of idiosyncratic demand shocks (annual, fractional)."""

    jit_amplifier: float = 1.4
    """Just-in-time / inventory cycle amplification factor.
    Demand for shipping overshoots underlying consumption changes by this
    multiple in both directions.
    Source: Stopford (2009), p. 122."""

    # --- Initial conditions ------------------------------------------------
    initial_demand: float = 100.0
    """Base demand level at t=0 (index, not absolute TEU).
    Set to 100 for easy interpretation of percentage changes."""

    # --- Geopolitical shock process ----------------------------------------
    shock_probability: float = 0.08
    """Annual probability of a major geopolitical disruption affecting
    effective capacity (ton-miles). Calibrated on post-2000 frequency:
    roughly one major event per 12 years on average."""

    shock_severity_mean: float = 0.12
    """Mean reduction in effective demand during a disruption (fractional).
    Red Sea 2023-24: ~12% effective capacity compression via rerouting."""

    shock_severity_std: float = 0.05
    """Standard deviation of shock severity."""

    shock_duration_mean: float = 1.5
    """Mean duration of disruption in years (Poisson-distributed)."""

    # --- Random seed -------------------------------------------------------
    seed: Optional[int] = 42
    """Random seed for reproducibility. Set to None for stochastic runs."""


# ---------------------------------------------------------------------------
# Demand simulator
# ---------------------------------------------------------------------------

class DemandProcess:
    """
    Simulates container shipping demand over a given horizon.

    The process generates three output series:
      - ``nominal_demand``:  trade volumes in TEU (unaffected by disruptions)
      - ``effective_demand``: ton-mile-adjusted demand (compressed by shocks)
      - ``gdp_path``:        underlying GDP index for reference

    Example
    -------
    >>> params = DemandParameters(seed=0)
    >>> proc = DemandProcess(params)
    >>> results = proc.simulate(n_periods=40)
    >>> results['effective_demand'].shape
    (40,)
    """

    def __init__(self, params: DemandParameters):
        self.p = params
        self.rng = np.random.default_rng(params.seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def simulate(self, n_periods: int) -> dict:
        """
        Run the demand simulation for ``n_periods`` years.

        Returns
        -------
        dict with keys:
            'nominal_demand'    : np.ndarray, shape (n_periods,)
            'effective_demand'  : np.ndarray, shape (n_periods,)
            'gdp_index'         : np.ndarray, shape (n_periods,)
            'gdp_growth_rates'  : np.ndarray, shape (n_periods,)
            'trade_elasticity'  : np.ndarray, shape (n_periods,)
            'shock_active'      : np.ndarray bool, shape (n_periods,)
            'shock_severity'    : np.ndarray, shape (n_periods,)
        """
        gdp_growth   = self._generate_gdp_path(n_periods)
        elasticity   = self._elasticity_path(n_periods)
        nominal      = self._generate_nominal_demand(gdp_growth, elasticity, n_periods)
        shock_active, shock_severity = self._generate_shocks(n_periods)
        effective    = nominal * (1.0 - shock_severity)

        # Build GDP index (base 100 at t=0)
        gdp_index = np.cumprod(1.0 + gdp_growth) * 100.0 / (1.0 + gdp_growth[0])

        return {
            'nominal_demand'   : nominal,
            'effective_demand' : effective,
            'gdp_index'        : gdp_index,
            'gdp_growth_rates' : gdp_growth,
            'trade_elasticity' : elasticity,
            'shock_active'     : shock_active,
            'shock_severity'   : shock_severity,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_gdp_path(self, n_periods: int) -> np.ndarray:
        """
        Draw GDP growth rates from a normal distribution.

        In a richer model this could be replaced by a Markov-switching
        process to capture recession regimes explicitly.
        """
        raw = self.rng.normal(
            loc   = self.p.gdp_growth_mean,
            scale = self.p.gdp_growth_std,
            size  = n_periods
        )
        # GDP growth is bounded below at -0.10 (no Great Depression scenarios)
        return np.clip(raw, -0.10, None)

    def _elasticity_path(self, n_periods: int) -> np.ndarray:
        """
        Step-shift elasticity at the transition period.

        Extension: could be made into a smooth logistic transition.
        """
        elasticity = np.full(n_periods, self.p.trade_elasticity_high)
        if self.p.elasticity_transition_year < n_periods:
            elasticity[self.p.elasticity_transition_year:] = self.p.trade_elasticity_low
        return elasticity

    def _generate_nominal_demand(
        self,
        gdp_growth: np.ndarray,
        elasticity: np.ndarray,
        n_periods: int
    ) -> np.ndarray:
        """
        Build nominal demand path via:

            D(t) = D(t-1) * [1 + elasticity(t) * g_GDP(t) + jit * eps(t)]

        where eps(t) is an AR(1) shock process capturing inventory cycles.

        The JIT amplifier scales the shock term — not the trend — because
        inventory cycles amplify *deviations* from expected demand, not
        the underlying growth rate itself.
        """
        # AR(1) shock process
        eps = np.zeros(n_periods)
        innovations = self.rng.normal(0, self.p.demand_shock_std, n_periods)
        eps[0] = innovations[0]
        for t in range(1, n_periods):
            eps[t] = self.p.ar1_coefficient * eps[t-1] + innovations[t]

        # Build demand level
        demand = np.zeros(n_periods)
        demand[0] = self.p.initial_demand
        for t in range(1, n_periods):
            growth_rate = (
                elasticity[t] * gdp_growth[t]          # long-run trend
                + self.p.jit_amplifier * eps[t]         # amplified cycle
            )
            demand[t] = demand[t-1] * (1.0 + growth_rate)

        return demand

    def _generate_shocks(self, n_periods: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate geopolitical disruption events.

        Each period, a shock occurs with probability ``shock_probability``.
        Shock severity (fractional reduction in effective demand) is drawn
        from a truncated normal. Duration is Poisson-distributed, so a shock
        can persist across multiple periods.

        Returns
        -------
        shock_active   : bool array, True in periods with active disruption
        shock_severity : float array, severity in [0, 1] per period
        """
        shock_active   = np.zeros(n_periods, dtype=bool)
        shock_severity = np.zeros(n_periods)

        t = 0
        while t < n_periods:
            if self.rng.random() < self.p.shock_probability:
                # Draw severity and duration
                severity = float(np.clip(
                    self.rng.normal(
                        self.p.shock_severity_mean,
                        self.p.shock_severity_std
                    ), 0.0, 0.40  # cap at 40% demand compression
                ))
                duration = max(1, int(self.rng.poisson(self.p.shock_duration_mean)))
                end = min(t + duration, n_periods)
                shock_active[t:end]   = True
                shock_severity[t:end] = severity
                t = end  # skip to after shock (no overlapping shocks)
            else:
                t += 1

        return shock_active, shock_severity


# ---------------------------------------------------------------------------
# Quick diagnostic
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params  = DemandParameters(seed=42)
    process = DemandProcess(params)
    results = process.simulate(n_periods=50)

    years = np.arange(2000, 2050)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Demand paths
    axes[0].plot(years, results['nominal_demand'],
                 label='Nominal demand (TEU index)', color='steelblue', lw=2)
    axes[0].plot(years, results['effective_demand'],
                 label='Effective demand (ton-mile adjusted)', color='firebrick',
                 lw=2, linestyle='--')
    # Shade disruption periods
    for t, active in enumerate(results['shock_active']):
        if active:
            axes[0].axvspan(years[t], years[t]+1, alpha=0.15, color='orange')
    axes[0].set_ylabel('Demand index (base 100)')
    axes[0].set_title('Container Shipping Demand Process — Simulation (n=50 periods)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: GDP growth and trade elasticity
    ax2 = axes[1]
    ax2.bar(years, results['gdp_growth_rates'] * 100,
            alpha=0.5, color='steelblue', label='GDP growth (%)')
    ax2.set_ylabel('GDP growth (%)')
    ax2.set_xlabel('Year')

    ax3 = ax2.twinx()
    ax3.step(years, results['trade_elasticity'],
             color='darkorange', lw=2, label='Trade elasticity')
    ax3.set_ylabel('Trade-to-GDP elasticity')
    ax3.set_ylim(0, 3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/demand_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Diagnostic plot saved to data/demand_diagnostic.png")

    # Summary statistics
    print("\n--- Demand Process Summary ---")
    print(f"Initial demand:           {results['nominal_demand'][0]:.1f}")
    print(f"Final nominal demand:     {results['nominal_demand'][-1]:.1f}")
    print(f"Total nominal growth:     {(results['nominal_demand'][-1]/results['nominal_demand'][0] - 1)*100:.1f}%")
    print(f"Number of shock periods:  {results['shock_active'].sum()}")
    print(f"Mean shock severity:      {results['shock_severity'][results['shock_active']].mean()*100:.1f}%"
          if results['shock_active'].any() else "No shocks occurred")
