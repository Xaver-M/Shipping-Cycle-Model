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
      2.2 for 1986-2000, declining to ~1.3 post-2012.

  Layer 2 — Short-run cycle (Markov-Switching):
      GDP growth follows a two-state Hamilton (1989) Markov-switching
      process. State 0 = expansion (high mean, low variance); State 1 =
      recession (low/negative mean, high variance). Transition
      probabilities are calibrated on post-1980 global recession
      frequency (~one recession per 8-10 years, lasting ~1-2 years).

  Exogenous shocks — two types:
      Type A (Negative / capacity compressing):
          Geopolitical disruptions that REDUCE effective demand by
          forcing longer routes (e.g. Suez blockage 2021 — short-lived
          supply disruption). Modelled as transitory Poisson events.

      Type B (Positive / tonne-mile expanding):
          Route disruptions that INCREASE tonne-mile demand by forcing
          vessels on longer alternative routes (e.g. Red Sea / Houthi
          2023-24). These are positive demand shocks with persistent
          duration (mean ~2 years), because rerouting can last as long
          as the geopolitical conflict. UNCTAD RMT 2025 documents the
          Red Sea crisis adding ~12-15% effective tonne-miles demand via
          Cape of Good Hope rerouting.

      This asymmetry is empirically important: treating all disruptions
      as negative demand shocks misrepresents the Red Sea episode, which
      was a *capacity demand boom* for vessel operators despite being a
      supply-chain cost for shippers.

Key references
--------------
- Constantinescu, Mattoo & Ruta (2020): long-run elasticity estimates,
  pp. 121-124 and 134-138.
- Hamilton (1989): Markov-switching GDP model, J. Political Economy.
- Stopford (2009): inventory cycle amplification, p. 122.
- Luo, Fan & Liu (2009): demand enters supply equation exogenously,
  pp. 512-514.
- UNCTAD RMT (2025): Red Sea crisis tonne-mile impact, ch. III.

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
    trade_elasticity_high: float = 2.2
    """Trade-to-GDP elasticity, high-globalisation regime (1986-2000).
    Source: Constantinescu et al. (2020), p. 124."""

    trade_elasticity_low: float = 1.3
    """Trade-to-GDP elasticity, post-2012 slowdown regime.
    Source: Constantinescu et al. (2020), p. 124."""

    elasticity_transition_year: int = 30
    """Simulation period (0-indexed) at which elasticity shifts from high to low.
    Default: period 30 ≈ year 2012 if simulation starts at 1982."""

    # --- Markov-switching GDP process (Hamilton 1989) ----------------------
    gdp_expansion_mean: float = 0.031
    """Mean annual real GDP growth in expansion state.
    World Bank average 2000-2023 excl. recession years."""

    gdp_expansion_std: float = 0.012
    """GDP growth volatility in expansion state (lower — expansions are stable)."""

    gdp_recession_mean: float = -0.012
    """Mean annual real GDP growth in recession state.
    Calibrated on 2009 (-1.7%) and 2020 (-3.1%) averaged with milder recessions."""

    gdp_recession_std: float = 0.030
    """GDP growth volatility in recession state (higher — recessions vary widely)."""

    p_expansion_to_recession: float = 0.10
    """Transition probability: expansion → recession per period.
    Implies mean expansion duration ~10 years. Hamilton (1989), p. 363."""

    p_recession_to_expansion: float = 0.60
    """Transition probability: recession → expansion per period.
    Implies mean recession duration ~1.7 years. Consistent with post-1980 data."""

    initial_regime: int = 0
    """Starting regime: 0 = expansion, 1 = recession."""

    # --- Short-run inventory cycle -----------------------------------------
    ar1_coefficient: float = 0.45
    """AR(1) autocorrelation of demand shocks around trend.
    Captures the persistence of inventory cycles."""

    demand_shock_std: float = 0.030
    """Standard deviation of idiosyncratic demand shocks (annual, fractional)."""

    jit_amplifier: float = 1.4
    """Just-in-time / inventory cycle amplification factor.
    Source: Stopford (2009), p. 122."""

    # --- Initial conditions ------------------------------------------------
    initial_demand: float = 100.0
    """Base demand level at t=0 (index, not absolute TEU)."""

    # --- Geopolitical shock process — Type A: Negative (compressing) ------
    neg_shock_probability: float = 0.06
    """Annual probability of a negative geopolitical disruption
    (e.g. port blockage, pandemic logistics crunch reducing effective demand).
    Calibrated on post-2000 frequency: ~1 event per 15 years."""

    neg_shock_severity_mean: float = 0.07
    """Mean reduction in effective demand during negative disruption (fractional)."""

    neg_shock_severity_std: float = 0.03
    """Std dev of negative shock severity."""

    neg_shock_duration_mean: float = 1.0
    """Mean duration of negative disruption (years). Transitory by nature."""

    # --- Geopolitical shock process — Type B: Positive (tonne-mile boom) --
    pos_shock_probability: float = 0.05
    """Annual probability of a positive tonne-mile shock
    (e.g. Red Sea rerouting — vessel demand increases via longer routes).
    Slightly lower frequency than negative shocks; rarer but more impactful.
    UNCTAD RMT 2025: Red Sea crisis boosted effective tonne-mile demand ~12-15%."""

    pos_shock_severity_mean: float = 0.13
    """Mean INCREASE in effective demand during positive rerouting shock.
    Calibrated on Red Sea 2023-24: ~12% tonne-mile increase via Cape rerouting.
    UNCTAD RMT 2025, ch. III."""

    pos_shock_severity_std: float = 0.04
    """Std dev of positive shock severity."""

    pos_shock_duration_mean: float = 2.0
    """Mean duration of positive tonne-mile shock (years).
    Positive shocks persist longer — geopolitical conflicts sustaining
    rerouting typically last 1-3 years (Red Sea: ongoing as of mid-2025).
    Source: UNCTAD RMT 2025, ch. III."""

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
      - ``nominal_demand``:   trade volumes in TEU (unaffected by disruptions)
      - ``effective_demand``: ton-mile-adjusted demand (modified by shocks)
      - ``gdp_path``:         underlying GDP index for reference
      - ``regime_path``:      Markov state per period (0=expansion, 1=recession)

    Shock sign convention
    ---------------------
    Negative shocks (Type A) compress effective demand below nominal.
    Positive shocks (Type B) expand effective demand above nominal.
    Both are applied as a net multiplier:
        effective_demand(t) = nominal_demand(t) * (1 + net_shock(t))
    where net_shock ∈ (−0.40, +0.40).

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
            'regime_path'       : np.ndarray int, shape (n_periods,)
              (0 = expansion, 1 = recession)
            'shock_type'        : np.ndarray int, shape (n_periods,)
              (0 = none, 1 = negative/compressing, 2 = positive/tonne-mile)
            'shock_net'         : np.ndarray, shape (n_periods,)
              (signed net shock multiplier applied to effective demand)
        """
        regime_path, gdp_growth = self._generate_gdp_markov(n_periods)
        elasticity              = self._elasticity_path(n_periods)
        nominal                 = self._generate_nominal_demand(gdp_growth, elasticity, n_periods)
        shock_type, shock_net   = self._generate_shocks(n_periods)
        effective               = nominal * (1.0 + shock_net)

        # Build GDP index (base 100 at t=0)
        gdp_index = np.cumprod(1.0 + gdp_growth) * 100.0 / (1.0 + gdp_growth[0])

        return {
            'nominal_demand'   : nominal,
            'effective_demand' : effective,
            'gdp_index'        : gdp_index,
            'gdp_growth_rates' : gdp_growth,
            'trade_elasticity' : elasticity,
            'regime_path'      : regime_path,
            'shock_type'       : shock_type,
            'shock_net'        : shock_net,
            # Backwards compatibility aliases
            'shock_active'     : shock_type != 0,
            'shock_severity'   : -shock_net * (shock_type == 1),  # only neg shocks
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_gdp_markov(self, n_periods: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Two-state Hamilton (1989) Markov-switching GDP process.

        State 0 = expansion: GDP ~ N(mu_exp, sigma_exp)
        State 1 = recession:  GDP ~ N(mu_rec, sigma_rec)

        Transition matrix:
            P = [[1 - p01,  p01],
                 [p10,      1 - p10]]

        Returns
        -------
        regime_path : int array of regime states (0 or 1)
        gdp_growth  : float array of GDP growth rates
        """
        p = self.p
        regime      = p.initial_regime
        regime_path = np.zeros(n_periods, dtype=int)
        gdp_growth  = np.zeros(n_periods)

        trans = np.array([
            [1.0 - p.p_expansion_to_recession, p.p_expansion_to_recession],
            [p.p_recession_to_expansion,        1.0 - p.p_recession_to_expansion],
        ])

        means  = [p.gdp_expansion_mean, p.gdp_recession_mean]
        stds   = [p.gdp_expansion_std,  p.gdp_recession_std]

        for t in range(n_periods):
            regime_path[t] = regime
            raw = self.rng.normal(means[regime], stds[regime])
            gdp_growth[t]  = np.clip(raw, -0.15, None)  # no worse than -15%

            # Markov transition
            if self.rng.random() < trans[regime, 1 - regime]:
                regime = 1 - regime

        return regime_path, gdp_growth

    def _elasticity_path(self, n_periods: int) -> np.ndarray:
        """
        Step-shift elasticity at the transition period.
        Extended: smooth logistic transition over 5 years around the break,
        consistent with the gradual nature of the trade slowdown
        (Constantinescu et al. 2020 show the break as a continuous process).
        """
        t_break = self.p.elasticity_transition_year
        hi      = self.p.trade_elasticity_high
        lo      = self.p.trade_elasticity_low
        t_arr   = np.arange(n_periods, dtype=float)

        # Logistic transition (k=1.5 → transition over ~5 periods)
        k = 1.5
        elasticity = lo + (hi - lo) / (1.0 + np.exp(k * (t_arr - t_break)))
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
        """
        eps          = np.zeros(n_periods)
        innovations  = self.rng.normal(0, self.p.demand_shock_std, n_periods)
        eps[0]       = innovations[0]
        for t in range(1, n_periods):
            eps[t] = self.p.ar1_coefficient * eps[t-1] + innovations[t]

        demand    = np.zeros(n_periods)
        demand[0] = self.p.initial_demand
        for t in range(1, n_periods):
            growth_rate = (
                elasticity[t] * gdp_growth[t]
                + self.p.jit_amplifier * eps[t]
            )
            demand[t] = demand[t-1] * (1.0 + growth_rate)

        return demand

    def _generate_shocks(self, n_periods: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate two types of geopolitical disruption events.

        Type A (negative): reduces effective demand — port blockages,
            logistics disruptions, sudden trade volume drops.
        Type B (positive): increases tonne-mile demand — rerouting
            events that force longer voyages (Red Sea / Cape of Good Hope).

        The two processes are independent Poisson arrivals. If both fire
        in the same period, the net effect is computed (partial offset is
        possible but extreme simultaneous events are rare).

        Returns
        -------
        shock_type : int array (0=none, 1=negative, 2=positive)
        shock_net  : float array (signed multiplier on effective demand)
        """
        shock_type = np.zeros(n_periods, dtype=int)
        shock_net  = np.zeros(n_periods)

        # --- Type A: Negative shocks ---
        t = 0
        while t < n_periods:
            if self.rng.random() < self.p.neg_shock_probability:
                severity = float(np.clip(
                    self.rng.normal(self.p.neg_shock_severity_mean,
                                    self.p.neg_shock_severity_std),
                    0.01, 0.35
                ))
                duration = max(1, int(self.rng.poisson(self.p.neg_shock_duration_mean)))
                end = min(t + duration, n_periods)
                for tt in range(t, end):
                    if shock_type[tt] == 0:
                        shock_type[tt] = 1
                        shock_net[tt]  = -severity  # negative: compresses demand
                    # If already a positive shock active: net out (partial offset)
                    elif shock_type[tt] == 2:
                        shock_net[tt] -= severity
                t = end
            else:
                t += 1

        # --- Type B: Positive shocks (independent Poisson) ---
        t = 0
        while t < n_periods:
            if self.rng.random() < self.p.pos_shock_probability:
                severity = float(np.clip(
                    self.rng.normal(self.p.pos_shock_severity_mean,
                                    self.p.pos_shock_severity_std),
                    0.02, 0.40
                ))
                # Persistent duration: geopolitical conflicts last longer
                duration = max(1, int(self.rng.poisson(self.p.pos_shock_duration_mean)))
                end = min(t + duration, n_periods)
                for tt in range(t, end):
                    if shock_type[tt] == 0:
                        shock_type[tt] = 2
                        shock_net[tt]  = +severity  # positive: expands tonne-miles
                    # If already a negative shock active: net out
                    elif shock_type[tt] == 1:
                        shock_net[tt] += severity
                        # If net is positive, reclassify
                        if shock_net[tt] > 0:
                            shock_type[tt] = 2
                t = end
            else:
                t += 1

        # Clip net shock to physically meaningful bounds
        shock_net = np.clip(shock_net, -0.40, 0.40)
        return shock_type, shock_net


# ---------------------------------------------------------------------------
# Quick diagnostic
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params  = DemandParameters(seed=42)
    process = DemandProcess(params)
    results = process.simulate(n_periods=50)

    years = np.arange(2000, 2050)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # Panel 1: Demand paths + shock shading
    axes[0].plot(years, results['nominal_demand'],
                 label='Nominal demand (TEU index)', color='steelblue', lw=2)
    axes[0].plot(years, results['effective_demand'],
                 label='Effective demand (tonne-mile adjusted)',
                 color='firebrick', lw=2, linestyle='--')
    for t in range(len(years)):
        if results['shock_type'][t] == 1:
            axes[0].axvspan(years[t], years[t]+1, alpha=0.18, color='firebrick',
                            label='_neg_shock')
        elif results['shock_type'][t] == 2:
            axes[0].axvspan(years[t], years[t]+1, alpha=0.18, color='seagreen',
                            label='_pos_shock')
    # Add legend patches manually
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0],[0], color='steelblue', lw=2, label='Nominal demand'),
        plt.Line2D([0],[0], color='firebrick', lw=2, ls='--', label='Effective demand'),
        Patch(facecolor='firebrick', alpha=0.3, label='Neg. shock (compressing)'),
        Patch(facecolor='seagreen',  alpha=0.3, label='Pos. shock (tonne-mile boom)'),
    ]
    axes[0].legend(handles=legend_elements, fontsize=8)
    axes[0].set_ylabel('Demand index (base 100)')
    axes[0].set_title('Container Shipping Demand — Markov-Switching GDP + Asymmetric Shocks')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: GDP growth coloured by regime
    regime = results['regime_path']
    gdp    = results['gdp_growth_rates'] * 100
    colors = ['steelblue' if r == 0 else 'firebrick' for r in regime]
    axes[1].bar(years, gdp, color=colors, alpha=0.7)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_ylabel('GDP growth (%)')
    axes[1].set_title('GDP Growth by Regime  (blue = expansion, red = recession)')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Net shock effect
    axes[2].bar(years, results['shock_net'] * 100,
                color=['seagreen' if s > 0 else 'firebrick' if s < 0 else 'lightgray'
                       for s in results['shock_net']],
                alpha=0.7)
    axes[2].axhline(0, color='black', lw=0.8)
    axes[2].set_ylabel('Net shock effect (%)')
    axes[2].set_xlabel('Year')
    axes[2].set_title('Asymmetric Shock Effects on Effective Demand')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    import os
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/demand_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("=== Demand Process Summary ===")
    print(f"Periods in expansion:  {(results['regime_path']==0).sum()}")
    print(f"Periods in recession:  {(results['regime_path']==1).sum()}")
    n_neg = (results['shock_type']==1).sum()
    n_pos = (results['shock_type']==2).sum()
    print(f"Negative shock periods: {n_neg}")
    print(f"Positive shock periods: {n_pos}")
    if n_neg > 0:
        print(f"Mean neg shock:         {results['shock_net'][results['shock_type']==1].mean()*100:.1f}%")
    if n_pos > 0:
        print(f"Mean pos shock:         {results['shock_net'][results['shock_type']==2].mean()*100:.1f}%")
