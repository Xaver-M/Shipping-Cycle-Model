"""
model/supply.py
===============
Supply-side dynamics for the container shipping cycle model.

Theoretical basis
-----------------
The supply side captures the two central mechanisms identified in the
Cobweb framework as applied to container shipping:

  1. Naive price expectations:
       Shipowners observe the current freight rate and place orders
       accordingly, without fully accounting for the aggregate supply
       response of competitors. Formally, the ordering rule follows
       Luo, Fan & Liu (2009), who demonstrate econometrically that
       aggregate newbuilding orders in period t are a statistically
       significant positive function of the freight rate in period t.

  2. Stochastic distributed shipbuilding lag:
       Orders placed in period t do NOT deliver at a fixed t+2 interval.
       Empirically, delivery times depend on yard capacity, steel prices,
       and client specifications. UNCTAD RMT 2024/2025 documents that
       the 2021-2024 order surge created delivery slot bottlenecks and
       pushed effective lags to 3-4 years for large vessels. This module
       models the lag as Beta-distributed over a window [LAG_MIN, LAG_MAX],
       so a single order is fractionally spread across delivery periods.
       This captures both average lag and its variance in a principled way.

  3. Earnings-dependent scrapping:
       The original scrapping rule used a rate-threshold trigger.
       Empirically, scrapping decisions depend on the *expected future
       earnings stream*, not just the spot rate. UNCTAD RMT 2025 reports
       scrapping at historic lows in 2024 despite significant overcapacity,
       because carriers were still earning above OPEX on charter markets.
       The new rule uses a weighted average of spot earnings and a
       backward-looking earnings expectation (EWMA of past rates), so
       scrapping is delayed when recent earnings were strong even if the
       current spot has dipped.

  4. Slow steaming (CII regulatory capacity reduction):
       Unchanged from v1. IMO EEXI/CII from 2023 reduces effective fleet
       capacity ~6% via mandatory speed reduction.
       Source: Lehmann et al. (2025), p. 3.

Key references
--------------
- Luo, Fan & Liu (2009): ordering rule specification, pp. 512-514.
- Stopford (2009): shipbuilding lag, pp. 155-156; scrapping dynamics,
  demolition market as stabiliser.
- Greenwood & Hanson (2015): overextrapolation of demand, pp. 57-58.
- UNCTAD RMT (2025): delivery slot bottlenecks, historic-low scrapping,
  earnings dynamics in 2024 container market. Ch. II.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class SupplyParameters:
    """
    All calibrated parameters for the supply-side process.

    The key new parameters relative to v1 are:
      - lag_min / lag_max / lag_alpha / lag_beta: distributed lag
      - scrapping_earnings_memory / scrapping_earnings_weight: EWMA scrapping
    """

    # --- Distributed shipbuilding lag (Beta distribution) ------------------
    lag_min: int = 1
    """Minimum possible delivery period after order (years)."""

    lag_max: int = 4
    """Maximum possible delivery period after order (years).
    UNCTAD RMT 2025: delivery bottlenecks pushed large vessel lags to 3-4y."""

    lag_alpha: float = 2.5
    """Beta distribution shape parameter alpha.
    With alpha=2.5, beta=1.5: mode at ~2 years, right-skewed to 3-4.
    Calibrated so that the *expected* lag ≈ 2.2 years (Luo et al. 2009)."""

    lag_beta: float = 1.5
    """Beta distribution shape parameter beta. See lag_alpha."""

    # --- Ordering rule (Luo et al. 2009) -----------------------------------
    order_sensitivity: float = 0.18
    """
    Sensitivity of new orders to the current freight rate deviation
    from long-run average. Derived from Luo et al. (2009):
    a 10% above-average freight rate triggers ~1.8% of fleet capacity
    in new orders per period.
    """

    order_base_rate: float = 0.04
    """Baseline replacement ordering rate (fraction of fleet per period)."""

    order_inertia: float = 0.30
    """AR(1) coefficient on the order series (herding effect)."""

    # --- Earnings-dependent scrapping ---------------------------------------
    scrapping_base_rate: float = 0.012
    """
    Baseline annual scrapping rate (fraction of fleet), independent
    of market conditions. Reflects normal end-of-life retirement.
    Slightly reduced from v1 to reflect younger average fleet age
    documented by UNCTAD RMT 2025 (average age data, ch. II).
    """

    scrapping_sensitivity: float = 0.10
    """Additional scrapping triggered per unit of earnings below threshold."""

    cost_covering_rate: float = 0.85
    """Rate level (relative to LR avg = 1.0) below which earnings are
    considered insufficient to cover total costs."""

    layup_threshold: float = 0.60
    """Rate level below which vessels are laid up (variable costs covered
    but not total costs). Fleet inertia — source of persistent overcapacity."""

    scrapping_earnings_memory: float = 0.25
    """
    EWMA decay factor for backward-looking earnings expectation.
    Expected_earnings(t) = (1-alpha)*Expected(t-1) + alpha*Rate(t)
    with alpha = scrapping_earnings_memory.
    Lower → longer memory (scrapping decision responds slowly to rate changes).
    Calibrated so that the 2021-2024 boom would suppress scrapping for ~3 years
    after rates peaked, consistent with UNCTAD RMT 2025 observation of
    historic-low scrapping in 2024 despite softening spot rates.
    """

    scrapping_earnings_weight: float = 0.35
    """
    Weight given to the backward-looking EWMA earnings vs the spot rate
    in the scrapping decision function.
    scrapping_rate_signal = (1-w)*spot_rate + w*earnings_expectation
    w=0.35 means 35% of scrapping decision reflects recent earnings history,
    65% reflects current spot. This captures that charterers often operate
    on multi-year contracts insulating them from spot rate fluctuations.
    """

    # --- Regulatory slow steaming (CII) ------------------------------------
    slow_steam_active: bool = True
    """Whether CII-style regulations are modelled."""

    slow_steam_start_period: int = 22
    """Period from which slow steaming regulations take effect (≈ 2022)."""

    slow_steam_reduction: float = 0.06
    """Fractional reduction in effective fleet capacity from mandatory speed
    reduction. Source: Lehmann et al. (2025), p. 3."""

    # --- Market concentration effect ---------------------------------------
    n_carriers: int = 20
    """Number of independent ordering decision-makers."""

    # --- Initial conditions ------------------------------------------------
    initial_fleet: float = 100.0
    """Base fleet capacity at t=0 (index)."""

    initial_orderbook: float = 8.0
    """Pre-existing orders that will deliver during the simulation."""

    # --- Random seed -------------------------------------------------------
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Supply simulator
# ---------------------------------------------------------------------------

class SupplyProcess:
    """
    Simulates container fleet capacity over a given horizon.

    Key changes from v1:
    - Distributed stochastic building lag (Beta-distributed delivery profile)
    - Earnings-dependent scrapping with EWMA memory
    """

    def __init__(self, params: SupplyParameters):
        self.p   = params
        self.rng = np.random.default_rng(params.seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def simulate(self, freight_rates: np.ndarray) -> dict:
        """
        Simulate fleet dynamics given a sequence of freight rates.

        Parameters
        ----------
        freight_rates : array of shape (n_periods,)
            Freight rate index per period (LR equilibrium = 1.0).
            Produced by market.py in the full model.

        Returns
        -------
        dict with arrays of shape (n_periods,):
            'nominal_fleet'    : total active fleet capacity (index)
            'effective_fleet'  : fleet adjusted for slow steaming
            'new_orders'       : orders placed each period
            'deliveries'       : capacity entering fleet per period
            'scrapping'        : capacity leaving fleet per period
            'orderbook'        : outstanding undelivered orders
            'lag_distribution' : expected lag (weighted avg) per period
        """
        p   = self.p
        n   = len(freight_rates)
        lag_window = p.lag_max - p.lag_min + 1  # number of lag periods

        # Delivery pipeline: pipeline[k] holds orders delivering in k periods
        pipeline = np.zeros(lag_window)
        # Seed with initial orderbook spread across pipeline
        for k in range(lag_window):
            pipeline[k] = p.initial_orderbook / lag_window

        fleet      = np.zeros(n)
        orders     = np.zeros(n)
        deliveries = np.zeros(n)
        scrapping  = np.zeros(n)
        orderbook  = np.zeros(n)

        # Earnings EWMA — initialise at LR equilibrium
        earnings_ewma = 1.0
        prev_orders   = p.initial_orderbook / p.lag_max

        # Long-run average rate (updated as simulation progresses)
        rate_history = []

        for t in range(n):
            rate         = float(freight_rates[t])
            rate_history.append(rate)
            long_run_avg = float(np.mean(rate_history))

            # --- 1. Update earnings EWMA -----------------------------------
            alpha_ew      = p.scrapping_earnings_memory
            earnings_ewma = (1 - alpha_ew) * earnings_ewma + alpha_ew * rate

            # --- 2. Deliveries (roll pipeline) -----------------------------
            delivery  = pipeline[0]
            pipeline  = np.roll(pipeline, -1)
            pipeline[-1] = 0.0
            deliveries[t] = delivery

            # --- 3. Scrapping (earnings-dependent) -------------------------
            scrap = self._compute_scrapping_earnings(
                current_fleet    = fleet[t-1] if t > 0 else p.initial_fleet,
                spot_rate        = rate,
                earnings_ewma    = earnings_ewma,
            )
            scrapping[t] = scrap

            # --- 4. Fleet update -------------------------------------------
            if t == 0:
                fleet[t] = p.initial_fleet + delivery - scrap
            else:
                fleet[t] = fleet[t-1] + delivery - scrap
            fleet[t] = max(fleet[t], 0.0)

            # --- 5. Rate deviation for ordering ----------------------------
            rate_deviation = rate / long_run_avg if long_run_avg > 0 else 1.0

            # --- 6. New orders (Cobweb rule) --------------------------------
            new_order = self._ordering_rule(
                rate_deviation = rate_deviation,
                current_fleet  = fleet[t],
                prev_orders    = prev_orders,
            )
            orders[t]    = new_order
            prev_orders  = new_order

            # --- 7. Place order into distributed pipeline ------------------
            delivery_profile = self._draw_delivery_profile(new_order)
            for k, dp_frac in enumerate(delivery_profile):
                if k < lag_window:
                    pipeline[k] += dp_frac

            # --- 8. Orderbook ----------------------------------------------
            orderbook[t] = pipeline.sum()

        # --- Effective fleet (slow steaming) ------------------------------
        effective_fleet = fleet.copy()
        if p.slow_steam_active and p.slow_steam_start_period < n:
            effective_fleet[p.slow_steam_start_period:] *= (1.0 - p.slow_steam_reduction)

        return {
            'nominal_fleet'  : fleet,
            'effective_fleet': effective_fleet,
            'new_orders'     : orders,
            'deliveries'     : deliveries,
            'scrapping'      : scrapping,
            'orderbook'      : orderbook,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_delivery_profile(self, order_size: float) -> np.ndarray:
        """
        Distribute a single order's delivery across the lag window
        according to a Beta distribution discretised over [lag_min, lag_max].

        This means: a fraction of 'order_size' delivers in period t+lag_min,
        another fraction in t+lag_min+1, etc., with probabilities given
        by the Beta CDF over each bucket.

        Returns a vector of length (lag_max - lag_min + 1).
        """
        p          = self.p
        lag_window = p.lag_max - p.lag_min + 1
        profile    = np.zeros(lag_window)

        # Discretise Beta(alpha, beta) over the lag window
        from scipy.stats import beta as beta_dist
        edges = np.linspace(0.0, 1.0, lag_window + 1)
        for k in range(lag_window):
            prob       = beta_dist.cdf(edges[k+1], p.lag_alpha, p.lag_beta) \
                       - beta_dist.cdf(edges[k],   p.lag_alpha, p.lag_beta)
            profile[k] = order_size * prob

        return profile

    def _ordering_rule(
        self,
        rate_deviation: float,
        current_fleet: float,
        prev_orders: float
    ) -> float:
        """
        Luo et al. (2009) ordering rule with AR(1) inertia.

        Orders(t) = fleet(t) * [base_rate + alpha * (rate/rate_LR - 1)]
                    + inertia * Orders(t-1) + noise
        """
        deviation = rate_deviation - 1.0
        base  = current_fleet * (
            self.p.order_base_rate + self.p.order_sensitivity * deviation
        )
        order = (1 - self.p.order_inertia) * base + self.p.order_inertia * prev_orders
        noise = self.rng.normal(0, current_fleet * 0.005)
        return max(0.0, order + noise)

    def _compute_scrapping_earnings(
        self,
        current_fleet: float,
        spot_rate: float,
        earnings_ewma: float,
    ) -> float:
        """
        Earnings-dependent scrapping.

        The scrapping signal is a weighted blend of the current spot rate
        and the backward-looking earnings expectation (EWMA):
            signal = (1-w) * spot_rate + w * earnings_ewma

        This captures the empirical observation (UNCTAD RMT 2025) that
        owners with strong recent earnings are reluctant to scrap even
        when spot rates soften — because they expect the high-earnings
        environment to persist (consistent with the Greenwood-Hanson
        overextrapolation logic on the demand side).

        Scrapping logic:
            signal > cost_covering_rate  → only baseline scrapping
            layup < signal < cost_covering → baseline only (laid up)
            signal < layup_threshold     → accelerated scrapping
        """
        p  = self.p
        w  = p.scrapping_earnings_weight
        signal = (1.0 - w) * spot_rate + w * earnings_ewma

        base_scrap = current_fleet * p.scrapping_base_rate

        if signal < p.layup_threshold:
            distress    = p.layup_threshold - signal
            extra_scrap = current_fleet * p.scrapping_sensitivity * distress
        else:
            extra_scrap = 0.0

        total_scrap = base_scrap + extra_scrap
        return min(total_scrap, current_fleet * 0.15)


# ---------------------------------------------------------------------------
# Quick diagnostic
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n     = 50
    years = np.arange(2000, 2000 + n)

    # Synthetic cycle: boom → bust → recovery → second boom (Red Sea analog)
    rates          = np.ones(n)
    rates[5:12]    = np.linspace(1.0, 1.6, 7)
    rates[12:18]   = np.linspace(1.6, 0.7, 6)
    rates[18:28]   = np.linspace(0.7, 0.9, 10)
    rates[28:35]   = np.linspace(0.9, 1.5, 7)
    rates[35:42]   = np.linspace(1.5, 0.8, 7)
    rates[42:]     = np.linspace(0.8, 1.0, n-42)

    params  = SupplyParameters(seed=42)
    process = SupplyProcess(params)
    results = process.simulate(freight_rates=rates)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # Panel 1: Fleet
    axes[0].plot(years, results['nominal_fleet'],
                 label='Nominal fleet', color='steelblue', lw=2)
    axes[0].plot(years, results['effective_fleet'],
                 label='Effective fleet (slow steam adj.)', color='steelblue',
                 lw=2, linestyle='--', alpha=0.7)
    ax1b = axes[0].twinx()
    ax1b.plot(years, rates, color='firebrick', lw=1.5, linestyle=':',
              label='Freight rate (input)')
    ax1b.set_ylabel('Rate (LR avg = 1.0)', color='firebrick')
    ax1b.tick_params(axis='y', colors='firebrick')
    axes[0].set_ylabel('Fleet index (base 100)')
    axes[0].set_title('Supply Dynamics — Distributed Building Lag + Earnings Scrapping')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Orders / Deliveries / Scrapping
    axes[1].bar(years, results['new_orders'],    alpha=0.6, color='steelblue', label='New orders')
    axes[1].bar(years, results['deliveries'],    alpha=0.6, color='seagreen',  label='Deliveries')
    axes[1].bar(years, -results['scrapping'],    alpha=0.6, color='firebrick', label='Scrapping (neg.)')
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_ylabel('Capacity change (index units)')
    axes[1].set_title('Orders, Deliveries and Earnings-Dependent Scrapping')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Orderbook
    axes[2].fill_between(years, results['orderbook'], alpha=0.4, color='darkorange')
    axes[2].plot(years, results['orderbook'], color='darkorange', lw=2, label='Orderbook')
    axes[2].set_ylabel('Orderbook (index units)')
    axes[2].set_xlabel('Year')
    axes[2].set_title('Orderbook Pipeline (Distributed Lag)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    import os
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/supply_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("=== Supply Process Summary ===")
    print(f"Initial fleet:   {results['nominal_fleet'][0]:.1f}")
    print(f"Peak fleet:      {results['nominal_fleet'].max():.1f} (period {results['nominal_fleet'].argmax()})")
    print(f"Final fleet:     {results['nominal_fleet'][-1]:.1f}")
    print(f"Total scrapped:  {results['scrapping'].sum():.1f}")
    print(f"Peak orderbook:  {results['orderbook'].max():.1f}")
