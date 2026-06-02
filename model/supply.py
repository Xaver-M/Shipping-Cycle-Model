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

  2. Shipbuilding lag:
       Orders placed in period t only enter the active fleet in period
       t + LAG, where LAG ≈ 2 years (Stopford 2009, pp. 155-156;
       Luo et al. 2009, pp. 512-514). This structural delay is the
       mechanism through which naive expectations generate endogenous
       overshooting.

The module also models:
  - Scrapping / demolition: vessels are retired when freight rates fall
    below a cost-covering threshold, or when they reach end-of-life age.
    This is the partial stabilising corrective identified by Stopford
    (2009) via the demolition market.
  - Slow steaming: CII-style regulatory constraints reduce *effective*
    fleet capacity below nominal TEU, distorting the price signal.
  - Fleet inertia: even at negative margins, vessels continue operating
    if the freight rate covers variable costs (above laid-up threshold),
    because high capital costs are sunk. This produces the persistent
    overcapacity documented empirically.

Key references
--------------
- Luo, Fan & Liu (2009): ordering rule specification, pp. 512-514.
- Stopford (2009): shipbuilding lag, pp. 155-156; scrapping dynamics,
  demolition market as stabiliser, Four-Market Model.
- Greenwood & Hanson (2015): overextrapolation of demand, pp. 57-58.

Units
-----
Fleet capacity is expressed as an index (base 100 at t=0), consistent
with the demand index in demand.py. This allows direct comparison of
supply and demand growth rates without unit conversion.
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

    Defaults are calibrated on empirical shipping data. The ordering
    sensitivity (alpha) and lag length are the most policy-relevant
    parameters — vary them first in sensitivity analyses.
    """

    # --- Shipbuilding lag --------------------------------------------------
    building_lag: int = 2
    """
    Number of periods between order placement and fleet entry.
    Calibrated on Luo et al. (2009), pp. 512-514: orders placed at t
    deliver at t+2. Stopford (2009, pp. 155-156) identifies this as
    fundamental and unavoidable.
    """

    # --- Ordering rule (Luo et al. 2009) -----------------------------------
    order_sensitivity: float = 0.18
    """
    Sensitivity of new orders to the current freight rate deviation
    from long-run average. Derived from Luo et al. (2009):
    a 10% above-average freight rate triggers ~1.8% of fleet capacity
    in new orders per period.

    This is the key Cobweb parameter: higher values → larger overshooting.
    """

    order_base_rate: float = 0.04
    """
    Baseline replacement ordering rate (fraction of fleet per period),
    independent of freight rate signals. Covers normal fleet renewal
    at end-of-life without cycle-driven investment.
    """

    order_inertia: float = 0.30
    """
    AR(1) coefficient on the order series. Captures momentum in
    shipyard bookings: once yards are busy, orders tend to cluster.
    Greenwood & Hanson (2015, pp. 57-58) document this herding effect.
    """

    # --- Scrapping / demolition --------------------------------------------
    scrapping_base_rate: float = 0.015
    """
    Baseline annual scrapping rate (fraction of fleet), independent
    of market conditions. Reflects normal end-of-life retirement.
    """

    scrapping_sensitivity: float = 0.12
    """
    Additional scrapping triggered per unit of freight rate below the
    cost-covering threshold. When rates are depressed, the demolition
    market provides a partial corrective — but slowly.
    Source: Stopford (2009), Four-Market Model.
    """

    cost_covering_rate: float = 0.85
    """
    Freight rate level (relative to long-run average = 1.0) below which
    additional scrapping is triggered. Vessels that cannot cover total
    costs begin to exit, but only gradually due to capital sunk costs.
    """

    layup_threshold: float = 0.60
    """
    Freight rate level below which vessels are laid up (idled) rather
    than scrapped immediately. Between layup_threshold and
    cost_covering_rate, vessels continue operating at a loss because
    variable costs are still covered. This is fleet inertia — the
    mechanism that sustains persistent overcapacity.
    """

    # --- Regulatory slow steaming (CII) ------------------------------------
    slow_steam_active: bool = True
    """
    Whether CII-style regulations are modelled. If True, effective fleet
    capacity is reduced by slow_steam_reduction during the regulatory
    period, distorting price signals as documented by Lehmann et al. (2025).
    """

    slow_steam_start_period: int = 22
    """
    Period from which slow steaming regulations take effect.
    Default: period 22 ≈ 2022, when EEXI/CII entered into force.
    """

    slow_steam_reduction: float = 0.06
    """
    Fractional reduction in effective fleet capacity due to mandatory
    speed reduction. Lehmann et al. (2025, p. 3): CII compliance can
    increase sailing time by up to 400 hours/year, equivalent to a
    ~6% effective capacity reduction.
    """

    # --- Market concentration effect ---------------------------------------
    n_carriers: int = 20
    """
    Number of independent ordering decision-makers. As concentration
    increases (fewer carriers), the Prisoner's Dilemma structure of
    overordering is unchanged but the number of simultaneous
    synchronized orders per period decreases.
    Cariou & Guillotreau (2022): even 4→5 carriers shows no
    measurable improvement in capacity discipline.
    """

    # --- Initial conditions ------------------------------------------------
    initial_fleet: float = 100.0
    """Base fleet capacity at t=0 (index). Matches initial_demand=100."""

    initial_orderbook: float = 8.0
    """
    Orders placed before t=0 that will deliver during the simulation.
    Calibrated on historical orderbook-to-fleet ratios (~8% at cycle trough).
    """

    # --- Random seed -------------------------------------------------------
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Supply simulator
# ---------------------------------------------------------------------------

class SupplyProcess:
    """
    Simulates container fleet capacity over a given horizon.

    The supply process takes freight rates as input (produced by
    market.py) and generates fleet capacity as output. This makes
    the supply module stateless between periods — the full simulation
    loop is managed by market.py.

    For standalone testing, a synthetic freight rate series can be
    passed directly (see __main__ block below).

    Outputs
    -------
    'nominal_fleet'    : TEU index, total ordered + delivered capacity
    'effective_fleet'  : nominal_fleet adjusted for slow steaming
    'new_orders'       : orders placed each period
    'deliveries'       : capacity entering fleet each period
    'scrapping'        : capacity retired each period
    'orderbook'        : outstanding orders at end of each period
    """

    def __init__(self, params: SupplyParameters):
        self.p = params
        self.rng = np.random.default_rng(params.seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def simulate(
        self,
        freight_rates: np.ndarray,
        long_run_rate: float = 1.0
    ) -> dict:
        """
        Simulate fleet dynamics given a freight rate series.

        Parameters
        ----------
        freight_rates : array of shape (n_periods,)
            Freight rate index each period. Long-run equilibrium = 1.0.
        long_run_rate : float
            Reference rate for deviation calculations. Default 1.0.

        Returns
        -------
        dict with keys listed in class docstring.
        """
        n = len(freight_rates)
        lag = self.p.building_lag

        # Initialise arrays
        fleet      = np.zeros(n)
        orders     = np.zeros(n)
        deliveries = np.zeros(n)
        scrapping  = np.zeros(n)
        orderbook  = np.zeros(n)

        fleet[0] = self.p.initial_fleet

        # Pre-fill orderbook pipeline for periods before t=0
        # These represent orders placed before the simulation starts
        pipeline = np.zeros(lag)
        pipeline[-1] = self.p.initial_orderbook

        prev_orders = self.p.initial_orderbook  # for AR(1) inertia

        for t in range(n):

            # --- 1. Deliveries from pipeline -------------------------------
            delivery = pipeline[0]
            deliveries[t] = delivery

            # Roll pipeline forward
            pipeline = np.roll(pipeline, -1)
            pipeline[-1] = 0.0

            # --- 2. Scrapping ----------------------------------------------
            rate_deviation = freight_rates[t] / long_run_rate
            scrap = self._compute_scrapping(fleet[t-1] if t > 0 else self.p.initial_fleet,
                                            rate_deviation)
            scrapping[t] = scrap

            # --- 3. Fleet update -------------------------------------------
            if t == 0:
                fleet[t] = self.p.initial_fleet + delivery - scrap
            else:
                fleet[t] = fleet[t-1] + delivery - scrap
            fleet[t] = max(fleet[t], 0.0)  # fleet cannot go negative

            # --- 4. New orders (Cobweb ordering rule) ----------------------
            new_order = self._ordering_rule(
                rate_deviation = rate_deviation,
                current_fleet  = fleet[t],
                prev_orders    = prev_orders
            )
            orders[t]    = new_order
            prev_orders  = new_order

            # Place into pipeline (delivers in `lag` periods)
            if t + lag < n:
                pipeline[-1] += new_order  # will roll into position over lag periods
            # Note: orders placed near end of simulation are tracked but
            # don't deliver within the horizon — this is intentional.

            # --- 5. Orderbook ----------------------------------------------
            orderbook[t] = pipeline.sum()

        # --- Effective fleet (slow steaming adjustment) -------------------
        effective_fleet = fleet.copy()
        if self.p.slow_steam_active:
            start = self.p.slow_steam_start_period
            if start < n:
                effective_fleet[start:] *= (1.0 - self.p.slow_steam_reduction)

        return {
            'nominal_fleet'   : fleet,
            'effective_fleet' : effective_fleet,
            'new_orders'      : orders,
            'deliveries'      : deliveries,
            'scrapping'       : scrapping,
            'orderbook'       : orderbook,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ordering_rule(
        self,
        rate_deviation: float,
        current_fleet: float,
        prev_orders: float
    ) -> float:
        """
        Luo et al. (2009) ordering rule with AR(1) inertia.

        Orders(t) = fleet(t) * [base_rate + alpha * (rate - r_LR)/r_LR]
                    + inertia * Orders(t-1)
                    + noise

        The noise term captures idiosyncratic yard-level randomness and
        prevents perfect synchronisation across carriers — though in
        practice, as Greenwood & Hanson (2015) show, synchronisation
        is already very high from the shared price signal alone.
        """
        # Rate deviation from long-run (positive = boom, negative = bust)
        deviation = rate_deviation - 1.0

        # Base order volume
        base = current_fleet * (
            self.p.order_base_rate
            + self.p.order_sensitivity * deviation
        )

        # Inertia from previous period orders
        order = (1 - self.p.order_inertia) * base + self.p.order_inertia * prev_orders

        # Small idiosyncratic noise
        noise = self.rng.normal(0, current_fleet * 0.005)
        order = max(0.0, order + noise)  # orders cannot be negative

        return order

    def _compute_scrapping(
        self,
        current_fleet: float,
        rate_deviation: float
    ) -> float:
        """
        Scrapping rate rises when freight rates fall below cost-covering level.

        Above cost_covering_rate: only baseline scrapping (end-of-life).
        Between layup and cost-covering: baseline only (vessels laid up,
          not scrapped — fleet inertia).
        Below layup_threshold: accelerated scrapping kicks in.

        This asymmetry — slow to scrap, fast to order — is the structural
        source of persistent overcapacity documented by Stopford (2009).
        """
        base_scrap = current_fleet * self.p.scrapping_base_rate

        if rate_deviation < self.p.layup_threshold:
            # Severe distress: accelerated scrapping
            distress = self.p.layup_threshold - rate_deviation
            extra_scrap = current_fleet * self.p.scrapping_sensitivity * distress
        elif rate_deviation < self.p.cost_covering_rate:
            # Vessels laid up but not scrapped — no additional scrapping
            extra_scrap = 0.0
        else:
            # Normal conditions
            extra_scrap = 0.0

        total_scrap = base_scrap + extra_scrap
        # Cannot scrap more than exists
        return min(total_scrap, current_fleet * 0.15)


# ---------------------------------------------------------------------------
# Quick diagnostic (standalone test with synthetic freight rates)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Synthetic freight rate: boom-bust cycle to stress-test supply response
    n = 50
    years = np.arange(2000, 2000 + n)

    # Simulate a boom (rates rise to 1.6x for 5 years) then bust (fall to 0.7x)
    # then recovery — a stylised version of the 2003-2016 cycle
    rates = np.ones(n)
    rates[5:12]  = np.linspace(1.0, 1.6, 7)   # boom
    rates[12:18] = np.linspace(1.6, 0.7, 6)   # collapse
    rates[18:28] = np.linspace(0.7, 0.9, 10)  # slow recovery
    rates[28:35] = np.linspace(0.9, 1.5, 7)   # second boom (Red Sea analog)
    rates[35:42] = np.linspace(1.5, 0.8, 7)   # second bust
    rates[42:]   = np.linspace(0.8, 1.0, n-42)

    params  = SupplyParameters(seed=42)
    process = SupplyProcess(params)
    results = process.simulate(freight_rates=rates)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # Panel 1: Fleet vs freight rates
    ax1 = axes[0]
    ax1.plot(years, results['nominal_fleet'],
             label='Nominal fleet capacity', color='steelblue', lw=2)
    ax1.plot(years, results['effective_fleet'],
             label='Effective fleet (slow steam adj.)', color='steelblue',
             lw=2, linestyle='--', alpha=0.7)
    ax1.set_ylabel('Fleet index (base 100)')
    ax1.set_title('Supply-Side Dynamics — Shipbuilding Lag & Cobweb Ordering')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(years, rates, color='firebrick', lw=1.5,
              linestyle=':', label='Freight rate (input)')
    ax1b.axhline(1.0, color='firebrick', lw=0.8, alpha=0.4, linestyle='--')
    ax1b.set_ylabel('Freight rate (LR avg = 1.0)', color='firebrick')
    ax1b.tick_params(axis='y', colors='firebrick')
    ax1b.legend(loc='upper right')

    # Panel 2: Orders, deliveries, scrapping
    ax2 = axes[1]
    ax2.bar(years, results['new_orders'],
            alpha=0.6, color='steelblue', label='New orders')
    ax2.bar(years, results['deliveries'],
            alpha=0.6, color='seagreen', label='Deliveries', bottom=0)
    ax2.bar(years, -results['scrapping'],
            alpha=0.6, color='firebrick', label='Scrapping (negative)')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_ylabel('Capacity change (index units)')
    ax2.set_title('Orders, Deliveries and Scrapping')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Orderbook
    ax3 = axes[2]
    ax3.fill_between(years, results['orderbook'],
                     alpha=0.4, color='darkorange', label='Outstanding orderbook')
    ax3.plot(years, results['orderbook'],
             color='darkorange', lw=2)
    ax3.set_ylabel('Orderbook (index units)')
    ax3.set_xlabel('Year')
    ax3.set_title('Orderbook Pipeline')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/supply_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Diagnostic plot saved to data/supply_diagnostic.png")
    print("\n--- Supply Process Summary ---")
    print(f"Initial fleet:        {results['nominal_fleet'][0]:.1f}")
    print(f"Peak fleet:           {results['nominal_fleet'].max():.1f} (period {results['nominal_fleet'].argmax()})")
    print(f"Final fleet:          {results['nominal_fleet'][-1]:.1f}")
    print(f"Total fleet growth:   {(results['nominal_fleet'][-1]/results['nominal_fleet'][0]-1)*100:.1f}%")
    print(f"Peak orderbook:       {results['orderbook'].max():.1f}")
    print(f"Total scrapped:       {results['scrapping'].sum():.1f}")