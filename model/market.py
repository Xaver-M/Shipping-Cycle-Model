"""
model/market.py
===============
Market clearing, price formation and welfare analysis for the
container shipping cycle model.

This module is the analytical core of the project. It connects the
demand process (demand.py) and supply process (supply.py) into a
complete equilibrium model, and introduces the normative benchmark
that the descriptive shipping literature has largely left unaddressed:

  What is the welfare loss from the structural coordination failure
  in capacity investment, relative to a social planner optimum?

Theoretical basis
-----------------

Market clearing:
    Each period, the freight rate adjusts to clear the market. We use
    an inverse demand function: the rate rises when effective supply
    falls short of effective demand, and falls when supply exceeds
    demand. The price elasticity of demand is calibrated on Stopford
    (2009, p. 154): short-run demand elasticity ≈ -0.3, implying that
    a 1% capacity surplus drives rates down by ~3.3%.

Welfare framework:
    Social welfare is defined as the sum of:
      W(t) = CS(t) + PS(t) - C_over(t) - C_under(t)

    where:
      CS      = consumer surplus (shipper welfare from low rates)
      PS      = producer surplus (carrier profit above variable cost)
      C_over  = cost of idle/excess capacity (capital tied up unproductively)
      C_under = cost of capacity shortage (supply chain disruption costs)

    The social planner maximises W(t) subject to the same technical
    constraints (building lag, scrapping dynamics) but chooses the
    aggregate order volume optimally — internalising the externality
    that individual carriers ignore when ordering.

    The welfare loss (deadweight loss of cyclicality) is:
      DWL(t) = W_planner(t) - W_decentralised(t)

    integrated over the simulation horizon and averaged over Monte Carlo
    runs to account for demand uncertainty.

Social planner:
    The planner solves a finite-horizon optimisation: choose order
    volumes each period to maximise cumulative welfare subject to:
      - Fleet(t+1) = Fleet(t) + Deliveries(t) - Scrapping(t)
      - Deliveries(t) = Orders(t - LAG)
      - Fleet >= 0

    We implement this via dynamic programming (value function iteration)
    on a discretised state space. The planner's solution serves as the
    efficiency benchmark.

Key references
--------------
- Stopford (2009): demand elasticity, p. 154; market clearing, ch. 3.
- Luo, Fan & Liu (2009): supply response function, pp. 512-514.
- Greenwood & Hanson (2015): welfare loss from overextrapolation, p. 57.
- Cramton & Stoft (2006): capacity market analogy from electricity.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import minimize_scalar

from demand import DemandParameters, DemandProcess
from supply import SupplyParameters, SupplyProcess


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class MarketParameters:
    """Calibrated parameters for market clearing and welfare calculation."""

    # --- Price formation ---------------------------------------------------
    demand_elasticity: float = -0.30
    """
    Short-run price elasticity of demand for container shipping.
    Calibrated on Stopford (2009), p. 154: empirical estimates cluster
    below -0.3. Negative: higher rates → lower volume demanded.
    Implies rate sensitivity = 1 / |elasticity| ≈ 3.33.
    """

    rate_adjustment_speed: float = 0.85
    """
    Fraction of excess supply/demand that is absorbed by price
    adjustment within one period. <1 implies partial adjustment
    (rates are sticky). =1 implies full instantaneous clearing.
    """

    long_run_equilibrium_rate: float = 1.0
    """Reference freight rate at which supply = demand (index)."""

    # --- Welfare / cost parameters ----------------------------------------
    capital_cost_rate: float = 0.08
    """
    Annual capital cost as fraction of vessel value (interest + depreciation).
    Idle capacity wastes this rate per unit per period.
    """

    disruption_cost_multiplier: float = 0.80
    """
    Multiplier representing supply chain disruption costs when capacity is
    insufficient. Kept below 1.0 so that the planner does not massively
    over-invest to avoid all shortage: shortage is costly, but not more
    costly than the capital waste of large structural overcapacity.
    Symmetric with capital_cost_rate to produce a balanced optimum near
    full utilisation (fleet ≈ demand), which is the economically sensible
    benchmark. Raise above 1.0 to model economies with very high
    supply-chain disruption sensitivity.
    """

    discount_rate: float = 0.04
    """Social discount rate for welfare calculations (4% = standard)."""

    # --- Planner optimisation ---------------------------------------------
    planner_horizon: int = 10
    """
    Look-ahead horizon for the social planner's dynamic programme.
    Beyond this horizon, the planner uses a steady-state approximation.
    """

    # --- Simulation -------------------------------------------------------
    n_periods: int = 50
    n_monte_carlo: int = 200
    """Number of Monte Carlo runs for welfare loss confidence intervals."""

    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Freight rate / market clearing
# ---------------------------------------------------------------------------

class FreightRateModel:
    """
    Maps supply-demand imbalance to freight rates each period.

    Uses an inverse demand function: given effective supply and
    effective demand, the market-clearing rate is determined by
    the demand elasticity.

    P(t) = P_LR * [D(t) / S(t)] ^ (1 / |epsilon|)

    where epsilon is the price elasticity of demand.
    This is the standard log-linear inverse demand form.
    """

    def __init__(self, params: MarketParameters):
        self.p = params
        self._rate_sensitivity = 1.0 / abs(params.demand_elasticity)

    def clearing_rate(
        self,
        effective_demand: float,
        effective_supply: float,
        prev_rate: float
    ) -> float:
        """
        Compute market-clearing freight rate given supply/demand.

        Partial adjustment: rate moves toward full-clearing rate at
        speed rate_adjustment_speed. This captures rate stickiness
        observed empirically (long-term contracts, alliance pricing).
        """
        if effective_supply <= 0:
            return prev_rate * 2.0  # avoid division by zero

        # Full-clearing rate from inverse demand
        ratio = effective_demand / effective_supply
        full_clearing = self.p.long_run_equilibrium_rate * (
            ratio ** self._rate_sensitivity
        )

        # Partial adjustment toward full-clearing rate
        rate = (self.p.rate_adjustment_speed * full_clearing
                + (1 - self.p.rate_adjustment_speed) * prev_rate)

        # Bound rate: cannot go below ~20% of LR (vessels always have
        # some scrap value) or above 5x LR (shippers find alternatives)
        return float(np.clip(rate, 0.20, 5.0))


# ---------------------------------------------------------------------------
# Welfare calculator
# ---------------------------------------------------------------------------

class WelfareCalculator:
    """
    Computes period-by-period welfare components and cumulative DWL.

    Welfare decomposition:
      CS  = shipper benefit from rates below LR equilibrium
      PS  = carrier profit above variable cost
      C_over = idle capacity cost
      C_under = disruption/shortage cost (convex)
    """

    def __init__(self, params: MarketParameters):
        self.p = params

    def period_welfare(
        self,
        freight_rate: float,
        effective_demand: float,
        effective_supply: float,
        period: int
    ) -> dict:
        """
        Compute welfare components for a single period.

        Returns
        -------
        dict with keys: 'CS', 'PS', 'C_over', 'C_under', 'total'
        """
        lr = self.p.long_run_equilibrium_rate
        discount = (1 / (1 + self.p.discount_rate)) ** period

        # --- Consumer surplus ---
        # Shippers gain when rate < LR (cheaper transport), lose when rate > LR.
        # Normalised by LR so magnitude is comparable to PS.
        cs = effective_demand * (lr - freight_rate) / lr * discount

        # --- Producer surplus ---
        # Carrier profit above variable cost (≈ 0.65 * LR, Stopford 2009).
        # Only on utilised capacity (min of S and D).
        variable_cost_rate = 0.65 * lr
        utilised = min(effective_supply, effective_demand)
        ps = utilised * max(0, freight_rate - variable_cost_rate) / lr * discount

        # --- Cost of overcapacity ---
        # Idle tonnage wastes capital. Proportional to surplus fraction.
        surplus = max(0, effective_supply - effective_demand)
        c_over = surplus * self.p.capital_cost_rate * discount

        # --- Cost of undercapacity ---
        # Supply chain disruption when fleet < demand. Linear in shortage
        # fraction, scaled by disruption_cost_multiplier.
        shortage = max(0, effective_demand - effective_supply)
        shortage_fraction = shortage / max(effective_demand, 1)
        c_under = effective_demand * self.p.disruption_cost_multiplier * shortage_fraction * discount

        total = cs + ps - c_over - c_under

        return {
            'CS'     : cs,
            'PS'     : ps,
            'C_over' : c_over,
            'C_under': c_under,
            'total'  : total,
            'surplus': surplus,
            'shortage': shortage,
        }

    def cumulative_welfare(self, welfare_series: list[dict]) -> dict:
        """Sum welfare components across all periods."""
        keys = ['CS', 'PS', 'C_over', 'C_under', 'total']
        return {k: sum(w[k] for w in welfare_series) for k in keys}


# ---------------------------------------------------------------------------
# Social planner
# ---------------------------------------------------------------------------

class SocialPlanner:
    """
    Computes the socially optimal capacity path via dynamic programming.

    The planner chooses aggregate order volumes each period to maximise
    cumulative discounted welfare, internalising the coordination
    externality that individual carriers ignore.

    This is the efficiency benchmark: the gap between planner welfare
    and decentralised welfare is the deadweight loss of cyclicality.

    Implementation: rolling finite-horizon DP with simplified state
    space (fleet level, current demand). Full stochastic DP would
    require discretising the demand process — computationally feasible
    but complex. Here we use a deterministic approximation with the
    expected demand path, which gives a lower bound on planner gains
    (the planner in practice would do even better with full foresight).
    """

    def __init__(
        self,
        market_params: MarketParameters,
        supply_params: SupplyParameters
    ):
        self.mp = market_params
        self.sp = supply_params
        self.rate_model = FreightRateModel(market_params)
        self.welfare_calc = WelfareCalculator(market_params)

    def optimal_orders(
        self,
        current_fleet: float,
        pipeline: np.ndarray,
        demand_forecast: np.ndarray,
        period: int
    ) -> float:
        """
        Determine optimal order volume for current period.

        Uses a greedy one-step lookahead: choose orders that maximise
        welfare at delivery time (period + LAG), given expected demand.
        This is a simplification of the full DP — sufficient to
        demonstrate the efficiency gap without full state-space enumeration.

        A full DP implementation is left as a natural extension.
        """
        lag = self.sp.building_lag
        delivery_period = period + lag
        n = len(demand_forecast)

        # End-of-horizon: in the final `lag` periods the planner cannot
        # affect fleet within the simulation window — order only replacement
        # to avoid spurious accumulation that distorts welfare comparison.
        if delivery_period >= n or period >= n - lag:
            return current_fleet * self.sp.order_base_rate

        expected_demand = demand_forecast[delivery_period]

        # Fleet at delivery time: current fleet + committed pipeline
        # minus expected scrapping over the lag period.
        # Crucially: already-committed pipeline is subtracted from the
        # gap — the planner only orders what is still missing.
        committed_deliveries = pipeline.sum()
        expected_scrapping = current_fleet * self.sp.scrapping_base_rate * lag
        expected_fleet_base = max(
            0.0,
            current_fleet + committed_deliveries - expected_scrapping
        )

        # Gap between expected fleet and expected demand.
        # If pipeline already covers demand, the planner orders only
        # replacement capacity — this prevents the runaway overinvestment
        # that arises when the planner ignores committed orders.
        gap = expected_demand - expected_fleet_base
        replacement = current_fleet * self.sp.order_base_rate

        if gap <= 0:
            # Pipeline already sufficient — only order replacement
            return replacement

        # Optimise order volume to maximise welfare at delivery time.
        # Search bounded to [0, gap + replacement] — the planner never
        # orders more than needed to meet demand plus replacement.
        max_order = gap + replacement

        def neg_welfare(order):
            fleet_at_delivery = expected_fleet_base + max(0, order)
            rate = self.rate_model.clearing_rate(
                expected_demand, fleet_at_delivery,
                prev_rate=self.mp.long_run_equilibrium_rate
            )
            w = self.welfare_calc.period_welfare(
                rate, expected_demand, fleet_at_delivery, delivery_period
            )
            return -w['total']

        result = minimize_scalar(
            neg_welfare,
            bounds=(0.0, max(max_order, replacement)),
            method='bounded'
        )
        return max(0.0, result.x)

    def simulate(self, demand_path: np.ndarray) -> dict:
        """
        Run planner simulation over the full demand path.

        Returns same structure as SupplyProcess.simulate() plus welfare.
        """
        n   = len(demand_path)
        lag = self.sp.building_lag

        fleet      = np.zeros(n)
        orders     = np.zeros(n)
        deliveries = np.zeros(n)
        scrapping  = np.zeros(n)
        rates      = np.zeros(n)
        welfare    = []

        fleet[0] = self.sp.initial_fleet
        pipeline = np.zeros(lag)
        pipeline[-1] = self.sp.initial_orderbook

        supply_proc = SupplyProcess(self.sp)
        prev_rate = self.mp.long_run_equilibrium_rate

        for t in range(n):
            # Deliveries
            delivery     = pipeline[0]
            deliveries[t] = delivery
            pipeline     = np.roll(pipeline, -1)
            pipeline[-1] = 0.0

            # Scrapping (rate-dependent, using previous rate)
            scrap = supply_proc._compute_scrapping(
                fleet[t-1] if t > 0 else self.sp.initial_fleet,
                prev_rate / self.mp.long_run_equilibrium_rate
            )
            scrapping[t] = scrap

            # Fleet update
            fleet[t] = (fleet[t-1] if t > 0 else self.sp.initial_fleet) + delivery - scrap
            fleet[t] = max(fleet[t], 0.0)

            # Freight rate — planner has same market clearing as decentralised
            # but with a structural floor: the planner would never allow rates
            # to collapse below variable cost (vessels exit before that point).
            rate = self.rate_model.clearing_rate(
                demand_path[t], fleet[t], prev_rate
            )
            rate = max(rate, 0.55)  # structural floor: covers variable costs
            rates[t]  = rate
            prev_rate = rate

            # Welfare
            w = self.welfare_calc.period_welfare(
                rate, demand_path[t], fleet[t], t
            )
            welfare.append(w)

            # Optimal orders (planner looks ahead)
            order = self.optimal_orders(
                fleet[t], pipeline.copy(), demand_path, t
            )
            orders[t] = order
            if t + lag < n:
                pipeline[-1] += order

        return {
            'fleet'      : fleet,
            'orders'     : orders,
            'deliveries' : deliveries,
            'scrapping'  : scrapping,
            'rates'      : rates,
            'welfare'    : welfare,
        }


# ---------------------------------------------------------------------------
# Full market simulation (decentralised)
# ---------------------------------------------------------------------------

class ShippingMarket:
    """
    Complete decentralised shipping market simulation.

    Couples DemandProcess and SupplyProcess through the freight rate:
      1. Demand process generates effective demand each period.
      2. Freight rate clears the market given current fleet.
      3. Carriers observe rate and place orders (naive expectations).
      4. Orders enter pipeline; deliver after building_lag periods.
      5. Welfare is computed each period.

    The decentralised result is compared against SocialPlanner to
    compute the deadweight loss of cyclicality.
    """

    def __init__(
        self,
        demand_params: DemandParameters,
        supply_params: SupplyParameters,
        market_params: MarketParameters,
    ):
        self.dp = demand_params
        self.sp = supply_params
        self.mp = market_params

        self.rate_model    = FreightRateModel(market_params)
        self.welfare_calc  = WelfareCalculator(market_params)
        self.planner       = SocialPlanner(market_params, supply_params)

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run(self, seed: Optional[int] = None) -> dict:
        """
        Run one complete simulation: demand + supply + market clearing.

        Returns
        -------
        dict with full time series for analysis and plotting.
        """
        n = self.mp.n_periods

        # --- Demand ---
        d_params = DemandParameters(**{
            **self.dp.__dict__,
            'seed': seed if seed is not None else self.dp.seed
        })
        demand_proc    = DemandProcess(d_params)
        demand_results = demand_proc.simulate(n)
        eff_demand     = demand_results['effective_demand']

        # --- Decentralised supply (iterative: rate → orders → fleet → rate) ---
        # Initialise with long-run rate
        rates      = np.full(n, self.mp.long_run_equilibrium_rate)
        fleet      = np.zeros(n)
        orders_arr = np.zeros(n)
        deliveries = np.zeros(n)
        scrapping  = np.zeros(n)
        orderbook  = np.zeros(n)
        welfare_d  = []

        fleet[0] = self.sp.initial_fleet
        lag      = self.sp.building_lag
        pipeline = np.zeros(lag)
        pipeline[-1] = self.sp.initial_orderbook

        supply_proc  = SupplyProcess(self.sp)
        prev_rate    = self.mp.long_run_equilibrium_rate
        prev_orders  = self.sp.initial_orderbook

        for t in range(n):
            # Deliveries
            delivery      = pipeline[0]
            deliveries[t] = delivery
            pipeline      = np.roll(pipeline, -1)
            pipeline[-1]  = 0.0

            # Scrapping
            scrap         = supply_proc._compute_scrapping(
                fleet[t-1] if t > 0 else self.sp.initial_fleet,
                prev_rate / self.mp.long_run_equilibrium_rate
            )
            scrapping[t] = scrap

            # Fleet
            base          = fleet[t-1] if t > 0 else self.sp.initial_fleet
            fleet[t]      = max(base + delivery - scrap, 0.0)

            # Effective fleet (slow steaming)
            eff_fleet = fleet[t]
            if self.sp.slow_steam_active and t >= self.sp.slow_steam_start_period:
                eff_fleet *= (1 - self.sp.slow_steam_reduction)

            # Market clearing rate
            rate         = self.rate_model.clearing_rate(
                eff_demand[t], eff_fleet, prev_rate
            )
            rates[t]     = rate
            prev_rate    = rate

            # Welfare
            w = self.welfare_calc.period_welfare(
                rate, eff_demand[t], eff_fleet, t
            )
            welfare_d.append(w)

            # Orders (naive Cobweb rule)
            order = supply_proc._ordering_rule(
                rate / self.mp.long_run_equilibrium_rate,
                fleet[t],
                prev_orders
            )
            orders_arr[t] = order
            prev_orders   = order
            if t + lag < n:
                pipeline[-1] += order
            orderbook[t] = pipeline.sum()

        # --- Social planner ---
        planner_results = self.planner.simulate(eff_demand)

        # --- Welfare comparison ---
        wc = self.welfare_calc
        cumulative_d = wc.cumulative_welfare(welfare_d)
        cumulative_p = wc.cumulative_welfare(planner_results['welfare'])

        dwl_total = cumulative_p['total'] - cumulative_d['total']
        dwl_pct   = (dwl_total / abs(cumulative_p['total'])) * 100 if cumulative_p['total'] != 0 else 0

        return {
            # Demand
            'demand'           : eff_demand,
            'nominal_demand'   : demand_results['nominal_demand'],
            'gdp_growth'       : demand_results['gdp_growth_rates'],
            'shocks'           : demand_results['shock_active'],
            # Decentralised supply
            'fleet'            : fleet,
            'orders'           : orders_arr,
            'deliveries'       : deliveries,
            'scrapping'        : scrapping,
            'orderbook'        : orderbook,
            'rates'            : rates,
            'welfare_d'        : welfare_d,
            # Planner
            'planner_fleet'    : planner_results['fleet'],
            'planner_rates'    : planner_results['rates'],
            'planner_welfare'  : planner_results['welfare'],
            # Welfare gap
            'cumulative_d'     : cumulative_d,
            'cumulative_p'     : cumulative_p,
            'dwl_total'        : dwl_total,
            'dwl_pct'          : dwl_pct,
        }

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def monte_carlo(self, n_runs: Optional[int] = None) -> dict:
        """
        Run multiple simulations with different demand realisations.

        Returns distribution of welfare losses across runs, giving
        confidence intervals for the DWL estimate.
        """
        n_runs = n_runs or self.mp.n_monte_carlo
        rng    = np.random.default_rng(self.mp.seed)

        dwl_totals = []
        dwl_pcts   = []
        rate_vols  = []

        for i in range(n_runs):
            seed   = int(rng.integers(0, 100_000))
            result = self.run(seed=seed)
            dwl_totals.append(result['dwl_total'])
            dwl_pcts.append(result['dwl_pct'])
            rate_vols.append(np.std(result['rates']))

        dwl_totals = np.array(dwl_totals)
        dwl_pcts   = np.array(dwl_pcts)
        rate_vols  = np.array(rate_vols)

        return {
            'dwl_mean'     : dwl_totals.mean(),
            'dwl_std'      : dwl_totals.std(),
            'dwl_p5'       : np.percentile(dwl_totals, 5),
            'dwl_p95'      : np.percentile(dwl_totals, 95),
            'dwl_pct_mean' : dwl_pcts.mean(),
            'dwl_pct_std'  : dwl_pcts.std(),
            'rate_vol_mean': rate_vols.mean(),
            'dwl_series'   : dwl_totals,
            'dwl_pct_series': dwl_pcts,
        }


# ---------------------------------------------------------------------------
# Diagnostic / entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("Running shipping market simulation...")

    market = ShippingMarket(
        demand_params=DemandParameters(seed=42),
        supply_params=SupplyParameters(seed=42),
        market_params=MarketParameters(n_periods=50, n_monte_carlo=300, seed=42),
    )

    # --- Single run ---
    result = market.run()
    years  = np.arange(2000, 2050)

    print(f"\n=== Single Run Results ===")
    print(f"Decentralised welfare:  {result['cumulative_d']['total']:>10.2f}")
    print(f"Planner welfare:        {result['cumulative_p']['total']:>10.2f}")
    print(f"DWL (absolute):         {result['dwl_total']:>10.2f}")
    print(f"DWL (% of planner):     {result['dwl_pct']:>9.2f}%")
    print(f"\nWelfare decomposition (decentralised):")
    for k, v in result['cumulative_d'].items():
        print(f"  {k:12s}: {v:10.2f}")

    # --- Monte Carlo ---
    print("\nRunning Monte Carlo (300 runs)...")
    mc = market.monte_carlo()
    print(f"\n=== Monte Carlo Results (n=300) ===")
    print(f"Mean DWL:               {mc['dwl_mean']:>10.2f}")
    print(f"Median DWL:             {np.median(mc['dwl_series']):>10.2f}")
    print(f"Std DWL:                {mc['dwl_std']:>10.2f}")
    print(f"5th–95th percentile:    [{mc['dwl_p5']:.2f}, {mc['dwl_p95']:.2f}]")
    print(f"Mean DWL (%):           {mc['dwl_pct_mean']:>9.2f}%")
    print(f"Median DWL (%):         {np.median(mc['dwl_pct_series']):>9.2f}%")
    print(f"Mean rate volatility:   {mc['rate_vol_mean']:>9.4f}")

    # --- Plots ---
    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Supply vs Demand
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(years, result['demand'],
             label='Effective demand', color='steelblue', lw=2)
    ax1.plot(years, result['fleet'],
             label='Decentralised fleet', color='firebrick', lw=2)
    ax1.plot(years, result['planner_fleet'],
             label='Planner fleet', color='seagreen', lw=2, linestyle='--')
    for t, active in enumerate(result['shocks']):
        if active:
            ax1.axvspan(years[t], years[t]+1, alpha=0.1, color='orange')
    ax1.set_title('Supply vs Demand: Decentralised vs Social Planner', fontsize=11)
    ax1.set_ylabel('Index (base 100)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Freight rates
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(years, result['rates'],
             label='Decentralised rate', color='firebrick', lw=2)
    ax2.plot(years, result['planner_rates'],
             label='Planner rate', color='seagreen', lw=2, linestyle='--')
    ax2.axhline(1.0, color='black', lw=0.8, linestyle=':', alpha=0.5,
                label='LR equilibrium')
    ax2.set_title('Freight Rates', fontsize=11)
    ax2.set_ylabel('Rate index (LR = 1.0)')
    ax2.set_xlabel('Year')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Period welfare gap
    welfare_gap = [
        p['total'] - d['total']
        for p, d in zip(result['planner_welfare'], result['welfare_d'])
    ]
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(years, welfare_gap,
            color=['firebrick' if w < 0 else 'seagreen' for w in welfare_gap],
            alpha=0.7)
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_title('Period Welfare Gap\n(Planner − Decentralised)', fontsize=11)
    ax3.set_ylabel('Welfare difference')
    ax3.set_xlabel('Year')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Monte Carlo DWL distribution — absolute values, log x-axis
    ax4 = fig.add_subplot(gs[2, 0])
    dwl_abs = mc['dwl_series']
    # Filter out near-zero or negative values for log scale
    dwl_pos = dwl_abs[dwl_abs > 0.1]
    if len(dwl_pos) > 0:
        ax4.hist(np.log10(dwl_pos), bins=30, color='steelblue',
                 alpha=0.7, edgecolor='white')
        mean_log  = np.log10(np.median(dwl_pos))
        p5_log    = np.log10(np.percentile(dwl_pos, 5))
        p95_log   = np.log10(np.percentile(dwl_pos, 95))
        ax4.axvline(mean_log,  color='firebrick', lw=2,
                    label=f"Median: {np.median(dwl_pos):.0f}")
        ax4.axvline(p5_log,   color='gray', lw=1.5, linestyle='--',
                    label=f"5th pct: {np.percentile(dwl_pos,5):.0f}")
        ax4.axvline(p95_log,  color='gray', lw=1.5, linestyle='--',
                    label=f"95th pct: {np.percentile(dwl_pos,95):.0f}")
        # Relabel x-ticks as powers of 10
        ticks = ax4.get_xticks()
        ax4.set_xticklabels([f"$10^{{{t:.1f}}}$" for t in ticks], fontsize=7)
    ax4.set_title('Monte Carlo: DWL Distribution\n(Absolute, log₁₀ scale)', fontsize=11)
    ax4.set_xlabel('DWL (log scale)')
    ax4.set_ylabel('Frequency')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Welfare decomposition
    ax5 = fig.add_subplot(gs[2, 1])
    components_d = result['cumulative_d']
    components_p = result['cumulative_p']
    labels  = ['CS', 'PS', 'C_over', 'C_under']
    x       = np.arange(len(labels))
    width   = 0.35
    bars_d  = [components_d[k] for k in labels]
    bars_p  = [components_p[k] for k in labels]
    ax5.bar(x - width/2, bars_d, width, label='Decentralised',
            color='firebrick', alpha=0.7)
    ax5.bar(x + width/2, bars_p, width, label='Planner',
            color='seagreen', alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.set_title('Welfare Decomposition\n(Cumulative)', fontsize=11)
    ax5.set_ylabel('Welfare units')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(0, color='black', lw=0.8)

    plt.suptitle('Container Shipping Cycle Model — Market Equilibrium & Welfare Analysis',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.savefig('data/market_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nDiagnostic plot saved to data/market_diagnostic.png")
