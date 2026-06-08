"""
model/heterogeneous_carriers.py
================================
Heterogeneous carrier market structure for the container shipping
cycle model.

Motivation
----------
The baseline consolidation module (interventions/consolidation.py)
treats all carriers as symmetric Cournot players. This module
relaxes that assumption by introducing two distinct carrier types:

  Type S — Strategic carriers (large alliance members):
      MSC, Maersk, CMA CGM, COSCO collectively control ~65% of
      global TEU capacity (UNCTAD RMT 2025). They:
      - Are aware of their price impact (non-zero conjectural variation)
      - Have access to long-term charter markets insulating them from
        spot rate volatility
      - Face capital cost structures that disincentivise rapid ordering
        (large ULCV vessels cost $180-200m each)
      - React to both spot rates AND forward-looking market intelligence
        (digitalisation / orderbook transparency)

  Type F — Fringe carriers (small independents):
      The remaining ~35% of capacity, mostly mid-size operators and
      charter ship owners. They:
      - Are price-takers: take the market rate as given
      - Follow the naive Cobweb ordering rule (Luo et al. 2009) without
        market power correction
      - Have shorter investment horizons and smaller vessels
      - Historically amplify cycle peaks/troughs by behaving procyclically

The key result this module tests:
  Consolidation does NOT reduce cycle amplitude because the fringe
  carriers continue to behave naively even as strategic carriers
  exercise partial capacity discipline. The strategic carriers'
  discipline is insufficient to offset fringe amplification,
  particularly when transparency allows all fringe players to
  observe the same boom signal simultaneously (Greenwood-Hanson
  synchronisation paradox).

Key references
--------------
- Cariou & Guillotreau (2022): capacity discipline in liner alliances.
- UNCTAD RMT 2025: market share of top carriers, ch. II.
- Ghorbani et al. (2022): heterogeneous agent models in shipping.
- Greenwood & Hanson (2015): overextrapolation and synchronisation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from demand import DemandParameters, DemandProcess
from supply import SupplyParameters, SupplyProcess
from market import MarketParameters, FreightRateModel, WelfareCalculator, SocialPlanner


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass
class StrategicCarrierParams:
    """Parameters for the strategic (large alliance) carrier block."""

    n_strategic: int = 4
    """Number of strategic carriers.
    Calibrated on current alliance structure: THE Alliance, Ocean Alliance,
    Gemini (Maersk-Hapag) post-2025. Source: UNCTAD RMT 2025, ch. II."""

    market_share: float = 0.65
    """Combined capacity share of strategic carriers.
    Source: UNCTAD RMT 2025: top 5 carriers hold ~65% TEU capacity."""

    conjectural_variation: float = 0.45
    """Degree to which strategic carriers internalise their collective
    price impact. 0 = Nash conjecture (no awareness); 1 = full collusion.
    Intermediate value reflects tacit coordination within alliances
    subject to antitrust constraint (OECD 2018: explicit coordination
    prohibited). Calibrated so that strategic carriers behave between
    Cournot (1/n correction) and monopoly."""

    order_sensitivity_reduction: float = 0.30
    """Fraction by which strategic carriers reduce their Cobweb order
    sensitivity below the baseline. They are less reactive to spot
    rate signals because they can smooth investment across vessels.
    Represents the partial discipline that alliances do achieve
    (Cariou & Guillotreau 2022, p. 63)."""

    charter_market_insulation: float = 0.40
    """Fraction of strategic carrier revenues from long-term charters
    (insulated from spot volatility). Charter contracts prevent
    immediate scrapping and moderate ordering during booms."""

    forward_looking_weight: float = 0.25
    """Weight strategic carriers place on the orderbook signal
    (forward-looking) vs the current spot rate (backward-looking)
    when making ordering decisions. Represents use of market
    intelligence platforms. Set to 0 to replicate naive expectations."""


@dataclass
class FringeCarrierParams:
    """Parameters for the fringe (price-taking) carrier block."""

    market_share: float = 0.35
    """Combined capacity share of fringe carriers (= 1 - strategic share)."""

    overextrapolation_bias: float = 0.30
    """
    Greenwood-Hanson (2015) overextrapolation of recent freight rate trend.
    Fringe carriers extrapolate recent rate movements when forming
    expectations, amplifying both booms and busts.
    Source: Greenwood & Hanson (2015), pp. 57-58.
    """

    synchronisation_factor: float = 0.70
    """
    Degree to which fringe carriers order simultaneously in response
    to the same public rate signal. 1.0 = perfectly synchronised
    (all fringe carriers respond identically to the same signal);
    0.0 = fully idiosyncratic (random timing). Transparency increases
    this — the Greenwood-Hanson paradox in action.
    """


@dataclass
class HeterogeneousMarketParams:
    """Top-level parameters for the two-type carrier market."""

    strategic: StrategicCarrierParams = field(
        default_factory=StrategicCarrierParams
    )
    fringe: FringeCarrierParams = field(
        default_factory=FringeCarrierParams
    )
    n_periods: int = 50
    n_monte_carlo: int = 200
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Heterogeneous carrier market simulation
# ---------------------------------------------------------------------------

class HeterogeneousMarket:
    """
    Container shipping market with strategic and fringe carrier types.

    Each period:
      1. Demand is realised (from DemandProcess).
      2. Strategic carriers choose orders via modified Cobweb rule
         (partial market power internalisation + forward-looking signal).
      3. Fringe carriers choose orders via naive Cobweb with
         overextrapolation and synchronisation amplification.
      4. Market clears: freight rate adjusts to clear supply and demand.
      5. Welfare is computed relative to the social planner benchmark.

    The key comparison is to the homogeneous baseline (all carriers
    are symmetric fringe-type), to isolate the pure effect of having
    strategic players in the market.
    """

    def __init__(
        self,
        demand_params: DemandParameters,
        supply_params: SupplyParameters,
        market_params: MarketParameters,
        het_params: HeterogeneousMarketParams,
    ):
        self.dp  = demand_params
        self.sp  = supply_params
        self.mp  = market_params
        self.hp  = het_params
        self.rate_model = FreightRateModel(market_params)
        self.welfare    = WelfareCalculator(market_params)

    def run(self, seed: Optional[int] = None) -> dict:
        """
        Single simulation run with heterogeneous carrier types.

        Returns the same output structure as ShippingMarket.run(),
        plus:
          'strategic_orders'  : orders from strategic carriers
          'fringe_orders'     : orders from fringe carriers
          'utilisation_rate'  : fleet utilisation per period
        """
        seed = seed if seed is not None else self.mp.seed

        # Demand
        dp_run   = DemandParameters(**{
            k: getattr(self.dp, k) for k in self.dp.__dataclass_fields__
        })
        dp_run.seed = seed
        demand_proc = DemandProcess(dp_run)
        dem_result  = demand_proc.simulate(self.mp.n_periods)
        demand      = dem_result['effective_demand']

        n   = self.mp.n_periods
        sp  = self.sp
        hp  = self.hp
        sc  = hp.strategic
        fr  = hp.fringe

        # Fleet initialisation
        strategic_fleet = np.zeros(n)
        fringe_fleet    = np.zeros(n)

        s_init = sp.initial_fleet * sc.market_share
        f_init = sp.initial_fleet * fr.market_share

        strategic_fleet[0] = s_init
        fringe_fleet[0]    = f_init

        # Pipelines (distributed lag like supply.py)
        from scipy.stats import beta as beta_dist
        lag_window = sp.lag_max - sp.lag_min + 1
        edges      = np.linspace(0.0, 1.0, lag_window + 1)
        lag_probs  = np.array([
            beta_dist.cdf(edges[k+1], sp.lag_alpha, sp.lag_beta) -
            beta_dist.cdf(edges[k],   sp.lag_alpha, sp.lag_beta)
            for k in range(lag_window)
        ])

        s_pipeline = np.full(lag_window, sp.initial_orderbook * sc.market_share / lag_window)
        f_pipeline = np.full(lag_window, sp.initial_orderbook * fr.market_share / lag_window)

        rates            = np.zeros(n)
        s_orders_arr     = np.zeros(n)
        f_orders_arr     = np.zeros(n)
        deliveries_arr   = np.zeros(n)
        scrapping_arr    = np.zeros(n)
        orderbook_arr    = np.zeros(n)
        welfare_d        = []

        prev_rate        = 1.0
        earnings_ewma    = 1.0
        prev_s_order     = sp.initial_orderbook * sc.market_share / sp.lag_max
        prev_f_order     = sp.initial_orderbook * fr.market_share / sp.lag_max
        rate_history     = []

        rng = np.random.default_rng(seed)

        for t in range(n):
            # Deliveries
            s_delivery = s_pipeline[0]
            f_delivery = f_pipeline[0]
            s_pipeline = np.roll(s_pipeline, -1); s_pipeline[-1] = 0.0
            f_pipeline = np.roll(f_pipeline, -1); f_pipeline[-1] = 0.0

            total_delivery = s_delivery + f_delivery
            deliveries_arr[t] = total_delivery

            # Fleet update
            if t > 0:
                # Earnings EWMA for scrapping
                alpha_ew      = sp.scrapping_earnings_memory
                earnings_ewma = (1 - alpha_ew) * earnings_ewma + alpha_ew * prev_rate

                # Scrapping (applied to total fleet proportionally)
                total_prev = strategic_fleet[t-1] + fringe_fleet[t-1]
                w_scrap    = sp.scrapping_earnings_weight
                signal     = (1-w_scrap)*prev_rate + w_scrap*earnings_ewma

                base_scrap = total_prev * sp.scrapping_base_rate
                if signal < sp.layup_threshold:
                    extra = total_prev * sp.scrapping_sensitivity * (sp.layup_threshold - signal)
                else:
                    extra = 0.0
                total_scrap = min(base_scrap + extra, total_prev * 0.15)
                scrapping_arr[t] = total_scrap

                strategic_fleet[t] = (
                    strategic_fleet[t-1] + s_delivery - total_scrap * sc.market_share
                )
                fringe_fleet[t] = (
                    fringe_fleet[t-1] + f_delivery - total_scrap * fr.market_share
                )
            else:
                strategic_fleet[t] = s_init + s_delivery
                fringe_fleet[t]    = f_init + f_delivery

            strategic_fleet[t] = max(strategic_fleet[t], 0.0)
            fringe_fleet[t]    = max(fringe_fleet[t], 0.0)

            # CII slow steaming
            slow_factor = 1.0
            if sp.slow_steam_active and t >= sp.slow_steam_start_period:
                slow_factor = 1.0 - sp.slow_steam_reduction

            eff_fleet = (strategic_fleet[t] + fringe_fleet[t]) * slow_factor

            # Market clearing
            rate = self.rate_model.clearing_rate(demand[t], eff_fleet, prev_rate)
            rates[t] = rate
            rate_history.append(rate)
            long_run = float(np.mean(rate_history))
            rate_dev = rate / long_run if long_run > 0 else 1.0

            # Strategic carrier ordering (modified Cobweb)
            # Forward-looking component: use orderbook signal to temper orders
            orderbook_signal = (s_pipeline.sum() + f_pipeline.sum()) / eff_fleet \
                               if eff_fleet > 0 else 0.1
            # Blend spot signal with orderbook awareness
            fw     = sc.forward_looking_weight
            s_signal_rate = (1 - fw) * rate_dev + fw * (1.0 / (1.0 + orderbook_signal))

            # Market power correction: reduce ordering sensitivity by CV
            s_sensitivity = sp.order_sensitivity * (1.0 - sc.order_sensitivity_reduction)
            s_deviation   = s_signal_rate - 1.0

            s_base  = strategic_fleet[t] * (sp.order_base_rate + s_sensitivity * s_deviation)
            s_order = (1 - sp.order_inertia) * s_base + sp.order_inertia * prev_s_order
            s_order = max(0.0, s_order + rng.normal(0, strategic_fleet[t] * 0.005))

            # Fringe carrier ordering (naive Cobweb + overextrapolation)
            # Overextrapolation: fringe carriers extrapolate recent trend
            if t >= 2:
                recent_trend = rates[t] - rates[t-2]
            elif t >= 1:
                recent_trend = rates[t] - rates[t-1]
            else:
                recent_trend = 0.0
            f_perceived_rate = rate_dev + fr.overextrapolation_bias * recent_trend
            f_deviation       = f_perceived_rate - 1.0

            # Synchronisation: fringe carriers order simultaneously
            # sync_factor amplifies the common shock component
            sync  = fr.synchronisation_factor
            f_base = fringe_fleet[t] * (sp.order_base_rate + sp.order_sensitivity * f_deviation)
            # Idiosyncratic noise is reduced by synchronisation (they all do the same thing)
            f_noise = rng.normal(0, fringe_fleet[t] * 0.005 * (1.0 - sync))
            f_order = max(0.0, (1 - sp.order_inertia) * f_base + sp.order_inertia * prev_f_order + f_noise)

            s_orders_arr[t] = s_order
            f_orders_arr[t] = f_order

            # Place orders into pipelines
            for k, prob in enumerate(lag_probs):
                if k < lag_window:
                    s_pipeline[k] += s_order * prob
                    f_pipeline[k] += f_order * prob

            orderbook_arr[t] = s_pipeline.sum() + f_pipeline.sum()

            # Welfare
            w = self.welfare.period_welfare(rate, demand[t], eff_fleet, t)
            welfare_d.append(w)

            prev_rate    = rate
            prev_s_order = s_order
            prev_f_order = f_order

        # Social planner (from market.py)
        planner       = SocialPlanner(self.mp, self.sp)
        plan_result   = planner.simulate(demand)

        wc = WelfareCalculator(self.mp)
        cumulative_d  = wc.cumulative_welfare(welfare_d)
        cumulative_p  = wc.cumulative_welfare(plan_result['welfare'])
        dwl_total     = max(0.0, cumulative_p['total'] - cumulative_d['total'])
        dwl_pct       = (dwl_total / cumulative_p['total'] * 100.0
                         if cumulative_p['total'] > 0 else 0.0)

        total_orders = s_orders_arr + f_orders_arr

        return {
            'demand'           : demand,
            'shocks'           : dem_result['shock_type'] != 0,
            'strategic_fleet'  : strategic_fleet,
            'fringe_fleet'     : fringe_fleet,
            'fleet'            : strategic_fleet + fringe_fleet,
            'rates'            : rates,
            'strategic_orders' : s_orders_arr,
            'fringe_orders'    : f_orders_arr,
            'total_orders'     : total_orders,
            'deliveries'       : deliveries_arr,
            'scrapping'        : scrapping_arr,
            'orderbook'        : orderbook_arr,
            'welfare_d'        : welfare_d,
            'planner_fleet'    : plan_result['fleet'],
            'planner_rates'    : plan_result['rates'],
            'planner_welfare'  : plan_result['welfare'],
            'cumulative_d'     : cumulative_d,
            'cumulative_p'     : cumulative_p,
            'dwl_total'        : dwl_total,
            'dwl_pct'          : dwl_pct,
        }

    def monte_carlo(self, n_runs: Optional[int] = None) -> dict:
        """Run Monte Carlo and return DWL distribution."""
        n_runs = n_runs or self.hp.n_monte_carlo
        rng    = np.random.default_rng(self.hp.seed)

        dwl_totals      = []
        dwl_pcts        = []
        rate_vols       = []
        fringe_order_share = []

        for _ in range(n_runs):
            seed   = int(rng.integers(0, 100_000))
            result = self.run(seed=seed)
            dwl_totals.append(result['dwl_total'])
            dwl_pcts.append(result['dwl_pct'])
            rate_vols.append(np.std(result['rates']))
            total_o = result['total_orders'].sum()
            fringe_o = result['fringe_orders'].sum()
            fringe_order_share.append(fringe_o / total_o if total_o > 0 else 0)

        dwl_totals = np.array(dwl_totals)
        return {
            'dwl_mean'              : dwl_totals.mean(),
            'dwl_std'               : dwl_totals.std(),
            'dwl_p5'                : np.percentile(dwl_totals, 5),
            'dwl_p95'               : np.percentile(dwl_totals, 95),
            'dwl_pct_mean'          : np.array(dwl_pcts).mean(),
            'rate_vol_mean'         : np.array(rate_vols).mean(),
            'fringe_order_share_mean': np.array(fringe_order_share).mean(),
            'dwl_series'            : dwl_totals,
            'dwl_pct_series'        : np.array(dwl_pcts),
        }


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("Running heterogeneous carrier market simulation...")

    het_params = HeterogeneousMarketParams(
        strategic=StrategicCarrierParams(n_strategic=4, market_share=0.65),
        fringe=FringeCarrierParams(market_share=0.35),
        n_periods=50,
        n_monte_carlo=100,
        seed=42,
    )

    market = HeterogeneousMarket(
        demand_params=DemandParameters(seed=42),
        supply_params=SupplyParameters(seed=42),
        market_params=MarketParameters(n_periods=50, seed=42),
        het_params=het_params,
    )

    result = market.run()
    years  = np.arange(2000, 2050)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Fleet decomposition
    ax1 = fig.add_subplot(gs[0, :])
    ax1.stackplot(years,
                  result['strategic_fleet'],
                  result['fringe_fleet'],
                  labels=['Strategic carriers (65%)', 'Fringe carriers (35%)'],
                  colors=['steelblue', 'lightskyblue'], alpha=0.8)
    ax1.plot(years, result['demand'], color='firebrick', lw=2,
             label='Effective demand')
    ax1.plot(years, result['planner_fleet'], color='seagreen', lw=2, ls='--',
             label='Social planner fleet')
    ax1.set_ylabel('Fleet / Demand index (base 100)')
    ax1.set_title('Heterogeneous Carrier Market: Strategic vs Fringe Fleet Dynamics')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Order decomposition
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(years, result['strategic_orders'], alpha=0.7, color='steelblue',
            label='Strategic orders')
    ax2.bar(years, result['fringe_orders'], alpha=0.7, color='lightskyblue',
            bottom=result['strategic_orders'], label='Fringe orders')
    ax2.set_ylabel('New orders (index units)')
    ax2.set_xlabel('Year')
    ax2.set_title('Order Decomposition by Carrier Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Freight rates
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(years, result['rates'], color='firebrick', lw=2, label='Decentralised rate')
    ax3.plot(years, result['planner_rates'], color='seagreen', lw=2, ls='--',
             label='Planner rate')
    ax3.axhline(1.0, color='black', lw=0.8, ls=':', alpha=0.5)
    ax3.set_ylabel('Rate index (LR = 1.0)')
    ax3.set_xlabel('Year')
    ax3.set_title('Freight Rates: Market vs Planner')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Heterogeneous Carrier Model — Strategic vs Fringe Dynamics',
                 fontsize=12, fontweight='bold')
    import os
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/heterogeneous_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n=== Single Run Results ===")
    print(f"DWL (absolute):       {result['dwl_total']:.2f}")
    print(f"DWL (% of planner):   {result['dwl_pct']:.2f}%")
    fshare = result['fringe_orders'].sum() / result['total_orders'].sum() * 100
    print(f"Fringe order share:   {fshare:.1f}%")

    print("\nRunning Monte Carlo (100 runs)...")
    mc = market.monte_carlo(n_runs=100)
    print(f"Mean DWL:             {mc['dwl_mean']:.2f}")
    print(f"Median DWL:           {np.median(mc['dwl_series']):.2f}")
    print(f"5th-95th pct:         [{mc['dwl_p5']:.2f}, {mc['dwl_p95']:.2f}]")
    print(f"Mean DWL (%):         {mc['dwl_pct_mean']:.2f}%")
    print(f"Fringe order share:   {mc['fringe_order_share_mean']*100:.1f}%")
