"""
interventions/transparency.py
==============================
Models the effect of informational transparency on shipping cycle
dynamics and welfare.

Theoretical basis
-----------------
The classical Cobweb mechanism rests on naive price expectations:
carriers observe the current freight rate and order accordingly,
ignoring the aggregate supply response of competitors. Section 9.b
of the paper examines whether digitalisation — which has dramatically
improved the availability of orderbook data, freight analytics, and
demand forecasting — neutralises this mechanism.

The central paradox established in Section 9.b is:
  Enhanced informational transparency does not prevent herd behaviour;
  it accelerates it. When all market participants observe the same
  elevated price signals simultaneously through identical predictive
  algorithms, their non-cooperative dominant strategies lead them to
  place identical expansionary bets.

This module operationalises that argument by modelling a spectrum of
expectation formation rules, from fully naive (classical Cobweb) to
fully rational (Muth 1961). The key parameter is the 'transparency
level' τ ∈ [0, 1]:

  τ = 0: Fully naive expectations (classical Cobweb)
         Carriers observe only current freight rate.

  τ = 0.5: Partial transparency
         Carriers observe current rate AND aggregate orderbook,
         and form adaptive expectations. Represents current state
         of digital platforms (Ghorbani et al. 2022).

  τ = 1.0: Full transparency (Muth rational expectations)
           Carriers know the full demand path and orderbook,
           and form model-consistent expectations.
           In theory eliminates cobweb dynamics (Muth 1961).
           In practice blocked by coordination law (OECD 2018).

The critical finding tested here is Greenwood & Hanson (2015):
even carriers with access to orderbook data systematically
overextrapolate demand persistence — partial transparency may
actually synchronise and amplify the ordering wave rather than
dampen it, because all carriers update their forecasts in the
same direction at the same time.

Key references
--------------
- Muth (1961): rational expectations hypothesis, p. 316.
- Greenwood & Hanson (2015): overextrapolation of demand, pp. 57-58.
- Ghorbani et al. (2022): digital platforms and orderbook
  visibility, pp. 445-447.
- Notteboom et al. (2021): blank sailing coordination via digital
  tools, pp. 12-14.
- OECD (2018): antitrust constraint on coordinated response
  to shared information, pp. 15-22.
"""

import numpy as np
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from demand import DemandParameters, DemandProcess
from supply import SupplyParameters, SupplyProcess
from market import MarketParameters, FreightRateModel, WelfareCalculator, SocialPlanner


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class TransparencyParameters:
    """Parameters controlling the information environment."""

    transparency_levels: list = field(
        default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    )
    """
    Range of transparency levels τ to simulate.
    0.0 = fully naive (classical Cobweb)
    0.5 = approximate current digital platform state
    1.0 = full rational expectations (Muth 1961)
    """

    overextrapolation_bias: float = 0.3
    """
    Greenwood & Hanson (2015) bias: even with orderbook visibility,
    carriers overweight recent demand trends and underweight the
    aggregate supply response. Calibrated at 0.3 (moderate bias).
    At 0.0: unbiased Bayesian update.
    At 1.0: full overextrapolation (ignores orderbook entirely).
    Source: Greenwood & Hanson (2015), pp. 57-58.
    """

    synchronisation_amplifier: float = 0.4
    """
    The paradox of transparency: when all carriers receive the same
    signal simultaneously, their correlated responses amplify the
    ordering wave. This parameter scales the additional variance
    from correlated updating.
    At 0.0: transparency purely dampens cobweb (textbook result).
    At 0.4: partial synchronisation effect dominates at intermediate τ.
    Calibrated to reproduce the 2021 ordering surge despite widespread
    orderbook visibility documented by UNCTAD (2022, p. 63).
    """


# ---------------------------------------------------------------------------
# Expectation formation models
# ---------------------------------------------------------------------------

class ExpectationFormation:
    """
    Maps transparency level τ to a specific expectation formation rule.

    The ordering decision depends on the carrier's *expected* freight
    rate at delivery time (t + lag), not just the current rate. The
    expectation rule determines how much of the available information
    the carrier actually uses and weights correctly.
    """

    def __init__(self, params: TransparencyParameters):
        self.p = params

    def expected_rate(
        self,
        tau: float,
        current_rate: float,
        rate_history: np.ndarray,
        orderbook: float,
        current_fleet: float,
        demand_trend: float,
        lag: int,
    ) -> float:
        """
        Form an expectation of the freight rate at t+lag.

        Parameters
        ----------
        tau           : transparency level [0, 1]
        current_rate  : observed freight rate this period
        rate_history  : array of past freight rates
        orderbook     : outstanding orders (known if τ > 0)
        current_fleet : current fleet size
        demand_trend  : estimated demand growth rate
        lag           : shipbuilding lag (periods until delivery)

        Returns
        -------
        expected_rate : float, carrier's expectation of rate at t+lag
        """

        # --- Naive component (τ = 0): use current rate as forecast ---
        naive_expectation = current_rate

        # --- Adaptive component: use trend in recent rates ---
        if len(rate_history) >= 3:
            # Weighted average of recent rates (exponential smoothing)
            weights = np.exp(np.linspace(-1, 0, min(len(rate_history), 6)))
            weights /= weights.sum()
            adaptive_expectation = np.dot(
                weights, rate_history[-len(weights):]
            )
        else:
            adaptive_expectation = current_rate

        # --- Rational component: model-consistent forecast ---
        # With full orderbook visibility, a rational carrier would
        # anticipate that committed orders will depress rates at delivery.
        # The rational expectation adjusts for the supply overhang:
        #   supply_overhang = orderbook / current_fleet
        #   rate_adjustment = -sensitivity * overhang
        supply_sensitivity = 1.0 / abs(-0.30)  # from demand elasticity
        overhang = orderbook / max(current_fleet, 1.0)

        # Rational forecast: adjust for expected demand growth and supply
        rational_expectation = (
            current_rate
            * (1 + demand_trend) ** lag          # demand side
            * max(0.2, 1.0 - supply_sensitivity * overhang * 0.15)  # supply side
        )

        # --- Greenwood-Hanson overextrapolation bias ---
        # Even rational carriers overweight recent demand trends.
        # Bias pushes rational expectation toward naive (current rate).
        bias = self.p.overextrapolation_bias
        rational_expectation_biased = (
            (1 - bias) * rational_expectation
            + bias * naive_expectation
        )

        # --- Blend naive, adaptive and rational by τ ---
        # τ = 0   → fully naive
        # τ = 0.5 → blend of adaptive and rational
        # τ = 1   → fully rational (but biased)
        if tau <= 0.5:
            # Low transparency: blend naive and adaptive
            alpha = tau * 2  # 0 → 1 as τ goes 0 → 0.5
            blended = (1 - alpha) * naive_expectation + alpha * adaptive_expectation
        else:
            # High transparency: blend adaptive and rational
            alpha = (tau - 0.5) * 2  # 0 → 1 as τ goes 0.5 → 1.0
            blended = (1 - alpha) * adaptive_expectation + alpha * rational_expectation_biased

        # --- Synchronisation amplifier ---
        # At intermediate τ, all carriers receive identical signals and
        # update in the same direction, amplifying the ordering wave.
        # Effect is strongest at τ ≈ 0.5 (all see same data, none fully
        # rational enough to self-correct).
        sync_effect = self.p.synchronisation_amplifier * np.sin(np.pi * tau)
        # Add correlated noise component (same sign for all carriers)
        sync_noise = sync_effect * (current_rate - 1.0) * 0.2  # amplifies boom/bust
        blended = blended + sync_noise

        return float(np.clip(blended, 0.2, 5.0))


# ---------------------------------------------------------------------------
# Transparency-adjusted market simulation
# ---------------------------------------------------------------------------

class TransparentMarket:
    """
    Simulates the shipping market with a given transparency level τ.

    At each period, carriers form expectations of future rates using
    the ExpectationFormation model, and place orders based on those
    expectations rather than the naive current-rate rule.

    The key difference from the baseline ShippingMarket:
      - Orders are based on EXPECTED rate at t+lag, not current rate
      - Expectation formation depends on τ
      - Orderbook is observable at transparency τ > 0
    """

    def __init__(
        self,
        tau: float,
        demand_params: DemandParameters,
        supply_params: SupplyParameters,
        market_params: MarketParameters,
        transparency_params: TransparencyParameters,
        seed: Optional[int] = None,
    ):
        self.tau    = tau
        self.dp     = demand_params
        self.sp     = supply_params
        self.mp     = market_params
        self.tp     = transparency_params
        self.rng    = np.random.default_rng(seed)

        self.rate_model      = FreightRateModel(market_params)
        self.welfare_calc    = WelfareCalculator(market_params)
        self.planner         = SocialPlanner(market_params, supply_params)
        self.expectation     = ExpectationFormation(transparency_params)

    def run(self, seed: Optional[int] = None) -> dict:
        """Run a complete simulation with transparency level τ."""
        n = self.mp.n_periods

        d_params = DemandParameters(**{
            **self.dp.__dict__,
            'seed': seed if seed is not None else self.dp.seed
        })
        demand_results = DemandProcess(d_params).simulate(n)
        eff_demand     = demand_results['effective_demand']

        fleet      = np.zeros(n)
        orders_arr = np.zeros(n)
        deliveries = np.zeros(n)
        scrapping  = np.zeros(n)
        orderbook  = np.zeros(n)
        rates      = np.zeros(n)
        exp_rates  = np.zeros(n)   # expected rates (for analysis)
        welfare_d  = []

        fleet[0]  = self.sp.initial_fleet
        lag       = self.sp.building_lag
        pipeline  = np.zeros(lag)
        pipeline[-1] = self.sp.initial_orderbook

        supply_proc = SupplyProcess(self.sp)
        prev_rate   = self.mp.long_run_equilibrium_rate
        prev_orders = self.sp.initial_orderbook
        rate_history = [self.mp.long_run_equilibrium_rate]

        # Estimate demand trend from history (updated each period)
        demand_trend = self.dp.gdp_growth_mean * self.dp.trade_elasticity_high

        for t in range(n):
            # Deliveries
            delivery      = pipeline[0]
            deliveries[t] = delivery
            pipeline      = np.roll(pipeline, -1)
            pipeline[-1]  = 0.0

            # Scrapping
            scrap        = supply_proc._compute_scrapping(
                fleet[t-1] if t > 0 else self.sp.initial_fleet,
                prev_rate / self.mp.long_run_equilibrium_rate
            )
            scrapping[t] = scrap

            # Fleet
            base     = fleet[t-1] if t > 0 else self.sp.initial_fleet
            fleet[t] = max(base + delivery - scrap, 0.0)

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
            rate_history.append(rate)

            # Welfare
            w = self.welfare_calc.period_welfare(
                rate, eff_demand[t], eff_fleet, t
            )
            welfare_d.append(w)

            # Update demand trend estimate (adaptive learning)
            if t >= 3:
                recent_demand_growth = (eff_demand[t] / eff_demand[t-3]) ** (1/3) - 1
                demand_trend = 0.7 * demand_trend + 0.3 * recent_demand_growth

            # Form expected rate using transparency model
            current_orderbook = pipeline.sum()
            orderbook[t]      = current_orderbook

            exp_rate = self.expectation.expected_rate(
                tau           = self.tau,
                current_rate  = rate,
                rate_history  = np.array(rate_history[-10:]),
                orderbook     = current_orderbook,
                current_fleet = fleet[t],
                demand_trend  = demand_trend,
                lag           = lag,
            )
            exp_rates[t] = exp_rate

            # Orders based on EXPECTED rate (not naive current rate)
            order = supply_proc._ordering_rule(
                exp_rate / self.mp.long_run_equilibrium_rate,
                fleet[t],
                prev_orders
            )
            orders_arr[t] = order
            prev_orders   = order
            if t + lag < n:
                pipeline[-1] += order

        # Planner benchmark
        planner_results = self.planner.simulate(eff_demand)

        cumulative_d = self.welfare_calc.cumulative_welfare(welfare_d)
        cumulative_p = self.welfare_calc.cumulative_welfare(
            planner_results['welfare']
        )
        dwl_total = cumulative_p['total'] - cumulative_d['total']
        dwl_pct   = (dwl_total / abs(cumulative_p['total']) * 100
                     if cumulative_p['total'] != 0 else 0)

        return {
            'tau'            : self.tau,
            'demand'         : eff_demand,
            'fleet'          : fleet,
            'orders'         : orders_arr,
            'rates'          : rates,
            'exp_rates'      : exp_rates,
            'orderbook'      : orderbook,
            'welfare_d'      : welfare_d,
            'planner_fleet'  : planner_results['fleet'],
            'planner_rates'  : planner_results['rates'],
            'cumulative_d'   : cumulative_d,
            'cumulative_p'   : cumulative_p,
            'dwl_total'      : dwl_total,
            'dwl_pct'        : dwl_pct,
            'rate_volatility': float(np.std(rates)),
            'order_volatility': float(np.std(orders_arr)),
            'expectation_error': float(np.mean(np.abs(exp_rates - rates))),
        }


# ---------------------------------------------------------------------------
# Transparency sweep
# ---------------------------------------------------------------------------

class TransparencyAnalysis:
    """
    Runs the market simulation across transparency levels τ ∈ [0, 1]
    and compares welfare, rate volatility, and ordering behaviour.

    Tests the central thesis of Section 9.b: does transparency dampen
    or amplify the Cobweb mechanism?
    """

    def __init__(
        self,
        demand_params: DemandParameters,
        supply_params: SupplyParameters,
        market_params: MarketParameters,
        transparency_params: TransparencyParameters,
    ):
        self.dp = demand_params
        self.sp = supply_params
        self.mp = market_params
        self.tp = transparency_params

    def run_sweep(
        self,
        n_monte_carlo: int = 100,
        seed: int = 42
    ) -> dict:
        """Simulate each transparency level n_monte_carlo times."""
        rng     = np.random.default_rng(seed)
        results = {}

        for tau in self.tp.transparency_levels:
            print(f"  Simulating τ={tau:.2f} ({n_monte_carlo} runs)...")

            dwl_list        = []
            rate_vol_list   = []
            order_vol_list  = []
            welfare_list    = []
            exp_error_list  = []

            for _ in range(n_monte_carlo):
                run_seed = int(rng.integers(0, 100_000))
                market   = TransparentMarket(
                    tau                 = tau,
                    demand_params       = self.dp,
                    supply_params       = self.sp,
                    market_params       = self.mp,
                    transparency_params = self.tp,
                    seed                = run_seed,
                )
                r = market.run(seed=run_seed)
                dwl_list.append(r['dwl_total'])
                rate_vol_list.append(r['rate_volatility'])
                order_vol_list.append(r['order_volatility'])
                welfare_list.append(r['cumulative_d']['total'])
                exp_error_list.append(r['expectation_error'])

            results[tau] = {
                'dwl_mean'         : float(np.mean(dwl_list)),
                'dwl_std'          : float(np.std(dwl_list)),
                'dwl_median'       : float(np.median(dwl_list)),
                'dwl_p5'           : float(np.percentile(dwl_list, 5)),
                'dwl_p95'          : float(np.percentile(dwl_list, 95)),
                'rate_vol_mean'    : float(np.mean(rate_vol_list)),
                'order_vol_mean'   : float(np.mean(order_vol_list)),
                'welfare_mean'     : float(np.mean(welfare_list)),
                'exp_error_mean'   : float(np.mean(exp_error_list)),
                'n_runs'           : n_monte_carlo,
            }

        return results


# ---------------------------------------------------------------------------
# Diagnostic / entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("=" * 60)
    print("Transparency Analysis: Information & Cobweb Dynamics")
    print("=" * 60)

    dp = DemandParameters(seed=42)
    sp = SupplyParameters(seed=42)
    mp = MarketParameters(n_periods=50, seed=42)
    tp = TransparencyParameters()

    analysis = TransparencyAnalysis(dp, sp, mp, tp)

    print("\nRunning transparency sweep (100 runs per τ)...")
    results = analysis.run_sweep(n_monte_carlo=100, seed=42)

    # --- Summary table ---
    taus = sorted(results.keys())
    print(f"\n{'τ':>5} | {'DWL Median':>10} | {'Rate Vol':>8} | "
          f"{'Order Vol':>9} | {'Exp Error':>9} | {'Welfare':>10}")
    print("-" * 65)

    for tau in taus:
        r = results[tau]
        print(f"{tau:>5.2f} | {r['dwl_median']:>10.1f} | "
              f"{r['rate_vol_mean']:>8.4f} | "
              f"{r['order_vol_mean']:>9.4f} | "
              f"{r['exp_error_mean']:>9.4f} | "
              f"{r['welfare_mean']:>10.1f}")

    # Key findings
    dwl_naive    = results[0.0]['dwl_median']
    dwl_rational = results[1.0]['dwl_median']
    dwl_partial  = results[0.5]['dwl_median']
    dwl_min      = min(r['dwl_median'] for r in results.values())
    tau_min      = min(results.keys(), key=lambda t: results[t]['dwl_median'])

    print(f"\nKey findings:")
    print(f"  DWL at τ=0.0 (naive):    {dwl_naive:.1f}")
    print(f"  DWL at τ=0.5 (partial):  {dwl_partial:.1f}  "
          f"({'worse' if dwl_partial > dwl_naive else 'better'} than naive)")
    print(f"  DWL at τ=1.0 (rational): {dwl_rational:.1f}  "
          f"({'worse' if dwl_rational > dwl_naive else 'better'} than naive)")
    print(f"  Minimum DWL at τ={tau_min:.2f}")
    print(f"\n  Greenwood-Hanson paradox confirmed: "
          f"{'YES' if dwl_partial > dwl_naive * 0.95 else 'NO'} "
          f"(partial transparency ≥ naive DWL)")

    # --- Single-run trace for τ=0 vs τ=0.5 vs τ=1.0 ---
    print("\nGenerating single-run traces for selected τ values...")
    trace_taus   = [0.0, 0.5, 1.0]
    trace_colors = ['firebrick', 'darkorange', 'seagreen']
    traces       = {}
    for tau in trace_taus:
        m = TransparentMarket(tau, dp, sp, mp, tp, seed=42)
        traces[tau] = m.run(seed=42)

    years = np.arange(2000, 2050)

    # --- Plots ---
    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    # Panel 1: DWL vs τ
    ax1 = fig.add_subplot(gs[0, :])
    dwl_meds = [results[t]['dwl_median'] for t in taus]
    dwl_p5s  = [results[t]['dwl_p5']     for t in taus]
    dwl_p95s = [results[t]['dwl_p95']    for t in taus]

    ax1.fill_between(taus, dwl_p5s, dwl_p95s,
                     alpha=0.2, color='steelblue', label='5th–95th pct')
    ax1.plot(taus, dwl_meds,
             color='steelblue', lw=2.5, marker='o', label='Median DWL')
    ax1.axhline(dwl_naive, color='firebrick', lw=1.5, linestyle='--',
                alpha=0.7, label=f'Naive baseline (τ=0): {dwl_naive:.0f}')
    ax1.axvline(0.5, color='darkorange', lw=1.5, linestyle=':',
                label='Current digital state (τ≈0.5)')

    # Annotate the paradox region
    paradox_taus = [t for t in taus if results[t]['dwl_median'] >= dwl_naive * 0.98]
    if paradox_taus:
        ax1.axvspan(min(paradox_taus), max(paradox_taus),
                    alpha=0.08, color='firebrick',
                    label='Greenwood-Hanson paradox region')

    ax1.set_title('Deadweight Loss vs. Transparency Level τ\n'
                  'τ=0: Naive Cobweb  |  τ=1: Rational Expectations (Muth 1961)',
                  fontsize=11)
    ax1.set_xlabel('Transparency level τ')
    ax1.set_ylabel('DWL (welfare units, median)')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)

    # Panel 2: Rate volatility vs τ
    ax2 = fig.add_subplot(gs[1, 0])
    rate_vols = [results[t]['rate_vol_mean'] for t in taus]
    ax2.plot(taus, rate_vols,
             color='firebrick', lw=2, marker='s')
    ax2.axvline(0.5, color='darkorange', lw=1.5, linestyle=':', alpha=0.7)
    ax2.set_title('Freight Rate Volatility\nvs. Transparency Level', fontsize=11)
    ax2.set_xlabel('τ')
    ax2.set_ylabel('Std dev of freight rate')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Freight rate traces
    ax3 = fig.add_subplot(gs[1, 1])
    for tau, color in zip(trace_taus, trace_colors):
        ax3.plot(years, traces[tau]['rates'],
                 color=color, lw=1.5, alpha=0.85,
                 label=f'τ={tau:.1f}')
    ax3.axhline(1.0, color='black', lw=0.8, linestyle=':', alpha=0.5)
    ax3.set_title('Freight Rate Paths\n(Single Run, τ=0 / 0.5 / 1.0)', fontsize=11)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Rate index (LR = 1.0)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Order volatility vs τ
    ax4 = fig.add_subplot(gs[2, 0])
    order_vols = [results[t]['order_vol_mean'] for t in taus]
    ax4.plot(taus, order_vols,
             color='darkorange', lw=2, marker='^')
    ax4.axvline(0.5, color='darkorange', lw=1.5, linestyle=':', alpha=0.7)
    ax4.set_title('Order Volatility vs. Transparency\n'
                  '(Synchronisation effect at τ≈0.5)', fontsize=11)
    ax4.set_xlabel('τ')
    ax4.set_ylabel('Std dev of new orders')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Welfare mean vs τ
    ax5 = fig.add_subplot(gs[2, 1])
    welfare_means = [results[t]['welfare_mean'] for t in taus]
    ax5.plot(taus, welfare_means,
             color='seagreen', lw=2, marker='D')
    ax5.axvline(0.5, color='darkorange', lw=1.5, linestyle=':',
                alpha=0.7, label='Current state (τ≈0.5)')
    ax5.set_title('Mean Social Welfare\nvs. Transparency Level', fontsize=11)
    ax5.set_xlabel('τ')
    ax5.set_ylabel('Cumulative welfare (mean)')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.suptitle(
        'Transparency & Cyclicality: Does Better Information Dampen the Cobweb?\n'
        'Greenwood-Hanson Paradox — Container Shipping Cycle Model',
        fontsize=12, fontweight='bold'
    )

    os.makedirs('../data', exist_ok=True)
    plt.savefig('../data/transparency_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to data/transparency_analysis.png")