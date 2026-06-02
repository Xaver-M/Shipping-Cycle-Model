"""
interventions/consolidation.py
================================
Models the effect of market consolidation on shipping cycle dynamics
and welfare.

Theoretical basis
-----------------
Section 9.a of the paper establishes the central paradox of consolidation:
while a reduced number of actors should theoretically enable capacity
discipline, antitrust law prohibits the coordination of newbuilding
decisions. The strategic interaction therefore remains a Prisoner's
Dilemma regardless of concentration.

This module operationalises that argument by modelling the n-carrier
oligopoly explicitly. Each carrier:
  1. Observes the common freight rate signal
  2. Places orders to maximise its own profit, taking competitor
     orders as given (Nash conjecture)
  3. Is subject to the same building lag as in the baseline

The key result from Cariou & Guillotreau (2022, pp. 61, 63):
moving from 5 to 4 alliances produces no measurable improvement
in capacity discipline. This module allows us to test that finding
across a continuous range of n.

The welfare comparison across consolidation levels answers:
  Does consolidation move the market closer to the social optimum?
  At what n does the Prisoner's Dilemma break down?

Key references
--------------
- Cariou & Guillotreau (2022): experimental evidence on oligopolistic
  capacity discipline, pp. 61, 63.
- Stopford (2009): market structure and cyclicality, pp. 139-142.
- OECD (2018): antitrust constraints on alliance coordination, pp. 15-22.
"""

import numpy as np
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

# Allow running from interventions/ or project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from demand import DemandParameters, DemandProcess
from supply import SupplyParameters, SupplyProcess
from market import MarketParameters, FreightRateModel, WelfareCalculator, SocialPlanner


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ConsolidationParameters:
    """Parameters controlling the oligopoly structure."""

    n_carriers_range: list = field(
        default_factory=lambda: [20, 10, 7, 5, 4, 3, 2]
    )
    """
    Range of carrier counts to simulate.
    20 = atomistic baseline (current paper assumption)
    4  = approximate current alliance structure
    2  = duopoly (theoretical extreme)
    """

    coordination_leakage: float = 0.0
    """
    Fraction of socially optimal order reduction that carriers achieve
    through tacit coordination (0 = pure Nash, 1 = full coordination).
    Antitrust law forces this toward 0; kept as sensitivity parameter.
    OECD (2018): explicit coordination is prohibited, pp. 15-22.
    """

    market_share_symmetry: bool = True
    """
    If True, all carriers have equal market share (1/n).
    If False, size distribution follows empirical Pareto (top carrier ~20%).
    """


# ---------------------------------------------------------------------------
# N-carrier Nash ordering model
# ---------------------------------------------------------------------------

class OligopolyMarket:
    """
    Simulates the shipping market with n symmetric Nash-competing carriers.

    Each carrier's ordering rule is derived from its individual profit
    maximisation problem, taking the aggregate orders of competitors
    as given. In the symmetric Nash equilibrium, each carrier orders:

        q_i*(t) = Q*(t) / n

    where Q*(t) is the aggregate Nash equilibrium order volume.

    The aggregate Nash order exceeds the social planner's order because
    each carrier ignores the negative price externality its capacity
    imposes on all others — the classic Cournot overproduction result
    applied to investment timing.

    As n → ∞ (atomistic), each carrier becomes a price-taker and the
    Cobweb naive-expectations result is recovered exactly.
    As n → 1 (monopoly), the carrier partially internalises the
    externality — but full internalisation requires n = 1 AND
    the monopolist acting as a welfare maximiser (which it does not).
    """

    def __init__(
        self,
        n_carriers: int,
        demand_params: DemandParameters,
        supply_params: SupplyParameters,
        market_params: MarketParameters,
        coordination_leakage: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.n          = n_carriers
        self.dp         = demand_params
        self.sp         = supply_params
        self.mp         = market_params
        self.leakage    = coordination_leakage
        self.rng        = np.random.default_rng(seed)

        self.rate_model   = FreightRateModel(market_params)
        self.welfare_calc = WelfareCalculator(market_params)
        self.planner      = SocialPlanner(market_params, supply_params)

    def _nash_order_scale(self, rate_deviation: float) -> float:
        """
        Compute the Nash equilibrium scaling factor relative to the
        atomistic (n→∞) ordering rule.

        In a symmetric Cournot model with n firms, each firm's quantity
        is 1/n of the competitive quantity at the Nash equilibrium.
        However, the *aggregate* Nash quantity exceeds the social optimum
        because each firm ignores the price impact of its own investment.

        The aggregate Nash order relative to the atomistic benchmark:
          Q_Nash / Q_atomistic = n / (n + 1)   [standard Cournot result]

        As n → ∞: ratio → 1   (competitive = atomistic)
        As n = 1:  ratio = 0.5 (monopolist produces half of competitive)

        With coordination leakage, some of the social optimum is recovered:
          Q_effective = Q_Nash * (1 - leakage) + Q_planner * leakage

        Note: this does NOT mean fewer orders necessarily — in the Cobweb
        context, the atomistic case already overshoots. The Nash correction
        reduces the *rate sensitivity* of orders, not the base rate.
        The key insight from Cariou & Guillotreau (2022): even at n=4,
        the Nash equilibrium is still far from the social optimum because
        the competitive incentive dominates at any n > 1.
        """
        # Cournot aggregate quantity ratio
        cournot_ratio = self.n / (self.n + 1)

        # Scale: at n=20 (baseline), ratio ≈ 0.952 ≈ 1.0
        # At n=4, ratio = 0.8 — 20% reduction in rate-sensitivity
        # The ORDER SENSITIVITY parameter is scaled, not the base replacement rate
        # (replacement ordering is independent of strategic interaction)

        # Apply coordination leakage toward social optimum
        # leakage=0: pure Nash; leakage=1: fully coordinated (prohibited)
        effective_ratio = cournot_ratio * (1 - self.leakage) + self.leakage * 0.5

        return effective_ratio

    def run(self, seed: Optional[int] = None) -> dict:
        """
        Simulate the market with n Nash-competing carriers.

        Returns the same structure as ShippingMarket.run() for
        direct comparison.
        """
        n_periods = self.mp.n_periods

        # Demand
        d_params = DemandParameters(**{
            **self.dp.__dict__,
            'seed': seed if seed is not None else self.dp.seed
        })
        demand_results = DemandProcess(d_params).simulate(n_periods)
        eff_demand     = demand_results['effective_demand']

        # Nash scaling — reduces order sensitivity relative to atomistic
        nash_scale = self._nash_order_scale(1.0)

        # Simulation loop (same structure as ShippingMarket.run)
        fleet      = np.zeros(n_periods)
        orders_arr = np.zeros(n_periods)
        deliveries = np.zeros(n_periods)
        scrapping  = np.zeros(n_periods)
        orderbook  = np.zeros(n_periods)
        rates      = np.zeros(n_periods)
        welfare_d  = []

        fleet[0]  = self.sp.initial_fleet
        lag       = self.sp.building_lag
        pipeline  = np.zeros(lag)
        pipeline[-1] = self.sp.initial_orderbook

        supply_proc = SupplyProcess(self.sp)
        prev_rate   = self.mp.long_run_equilibrium_rate
        prev_orders = self.sp.initial_orderbook

        # Scaled supply params: reduce order_sensitivity by Nash ratio
        sp_scaled = SupplyParameters(**{
            **self.sp.__dict__,
            'order_sensitivity': self.sp.order_sensitivity * nash_scale,
            'order_inertia'    : self.sp.order_inertia * nash_scale,
        })
        supply_proc_scaled = SupplyProcess(sp_scaled)

        for t in range(n_periods):
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

            # Effective fleet
            eff_fleet = fleet[t]
            if self.sp.slow_steam_active and t >= self.sp.slow_steam_start_period:
                eff_fleet *= (1 - self.sp.slow_steam_reduction)

            # Rate
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

            # Nash-adjusted orders
            order = supply_proc_scaled._ordering_rule(
                rate / self.mp.long_run_equilibrium_rate,
                fleet[t],
                prev_orders
            )
            orders_arr[t] = order
            prev_orders   = order
            if t + lag < n_periods:
                pipeline[-1] += order
            orderbook[t] = pipeline.sum()

        # Planner benchmark (same demand path)
        planner_results = self.planner.simulate(eff_demand)

        # Welfare comparison
        cumulative_d = self.welfare_calc.cumulative_welfare(welfare_d)
        cumulative_p = self.welfare_calc.cumulative_welfare(
            planner_results['welfare']
        )
        dwl_total = cumulative_p['total'] - cumulative_d['total']
        dwl_pct   = (dwl_total / abs(cumulative_p['total']) * 100
                     if cumulative_p['total'] != 0 else 0)

        return {
            'n_carriers'    : self.n,
            'nash_scale'    : nash_scale,
            'demand'        : eff_demand,
            'fleet'         : fleet,
            'orders'        : orders_arr,
            'rates'         : rates,
            'orderbook'     : orderbook,
            'welfare_d'     : welfare_d,
            'planner_fleet' : planner_results['fleet'],
            'planner_rates' : planner_results['rates'],
            'cumulative_d'  : cumulative_d,
            'cumulative_p'  : cumulative_p,
            'dwl_total'     : dwl_total,
            'dwl_pct'       : dwl_pct,
            'rate_volatility': float(np.std(rates)),
            'fleet_overshoot': float(np.mean(
                np.maximum(0, fleet - eff_demand) / np.maximum(eff_demand, 1)
            )),
        }


# ---------------------------------------------------------------------------
# Consolidation sweep
# ---------------------------------------------------------------------------

class ConsolidationAnalysis:
    """
    Runs the market simulation across a range of carrier counts and
    compares welfare outcomes, rate volatility, and fleet overshoot.

    This directly operationalises the research question from Section 9.a:
    does consolidation move the market toward the social optimum, and
    if so, by how much?
    """

    def __init__(
        self,
        demand_params: DemandParameters,
        supply_params: SupplyParameters,
        market_params: MarketParameters,
        consolidation_params: ConsolidationParameters,
    ):
        self.dp   = demand_params
        self.sp   = supply_params
        self.mp   = market_params
        self.cp   = consolidation_params

    def run_sweep(
        self,
        n_monte_carlo: int = 100,
        seed: int = 42
    ) -> dict:
        """
        Simulate each consolidation level n_monte_carlo times and
        return summary statistics.

        Returns
        -------
        dict keyed by n_carriers, each containing:
          'dwl_mean', 'dwl_std', 'dwl_median',
          'rate_vol_mean', 'fleet_overshoot_mean',
          'welfare_mean', 'raw_runs'
        """
        rng     = np.random.default_rng(seed)
        results = {}

        for n in self.cp.n_carriers_range:
            print(f"  Simulating n={n} carriers ({n_monte_carlo} runs)...")

            dwl_list       = []
            rate_vol_list  = []
            overshoot_list = []
            welfare_list   = []

            for _ in range(n_monte_carlo):
                run_seed = int(rng.integers(0, 100_000))
                market   = OligopolyMarket(
                    n_carriers          = n,
                    demand_params       = self.dp,
                    supply_params       = self.sp,
                    market_params       = self.mp,
                    coordination_leakage= self.cp.coordination_leakage,
                    seed                = run_seed,
                )
                r = market.run(seed=run_seed)
                dwl_list.append(r['dwl_total'])
                rate_vol_list.append(r['rate_volatility'])
                overshoot_list.append(r['fleet_overshoot'])
                welfare_list.append(r['cumulative_d']['total'])

            results[n] = {
                'dwl_mean'            : float(np.mean(dwl_list)),
                'dwl_std'             : float(np.std(dwl_list)),
                'dwl_median'          : float(np.median(dwl_list)),
                'dwl_p5'              : float(np.percentile(dwl_list, 5)),
                'dwl_p95'             : float(np.percentile(dwl_list, 95)),
                'rate_vol_mean'       : float(np.mean(rate_vol_list)),
                'rate_vol_std'        : float(np.std(rate_vol_list)),
                'fleet_overshoot_mean': float(np.mean(overshoot_list)),
                'welfare_mean'        : float(np.mean(welfare_list)),
                'n_runs'              : n_monte_carlo,
            }

        return results


# ---------------------------------------------------------------------------
# Diagnostic / entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("=" * 60)
    print("Consolidation Analysis: Oligopoly Structure & Welfare")
    print("=" * 60)

    dp = DemandParameters(seed=42)
    sp = SupplyParameters(seed=42)
    mp = MarketParameters(n_periods=50, seed=42)
    cp = ConsolidationParameters()

    analysis = ConsolidationAnalysis(dp, sp, mp, cp)

    print("\nRunning consolidation sweep (100 runs per n)...")
    results = analysis.run_sweep(n_monte_carlo=100, seed=42)

    # --- Summary table ---
    print(f"\n{'n':>4} | {'Nash Scale':>10} | {'DWL Median':>10} | "
          f"{'Rate Vol':>8} | {'Overshoot':>9} | {'Welfare':>10}")
    print("-" * 65)

    n_vals        = sorted(results.keys(), reverse=True)
    dwl_medians   = []
    rate_vols     = []
    overshoots    = []
    welfare_means = []
    nash_scales   = []

    for n in n_vals:
        r = results[n]
        # Compute Nash scale for display
        ns = n / (n + 1)
        nash_scales.append(ns)
        dwl_medians.append(r['dwl_median'])
        rate_vols.append(r['rate_vol_mean'])
        overshoots.append(r['fleet_overshoot_mean'])
        welfare_means.append(r['welfare_mean'])
        print(f"{n:>4} | {ns:>10.4f} | {r['dwl_median']:>10.1f} | "
              f"{r['rate_vol_mean']:>8.4f} | "
              f"{r['fleet_overshoot_mean']:>9.4f} | "
              f"{r['welfare_mean']:>10.1f}")

    # Key finding: DWL reduction from n=20 to n=4
    dwl_20 = results[20]['dwl_median'] if 20 in results else results[max(results.keys())]['dwl_median']
    dwl_4  = results[4]['dwl_median']  if 4  in results else results[min(results.keys())]['dwl_median']
    pct_reduction = (dwl_20 - dwl_4) / dwl_20 * 100 if dwl_20 > 0 else 0

    print(f"\nKey finding:")
    print(f"  DWL at n=20 (baseline): {dwl_20:.1f}")
    print(f"  DWL at n=4  (alliances): {dwl_4:.1f}")
    print(f"  Reduction:              {pct_reduction:.1f}%")
    print(f"  Consistent with Cariou & Guillotreau (2022): "
          f"{'YES — marginal improvement only' if pct_reduction < 20 else 'NO — larger than expected'}")

    # --- Plots ---
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    n_plot = list(reversed(n_vals))  # ascending order for plots

    # Panel 1: DWL vs n_carriers
    ax1 = fig.add_subplot(gs[0, :2])
    dwl_med_plot = [results[n]['dwl_median'] for n in n_plot]
    dwl_p5_plot  = [results[n]['dwl_p5']     for n in n_plot]
    dwl_p95_plot = [results[n]['dwl_p95']    for n in n_plot]

    ax1.fill_between(n_plot, dwl_p5_plot, dwl_p95_plot,
                     alpha=0.2, color='steelblue', label='5th–95th pct')
    ax1.plot(n_plot, dwl_med_plot,
             color='steelblue', lw=2.5, marker='o', label='Median DWL')
    ax1.axvline(4, color='firebrick', lw=1.5, linestyle='--',
                label='Current alliance structure (n≈4)')
    ax1.axvline(20, color='gray', lw=1.0, linestyle=':',
                label='Atomistic baseline (n=20)')
    ax1.set_title('Deadweight Loss vs. Number of Carriers\n'
                  'Cournot Nash Equilibrium', fontsize=11)
    ax1.set_xlabel('Number of independent carriers (n)')
    ax1.set_ylabel('DWL (welfare units, median)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Panel 2: Welfare gap closed (% of max possible improvement)
    ax2 = fig.add_subplot(gs[0, 2])
    dwl_baseline = results[max(n_plot)]['dwl_median']
    pct_closed   = [(dwl_baseline - results[n]['dwl_median']) / dwl_baseline * 100
                    for n in n_plot]
    ax2.barh([str(n) for n in n_plot], pct_closed,
             color='seagreen', alpha=0.7)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_title('% of DWL Closed\nvs. Atomistic Baseline', fontsize=11)
    ax2.set_xlabel('DWL reduction (%)')
    ax2.set_ylabel('n carriers')
    ax2.grid(True, alpha=0.3, axis='x')

    # Panel 3: Rate volatility vs n
    ax3 = fig.add_subplot(gs[1, 0])
    rate_vol_plot = [results[n]['rate_vol_mean'] for n in n_plot]
    ax3.plot(n_plot, rate_vol_plot,
             color='firebrick', lw=2, marker='s')
    ax3.axvline(4, color='firebrick', lw=1.5, linestyle='--', alpha=0.5)
    ax3.set_title('Freight Rate Volatility\nvs. Carrier Count', fontsize=11)
    ax3.set_xlabel('n carriers')
    ax3.set_ylabel('Std dev of freight rate')
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # Panel 4: Fleet overshoot vs n
    ax4 = fig.add_subplot(gs[1, 1])
    overshoot_plot = [results[n]['fleet_overshoot_mean'] for n in n_plot]
    ax4.plot(n_plot, overshoot_plot,
             color='darkorange', lw=2, marker='^')
    ax4.axvline(4, color='firebrick', lw=1.5, linestyle='--', alpha=0.5)
    ax4.set_title('Fleet Overshoot\nvs. Carrier Count', fontsize=11)
    ax4.set_xlabel('n carriers')
    ax4.set_ylabel('Mean excess capacity / demand')
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()

    # Panel 5: Welfare mean vs n
    ax5 = fig.add_subplot(gs[1, 2])
    welfare_plot = [results[n]['welfare_mean'] for n in n_plot]
    ax5.plot(n_plot, welfare_plot,
             color='seagreen', lw=2, marker='D')
    ax5.axvline(4, color='firebrick', lw=1.5, linestyle='--', alpha=0.5,
                label='n=4 (alliances)')
    ax5.set_title('Mean Social Welfare\nvs. Carrier Count', fontsize=11)
    ax5.set_xlabel('n carriers')
    ax5.set_ylabel('Cumulative welfare (mean)')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.invert_xaxis()

    plt.suptitle(
        'Consolidation & Cyclicality: Does Fewer Carriers Mean Less Overshooting?\n'
        'Cournot Nash Equilibrium — Container Shipping Cycle Model',
        fontsize=12, fontweight='bold'
    )

    os.makedirs('../data', exist_ok=True)
    plt.savefig('../data/consolidation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to data/consolidation_analysis.png")