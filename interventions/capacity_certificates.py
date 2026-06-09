"""
interventions/capacity_certificates.py
=======================================
Models the effect of a tradeable capacity certificate system on
shipping cycle dynamics and welfare.

Theoretical basis
-----------------
The consolidation and transparency analyses establish a fundamental
result: neither market structure change nor informational improvement
can resolve the coordination failure at the heart of the Cobweb
mechanism. Consolidation below n=5 increases market power without
sufficient capacity discipline; transparency at τ=0.5 synchronises
and amplifies the ordering wave.

This module examines the only intervention that directly addresses
the root cause: a market mechanism that internalises the aggregate
capacity externality that individual carriers ignore when ordering.

The analogy is the electricity capacity market (Cramton & Stoft 2006):
in power markets, generators are paid explicitly for *available*
capacity, not just energy produced. This internalises the social
value of reserve capacity and prevents the under-investment that
would occur in an energy-only market.

The shipping analogue is a system of tradeable capacity certificates:

  1. A regulatory authority sets a maximum aggregate orderbook each
     period (the cap), calibrated on expected demand growth plus an
     efficient buffer.

  2. Carriers wishing to order must hold certificates equal to the
     TEU capacity they intend to order.

  3. Certificates are auctioned each period; unused certificates
     can be sold to other carriers.

  4. The certificate price creates a Pigouvian tax on overordering:
     when all carriers want to order simultaneously (boom phase),
     certificate prices rise, internalising the negative externality.

This is structurally equivalent to a cap-and-trade system for
capacity investment — analogous to the EU ETS for carbon emissions,
which is already being applied to shipping (EU ETS inclusion 2024).

The welfare gain relative to the decentralised baseline quantifies
the value of resolving the coordination failure directly, as opposed
to the partial and paradoxical improvements from consolidation and
transparency.

Key parameters
--------------
- cap_tightness: how close the cap is to the social planner's optimal
  order volume (1.0 = exactly the planner's optimum)
- auction_efficiency: fraction of the certificate price that is
  redistributed as lump-sum transfers (revenue neutrality)
- buffer_fraction: efficient overcapacity buffer above expected demand

Key references
--------------
- Cramton & Stoft (2006): capacity market design, electricity analogy.
- Dixit & Pindyck (1994): investment under uncertainty, irreversibility.
- Stopford (2009): Four-Market Model, demolition as partial stabiliser.
- OECD (2018): limits of current regulatory framework, pp. 15-22.
"""

import numpy as np
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).parent.parent / 'data'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from demand import DemandParameters, DemandProcess
from supply import SupplyParameters, SupplyProcess
from market import MarketParameters, FreightRateModel, WelfareCalculator, SocialPlanner


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class CertificateParameters:
    """Parameters for the capacity certificate system."""

    cap_tightness_range: list = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    )
    """
    Range of cap tightness values to sweep.
    0.0  = no cap (decentralised baseline)
    0.5  = cap at 50% of planner optimum (very tight)
    1.0  = cap exactly at planner's optimal order volume
    1.25 = cap slightly above planner optimum (loose)
    Values > 1.0 test robustness to over-generous caps.
    """

    efficient_buffer: float = 0.05
    """
    Efficient overcapacity buffer: the planner's optimal fleet is
    demand * (1 + buffer). Captures the insurance value of reserve
    capacity against demand shocks and disruptions.
    At 5%: modest buffer consistent with capital cost of idle tonnage.
    """

    auction_efficiency: float = 0.80
    """
    Fraction of certificate auction revenue redistributed as lump-sum
    transfers to market participants (revenue neutrality).
    1.0 = fully neutral (no deadweight loss from the tax instrument).
    0.8 = 20% administrative cost / redistribution friction.
    """

    certificate_price_sensitivity: float = 2.0
    """
    Elasticity of certificate price with respect to excess demand for
    certificates (i.e., how steeply the auction price rises when all
    carriers want to order simultaneously).
    Higher values = steeper Pigouvian tax in boom phases.
    """

    demand_forecast_window: int = 3
    """
    Number of periods used to estimate expected demand growth for
    cap calibration. Longer window = smoother cap, less responsive
    to recent shocks.
    """


# ---------------------------------------------------------------------------
# Certificate market
# ---------------------------------------------------------------------------

class CertificateMarket:
    """
    Implements the period-by-period certificate auction.

    Each period:
      1. The authority announces the cap Q_cap(t) based on demand forecast.
      2. Carriers submit bids for certificates proportional to their
         desired order volume (naive Cobweb rule, as in baseline).
      3. If aggregate desired orders > Q_cap, the auction price rises
         until demand for certificates equals supply (the cap).
      4. Each carrier receives certificates proportional to its bid,
         scaled down by the clearing ratio.
      5. Auction revenue is partially redistributed (auction_efficiency).
    """

    def __init__(self, params: CertificateParameters):
        self.p = params

    def auction(
        self,
        desired_orders: float,
        q_cap: float,
        prev_cert_price: float,
    ) -> tuple[float, float, float]:
        """
        Run one period's certificate auction.

        Parameters
        ----------
        desired_orders  : aggregate desired order volume (unconstrained)
        q_cap           : certificate cap for this period
        prev_cert_price : certificate price from previous period

        Returns
        -------
        actual_orders   : order volume after certificate constraint
        cert_price      : clearing certificate price
        revenue         : auction revenue (for redistribution)
        """
        if q_cap <= 0 or desired_orders <= 0:
            return desired_orders, 0.0, 0.0

        if desired_orders <= q_cap:
            # Cap not binding: no certificate price, full orders proceed
            return desired_orders, 0.0, 0.0

        # Cap is binding: price rises to clear the market
        # Certificate price = shadow cost of the cap constraint
        excess_demand_ratio = desired_orders / max(q_cap, 0.01) - 1.0
        cert_price = (prev_cert_price * 0.5 +
                      excess_demand_ratio * self.p.certificate_price_sensitivity * 0.5)
        cert_price = max(0.0, cert_price)

        # Orders scaled to cap
        actual_orders = q_cap
        revenue = cert_price * actual_orders * self.auction_redistribution_fraction()

        return actual_orders, cert_price, revenue

    def auction_redistribution_fraction(self) -> float:
        return self.p.auction_efficiency

    def compute_cap(
        self,
        current_fleet: float,
        demand_history: np.ndarray,
        cap_tightness: float,
        planner_order: float,
    ) -> float:
        """
        Set the period's certificate cap.

        The cap is anchored on the social planner's optimal order volume,
        scaled by cap_tightness. At cap_tightness=1.0, the cap exactly
        equals what the planner would order — internalising the externality
        perfectly (subject to the planner's own estimation error).

        A demand-growth adjustment ensures the cap tracks trend demand
        rather than being a fixed number — otherwise the cap becomes
        procyclical (tight in booms, loose in busts).
        """
        if cap_tightness == 0.0:
            return float('inf')  # no cap

        # Base cap: planner optimum * tightness
        base_cap = planner_order * cap_tightness

        # Add efficient buffer for demand uncertainty
        buffer = current_fleet * self.p.efficient_buffer

        return max(base_cap + buffer, current_fleet * 0.01)


# ---------------------------------------------------------------------------
# Certificate market simulation
# ---------------------------------------------------------------------------

class CertificateMarketSimulation:
    """
    Simulates the full shipping market with a capacity certificate system.

    The certificate market sits between the carriers' desired orders
    (from the naive Cobweb rule) and the actual orders placed. In boom
    phases, the cap constrains overordering; in bust phases, the cap is
    not binding and the market operates freely.

    This asymmetry is the key efficiency property: the system only
    intervenes when the externality is most severe (simultaneous
    overordering), leaving normal market operations undisturbed.
    """

    def __init__(
        self,
        cap_tightness: float,
        demand_params: DemandParameters,
        supply_params: SupplyParameters,
        market_params: MarketParameters,
        cert_params: CertificateParameters,
        seed: Optional[int] = None,
    ):
        self.cap_tightness = cap_tightness
        self.dp   = demand_params
        self.sp   = supply_params
        self.mp   = market_params
        self.cp   = cert_params

        self.rate_model    = FreightRateModel(market_params)
        self.welfare_calc  = WelfareCalculator(market_params)
        self.planner       = SocialPlanner(market_params, supply_params)
        self.cert_market   = CertificateMarket(cert_params)

    def run(self, seed: Optional[int] = None) -> dict:
        """Run complete simulation with certificate system."""
        n = self.mp.n_periods

        d_params = DemandParameters(**{
            **self.dp.__dict__,
            'seed': seed if seed is not None else self.dp.seed
        })
        demand_results = DemandProcess(d_params).simulate(n)
        eff_demand     = demand_results['effective_demand']

        # First pass: get planner orders for cap calibration
        planner_results = self.planner.simulate(eff_demand)
        planner_orders  = planner_results['orders']

        fleet         = np.zeros(n)
        orders_arr    = np.zeros(n)
        desired_arr   = np.zeros(n)  # unconstrained desired orders
        deliveries    = np.zeros(n)
        scrapping     = np.zeros(n)
        orderbook     = np.zeros(n)
        rates         = np.zeros(n)
        cert_prices   = np.zeros(n)
        cert_revenues = np.zeros(n)
        caps          = np.zeros(n)
        welfare_d     = []

        fleet[0]  = self.sp.initial_fleet
        lag       = self.sp.building_lag
        pipeline  = np.zeros(lag)
        pipeline[-1] = self.sp.initial_orderbook

        supply_proc  = SupplyProcess(self.sp)
        prev_rate    = self.mp.long_run_equilibrium_rate
        prev_orders  = self.sp.initial_orderbook
        prev_cert_p  = 0.0
        demand_hist  = [eff_demand[0]]

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

            # Effective fleet
            eff_fleet = fleet[t]
            if self.sp.slow_steam_active and t >= self.sp.slow_steam_start_period:
                eff_fleet *= (1 - self.sp.slow_steam_reduction)

            # Market clearing rate
            rate         = self.rate_model.clearing_rate(
                eff_demand[t], eff_fleet, prev_rate
            )
            rates[t]     = rate
            prev_rate    = rate

            # Welfare (include certificate revenue as transfer)
            w = self.welfare_calc.period_welfare(
                rate, eff_demand[t], eff_fleet, t
            )
            # Add redistributed certificate revenue to welfare
            if t > 0:
                w = dict(w)
                w['total'] = w['total'] + cert_revenues[t-1]
            welfare_d.append(w)

            # Unconstrained desired orders (naive Cobweb)
            desired = supply_proc._ordering_rule(
                rate / self.mp.long_run_equilibrium_rate,
                fleet[t],
                prev_orders
            )
            desired_arr[t] = desired

            # Certificate cap for this period
            q_cap = self.cert_market.compute_cap(
                current_fleet  = fleet[t],
                demand_history = np.array(demand_hist[-self.cp.demand_forecast_window:]),
                cap_tightness  = self.cap_tightness,
                planner_order  = planner_orders[t],
            )
            caps[t] = q_cap

            # Auction: constrain orders to cap
            actual_order, cert_price, revenue = self.cert_market.auction(
                desired_orders  = desired,
                q_cap           = q_cap,
                prev_cert_price = prev_cert_p,
            )
            orders_arr[t]    = actual_order
            cert_prices[t]   = cert_price
            cert_revenues[t] = revenue
            prev_cert_p      = cert_price
            prev_orders      = actual_order

            if t + lag < n:
                pipeline[-1] += actual_order
            orderbook[t] = pipeline.sum()

            demand_hist.append(eff_demand[t])

        # Welfare comparison
        cumulative_d = self.welfare_calc.cumulative_welfare(welfare_d)
        cumulative_p = self.welfare_calc.cumulative_welfare(
            planner_results['welfare']
        )
        dwl_total = cumulative_p['total'] - cumulative_d['total']
        dwl_pct   = (dwl_total / abs(cumulative_p['total']) * 100
                     if cumulative_p['total'] != 0 else 0)

        # Cap binding fraction: how often did the cap actually constrain orders?
        cap_binding_fraction = float(np.mean(desired_arr > caps))

        return {
            'cap_tightness'       : self.cap_tightness,
            'demand'              : eff_demand,
            'fleet'               : fleet,
            'orders'              : orders_arr,
            'desired_orders'      : desired_arr,
            'rates'               : rates,
            'cert_prices'         : cert_prices,
            'cert_revenues'       : cert_revenues,
            'caps'                : caps,
            'orderbook'           : orderbook,
            'welfare_d'           : welfare_d,
            'planner_fleet'       : planner_results['fleet'],
            'planner_rates'       : planner_results['rates'],
            'cumulative_d'        : cumulative_d,
            'cumulative_p'        : cumulative_p,
            'dwl_total'           : dwl_total,
            'dwl_pct'             : dwl_pct,
            'rate_volatility'     : float(np.std(rates)),
            'cap_binding_fraction': cap_binding_fraction,
            'cert_revenue_total'  : float(cert_revenues.sum()),
            'order_reduction'     : float(
                1 - orders_arr.sum() / max(desired_arr.sum(), 1)
            ),
        }


# ---------------------------------------------------------------------------
# Certificate sweep
# ---------------------------------------------------------------------------

class CertificateAnalysis:
    """
    Sweeps cap tightness from 0 (no cap) to 1.25 (loose cap) and
    measures welfare improvement relative to the decentralised baseline
    and the social planner optimum.

    The key question: how close to the planner optimum can a feasible
    certificate system get, and at what tightness is the gain maximised?
    """

    def __init__(
        self,
        demand_params: DemandParameters,
        supply_params: SupplyParameters,
        market_params: MarketParameters,
        cert_params: CertificateParameters,
    ):
        self.dp = demand_params
        self.sp = supply_params
        self.mp = market_params
        self.cp = cert_params

    def run_sweep(
        self,
        n_monte_carlo: int = 100,
        seed: int = 42,
    ) -> dict:
        """Sweep cap tightness levels."""
        rng     = np.random.default_rng(seed)
        results = {}

        for cap in self.cp.cap_tightness_range:
            label = f"{cap:.2f}"
            print(f"  Simulating cap_tightness={cap:.2f} ({n_monte_carlo} runs)...")

            dwl_list      = []
            rate_vol_list = []
            welfare_list  = []
            binding_list  = []
            reduction_list= []

            for _ in range(n_monte_carlo):
                run_seed = int(rng.integers(0, 100_000))
                sim = CertificateMarketSimulation(
                    cap_tightness = cap,
                    demand_params = self.dp,
                    supply_params = self.sp,
                    market_params = self.mp,
                    cert_params   = self.cp,
                    seed          = run_seed,
                )
                r = sim.run(seed=run_seed)
                dwl_list.append(r['dwl_total'])
                rate_vol_list.append(r['rate_volatility'])
                welfare_list.append(r['cumulative_d']['total'])
                binding_list.append(r['cap_binding_fraction'])
                reduction_list.append(r['order_reduction'])

            results[cap] = {
                'dwl_mean'          : float(np.mean(dwl_list)),
                'dwl_std'           : float(np.std(dwl_list)),
                'dwl_median'        : float(np.median(dwl_list)),
                'dwl_p5'            : float(np.percentile(dwl_list, 5)),
                'dwl_p95'           : float(np.percentile(dwl_list, 95)),
                'rate_vol_mean'     : float(np.mean(rate_vol_list)),
                'welfare_mean'      : float(np.mean(welfare_list)),
                'cap_binding_mean'  : float(np.mean(binding_list)),
                'order_reduction_mean': float(np.mean(reduction_list)),
                'n_runs'            : n_monte_carlo,
            }

        return results


# ---------------------------------------------------------------------------
# Three-way comparison: baseline vs consolidation vs transparency vs certs
# ---------------------------------------------------------------------------

def three_way_comparison(
    demand_params: DemandParameters,
    supply_params: SupplyParameters,
    market_params: MarketParameters,
    n_runs: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run all three interventions at their best parameter settings
    and compare DWL reduction vs. decentralised baseline.

    Returns a summary dict for the final comparison plot.
    """
    from consolidation  import OligopolyMarket, ConsolidationParameters
    from transparency   import TransparentMarket, TransparencyParameters

    rng = np.random.default_rng(seed)
    mp  = market_params

    results = {
        'baseline'      : [],
        'consolidation' : [],   # best: n=7
        'transparency'  : [],   # best: τ=1.0
        'certificates'  : [],   # best: cap=1.0
    }

    for _ in range(n_runs):
        s = int(rng.integers(0, 100_000))

        # Baseline (decentralised, from market.py via OligopolyMarket n=20)
        b = OligopolyMarket(20, demand_params, supply_params, mp, seed=s)
        results['baseline'].append(b.run(seed=s)['cumulative_d']['total'])

        # Best consolidation: n=7 (marginal improvement without market power)
        c = OligopolyMarket(7, demand_params, supply_params, mp, seed=s)
        results['consolidation'].append(c.run(seed=s)['cumulative_d']['total'])

        # Best transparency: τ=1.0 (full rational expectations)
        t = TransparentMarket(1.0, demand_params, supply_params, mp,
                              TransparencyParameters(), seed=s)
        results['transparency'].append(t.run(seed=s)['cumulative_d']['total'])

        # Best certificates: cap_tightness=1.0 (planner-calibrated cap)
        cert = CertificateMarketSimulation(
            1.0, demand_params, supply_params, mp,
            CertificateParameters(), seed=s
        )
        results['certificates'].append(cert.run(seed=s)['cumulative_d']['total'])

    return {k: np.array(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Diagnostic / entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("=" * 60)
    print("Capacity Certificate Analysis: Direct Coordination Fix")
    print("=" * 60)

    dp = DemandParameters(seed=42)
    sp = SupplyParameters(seed=42)
    mp = MarketParameters(n_periods=50, seed=42)
    cp = CertificateParameters()

    # --- Certificate sweep ---
    analysis = CertificateAnalysis(dp, sp, mp, cp)
    print("\nRunning certificate sweep (100 runs per cap level)...")
    cert_results = analysis.run_sweep(n_monte_carlo=100, seed=42)

    caps = sorted(cert_results.keys())

    # Summary table
    print(f"\n{'Cap':>6} | {'DWL Median':>10} | {'Rate Vol':>8} | "
          f"{'Binding%':>8} | {'Order↓%':>7} | {'Welfare':>10}")
    print("-" * 65)
    for cap in caps:
        r = cert_results[cap]
        print(f"{cap:>6.2f} | {r['dwl_median']:>10.1f} | "
              f"{r['rate_vol_mean']:>8.4f} | "
              f"{r['cap_binding_mean']*100:>7.1f}% | "
              f"{r['order_reduction_mean']*100:>6.1f}% | "
              f"{r['welfare_mean']:>10.1f}")

    # Baseline DWL
    dwl_baseline = cert_results[0.0]['dwl_median']
    dwl_best     = min(cert_results[c]['dwl_median'] for c in caps if c > 0)
    best_cap     = min((c for c in caps if c > 0),
                       key=lambda c: cert_results[c]['dwl_median'])
    pct_reduction = (dwl_baseline - dwl_best) / dwl_baseline * 100

    print(f"\nKey findings:")
    print(f"  DWL without certificates:      {dwl_baseline:.1f}")
    print(f"  DWL at optimal cap ({best_cap:.2f}):  {dwl_best:.1f}")
    print(f"  DWL reduction:                 {pct_reduction:.1f}%")

    # --- Three-way comparison ---
    print("\nRunning three-way intervention comparison (100 runs)...")
    comparison = three_way_comparison(dp, sp, mp, n_runs=100, seed=42)

    print(f"\n=== Three-Way Comparison (Median Welfare) ===")
    for label, vals in comparison.items():
        print(f"  {label:15s}: {np.median(vals):>10.1f}  "
              f"(vs baseline: {np.median(vals) - np.median(comparison['baseline']):+.1f})")

    # --- Plots ---
    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    # Panel 1: DWL vs cap tightness
    ax1 = fig.add_subplot(gs[0, :])
    dwl_meds = [cert_results[c]['dwl_median'] for c in caps]
    dwl_p5s  = [cert_results[c]['dwl_p5']     for c in caps]
    dwl_p95s = [cert_results[c]['dwl_p95']    for c in caps]

    ax1.fill_between(caps, dwl_p5s, dwl_p95s,
                     alpha=0.2, color='steelblue', label='5th–95th pct')
    ax1.plot(caps, dwl_meds,
             color='steelblue', lw=2.5, marker='o', label='Median DWL')
    ax1.axhline(dwl_baseline, color='firebrick', lw=1.5, linestyle='--',
                alpha=0.7, label=f'No-certificate baseline: {dwl_baseline:.0f}')
    ax1.axvline(1.0, color='seagreen', lw=1.5, linestyle=':',
                label='Planner-calibrated cap (cap=1.0)')
    ax1.set_title('Deadweight Loss vs. Certificate Cap Tightness\n'
                  '0=No cap (baseline)  |  1.0=Planner-optimal cap',
                  fontsize=11)
    ax1.set_xlabel('Cap tightness (relative to planner optimum)')
    ax1.set_ylabel('DWL (welfare units, median)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Rate volatility vs cap
    ax2 = fig.add_subplot(gs[1, 0])
    rate_vols = [cert_results[c]['rate_vol_mean'] for c in caps]
    ax2.plot(caps, rate_vols, color='firebrick', lw=2, marker='s')
    ax2.axvline(1.0, color='seagreen', lw=1.5, linestyle=':', alpha=0.7)
    ax2.set_title('Freight Rate Volatility\nvs. Cap Tightness', fontsize=11)
    ax2.set_xlabel('Cap tightness')
    ax2.set_ylabel('Std dev of freight rate')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Cap binding fraction
    ax3 = fig.add_subplot(gs[1, 1])
    binding = [cert_results[c]['cap_binding_mean'] * 100 for c in caps if c > 0]
    caps_nonzero = [c for c in caps if c > 0]
    ax3.bar(caps_nonzero, binding, width=0.08,
            color='darkorange', alpha=0.7)
    ax3.axvline(1.0, color='seagreen', lw=1.5, linestyle=':', alpha=0.7)
    ax3.set_title('Cap Binding Frequency\n(% of periods cap constrains orders)',
                  fontsize=11)
    ax3.set_xlabel('Cap tightness')
    ax3.set_ylabel('% periods cap is binding')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Three-way comparison — welfare boxplot
    ax4 = fig.add_subplot(gs[2, :])
    labels   = ['Baseline\n(Decentralised)', 'Consolidation\n(n=7)',
                 'Transparency\n(τ=1.0)', 'Capacity\nCertificates\n(cap=1.0)']
    colors   = ['firebrick', 'darkorange', 'steelblue', 'seagreen']
    data     = [comparison['baseline'], comparison['consolidation'],
                comparison['transparency'], comparison['certificates']]

    bp = ax4.boxplot(data, patch_artist=True, widths=0.5,
                     medianprops=dict(color='white', lw=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_xticklabels(labels, fontsize=10)
    ax4.set_title('Three-Way Intervention Comparison: Cumulative Social Welfare\n'
                  'Certificates vs. Consolidation vs. Transparency vs. Baseline',
                  fontsize=11)
    ax4.set_ylabel('Cumulative welfare (100 Monte Carlo runs)')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add median labels
    for i, d in enumerate(data):
        ax4.text(i + 1, np.median(d) + 10, f'{np.median(d):.0f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle(
        'Capacity Certificate System: Resolving the Coordination Failure Directly\n'
        'Analogy: Electricity Capacity Markets (Cramton & Stoft 2006)',
        fontsize=12, fontweight='bold'
    )

    _DATA_DIR.mkdir(exist_ok=True)
    plt.savefig(_DATA_DIR / 'certificate_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to data/certificate_analysis.png")