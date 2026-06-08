# Shipping Cycle Model

A calibrated simulation of capacity cycle dynamics and welfare losses in container shipping markets.

The model quantifies the **deadweight loss from the structural coordination failure** in capacity investment — the mechanism through which individually rational ordering decisions by carriers produce collectively inefficient boom-bust cycles. A social planner benchmark isolates the efficiency gap, and three intervention modules evaluate policy responses.

---

## Motivation

Container shipping markets exhibit persistent cyclicality: periods of freight rate boom trigger simultaneous over-ordering across carriers, whose deliveries arrive years later into a bust. This Cobweb pattern (Ezekiel 1938; Luo, Fan & Liu 2009) is well-documented empirically but rarely evaluated through a welfare-economics lens. This project fills that gap.

A quasi-normative question motivates the model: *what is the welfare cost of the coordination failure, and which interventions reduce it?*

---

## Model Overview

### Core modules (`model/`)

| Module | Description |
|--------|-------------|
| `demand.py` | Markov-switching GDP process (Hamilton 1989) with two regimes (expansion/recession), logistic trade elasticity transition calibrated on Constantinescu et al. (2020), and asymmetric geopolitical shocks: Type A (negative, compressing) and Type B (positive tonne-mile boom, e.g. Red Sea rerouting) |
| `supply.py` | Beta-distributed stochastic building lag [1–4 periods], Cobweb ordering rule (Luo et al. 2009), earnings-dependent scrapping with EWMA memory (calibrated on UNCTAD RMT 2025 historic-low scrapping), CII slow-steaming capacity reduction |
| `market.py` | Market clearing via inverse demand function, welfare decomposition CS + PS − C_over − C_under, social planner benchmark via greedy lookahead optimisation, Monte Carlo welfare loss distribution |
| `heterogeneous_carriers.py` | Two-type carrier market: strategic carriers (large alliances, 65% share, partial conjectural variation, forward-looking orderbook signal) vs. fringe carriers (price-takers, Greenwood-Hanson overextrapolation, synchronisation amplifier) |
| `validation.py` | Method-of-moments validation against SCFI 2009-2024 (UNCTAD RMT 2025): 8-moment comparison including volatility, autocorrelation, excess kurtosis, boom/bust frequency and duration |
| `seeds.py` | Fixed seed list for reproducible Monte Carlo runs |

### Intervention modules (`interventions/`)

| Module | Mechanism | Key finding |
|--------|-----------|-------------|
| `consolidation.py` | Cournot-Nash oligopoly sweep across n carriers | Welfare deteriorates below n ≈ 5: market power dominates Cournot dampening |
| `transparency.py` | Expectation formation spectrum τ ∈ [0,1] with Greenwood-Hanson synchronisation | τ = 0.5 (partial transparency) is the *worst* point — synchronisation amplifies the Cobweb overshoot |
| `capacity_certificates.py` | Pigouvian cap-and-trade on newbuilding capacity | +4% welfare gain at cap = 1.0; applies to all carriers including fringe |

---

## Key Findings

### Baseline welfare loss
Monte Carlo median DWL ~30–35% of social planner welfare (P5–P95: ~[300, 2200] welfare units), driven primarily by geopolitical shock realisations.

### Intervention ranking (200 Monte Carlo runs each)

| Intervention | Best parameter | Welfare vs. baseline |
|---|---|---|
| Transparency | τ = 1.0 | +~9% |
| Capacity certificates | cap = 1.0 | +~4% |
| Consolidation | n = 4 | −2 to −3% (negative) |

### The Greenwood-Hanson paradox
Intermediate digitalisation (τ ≈ 0.5) is the worst point on the transparency curve. When all carriers observe the same orderbook data simultaneously, their non-cooperative responses synchronise the ordering wave — amplifying rather than dampening the Cobweb overshoot.

### The consolidation trap
Consolidation below n ≈ 5 reduces fleet overshoot but lowers social welfare below the atomistic baseline, because market power (higher rates → lower consumer surplus) dominates the Cournot capacity discipline effect. This confirms Cariou & Guillotreau (2022).

### Fringe dominance
In the heterogeneous carrier model, strategic carrier discipline becomes effective only when the fringe share falls below ~15–25% of total capacity. At the current ~35% fringe share, overextrapolation and synchronisation among price-taking carriers dominate, maintaining DWL close to the homogeneous baseline regardless of alliance structure.

---

## Calibration

| Parameter | Value | Source |
|---|---|---|
| `trade_elasticity_high` | 2.2 | Constantinescu et al. (2020), p. 124 |
| `trade_elasticity_low` | 1.3 | Constantinescu et al. (2020), p. 124 |
| `building_lag` (expected) | ~2.2 years | Luo et al. (2009), pp. 512–514 |
| `order_sensitivity` | 0.18 | Luo et al. (2009), pp. 512–514 |
| `demand_elasticity` | −0.30 | Stopford (2009), p. 154 |
| `slow_steam_reduction` | 0.06 | Lehmann et al. (2025), p. 3 |
| `overextrapolation_bias` | 0.30 | Greenwood & Hanson (2015), pp. 57–58 |
| `pos_shock_severity_mean` | 0.13 | UNCTAD RMT 2025, ch. III (Red Sea +12–15% tonne-miles) |
| `strategic_market_share` | 0.65 | UNCTAD RMT 2025, ch. II (top 5 carriers) |

---

## Quickstart

```bash
git clone https://github.com/Xaver-M/Shipping-Cycle-Model.git
cd Shipping-Cycle-Model

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run baseline model
cd model && python market.py

# Run interventions
cd ../interventions
python consolidation.py
python transparency.py
python capacity_certificates.py

# Open notebooks
cd .. && jupyter notebook notebooks/
```

---

## Notebooks

| Notebook | Content |
|----------|---------|
| `01_baseline_calibration.ipynb` | Demand process, supply dynamics, market equilibrium, Monte Carlo welfare distribution |
| `02_intervention_comparison.ipynb` | Consolidation sweep, transparency sweep, capacity certificates, three-way comparison, policy implications |
| `03_validation_and_heterogeneous.ipynb` | Method-of-moments validation against SCFI 2009-2024, heterogeneous carrier analysis, fringe dominance sweep |

---

## References

- Cariou, P. & Guillotreau, P. (2022). Capacity management in liner shipping alliances. *Maritime Economics & Logistics*.
- Constantinescu, C., Mattoo, A. & Ruta, M. (2020). The global trade slowdown. *World Bank Economic Review*, 34(1), 121–142.
- Cramton, P. & Stoft, S. (2006). *The Convergence of Market Designs for Adequate Generating Capacity*. MIT CEEPR.
- Ezekiel, M. (1938). The Cobweb theorem. *Quarterly Journal of Economics*, 52(2), 255–280.
- Ghorbani, A. et al. (2022). Agent-based modelling in maritime economics. *Maritime Policy & Management*.
- Greenwood, R. & Hanson, S. G. (2015). Waves in ship prices and investment. *Quarterly Journal of Economics*, 130(1), 55–109.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*, 57(2), 357–384.
- Lehmann, M. et al. (2025). CII compliance and effective fleet capacity. *Transportation Research Part D*.
- Luo, M., Fan, L. & Liu, L. (2009). An econometric analysis for container shipping market. *Maritime Policy & Management*, 36(6), 507–523.
- Stopford, M. (2009). *Maritime Economics* (3rd ed.). Routledge.
- UNCTAD (2025). *Review of Maritime Transport 2025*. United Nations.
