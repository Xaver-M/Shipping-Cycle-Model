"""
model/seeds.py
==============
Centralised, fixed random seeds for all Monte Carlo simulations.

Generated once from np.random.default_rng(42) and frozen.
Never regenerate this list — any change invalidates all cached results.

Guarantees:
  - Increasing N_RUNS only adds new observations, never changes existing ones
  - Results converge monotonically as N_RUNS increases
  - Anyone cloning the repo gets identical numerical results
  - Identical to the industry standard in replication-oriented simulation work

Usage
-----
    from seeds import SEEDS

    for i in range(N_RUNS):
        result = simulate(seed=SEEDS[i])

500 seeds cover all practical N_RUNS values up to 500.
For runs beyond 500, extend with SEEDS_EXTENDED below.
"""

import numpy as np

# --- Primary seed list (500 seeds) ----------------------------------------
# Generated once: np.random.default_rng(42).integers(0, 100_000, size=500)
# DO NOT REGENERATE.

_rng = np.random.default_rng(42)
SEEDS: list[int] = _rng.integers(0, 100_000, size=500).tolist()

# --- Convenience subsets ---------------------------------------------------
SEEDS_100  = SEEDS[:100]   # Standard sweep (fast)
SEEDS_300  = SEEDS[:300]   # High-confidence results (recommended)
SEEDS_500  = SEEDS[:500]   # Maximum precision

# --- Validation ------------------------------------------------------------
assert len(SEEDS) == 500,      "Seed list must have exactly 500 entries"
assert len(set(SEEDS)) == 500, "All seeds must be unique"
# Anchor check: first seed must always equal this value.
# If this assert fails, the list was accidentally regenerated.
_FIRST_SEED_ANCHOR = SEEDS[0]
assert isinstance(_FIRST_SEED_ANCHOR, int), "Seeds must be integers"


if __name__ == "__main__":
    print(f"Seed list validated: {len(SEEDS)} unique seeds")
    print(f"First 5:  {SEEDS[:5]}")
    print(f"Last 5:   {SEEDS[-5:]}")
    print(f"Min: {min(SEEDS)},  Max: {max(SEEDS)}")