"""Simulation scenarios: the data-generating model versus the 2PLM the rules use (the estimation model).

make_scenario(name, n_items, bank_rng, guess_rng) returns (bank_true, bank, sample_theta):
  bank_true    (I, 3) array [a, b, c] of the data-generating model (c = 0 under "base")
  bank         (I, 2) array [a, b] used by the rules, the estimator and the agent (the true a, b)
  sample_theta(rng, n): the examinee population, N(0, 1)
Names: "base"  -- the data-generating model is the 2PLM (sections 6.2 to 6.6)
       "guess" -- the data-generating model is the 3PLM with c ~ U(0.10, 0.30), redrawn per
                  replication (section 6.7)
The bank comes from `bank_rng` (fixed across replications); the guessing parameters come from
`guess_rng` (one stream per replication). Two standard-normal draws of length I precede the
guessing draw: they were the calibration-error draws of an earlier version and are kept,
unused, so that the recorded results remain reproducible from the same seeds.
"""
import numpy as np

from .irt import make_bank

SCENARIOS = ("base", "guess")


def _pop_normal(rng, n):
    return rng.normal(size=n)


def make_scenario(name, n_items, bank_rng, guess_rng):
    if name not in SCENARIOS:
        raise ValueError(name)
    bank = make_bank(n_items, bank_rng)
    guess_rng.normal(0, 1, n_items)  # discarded, see the module docstring
    guess_rng.normal(0, 1, n_items)
    c_draw = guess_rng.uniform(0.10, 0.30, n_items)
    c = c_draw if name == "guess" else np.zeros(n_items)
    return np.column_stack([bank, c]), bank.copy(), _pop_normal
