"""Inference-time cost of each selection rule: wall-clock time per examinee per step (section 6.8, tab:cost).

Measures the time to choose one item for a batch of examinees (batch 1 = real-time use,
batch 2000 = offline evaluation) after 10 responses, for bank sizes 500, 2000 and 8000.
The DQN uses an untrained network of the proposed architecture (64-64) and the proposed
state, so the posterior-variance feature is computed inside the timing; the time does not
depend on the weights. Outputs result/<experiment>/cost/cost.csv.
"""
import argparse
import os
import time
from experiment_runner import result_directory

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded NumPy/BLAS; must be set before importing numpy

import numpy as np
import pandas as pd
import torch

from .dqn import DQNAgent, DQNConfig
from .irt import eap, make_bank
from .rules import MEPVPolicy, MFIPolicy, WeightedInfoPolicy
from .simulate import prob
from .run_grid import PROPOSED

torch.set_num_threads(1)


def history(bank, theta, t, rng):
    """t administered items (random) and responses, plus the EAP estimate."""
    n = len(theta)
    items = np.array([rng.choice(bank.shape[0], t, replace=False) for _ in range(n)])
    resp = (rng.random((n, t)) <= prob(bank[items, 0], bank[items, 1], theta[:, None])).astype(int)
    th = eap(bank, items, resp)
    mask = np.zeros((n, bank.shape[0]), bool)
    mask[np.arange(n)[:, None], items] = True
    return items, resp, th, mask


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="output root (default: result); saves under <root>/<experiment>/cost")
    args = ap.parse_args(argv)
    out = result_directory(__package__, args.out) / "cost"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for n_items in [500, 2000, 8000]:
        bank = make_bank(n_items, rng)
        rules = {"MFI": MFIPolicy(bank), "FIWL": WeightedInfoPolicy(bank, "fiwl"),
                 "MPWI": WeightedInfoPolicy(bank, "mpwi"), "MEPV": MEPVPolicy(bank),
                 "DQN (proposed)": DQNAgent(bank, 40, DQNConfig(seed=0, **PROPOSED))}
        for batch in [1, 2000]:
            theta = rng.normal(size=batch)
            items, resp, th, mask = history(bank, theta, 10, rng)
            for name, rule in rules.items():
                reps = 200 if batch == 1 else 5
                rule(th, 10, mask, items, resp)  # warm-up
                t0 = time.perf_counter()
                for _ in range(reps):
                    rule(th, 10, mask, items, resp)
                per = (time.perf_counter() - t0) / reps / batch
                rows.append(dict(n_items=n_items, batch=batch, rule=name, us_per_examinee_step=per * 1e6))
                print(f"I={n_items:5d} batch={batch:4d} {name:15s} {per*1e6:9.1f} us / examinee-step", flush=True)
    pd.DataFrame(rows).to_csv(out / "cost.csv", index=False)
    import numpy as _np
    print("threads: torch", torch.get_num_threads(), "; BLAS env OMP/OPENBLAS/MKL =", os.environ["OMP_NUM_THREADS"], "; numpy", _np.__version__)


if __name__ == "__main__":
    main()
