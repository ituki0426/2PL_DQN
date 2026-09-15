"""Measure item-selection cost on an EXP005 real-data item bank."""
import argparse
import os
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import torch

from experiment_runner import PROJECT_ROOT, result_directory
from .data import DATASETS, SPLIT_SEED, load_dataset, split_respondents
from .dqn import DQNAgent, DQNConfig
from .irt import mle
from .run_grid import PROPOSED, classical_policies


torch.set_num_threads(1)


def history(bank, responses, test_length=10):
    """Build a fixed observed-response history excluded from timed selection."""
    n = len(responses)
    items = np.tile(np.arange(test_length), (n, 1))
    resp = np.take_along_axis(responses, items, axis=1)
    theta = mle(bank, items, resp, np.zeros(n))
    mask = np.zeros((n, len(bank)), dtype=bool)
    mask[:, :test_length] = True
    return items, resp, theta, mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-root", default=PROJECT_ROOT / "data")
    parser.add_argument("--responses")
    parser.add_argument("--screening", choices=["positive", "strict"], default="positive")
    parser.add_argument("--rep", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--test-length", type=int, default=40)
    parser.add_argument("--out")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    data = load_dataset(args.dataset, args.data_root, args.test_length, args.responses, args.screening)
    split = split_respondents(data, args.split_seed + args.rep - 1)
    test = data.responses[split.split.to_numpy() == "test"]
    if args.quick:
        test = test[:64]
    warmup_length = min(10, args.test_length - 1)
    if warmup_length < 1:
        parser.error("--test-length must be at least 2 for the cost benchmark")

    out = result_directory(__package__, args.out) / args.dataset
    if args.screening == "strict":
        out /= "strict"
    out /= "cost_quick" if args.quick else "cost"
    out.mkdir(parents=True, exist_ok=True)

    rules = classical_policies(data.bank)
    rules["DQN (proposed)"] = DQNAgent(
        data.bank, args.test_length, DQNConfig(train_seed=0, **PROPOSED))
    batch_sizes = sorted(set([1, min(2000, len(test))]))
    rows = []
    for batch in batch_sizes:
        responses = test[:batch]
        items, resp, theta, mask = history(data.bank, responses, warmup_length)
        for name, rule in rules.items():
            reps = 200 if batch == 1 else 5
            rule(theta, warmup_length, mask, items, resp)
            started = time.perf_counter()
            for _ in range(reps):
                rule(theta, warmup_length, mask, items, resp)
            elapsed = (time.perf_counter() - started) / reps / batch
            rows.append(dict(dataset=args.dataset, n_items=len(data.bank), batch=batch,
                             rule=name, us_per_examinee_step=elapsed * 1e6))
            print(f"I={len(data.bank):5d} batch={batch:4d} {name:15s} "
                  f"{elapsed * 1e6:9.1f} us / examinee-step", flush=True)
    pd.DataFrame(rows).to_csv(out / "cost.csv", index=False)
    print(f"saved: {out / 'cost.csv'}")


if __name__ == "__main__":
    main()
