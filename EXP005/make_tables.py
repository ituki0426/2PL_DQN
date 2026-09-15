"""Print EXP005 RMSE table rows against the full-response EAP reference."""
import argparse

import pandas as pd

from experiment_runner import result_directory
from .data import DATASETS
from .run_grid import CONDITIONS, LIMITATION, RULES, SHOW_STEPS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", nargs="?", default="main", choices=["main"])
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--out")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--screening", choices=["positive", "strict"], default="positive")
    args = parser.parse_args(argv)
    out = result_directory(__package__, args.out) / args.dataset
    if args.screening == "strict":
        out /= "strict"
    out /= "main_quick" if args.quick else "main"
    path = out / "mean.csv"
    if not path.exists():
        parser.error(f"Run --aggregate first: {path}")
    frame = pd.read_csv(path)
    steps = [s for s in SHOW_STEPS if s in set(frame.step)] or [int(frame.step.max())]
    print(f"% {args.dataset}: RMSE against theta_reference (full-response theta_EAP), steps={steps}")
    print(f"% {LIMITATION}")
    for condition in (*RULES, *CONDITIONS):
        sub = frame[frame.condition == condition].set_index("step")
        if sub.empty:
            continue
        cells = []
        for step in steps:
            if step not in sub.index:
                cells.append("--")
                continue
            value = sub.loc[step, "rmse_mean"]
            best = frame[frame.step == step].rmse_mean.min()
            cells.append(("\\textbf{%.3f}" if round(value, 3) == round(best, 3) else "%.3f") % value)
        print(condition + " & " + " & ".join(cells) + " \\\\")
        print(f"% {condition}: n_rep={int(sub.n_rep.min())}")


if __name__ == "__main__":
    main()
