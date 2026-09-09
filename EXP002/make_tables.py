# -*- coding: utf-8 -*-
"""Generate the LaTeX table rows and the paired differences transcribed into ../Paper/main.tex.

  .venv/bin/python make_tables.py main | sensitivity | state | ablation | guess | cost

  main         6.2  tab:main         sensitivity  6.3  tab:sensitivity   state  6.4  tab:state
  ablation     6.5  tab:ablation     guess        6.7  tab:guess         cost   6.8  tab:cost

Every number printed here is read from result/<experiment>/<grid>/ files (mean.csv written by
`run_grid.py --aggregate`, the per-replication rep*.csv, and result/<experiment>/cost/cost.csv); the
manuscript transcribes these outputs verbatim (see the %% comments in ../Paper/main.tex).
"""
import argparse

from experiment_runner import result_directory

import pandas as pd

LABEL = {"MFI": "MFI", "FIWL": "FIWL", "MPWI": "MPWI", "MEPV": "MEPV",
         "existing": "既存", "proposed": "提案"}
REWARD_LABEL = {"prec_gain": "提案手法（式\\eqref{eq:reward}）", "var_reduction": "対数分散減少",
                "fi_ref": "参照能力値FI", "fi_hat_prev": "既存手法（式\\eqref{eq:reward-prev}）",
                "fi_hat_post": "反応後推定値FI", "err_reduction_ref": "参照誤差減少", "neg_sq_err_ref": "参照負二乗誤差"}
FACTOR_LABEL = {"constraint": "非負制約", "state": "状態表現", "reward": "報酬", "gamma": "割引率", "learning": "学習設定群"}


RESULT_ROOT = result_directory(__package__)


def load(grid):
    agg = pd.read_csv(RESULT_ROOT / grid / "mean.csv")
    reps = pd.concat([pd.read_csv(f) for f in sorted((RESULT_ROOT / grid).glob("rep*.csv"))])
    return agg, reps


def cell(v, bold):
    return ("\\textbf{%.3f}" % v) if bold else ("%.3f" % v)


def is_best(v, best):
    """Bold rule: equal to the column minimum at the displayed precision (three decimals)."""
    return round(v, 3) == round(best, 3)


def rows(agg, conds, steps, labels, bold_steps=None, items_col=True):
    """One LaTeX row per condition; every value equal to the best at each step (three decimals) in bold."""
    bold_steps = steps if bold_steps is None else bold_steps
    vals = {c: [agg[(agg.condition == c) & (agg.step == s)].iloc[0] for s in steps] for c in conds}
    best = [min(vals[c][k].rmse_mean for c in conds) for k in range(len(steps))]
    out = []
    for c in conds:
        cells = [cell(v.rmse_mean, steps[k] in bold_steps and is_best(v.rmse_mean, best[k])) for k, v in enumerate(vals[c])]
        line = f"{labels.get(c, c)} & " + " & ".join(cells)
        if items_col:
            line += f" & {vals[c][-1].distinct_items_mean:.0f}"
        out.append(line + "\\\\")
    return out


def diffs(reps, base, conds, steps):
    """Paired differences (cond - base) of the RMSE on the same replications: mean, range, sign agreement."""
    piv = reps.pivot_table(index=["rep", "step"], columns="condition", values="rmse")
    for c in conds:
        d = (piv[c] - piv[base]).unstack("rep")
        for s in steps:
            row = d.loc[s]
            print(f"  {c:22s} - {base:8s} step {s:2d}: mean {row.mean():+.4f}  range [{row.min():+.4f}, {row.max():+.4f}]  "
                  f"all<0: {bool((row < 0).all())}  all>0: {bool((row > 0).all())}")


def maxabs_diff(reps, a, b, step_from, step_to):
    piv = reps.pivot_table(index=["rep", "step"], columns="condition", values="rmse")
    d = (piv[a] - piv[b]).groupby("step").mean().loc[step_from:step_to]
    return float(d.abs().max()), int(d.abs().idxmax())


def main_table():
    agg, reps = load("main")
    conds = ["MFI", "FIWL", "MPWI", "MEPV", "existing", "proposed"]
    steps = [1, 3, 5, 10, 20, 30, 40]
    print("\n".join(rows(agg, conds, steps, LABEL)))
    print("%% reps:", sorted(reps.rep.unique().tolist()), " reference:", reps.groupby("rep").reference.first().round(4).tolist())
    print("%% rmse SD@40:", {c: round(float(agg[(agg.condition == c) & (agg.step == 40)].rmse_sd.iloc[0]), 3) for c in conds})
    print("%% max |mean diff| FIWL-MFI steps 10-40:", maxabs_diff(reps, "FIWL", "MFI", 10, 40))
    print("%% paired differences vs MFI")
    diffs(reps, "MFI", [c for c in conds if c != "MFI"], [1, 3, 5, 10, 20, 40])
    print("%% paired differences vs MEPV")
    diffs(reps, "MEPV", ["proposed", "MPWI"], [5, 10, 20, 40])


def sensitivity_table():
    agg, reps = load("sensitivity")
    steps = [5, 10, 20, 40]
    rewards = ["prec_gain", "var_reduction", "fi_ref", "fi_hat_prev", "fi_hat_post", "err_reduction_ref", "neg_sq_err_ref"]
    gammas = [0.0, 0.5, 0.9, 1.0]
    conds = [f"{r}_g{g}" for r in rewards for g in gammas]
    vals = {c: [agg[(agg.condition == c) & (agg.step == s)].iloc[0] for s in steps] for c in conds}
    best = [min(vals[c][k].rmse_mean for c in conds) for k in range(4)]
    for r in rewards:
        for gi, g in enumerate(gammas):
            c = f"{r}_g{g}"
            cells = [cell(v.rmse_mean, is_best(v.rmse_mean, best[k])) for k, v in enumerate(vals[c])]
            print(f"{REWARD_LABEL[r] if gi == 0 else ''} & {g:g} & " + " & ".join(cells) + f" & {vals[c][-1].distinct_items_mean:.0f}\\\\")
        print("\\hline")
    print("%% MFI:", agg[(agg.condition == "MFI") & agg.step.isin(steps)].rmse_mean.round(3).tolist(),
          " MEPV:", agg[(agg.condition == "MEPV") & agg.step.isin(steps)].rmse_mean.round(3).tolist())
    print("%% SD@5 per reward (max over gamma):", {r: round(float(max(vals[f'{r}_g{g}'][0].rmse_sd for g in gammas)), 3) for r in rewards})
    print("%% paired diffs vs MFI")
    diffs(reps, "MFI", ["prec_gain_g0.0", "prec_gain_g0.5", "fi_ref_g0.0", "fi_ref_g0.5", "fi_hat_prev_g0.0"], steps)
    print("%% prec_gain gamma 0 vs 0.5")
    diffs(reps, "prec_gain_g0.5", ["prec_gain_g0.0"], steps)


def state_table():
    agg, reps = load("state")
    steps = [5, 10, 20, 40]
    conds = ["state_theta", "state_theta_step", "state_belief"]
    print("\n".join(rows(agg, conds, steps, {"state_theta": "A", "state_theta_step": "B", "state_belief": "C"})))
    print("%% MFI:", agg[(agg.condition == "MFI") & agg.step.isin(steps)].rmse_mean.round(3).tolist())
    diffs(reps, "state_belief", ["state_theta", "state_theta_step"], steps)
    diffs(reps, "MFI", conds, steps)


def ablation_table():
    agg, reps = load("ablation")
    steps = [5, 10, 20, 40]
    from .run_grid import ablation_grid
    conds = list(ablation_grid())
    labels = {}
    for c in conds:
        if c in ("existing", "proposed"):
            labels[c] = "既存" if c == "existing" else "提案"
        elif "+" in c:
            labels[c] = "既存＋" + FACTOR_LABEL[c.split("+")[1]]
        else:
            labels[c] = "提案−" + FACTOR_LABEL[c.split("-")[1]]  # U+2212 (JIS minus sign), as in the manuscript
    lines = rows(agg, ["MFI"] + conds, steps, {"MFI": "MFI", **labels})
    for i, l in enumerate(lines):
        print(l)
        if i in (0, 6):
            print("\\hline")
    print("%% grad_steps:", reps[reps.step == 40].groupby("condition").grad_steps.mean().dropna().round(0).to_dict())
    diffs(reps, "existing", [c for c in conds if c.startswith("existing+")], [5, 40])
    diffs(reps, "proposed", [c for c in conds if c.startswith("proposed-")], [5, 40])


def guess_table():
    """Data-generating model with guessing. Bold from step 3 on (all posterior rules tie at step 1)."""
    agg, reps = load("guess")
    conds = ["MFI", "FIWL", "MPWI", "MEPV", "proposed"]
    steps = [1, 3, 5, 10, 20, 30, 40]
    print("\n".join(rows(agg, conds, steps, LABEL, bold_steps=steps[1:])))
    print("%% reps:", sorted(reps.rep.unique().tolist()))
    print("%% paired differences")
    diffs(reps, "MFI", ["MPWI", "MEPV", "proposed"], steps)
    diffs(reps, "MEPV", ["proposed", "MPWI"], steps)


def cost_table():
    df = pd.read_csv(RESULT_ROOT / "cost" / "cost.csv")
    for rule in ["MFI", "FIWL", "MPWI", "MEPV", "DQN (proposed)"]:
        cells = []
        for batch in [1, 2000]:
            for n in [500, 2000, 8000]:
                v = df[(df.rule == rule) & (df.batch == batch) & (df.n_items == n)].us_per_examinee_step.iloc[0]
                cells.append(f"{v:.0f}" if v >= 10 else f"{v:.1f}")
        print(f"{'提案' if rule.startswith('DQN') else rule} & " + " & ".join(cells) + "\\\\")


def main(argv=None):
    global RESULT_ROOT
    tables = {"main": main_table, "sensitivity": sensitivity_table, "state": state_table,
              "ablation": ablation_table, "guess": guess_table, "cost": cost_table}
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", choices=list(tables))
    ap.add_argument("--out", help="input result root (default: result); reads <root>/<experiment>/<grid>")
    args = ap.parse_args(argv)
    RESULT_ROOT = result_directory(__package__, args.out)
    tables[args.grid]()


if __name__ == "__main__":
    main()
