"""Run a named grid of conditions on one simulated 2PLM bank, one replication per process.

  .venv/bin/python run_grid.py --grid main --rep 4            # one replication
  .venv/bin/python run_grid.py --grid main --aggregate        # mean / SD over replications

Grids (results go to result/<experiment>/<grid>/; section numbers refer to ../Paper/main.tex):
  main         existing and proposed                                        6.2  confirmatory, reps 4-6
  sensitivity  reward (7) x gamma (4) under the proposed state and learning  6.3  exploratory,  reps 1-3
  state        state A / B / C under the proposed reward                     6.4  exploratory,  reps 1-3
  ablation     one-factor swaps between existing and proposed (12 conditions) 6.5  exploratory,  reps 1-3
  guess        data-generating model = 3PLM with guessing, estimation model = 2PLM with the true
               a, b; the proposed agent learns from the 3PLM responses       6.7  confirmatory, reps 4-6
Every grid also runs the analytic rules MFI, FIWL, MPWI and MEPV on the same examinees.
  --conditions a,b   run only these DQN conditions and merge their rows into an existing rep<k>.csv
                     (the analytic rules are re-run only when the file does not exist)

Random streams. The bank is drawn from (base seed, 99) and is fixed. Replication k uses
seed_k = base + k - 1 for: the guessing draws ([seed_k, 30]), the evaluation examinees
([seed_k, 20]), the evaluation responses (run_cat streams [seed_k + 100, 0/1]) and the DQN streams.
"""
import argparse
import glob
import time
from experiment_runner import result_directory

import numpy as np
import pandas as pd
import torch

from .dqn import DQNAgent, DQNConfig
from .irt import info
from .rules import MEPVPolicy, MFIPolicy, WeightedInfoPolicy
from .scenarios import make_scenario
from .simulate import rmse_by_step, run_cat

# --- condition definitions --------------------------------------------------------------
EXISTING = dict(  # Wang et al. (2024) as described in their paper text
    state="theta", reward="fi_hat_prev", positive="all",
    hidden=50, hidden2=30, gamma=0.1, buffer_size=1_000, n_episodes=1_000, n_env=1,
    target_every=40, eps_start=0.1, eps_end=0.1, eval_every=50, n_val=200,
)
PROPOSED = dict(  # proposed: state C (theta_hat, t/L, log posterior variance), precision-gain reward, no constraint
    state="belief", reward="prec_gain", positive="none",
    hidden=64, hidden2=0, gamma=0.5, buffer_size=50_000, n_episodes=20_000, n_env=32,
    target_every=500, eps_start=1.0, eps_end=0.05, eval_every=2_000, n_val=500,
)
REWARDS = ["prec_gain", "var_reduction", "fi_ref", "fi_hat_prev", "fi_hat_post", "err_reduction_ref", "neg_sq_err_ref"]
GAMMAS = [0.0, 0.5, 0.9, 1.0]
FACTORS = {  # factors separating EXISTING from PROPOSED (ablation grid)
    "constraint": ["positive"],
    "state": ["state"],
    "reward": ["reward"],
    "gamma": ["gamma"],
    "learning": ["hidden", "hidden2", "buffer_size", "n_episodes", "n_env", "target_every",
                 "eps_start", "eps_end", "eval_every", "n_val"],
}


def ablation_grid():
    def build(base, other, flipped):
        kw = dict(base)
        if flipped:
            for key in FACTORS[flipped]:
                kw[key] = other[key]
        return kw
    g = {"existing": build(EXISTING, PROPOSED, None)}
    for f in FACTORS:
        g[f"existing+{f}"] = build(EXISTING, PROPOSED, f)
    g["proposed"] = build(PROPOSED, EXISTING, None)
    for f in FACTORS:
        g[f"proposed-{f}"] = build(PROPOSED, EXISTING, f)
    return g


GRIDS = {
    "sensitivity": {f"{r}_g{g}": dict(PROPOSED, reward=r, gamma=g) for r in REWARDS for g in GAMMAS},
    "state": {f"state_{s}": dict(PROPOSED, state=s) for s in ["theta", "theta_step", "belief"]},
    "ablation": ablation_grid(),
    "main": {"existing": EXISTING, "proposed": PROPOSED},
    "guess": {"proposed": PROPOSED},
}
SCENARIO = {"guess": "guess"}  # every other grid uses the "base" scenario
SHOW_STEPS = [5, 10, 20, 40]


def classical_policies(bank):
    return {"MFI": MFIPolicy(bank), "FIWL": WeightedInfoPolicy(bank, "fiwl"),
            "MPWI": WeightedInfoPolicy(bank, "mpwi"), "MEPV": MEPVPolicy(bank)}


def run_rep(args, out):
    rep = args.rep
    seed = args.seed + rep - 1
    L = args.test_length
    bank_true, bank, sample_theta = make_scenario(
        SCENARIO.get(args.grid, "base"), args.n_items, np.random.default_rng([args.seed, 99]), np.random.default_rng([seed, 30]))
    theta = sample_theta(np.random.default_rng([seed, 20]), args.n_test)
    eval_seed = seed + 100
    np.savetxt(out / f"bank_true_rep{rep}.csv", bank_true, delimiter=",", header="a,b,c", comments="")
    rows = []

    def record(cond, hist, items, extra=None):
        rmse = rmse_by_step(hist, theta)
        for t in range(L):
            rows.append(dict(rep=rep, condition=cond, step=t + 1, rmse=rmse[t],
                             distinct_items=len(np.unique(items)), **(extra or {})))

    # asymptotic reference value: RMSE implied by the L most informative items at the true ability
    fi = info(bank_true[:, 0][None, :], bank_true[:, 1][None, :], theta[:, None])
    ref = float(np.sqrt(np.mean(1 / np.sort(fi, axis=1)[:, -L:].sum(axis=1))))
    print(f"rep {rep}: asymptotic reference value from true-theta information {ref:.3f}", flush=True)
    existing = out / f"rep{rep}.csv"
    conds = GRIDS[args.grid]
    if args.conditions:
        conds = {c: conds[c] for c in args.conditions.split(",")}
    if not (args.conditions and existing.exists()):
        for name, pol in classical_policies(bank).items():
            hist, items, _, _ = run_cat(pol, bank, theta, L, seed=eval_seed, bank_true=bank_true)
            record(name, hist, items, {"reference": ref})
            print(f"rep {rep}: {name:22s} RMSE@{L} {rows[-1]['rmse']:.3f}  items {rows[-1]['distinct_items']}", flush=True)

    for cond, kw in conds.items():
        cfg = DQNConfig(seed=seed, **kw)
        if args.quick:
            cfg.n_episodes, cfg.eval_every, cfg.n_val = 200, 100, 100
        t0 = time.time()
        agent = DQNAgent(bank, L, cfg, bank_true=bank_true, sample_theta=sample_theta)
        best = agent.train(log=lambda s: None)
        hist, items, _, _ = run_cat(agent, bank, theta, L, seed=eval_seed, bank_true=bank_true)
        record(cond, hist, items, {"reference": ref, "grad_steps": agent.grad_steps, "best_val_return": best})
        print(f"rep {rep}: {cond:22s} RMSE@{L} {rows[-1]['rmse']:.3f}  items {rows[-1]['distinct_items']}  "
              f"val return {best:.3f}  updates {agent.grad_steps}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if args.conditions and existing.exists():  # merge into the recorded replication
        old = pd.read_csv(existing)
        df = pd.concat([old[~old.condition.isin(conds)], df], ignore_index=True)
    df.to_csv(existing, index=False)


def aggregate(out):
    files = sorted(glob.glob(str(out / "rep*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files])
    g = df.groupby(["condition", "step"])
    agg = pd.DataFrame({"rmse_mean": g.rmse.mean(), "rmse_sd": g.rmse.std(ddof=1), "rmse_min": g.rmse.min(),
                        "rmse_max": g.rmse.max(), "distinct_items_mean": g.distinct_items.mean(), "n_rep": g.rmse.count()})
    agg = agg.reset_index()
    agg.to_csv(out / "mean.csv", index=False)
    order = list(dict.fromkeys(df.condition))
    print(f"RMSE mean (SD) over {df.rep.nunique()} replications (reps {sorted(df.rep.unique().tolist())}); steps {SHOW_STEPS}; last = distinct items")
    for cond in order:
        sub = agg[agg.condition == cond]
        cells = [f"{sub[sub.step==s].iloc[0].rmse_mean:.3f} ({sub[sub.step==s].iloc[0].rmse_sd:.3f})" for s in SHOW_STEPS]
        print(f"{cond:24s} " + "  ".join(cells) + f"  {sub[sub.step==40].iloc[0].distinct_items_mean:6.1f}")
    print(f"reference value (mean over reps): {df.groupby('rep').reference.first().mean():.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=list(GRIDS))
    ap.add_argument("--rep", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--n-items", type=int, default=500)
    ap.add_argument("--test-length", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--quick", action="store_true", help="a few hundred training episodes per condition (smoke test)")
    ap.add_argument("--conditions", default=None, help="comma-separated subset of the grid's DQN conditions to run")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", help="output root (default: result); saves under <root>/<experiment>/<grid>")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    out = result_directory(__package__, args.out) / args.grid
    out.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate(out)
    elif args.rep:
        run_rep(args, out)
    else:
        ap.error("give --rep K or --aggregate")


if __name__ == "__main__":
    main()
