"""How do the proposed DQN, MFI and MEPV differ in the items they select? (section 6.6, fig:selection)

Trains one proposed agent (PROPOSED config of run_grid.py, replication-4 seed), then runs
MFI, MEPV and the DQN on the same examinees with common random numbers and records, per
examinee and step, the selected item's a and b, the estimate used for selection, the true
ability, whether the response pattern so far was all-correct or all-incorrect, and the item
information at the true ability.

Outputs (result/<experiment>/selection/): selection_by_step.csv, extreme_patterns.csv, reward_scale.csv,
fig_selection.pdf, log.txt (stdout)
"""
import argparse
from experiment_runner import result_directory

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dqn import DQNAgent, DQNConfig
from .irt import info, posterior
from .rules import MEPVPolicy, MFIPolicy
from .scenarios import make_scenario
from .simulate import rmse_by_step, run_cat
from .run_grid import PROPOSED

METHODS = ["MFI", "MEPV", "DQN"]
LEGEND = {"MFI": "MFI", "MEPV": "MEPV", "DQN": "Proposed"}
COLORS = {"MFI": "#2a78d6", "DQN": "#eb6834", "MEPV": "#1baf7a"}
MARKERS = {"MFI": "o", "DQN": "s", "MEPV": "^"}


def trace_run(policy, bank, theta, L, seed):
    prev = []

    def wrapped(theta_hat, step, mask, items, resp):
        prev.append(theta_hat.copy())
        return policy(theta_hat, step, mask, items, resp)

    hist, items, resp, theta0 = run_cat(wrapped, bank, theta, L, seed=seed)
    return hist, items, resp, np.array(prev)


def records(method, bank, theta, hist, items, resp, prev):
    L, n = hist.shape
    a, b = bank[items, 0].T, bank[items, 1].T
    cum = np.cumsum(resp, axis=1).T
    n_before = np.arange(L)[:, None]
    correct_before = np.vstack([np.zeros((1, n)), cum[:-1]])
    extreme = (n_before > 0) & ((correct_before == 0) | (correct_before == n_before))
    return pd.DataFrame(dict(
        method=method, examinee=np.tile(np.arange(n), L), step=np.repeat(np.arange(1, L + 1), n),
        theta=np.tile(theta, L), theta_hat_prev=prev.ravel(), theta_hat_post=hist.ravel(),
        a=a.ravel(), b=b.ravel(), resp=resp.T.ravel(), extreme_at_selection=extreme.ravel(),
        fi_true=info(a, b, theta[None, :]).ravel(),
    ))


def by_step(df):
    out = []
    for (m, t), d in df.groupby(["method", "step"]):
        err2 = (d.theta_hat_post - d.theta) ** 2
        # With EAP, all initial estimates are zero; a constant predictor has no slope.
        slope = np.polyfit(d.theta_hat_prev, d.b, 1)[0] if d.theta_hat_prev.nunique() > 1 else np.nan
        out.append(dict(
            method=m, step=t, a_mean=d.a.mean(), b_mean=d.b.mean(),
            abs_b_minus_theta_mean=(d.b - d.theta).abs().mean(),
            slope_b_on_prev=slope,
            fi_true_mean=d.fi_true.mean(), p_correct=d.resp.mean(), rmse_post=np.sqrt(err2.mean()),
            extreme_frac=d.extreme_at_selection.mean(),
            extreme_share_of_sq_error=err2[d.extreme_at_selection].sum() / err2.sum() if err2.sum() > 0 else np.nan,
        ))
    return pd.DataFrame(out)


def reward_scale(bank, items, resp, hist, theta0):
    """Per-step mean and SD of the precision-gain and log-variance-reduction rewards along
    the MFI trajectories, to document the scale of the rewards over the test."""
    L = items.shape[1]
    v = np.column_stack([posterior(bank, items[:, :t], resp[:, :t])[1] for t in range(L + 1)])  # (n, L+1)
    prec = 1 / v[:, 1:] - 1 / v[:, :-1]
    logv = np.log(v[:, :-1]) - np.log(v[:, 1:])
    return pd.DataFrame(dict(step=np.arange(1, L + 1), prec_gain_mean=prec.mean(0), prec_gain_sd=prec.std(0),
                             prec_gain_neg_frac=(prec < 0).mean(0), logvar_mean=logv.mean(0), logvar_sd=logv.std(0)))


def style(ax):
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9891")
    ax.tick_params(colors="#52514e", labelsize=8)


def fig_selection(summary, out, max_step=20):
    panels = [("slope_b_on_prev", "Slope of selected b on current estimate"),
              ("abs_b_minus_theta_mean", "Mean |b - true theta| of selected item"),
              ("a_mean", "Mean a of selected item"),
              ("extreme_frac", "Fraction all-correct / all-incorrect at selection")]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for m in METHODS:
            d = summary[(summary.method == m) & (summary.step <= max_step)]
            ax.plot(d.step, d[col], color=COLORS[m], marker=MARKERS[m], markersize=3.5, linewidth=1.3, label=LEGEND[m])
        ax.set_title(title, fontsize=9, color="#0b0b0b")
        ax.set_xlabel("Step", fontsize=8, color="#52514e")
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_selection.pdf")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=500)
    ap.add_argument("--test-length", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--rep", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", help="output root (default: result); saves under <root>/<experiment>/selection")
    args = ap.parse_args(argv)
    out = result_directory(__package__, args.out) / "selection"
    out.mkdir(parents=True, exist_ok=True)
    L, seed = args.test_length, args.seed + args.rep - 1
    bank_true, bank, sample_theta = make_scenario("base", args.n_items, np.random.default_rng([args.seed, 99]), np.random.default_rng([seed, 30]))
    theta = sample_theta(np.random.default_rng([seed, 20]), args.n_test)
    eval_seed = seed + 100

    cfg = DQNConfig(seed=seed, **PROPOSED)
    if args.quick:
        cfg.n_episodes, cfg.eval_every, cfg.n_val = 200, 100, 100
    agent = DQNAgent(bank, L, cfg)
    best = agent.train(log=lambda s: None)
    print(f"proposed agent trained (best validation return {best:.3f}); config {PROPOSED}")

    dfs = []
    pols = {"MFI": MFIPolicy(bank), "MEPV": MEPVPolicy(bank), "DQN": agent}
    for name, pol in pols.items():
        hist, items, resp, prev = trace_run(pol, bank, theta, L, eval_seed)
        dfs.append(records(name, bank, theta, hist, items, resp, prev))
        if name == "MFI":
            reward_scale(bank, items, resp, hist, prev[0]).to_csv(out / "reward_scale.csv", index=False)
        print(f"{name}: RMSE@{L} {rmse_by_step(hist, theta)[-1]:.3f}")
    df = pd.concat(dfs, ignore_index=True)
    summary = by_step(df)
    summary.to_csv(out / "selection_by_step.csv", index=False)
    summary[summary.step.isin([2, 3, 5, 10])][["method", "step", "extreme_frac", "extreme_share_of_sq_error", "rmse_post"]] \
        .to_csv(out / "extreme_patterns.csv", index=False)
    fig_selection(summary, out)

    cols = ["step", "a_mean", "abs_b_minus_theta_mean", "slope_b_on_prev", "fi_true_mean", "rmse_post", "extreme_frac", "extreme_share_of_sq_error"]
    for m in METHODS:
        print(f"\n{m}")
        print(summary[(summary.method == m) & summary.step.isin([1, 2, 3, 4, 5, 7, 10, 15, 20, 40])][cols].to_string(index=False, float_format="%.3f"))
    rs = pd.read_csv(out / "reward_scale.csv")
    print("\nreward scale along MFI trajectories (mean, SD, fraction negative) at steps 1, 5, 10, 20, 40:")
    print(rs[rs.step.isin([1, 5, 10, 20, 40])].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
