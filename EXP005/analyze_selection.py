"""Analyze items selected from real EXP005 responses by MFI, MEPV and the proposed DQN."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiment_runner import PROJECT_ROOT, result_directory
from .data import DATASETS, SPLIT_SEED, load_dataset, split_respondents
from .dqn import DQNAgent, DQNConfig
from .irt import info, posterior
from .rules import MEPVPolicy, MFIPolicy
from .simulate import run_cat


METHODS = ("MFI", "MEPV", "DQN")
COLORS = {"MFI": "#2a78d6", "MEPV": "#1baf7a", "DQN": "#eb6834"}
LABELS = {"MFI": "MFI", "MEPV": "MEPV", "DQN": "Proposed"}


def trace_run(policy, bank, responses, test_length, batch_size):
    chunks = []

    def wrapped(theta_hat, step, mask, items, resp):
        chunks.append(theta_hat.copy())
        return policy(theta_hat, step, mask, items, resp)

    history, items, resp, _ = run_cat(
        wrapped, bank, responses, test_length, batch_size=batch_size)
    previous = np.empty_like(history)
    chunk = 0
    for start in range(0, len(responses), batch_size):
        stop = min(start + batch_size, len(responses))
        for step in range(test_length):
            previous[step, start:stop] = chunks[chunk]
            chunk += 1
    return history, items, resp, previous


def records(method, bank, item_ids, theta_reference, history, items, resp, previous):
    length, n = history.shape
    a, b = bank[items, 0].T, bank[items, 1].T
    correct = np.cumsum(resp, axis=1).T
    n_before = np.arange(length)[:, None]
    correct_before = np.vstack([np.zeros((1, n)), correct[:-1]])
    extreme = (n_before > 0) & ((correct_before == 0) | (correct_before == n_before))
    return pd.DataFrame(dict(
        method=method,
        examinee=np.tile(np.arange(n), length),
        step=np.repeat(np.arange(1, length + 1), n),
        theta_reference=np.tile(theta_reference, length),
        theta_hat_prev=previous.ravel(),
        theta_hat_post=history.ravel(),
        item=np.asarray(item_ids)[items].T.ravel(),
        a=a.ravel(),
        b=b.ravel(),
        resp=resp.T.ravel(),
        extreme_at_selection=extreme.ravel(),
        fi_reference=info(a, b, theta_reference[None, :]).ravel(),
    ))


def summarize(frame):
    rows = []
    for (method, step), data in frame.groupby(["method", "step"]):
        squared_error = (data.theta_hat_post - data.theta_reference) ** 2
        x = data.theta_hat_prev.to_numpy()
        slope = np.nan if np.all(x == x[0]) else np.polyfit(x, data.b, 1)[0]
        rows.append(dict(
            method=method,
            step=step,
            a_mean=data.a.mean(),
            b_mean=data.b.mean(),
            abs_b_minus_reference_mean=(data.b - data.theta_reference).abs().mean(),
            slope_b_on_prev=slope,
            fi_reference_mean=data.fi_reference.mean(),
            p_correct=data.resp.mean(),
            rmse_reference=np.sqrt(squared_error.mean()),
            extreme_frac=data.extreme_at_selection.mean(),
            extreme_share_of_sq_error=(
                squared_error[data.extreme_at_selection].sum() / squared_error.sum()
                if squared_error.sum() > 0 else np.nan),
        ))
    return pd.DataFrame(rows)


def reward_scale(bank, items, resp):
    length = items.shape[1]
    variance = np.column_stack([
        posterior(bank, items[:, :step], resp[:, :step])[1]
        for step in range(length + 1)
    ])
    precision = 1 / variance[:, 1:] - 1 / variance[:, :-1]
    log_variance = np.log(variance[:, :-1]) - np.log(variance[:, 1:])
    return pd.DataFrame(dict(
        step=np.arange(1, length + 1),
        prec_gain_mean=precision.mean(0),
        prec_gain_sd=precision.std(0),
        prec_gain_neg_frac=(precision < 0).mean(0),
        logvar_mean=log_variance.mean(0),
        logvar_sd=log_variance.std(0),
    ))


def plot_summary(summary, path, max_step=20):
    panels = [
        ("slope_b_on_prev", "Slope of selected b on current estimate"),
        ("abs_b_minus_reference_mean", "Mean |b - full-response EAP|"),
        ("a_mean", "Mean a of selected item"),
        ("extreme_frac", "Fraction all-correct / all-incorrect"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0))
    for ax, (column, title) in zip(axes.ravel(), panels):
        for method in METHODS:
            data = summary[(summary.method == method) & (summary.step <= max_step)]
            ax.plot(data.step, data[column], label=LABELS[method], color=COLORS[method],
                    linewidth=1.3, marker="o", markersize=3)
        ax.set(title=title, xlabel="Step")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_agent(model_path, bank, item_ids, dataset, test_length):
    if not model_path.is_file():
        raise FileNotFoundError(f"Run EXP005 proposed for this replication first: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("dataset") != dataset
            or checkpoint.get("test_length") != test_length
            or list(checkpoint.get("item_ids", [])) != list(item_ids)):
        raise ValueError(f"Model metadata does not match the selected data: {model_path}")
    agent = DQNAgent(bank, test_length, DQNConfig(**checkpoint["config"]))
    agent.q.load_state_dict(checkpoint["state_dict"])
    agent.q_target.load_state_dict(checkpoint["state_dict"])
    return agent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--rep", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--screening", choices=["positive", "strict"], default="positive")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--test-length", type=int, default=40)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    data = load_dataset(args.dataset, args.data_root, args.test_length, args.responses, args.screening)
    split = split_respondents(data, args.split_seed + args.rep - 1)
    test_indices = np.flatnonzero(split.split.to_numpy() == "test")
    if args.quick:
        test_indices = test_indices[:64]

    root = result_directory(__package__, args.out) / args.dataset
    if args.screening == "strict":
        root /= "strict"
    main_dir = root / ("main_quick" if args.quick else "main")
    output = root / ("selection_quick" if args.quick else "selection")
    output.mkdir(parents=True, exist_ok=True)
    agent = load_agent(main_dir / f"model_proposed_rep{args.rep}.pt", data.bank,
                       data.item_ids, args.dataset, args.test_length)

    frames = []
    for method, policy in (("MFI", MFIPolicy(data.bank)),
                           ("MEPV", MEPVPolicy(data.bank)), ("DQN", agent)):
        history, items, resp, previous = trace_run(
            policy, data.bank, data.responses[test_indices], args.test_length,
            args.eval_batch_size)
        frames.append(records(method, data.bank, data.item_ids,
                              data.theta_reference[test_indices], history, items, resp, previous))
        if method == "MFI":
            reward_scale(data.bank, items, resp).to_csv(output / "reward_scale.csv", index=False)
        rmse = np.sqrt(np.mean((history[-1] - data.theta_reference[test_indices]) ** 2))
        print(f"{method}: RMSE@{args.test_length} against full-response EAP {rmse:.3f}")

    detail = pd.concat(frames, ignore_index=True)
    summary = summarize(detail)
    detail.to_csv(output / "selection_records.csv", index=False)
    summary.to_csv(output / "selection_by_step.csv", index=False)
    summary[summary.step.isin([2, 3, 5, 10])].to_csv(
        output / "extreme_patterns.csv", index=False)
    plot_summary(summary, output / "fig_selection.pdf")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
