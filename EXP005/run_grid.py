"""Run the six main real-response conditions, one DQN replication per process."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from experiment_runner import PROJECT_ROOT, result_directory
from .data import DATASETS, SPLIT_SEED, load_dataset, split_respondents
from .dqn import DQNAgent, DQNConfig
from .rules import MEPVPolicy, MFIPolicy, WeightedInfoPolicy
from .simulate import metrics_by_step, run_cat


EXISTING = dict(state="theta", reward="fi_hat_prev", positive="all",
    hidden=50, hidden2=30, gamma=0.1, buffer_size=1_000, n_env=1,
    target_every=40, eps_start=0.1, eps_end=0.1)
PROPOSED = dict(state="belief", reward="prec_gain", positive="none",
    hidden=64, hidden2=0, gamma=0.5, buffer_size=50_000, n_env=32,
    target_every=500, eps_start=1.0, eps_end=0.05)
CONDITIONS = {"existing": EXISTING, "proposed": PROPOSED}
RULES = ("MFI", "FIWL", "MPWI", "MEPV")
SHOW_STEPS = [5, 10, 20, 40]
LIMITATION = (
    "Preliminary comparison on a fixed item bank calibrated using all respondents, including "
    "test respondents; evaluation is not fully independent of test data and does not establish "
    "generalization to an uncalibrated population. theta_reference is full-response theta_EAP, "
    "not observed latent ability. Observed answers are assumed invariant to CAT presentation order."
)


def classical_policies(bank):
    return {"MFI": MFIPolicy(bank), "FIWL": WeightedInfoPolicy(bank, "fiwl"),
            "MPWI": WeightedInfoPolicy(bank, "mpwi"), "MEPV": MEPVPolicy(bank)}


@contextmanager
def result_lock(path):
    """Protect shared baseline caches and same-replication updates on Linux/macOS."""
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save_csv(frame, path):
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def output_directory(args):
    out = result_directory(__package__, args.out) / args.dataset
    if args.screening == "strict":
        out /= "strict"
    return out / ("main_quick" if args.quick else "main")


def config_for(args, condition):
    return DQNConfig(**CONDITIONS[condition], n_epochs=1 if args.quick else args.n_epochs,
        eval_every=args.eval_every, eval_batch_size=args.eval_batch_size,
        train_seed=args.train_seed + args.rep - 1, shuffle_seed=args.shuffle_seed)


def rep_split_seed(args):
    return args.split_seed + args.rep - 1


def _drop_v1_layout(out):
    """v1 (shared split, cached analytic.csv) と v2 (per-rep split) は非互換。検知したら黙って一掃する。"""
    metadata_path = out / "metadata.json"
    is_v1 = (out / "analytic.csv").exists()
    if not is_v1 and metadata_path.exists():
        try:
            is_v1 = json.loads(metadata_path.read_text()).get("schema_version") == 1
        except json.JSONDecodeError:
            pass
    if not is_v1:
        return
    for name in ("analytic.csv", "splits.csv", "metadata.json", "item_bank.csv"):
        (out / name).unlink(missing_ok=True)
    for pattern in ("rep*.csv", "splits_rep*.csv", "validation_*.csv", "model_*.pt",
                    ".rep*.lock", ".analytic.lock", "mean.csv", "mean.png"):
        for path in out.glob(pattern):
            path.unlink()


def prepare(args, out):
    data = load_dataset(args.dataset, args.data_root, args.test_length, args.responses, args.screening)
    seed = rep_split_seed(args)
    splits = split_respondents(data, seed)
    indices = {s: np.flatnonzero(splits.split.to_numpy() == s) for s in ("train", "validation", "test")}
    if args.quick:
        # Keep the per-rep split; small diagnostic subsets are isolated in main_quick/.
        indices = {s: idx[:128 if s == "train" else 64] for s, idx in indices.items()}
    splits["used_in_run"] = False
    splits.loc[np.concatenate(list(indices.values())), "used_in_run"] = True
    configs = {c: asdict(config_for(args, c)) for c in CONDITIONS}
    for config in configs.values():
        config["train_seed"] = args.train_seed  # record the base, independent of --rep
    metadata = dict(schema_version=2, dataset=data.name, source_sha256=data.source_hashes,
        screening=args.screening, test_length=args.test_length, split_seed=args.split_seed,
        split_seed_scheme="split_seed + rep - 1",
        configs=configs, quick=args.quick, n_items=len(data.bank), n_respondents=len(data.responses),
        excluded_item_ids=data.excluded_item_ids, reference="theta_reference = full-response theta_EAP",
        model_selection="maximum mean validation episode return; RMSE is diagnostic only",
        planned_dqn_replications=5, limitation=LIMITATION)
    manifest = out / "metadata.json"
    with result_lock(out / ".metadata.lock"):
        _drop_v1_layout(out)
        if manifest.exists() and json.loads(manifest.read_text()) != metadata:
            raise ValueError("Existing results use different data/settings. Use a separate --out root.")
        manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        save_csv(splits, out / f"splits_rep{args.rep}.csv")
        save_csv(pd.DataFrame({"item": data.item_ids, "a": data.bank[:, 0], "b": data.bank[:, 1]}),
                 out / "item_bank.csv")
    print(f"{data.name}: {len(data.bank)} CAT items, {len(data.excluded_item_ids)} excluded; "
          + ", ".join(f"{s}={len(v)}" for s, v in indices.items())
          + f"; rep {args.rep} split_seed={seed}", flush=True)
    print(LIMITATION, flush=True)
    return data, indices


def metric_rows(condition, hist, items, theta_reference, extra=None):
    metrics = metrics_by_step(hist, theta_reference)
    return [dict(condition=condition, step=t + 1,
                 **{name: values[t] for name, values in metrics.items()},
                 distinct_items=len(np.unique(items)), **(extra or {})) for t in range(len(hist))]


def run_rep(args, out):
    data, indices = prepare(args, out)
    test_responses = data.responses[indices["test"]]
    reference = data.theta_reference[indices["test"]]
    seed = rep_split_seed(args)
    conditions = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    path = out / f"rep{args.rep}.csv"
    with result_lock(out / f".rep{args.rep}.lock"):
        analytic_rows = []
        for name, policy in classical_policies(data.bank).items():
            hist, items, _, _ = run_cat(policy, data.bank, test_responses,
                args.test_length, args.eval_batch_size)
            analytic_rows.extend(metric_rows(name, hist, items, reference,
                dict(dataset=data.name, rep=args.rep, split_seed=seed,
                     reference_kind="theta_full_EAP")))
            print(f"rep {args.rep}: {name:10s} RMSE@{args.test_length} "
                  f"{analytic_rows[-1]['rmse']:.3f}", flush=True)
        frame = pd.DataFrame(analytic_rows)
        if path.exists():
            old = pd.read_csv(path)
            frame = pd.concat([old[~old.condition.isin(RULES)], frame], ignore_index=True)
        save_csv(frame.sort_values(["condition", "step"]), path)
        for condition in conditions:
            started = time.monotonic()
            cfg = config_for(args, condition)
            print(f"rep {args.rep}: {condition}, train_seed={cfg.train_seed}, "
                  f"shuffle_seed={cfg.shuffle_seed}, epochs={cfg.n_epochs}", flush=True)
            agent = DQNAgent(data.bank, args.test_length, cfg)
            best = agent.train(data.responses[indices["train"]], data.responses[indices["validation"]],
                               data.theta_reference[indices["validation"]])
            hist, items, _, _ = run_cat(agent, data.bank, test_responses,
                                      args.test_length, args.eval_batch_size)
            rows = metric_rows(condition, hist, items, reference, dict(dataset=data.name,
                rep=args.rep, train_seed=cfg.train_seed, shuffle_seed=cfg.shuffle_seed,
                split_seed=seed, n_epochs=cfg.n_epochs, episodes=agent.episodes,
                transitions=agent.transitions, grad_steps=agent.grad_steps, best_val_return=best,
                best_epoch=agent.best_epoch, best_episodes=agent.best_episodes,
                reference_kind="theta_full_EAP"))
            frame = pd.DataFrame(rows)
            if path.exists():
                old = pd.read_csv(path)
                frame = pd.concat([old[old.condition != condition], frame], ignore_index=True)
            save_csv(frame.sort_values(["condition", "step"]), path)
            save_csv(pd.DataFrame(agent.validation_log), out / f"validation_{condition}_rep{args.rep}.csv")
            torch.save(dict(state_dict=agent.q.state_dict(), config=asdict(cfg),
                item_ids=data.item_ids.tolist(), dataset=data.name, test_length=args.test_length,
                source_sha256=data.source_hashes, best_val_return=best, best_epoch=agent.best_epoch,
                best_episodes=agent.best_episodes), out / f"model_{condition}_rep{args.rep}.pt")
            print(f"rep {args.rep}: {condition:10s} RMSE@{args.test_length} {rows[-1]['rmse']:.3f}; "
                  f"{agent.episodes} episodes, {agent.transitions} transitions, "
                  f"{time.monotonic() - started:.1f}s", flush=True)


def plot_mean(agg, out):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    fig = Figure(figsize=(9, 5.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for condition in (*RULES, *CONDITIONS):
        sub = agg[agg.condition == condition].sort_values("step")
        if len(sub):
            ax.plot(sub.step, sub.rmse_mean, label=condition, linewidth=1.8)
    ax.set(xlabel="step", ylabel="RMSE against full-response EAP",
           title=f"EXP005 / {out.parent.name} / {out.name}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.savefig(out / "mean.png", dpi=180)


def aggregate(out):
    paths = sorted(out.glob("rep[0-9]*.csv"))
    if not paths:
        raise ValueError(f"No results to aggregate in {out}")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if df.duplicated(["rep", "condition", "step"]).any():
        raise ValueError("Duplicate result rows")
    groups = df.groupby(["condition", "step"])
    fields = {"rmse_mean": groups.rmse.mean(), "rmse_sd": groups.rmse.std(ddof=1),
        "rmse_min": groups.rmse.min(), "rmse_max": groups.rmse.max(),
        "distinct_items_mean": groups.distinct_items.mean(), "n_rep": groups.rmse.count()}
    for metric in ("mae", "bias", "correlation"):
        values = groups[metric]
        fields[f"{metric}_mean"], fields[f"{metric}_sd"] = values.mean(), values.std(ddof=1)
    agg = pd.DataFrame(fields).reset_index()
    with result_lock(out / ".aggregate.lock"):
        save_csv(agg, out / "mean.csv")
        plot_mean(agg, out)
    print(LIMITATION)
    print("All conditions are re-evaluated per rep because the test split is resampled "
          "(split_seed = base + rep - 1). DQN target: 5 replications.")
    steps = [s for s in SHOW_STEPS if s in set(agg.step)] or [int(agg.step.max())]
    print(agg[agg.step.isin(steps)].to_string(index=False))
    return agg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--grid", default="main", choices=["main"])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--rep", type=int, choices=range(1, 6))
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--responses", type=Path, help="explicit wide response CSV (id column required)")
    parser.add_argument("--screening", choices=["positive", "strict"], default="positive",
                        help="strict is a separate sensitivity analysis: a>=0.2 and |b|<=4")
    parser.add_argument("--test-length", type=int, default=40)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--shuffle-seed", type=int, default=20260904)
    parser.add_argument("--train-seed", "--seed", type=int, default=20260904)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="additional validation interval in respondents; 0 = epoch ends only")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--conditions", help="existing,proposed or either condition")
    parser.add_argument("--quick", action="store_true", help="128/64/64 people, one epoch; separate main_quick output")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, help="output root; saves under <root>/EXP005/<dataset>/main")
    args = parser.parse_args(argv)
    if min(args.test_length, args.n_epochs, args.threads, args.eval_batch_size) < 1:
        parser.error("test length, epochs, threads and evaluation batch size must be positive")
    if min(args.split_seed, args.shuffle_seed, args.train_seed, args.eval_every) < 0:
        parser.error("seeds and eval-every must be nonnegative")
    if args.conditions:
        names = args.conditions.split(",")
        if len(set(names)) != len(names) or not set(names).issubset(CONDITIONS):
            parser.error("--conditions must be existing,proposed or either condition")
    torch.set_num_threads(args.threads)
    out = output_directory(args)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if args.aggregate:
            aggregate(out)
        else:
            run_rep(args, out)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
