"""Strict ID-aligned input and reproducible respondent splits for EXP005."""
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = tuple(f"choi_2026_cmsce_{year}" for year in ("2019_2", "2020_1", "2021_2"))
SPLIT_SEED = 20260904


@dataclass
class RealDataset:
    name: str
    bank: np.ndarray
    responses: np.ndarray
    respondent_ids: np.ndarray
    item_ids: np.ndarray
    theta_reference: np.ndarray
    excluded_item_ids: list[str]
    source_hashes: dict[str, str]


def _read_csv(path, required):
    # Check the original header before pandas can silently rename duplicate columns.
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    if not header or len(header) != len(set(header)) or any(not s.strip() for s in header):
        raise ValueError(f"{path}: empty or duplicate column IDs")
    if not set(required).issubset(header):
        raise ValueError(f"{path}: required columns are {required}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _check_ids(frame, column, path):
    ids = frame[column]
    if not len(ids) or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{path}: empty or duplicate {column} IDs")


def load_dataset(dataset, data_root, test_length=40, responses_path=None, screening="positive"):
    """Align all original IDs before filtering items; never infer or coerce IDs."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    if screening not in ("positive", "strict"):
        raise ValueError(f"unknown screening: {screening}")
    if test_length < 1:
        raise ValueError("test_length must be positive")
    directory = Path(data_root) / dataset
    item_path, person_path = directory / "item_parameters.csv", directory / "person_scores.csv"
    if responses_path is None:
        candidates = [directory / "responses.csv", directory / f"{dataset}_wide.csv"]
        responses_path = next((p for p in candidates if p.is_file()), None)
        if responses_path is None:
            raise FileNotFoundError(f"No response matrix in {directory}; provide --responses PATH")
    responses_path = Path(responses_path)
    items = _read_csv(item_path, ["item", "a", "b"])
    persons = _read_csv(person_path, ["id", "theta_EAP"])
    responses = _read_csv(responses_path, ["id"])
    for frame, column, path in ((items, "item", item_path), (persons, "id", person_path),
                                (responses, "id", responses_path)):
        _check_ids(frame, column, path)
    if set(persons.id) != set(responses.id):
        raise ValueError("person_scores.csv and response matrix respondent IDs do not match")
    if set(items.item) != set(responses.columns) - {"id"}:
        raise ValueError("item_parameters.csv and response matrix item IDs do not match")
    items = items.sort_values("item").reset_index(drop=True)
    persons = persons.sort_values("id").reset_index(drop=True)
    responses = responses.set_index("id").loc[persons.id, items.item]
    bank = items[["a", "b"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    theta_reference = pd.to_numeric(persons.theta_EAP, errors="raise").to_numpy(dtype=float)
    values = responses.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(bank).all() or not np.isfinite(theta_reference).all():
        raise ValueError("item parameters and theta_EAP must be finite, without missing values")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("response matrix must contain only 0/1, without missing values")
    keep = bank[:, 0] > 0
    if screening == "strict":
        keep &= (bank[:, 0] >= 0.2) & (np.abs(bank[:, 1]) <= 4)
    if keep.sum() < test_length:
        raise ValueError(f"Only {keep.sum()} usable items for test_length={test_length}")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
              (("items", item_path), ("persons", person_path), ("responses", responses_path))}
    return RealDataset(dataset, bank[keep], values[:, keep].astype(np.int8),
                       persons.id.to_numpy(), items.item.to_numpy()[keep], theta_reference,
                       items.item.to_numpy()[~keep].tolist(), hashes)


def split_respondents(data, split_seed=SPLIT_SEED):
    """70/10/20 split stratified by theta_EAP deciles, independent of training RNG.

    Repeated quantile edges are collapsed (constant scores form one stratum).
    Largest-remainder allocation preserves rounded global train/validation counts.
    """
    n = len(data.respondent_ids)
    if n < 3:
        raise ValueError("At least three respondents are needed for train/validation/test")
    strata = pd.qcut(pd.Series(data.theta_reference), 10, labels=False, duplicates="drop").fillna(0).to_numpy(int)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    counts = np.array([len(g) for g in groups])
    targets = [round(n * 0.7), round(n * 0.1)]
    if min(*targets, n - sum(targets)) < 1:
        raise ValueError("Too few respondents for nonempty 70/10/20 groups")

    def allocate(fraction, target, capacity):
        desired = counts * fraction
        result = np.minimum(np.floor(desired).astype(int), capacity)
        while result.sum() < target:
            priority = np.where(result < capacity, desired - result, -np.inf)
            result[np.argmax(priority)] += 1
        return result

    n_train = allocate(0.7, targets[0], counts)
    n_val = allocate(0.1, targets[1], counts - n_train)
    rng = np.random.default_rng(split_seed)
    labels = np.empty(n, dtype=object)
    for group, nt, nv in zip(groups, n_train, n_val):
        indices = rng.permutation(group)
        labels[indices[:nt]] = "train"
        labels[indices[nt:nt + nv]] = "validation"
        labels[indices[nt + nv:]] = "test"
    return pd.DataFrame({"id": data.respondent_ids, "theta_reference": data.theta_reference,
                         "stratum": strata, "split": labels})
