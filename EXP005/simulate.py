"""CAT evaluation using observed responses; no sampled abilities or answers."""
import numpy as np

from .irt import mle


def run_cat(select, bank, response_matrix, test_length=40, batch_size=256):
    """Return estimates (L,n), item/response histories (n,L), and zero initial estimates.

    Batching bounds posterior-grid memory without changing respondent-level policies.
    No reference ability is passed to this environment or to the policy.
    """
    responses = np.asarray(response_matrix)
    if responses.ndim != 2 or responses.shape[1] != len(bank) or not len(responses):
        raise ValueError("response matrix must be nonempty with one column per bank item")
    if not np.isin(responses, [0, 1]).all():
        raise ValueError("responses must be 0/1 without missing values")
    if not 1 <= test_length <= len(bank) or batch_size < 1:
        raise ValueError("invalid test_length or batch_size")
    n = len(responses)
    history = np.empty((test_length, n))
    all_items = np.empty((n, test_length), dtype=np.int64)
    all_resp = np.empty_like(all_items)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        values = responses[start:stop]
        size = len(values)
        rows = np.arange(size)
        theta_hat = np.zeros(size)
        mask = np.zeros((size, len(bank)), dtype=bool)
        items, resp = all_items[start:stop], all_resp[start:stop]
        for t in range(test_length):
            j = np.asarray(select(theta_hat, t, mask, items[:, :t], resp[:, :t]))
            if (j.shape != (size,) or not np.issubdtype(j.dtype, np.integer)
                    or np.any(j < 0) or np.any(j >= len(bank))):
                raise ValueError("policy returned invalid item indices")
            if mask[rows, j].any():
                raise ValueError("policy selected an already administered item")
            items[:, t], resp[:, t] = j, values[rows, j]
            mask[rows, j] = True
            theta_hat = mle(bank, items[:, :t + 1], resp[:, :t + 1], theta_hat)
            history[t, start:stop] = theta_hat
    return history, all_items, all_resp, np.zeros(n)


def rmse_by_step(history, theta_reference):
    return np.sqrt(((history - theta_reference[None, :]) ** 2).mean(axis=1))


def metrics_by_step(history, theta_reference):
    errors = history - theta_reference[None, :]
    centered = history - history.mean(axis=1, keepdims=True)
    ref_centered = theta_reference - theta_reference.mean()
    denom = np.sqrt((centered ** 2).sum(axis=1) * (ref_centered ** 2).sum())
    correlation = np.full(len(history), np.nan)
    np.divide(centered @ ref_centered, denom, out=correlation, where=denom > 0)
    return {"rmse": rmse_by_step(history, theta_reference),
            "mae": np.abs(errors).mean(axis=1), "bias": errors.mean(axis=1),
            "correlation": correlation}
