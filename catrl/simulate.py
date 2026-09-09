"""Vectorised CAT run for a batch of examinees under an arbitrary selection policy."""
import numpy as np

from .irt import mle, prob


def run_cat(select, bank, theta, test_length, seed, bank_true=None):
    """Administer `test_length` items to every examinee.

    select(theta_hat, step, mask, items, resp) -> item index per examinee, where
    items and resp are the (n, step) histories so far and mask marks used items.
    The initial estimate and the response uniforms come from streams derived from
    `seed` only, so different policies evaluated with the same seed face identical
    examinees and random numbers (common random numbers).
    Responses are generated from `bank_true` (the data-generating model, (I, 3) with guessing)
    when given, otherwise from `bank`; the estimator always uses `bank` (the estimation model).
    Returns theta_hat history (test_length, n), items (n, test_length), responses (n, test_length),
    and the initial estimates theta_hat_0 (n,).
    """
    if bank_true is None:
        bank_true = np.column_stack([bank, np.zeros(bank.shape[0])])
    n, n_items = len(theta), bank.shape[0]
    theta0 = np.random.default_rng([seed, 0]).uniform(-0.5, 0.5, size=n)
    uniforms = np.random.default_rng([seed, 1]).random((test_length, n))

    theta_hat = theta0.copy()
    mask = np.zeros((n, n_items), dtype=bool)
    items = np.zeros((n, test_length), dtype=np.int64)
    resp = np.zeros((n, test_length), dtype=np.int64)
    history = np.zeros((test_length, n))
    rows = np.arange(n)
    for t in range(test_length):
        j = select(theta_hat, t, mask, items[:, :t], resp[:, :t])
        p = prob(bank_true[j, 0], bank_true[j, 1], theta, bank_true[j, 2])
        items[:, t] = j
        resp[:, t] = uniforms[t] <= p
        mask[rows, j] = True
        theta_hat = mle(bank, items[:, : t + 1], resp[:, : t + 1], theta_hat)
        history[t] = theta_hat
    return history, items, resp, theta0


def rmse_by_step(history, theta):
    """RMSE of the estimates against theta at every step: (test_length,)."""
    return np.sqrt(((history - theta[None, :]) ** 2).mean(axis=1))
