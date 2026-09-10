"""2-parameter logistic model (2PLM): bank, probabilities, information, EAP, grid posterior."""
import numpy as np

THETA_MIN, THETA_MAX = -4.0, 4.0
GRID = np.linspace(THETA_MIN, THETA_MAX, 81)


def make_bank(n_items, rng, a_mean=1.2, a_sd=0.25, b_sd=1.0):
    """Item bank as an (n_items, 2) array of [a, b]. a is resampled until positive."""
    a = rng.normal(a_mean, a_sd, size=n_items)
    while np.any(a <= 0):
        bad = a <= 0
        a[bad] = rng.normal(a_mean, a_sd, size=bad.sum())
    b = rng.normal(0.0, b_sd, size=n_items)
    return np.column_stack([a, b])


def prob(a, b, theta, c=0.0):
    """P(correct) under the 3PLM (c = 0 gives the 2PLM). Arguments broadcast."""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def info(a, b, theta):
    """Fisher information of 2PLM items at theta."""
    p = prob(a, b, theta)
    return a * a * p * (1.0 - p)


def eap(bank, items, resp):
    """Posterior mean under N(0, 1), evaluated on the 81-point grid [-4, 4].

    The same estimator handles mixed and all-correct/all-incorrect histories.
    Before any responses, the symmetric prior has mean zero.
    """
    if items.shape[1] == 0:
        return np.zeros(items.shape[0])
    return posterior(bank, items, resp)[0]


def loglik_grid(bank, items, resp, grid=GRID):
    """Log-likelihood of each examinee's history on the theta grid: (n, G)."""
    n, t = items.shape
    if t == 0:
        return np.zeros((n, len(grid)))
    a = bank[items, 0][:, :, None]
    b = bank[items, 1][:, :, None]
    z = a * (grid[None, None, :] - b)
    logp = -np.logaddexp(0.0, -z)  # log P(correct)
    u = resp[:, :, None]
    return np.sum(u * logp + (1 - u) * (logp - z), axis=1)


def posterior(bank, items, resp, grid=GRID, prior_sd=1.0):
    """Grid posterior of theta under a N(0, prior_sd^2) prior.

    Returns (mean, var, w_lik, w_post): posterior mean and variance (n,), the normalised
    likelihood weights (n, G) used by FIWL, and the normalised posterior weights (n, G).
    """
    ll = loglik_grid(bank, items, resp, grid)
    w_lik = np.exp(ll - ll.max(axis=1, keepdims=True))
    w_lik /= w_lik.sum(axis=1, keepdims=True)
    log_prior = -0.5 * (grid / prior_sd) ** 2
    lp = ll + log_prior[None, :]
    w_post = np.exp(lp - lp.max(axis=1, keepdims=True))
    w_post /= w_post.sum(axis=1, keepdims=True)
    mean = w_post @ grid
    var = w_post @ (grid**2) - mean**2
    return mean, np.maximum(var, 1e-12), w_lik, w_post
