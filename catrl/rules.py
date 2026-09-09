"""Item selection rules: MFI, FIWL, MPWI and MEPV."""
import numpy as np

from .irt import GRID, info, posterior


class MFIPolicy:
    """Maximum Fisher information at the current estimate (Lord, 1980)."""

    def __init__(self, bank):
        self.bank = bank

    def __call__(self, theta_hat, step, mask, items, resp):
        fi = info(self.bank[:, 0][None, :], self.bank[:, 1][None, :], theta_hat[:, None])
        fi[mask] = -np.inf
        return fi.argmax(axis=1)


class WeightedInfoPolicy:
    """Information integrated over theta with history-dependent weights.

    kind='fiwl': likelihood weights, no prior (Veerkamp & Berger, 1997).
    kind='mpwi': posterior weights under a N(0,1) prior (van der Linden, 1998).
    With no responses yet the weights are flat (fiwl) or the prior (mpwi).
    """

    def __init__(self, bank, kind):
        self.bank, self.kind = bank, kind
        self.info_grid = info(bank[:, 0][None, :], bank[:, 1][None, :], GRID[:, None])  # (G, I)

    def __call__(self, theta_hat, step, mask, items, resp):
        _, _, w_lik, w_post = posterior(self.bank, items, resp)
        w = w_lik if self.kind == "fiwl" else w_post
        crit = w @ self.info_grid
        crit[mask] = -np.inf
        return crit.argmax(axis=1)


class MEPVPolicy:
    """Minimum expected posterior variance (Owen, 1975; van der Linden, 1998).

    For every candidate item the posterior variance after each of the two possible
    responses is computed in closed form from the grid posterior, and the item minimising
    the response-probability-weighted variance is selected. Fully vectorised: six
    (n, G) x (G, I) products per step.
    """

    def __init__(self, bank):
        from .irt import GRID, prob
        self.bank = bank
        p = prob(bank[:, 0][None, :], bank[:, 1][None, :], GRID[:, None])  # (G, I)
        g = GRID[:, None]
        self.P, self.Pt, self.Pt2 = p, g * p, g * g * p
        q = 1 - p
        self.Q, self.Qt, self.Qt2 = q, g * q, g * g * q

    def __call__(self, theta_hat, step, mask, items, resp):
        _, _, _, w = posterior(self.bank, items, resp)  # (n, G)
        z1, m1, s1 = w @ self.P, w @ self.Pt, w @ self.Pt2  # (n, I)
        z0, m0, s0 = w @ self.Q, w @ self.Qt, w @ self.Qt2
        var1 = s1 / z1 - (m1 / z1) ** 2
        var0 = s0 / z0 - (m0 / z0) ** 2
        expected = z1 * var1 + z0 * var0  # z1 + z0 = 1
        expected[mask] = np.inf
        return expected.argmin(axis=1)
