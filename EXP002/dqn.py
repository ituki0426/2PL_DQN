"""Deep Q-network item selection for CAT: the existing formulation (Wang, Liu, & Xu, 2024)
and the proposed one, selectable through DQNConfig.

State (cfg.state; A / B / C are the labels used in the paper):
  theta        [theta_hat]                                (existing; state A)
  theta_step   [theta_hat, t/L]                           (state B)
  belief       [theta_hat, t/L, log posterior variance]    (proposed; state C, the state that
                                                           reflects the uncertainty of the estimate)
Reward (cfg.reward):
  fi_hat_prev        information at the estimate used for selection (existing)
  fi_hat_post        information at the estimate after the response
  var_reduction      log V_{t-1} - log V_t, V = posterior variance under N(0, 1)
  prec_gain          1/V_t - 1/V_{t-1}  (proposed; the posterior-precision gain, sums to the
                     final posterior precision)
  fi_ref             information at the reference ability, i.e. the MLE from the examinee's
                     complete L-response record (assigned after the episode; sensitivity analysis only)
  err_reduction_ref  (theta_hat_{t-1} - ref)^2 - (theta_hat_t - ref)^2   (assigned after the episode)
  neg_sq_err_ref     -(theta_hat_t - ref)^2                               (assigned after the episode)
Constraint (cfg.positive): 'none' (proposed) or 'all' (every parameter clamped to be
non-negative after each optimiser step, as in the existing method).
Model selection: every `eval_every` training examinees the greedy policy is run on a fixed
validation set and the mean episodic return of the condition's own reward is recorded; the
parameters with the highest validation return are kept. No true ability is used anywhere.
Random streams: training and validation use streams derived from (cfg.seed, tag) so that they
never coincide with the evaluation streams of run_cat.
Device: use MPS when available, otherwise CPU. NumPy simulation and replay storage stay on
CPU; network inference and gradient updates run on the selected device.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .irt import info, mle, posterior, prob
from .simulate import rmse_by_step, run_cat


@dataclass
class DQNConfig:
    state: str = "belief"
    reward: str = "prec_gain"
    positive: str = "none"  # 'none' | 'all'
    hidden: int = 64
    hidden2: int = 0  # 0: same as hidden
    gamma: float = 0.5
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 50_000
    target_every: int = 500
    n_env: int = 32
    n_episodes: int = 20_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5
    n_val: int = 500
    eval_every: int = 2_000
    seed: int = 0


STATE_DIM = {"theta": 1, "theta_step": 2, "belief": 3}
DEFERRED = ("fi_ref", "err_reduction_ref", "neg_sq_err_ref")  # need the end-of-test estimate


class QNet(nn.Module):
    def __init__(self, n_in, h1, h2, n_items):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, n_items))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, bank, test_length, cfg: DQNConfig, bank_true=None, sample_theta=None):
        """bank: (I, 2) parameters of the estimation model (state, reward, estimator).
        bank_true: (I, 3) data-generating model of the training simulation (default: bank, no guessing).
        sample_theta(rng, n): training population (default N(0, 1))."""
        self.bank, self.L, self.cfg = bank, test_length, cfg
        self.bank_true = bank_true if bank_true is not None else np.column_stack([bank, np.zeros(bank.shape[0])])
        self.sample_theta = sample_theta or (lambda rng, n: rng.normal(size=n))
        self.deferred = cfg.reward in DEFERRED
        self.n_items = bank.shape[0]
        self.n_in = STATE_DIM[cfg.state]
        h2 = cfg.hidden2 or cfg.hidden
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.q = QNet(self.n_in, cfg.hidden, h2, self.n_items).to(self.device)
        self._clamp()  # constrain before the target copy so both nets start identical
        self.q_target = QNet(self.n_in, cfg.hidden, h2, self.n_items).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

    # ---- state / action -------------------------------------------------
    def features(self, theta_hat, step, log_var):
        cols = [theta_hat]
        if self.cfg.state != "theta":
            cols.append(np.full_like(theta_hat, step / self.L))
        if self.cfg.state == "belief":
            cols.append(log_var)
        return np.column_stack(cols).astype(np.float32)

    def greedy_from_features(self, feats, mask):
        with torch.no_grad():
            qv = self.q(torch.from_numpy(feats).to(self.device))
            qv = qv.masked_fill(torch.from_numpy(mask).to(self.device), -torch.inf)
            return qv.argmax(dim=1).cpu().numpy()

    def __call__(self, theta_hat, step, mask, items, resp):  # policy interface for run_cat
        log_var = np.log(posterior(self.bank, items, resp)[1]) if self.cfg.state == "belief" else None
        return self.greedy_from_features(self.features(theta_hat, step, log_var), mask)

    def _clamp(self):
        if self.cfg.positive == "all":
            with torch.no_grad():
                for p in self.q.parameters():
                    p.clamp_(min=0.0)

    def _reward(self, a, th_prev, th_post, log_var_prev, log_var_post, th_ref):
        a_i, b_i = self.bank[a, 0], self.bank[a, 1]
        kind = self.cfg.reward
        if kind == "prec_gain":
            return np.exp(-log_var_post) - np.exp(-log_var_prev)
        if kind == "var_reduction":
            return log_var_prev - log_var_post
        if kind == "fi_hat_prev":
            return info(a_i, b_i, th_prev)
        if kind == "fi_hat_post":
            return info(a_i, b_i, th_post)
        if kind == "fi_ref":
            return info(a_i, b_i, th_ref)
        if kind == "err_reduction_ref":
            return (th_prev - th_ref) ** 2 - (th_post - th_ref) ** 2
        if kind == "neg_sq_err_ref":
            return -((th_post - th_ref) ** 2)
        raise ValueError(f"unknown reward: {kind}")

    # ---- training --------------------------------------------------------
    def train(self, log=print):
        cfg, L, n_items = self.cfg, self.L, self.n_items
        rng = np.random.default_rng([cfg.seed, 10])
        theta_val = self.sample_theta(np.random.default_rng([cfg.seed, 11]), cfg.n_val)
        val_seed = cfg.seed * 7 + 12  # evaluation streams inside run_cat are [seed, 0/1]

        B = cfg.buffer_size
        buf_s = np.zeros((B, self.n_in), np.float32)
        buf_a = np.zeros(B, np.int64)
        buf_r = np.zeros(B, np.float32)
        buf_s2 = np.zeros((B, self.n_in), np.float32)
        buf_done = np.zeros(B, np.float32)
        buf_mask2 = np.zeros((B, n_items), bool)
        n_stored, ptr, grad_steps = 0, 0, 0

        best_score, best_state, episodes = -np.inf, None, 0
        rows = np.arange(cfg.n_env)
        n_batches = int(np.ceil(cfg.n_episodes / cfg.n_env))
        next_eval = cfg.eval_every
        for batch_idx in range(n_batches):
            frac = min(1.0, episodes / (cfg.eps_decay_frac * cfg.n_episodes))
            eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

            theta = self.sample_theta(rng, cfg.n_env)
            theta_hat = rng.uniform(-0.5, 0.5, size=cfg.n_env)
            log_var = np.zeros(cfg.n_env)  # prior variance 1
            episode = []
            mask = np.zeros((cfg.n_env, n_items), bool)
            items = np.zeros((cfg.n_env, L), np.int64)
            resp = np.zeros((cfg.n_env, L), np.int64)
            for t in range(L):
                s = self.features(theta_hat, t, log_var)
                a = self.greedy_from_features(s, mask)
                explore = rng.random(cfg.n_env) < eps
                if explore.any():
                    a[explore] = np.array([rng.choice(np.flatnonzero(~m)) for m in mask[explore]])
                bt = self.bank_true[a]
                u = rng.random(cfg.n_env) <= prob(bt[:, 0], bt[:, 1], theta, bt[:, 2])
                items[:, t], resp[:, t] = a, u
                mask[rows, a] = True
                th_prev, log_var_prev = theta_hat, log_var
                theta_hat = mle(self.bank, items[:, : t + 1], resp[:, : t + 1], theta_hat)
                log_var = np.log(posterior(self.bank, items[:, : t + 1], resp[:, : t + 1])[1])
                s2 = self.features(theta_hat, t + 1, log_var)
                done = float(t == L - 1)
                episode.append((s, a.copy(), th_prev, theta_hat, log_var_prev, log_var, s2, done, mask.copy()))
                if not self.deferred:
                    pending = [episode[-1]]
                elif done:
                    pending = episode  # reference-ability rewards need the end-of-test estimate
                else:
                    pending = []
                for (s_, a_, thp, thq, lvp, lvq, s2_, d_, m_) in pending:
                    r = self._reward(a_, thp, thq, lvp, lvq, th_ref=theta_hat)
                    idx = (ptr + np.arange(cfg.n_env)) % B
                    buf_s[idx], buf_a[idx], buf_r[idx], buf_s2[idx] = s_, a_, r, s2_
                    buf_done[idx], buf_mask2[idx] = d_, m_
                    ptr = (ptr + cfg.n_env) % B
                    n_stored = min(n_stored + cfg.n_env, B)

                if n_stored >= cfg.batch_size:
                    self._update(rng, n_stored, buf_s, buf_a, buf_r, buf_s2, buf_done, buf_mask2)
                    grad_steps += 1
                    if grad_steps % cfg.target_every == 0:
                        self.q_target.load_state_dict(self.q.state_dict())
            episodes += cfg.n_env

            if episodes >= next_eval or batch_idx == n_batches - 1:
                next_eval += cfg.eval_every
                hist, items_v, resp_v, theta0_v = run_cat(self, self.bank, theta_val, L, seed=val_seed, bank_true=self.bank_true)
                score = self.episode_return(items_v, resp_v, hist, theta0_v)
                log(f"  episodes {episodes:6d}  eps {eps:.2f}  val return {score:.3f}  "
                    f"(val RMSE@{L} {rmse_by_step(hist, theta_val)[-1]:.3f}, not used)")
                if score > best_score:
                    best_score = score
                    best_state = {k: v.clone() for k, v in self.q.state_dict().items()}
        self.q.load_state_dict(best_state)
        self.grad_steps = grad_steps
        return best_score

    def episode_return(self, items, resp, hist, theta0):
        """Mean undiscounted return of the condition's reward over completed episodes."""
        prev = np.vstack([theta0[None, :], hist[:-1]])  # estimate before each step (L, n)
        kind = self.cfg.reward
        if kind in ("prec_gain", "var_reduction"):
            v_end = posterior(self.bank, items, resp)[1]
            v_0 = posterior(self.bank, items[:, :0], resp[:, :0])[1]
            return float(np.mean(1 / v_end - 1 / v_0)) if kind == "prec_gain" else float(np.mean(np.log(v_0) - np.log(v_end)))
        a, b = self.bank[items, 0].T, self.bank[items, 1].T  # (L, n)
        if kind == "fi_hat_prev":
            return float(info(a, b, prev).sum(axis=0).mean())
        if kind == "fi_hat_post":
            return float(info(a, b, hist).sum(axis=0).mean())
        ref = hist[-1]  # reference ability: MLE from the complete record of the episode
        if kind == "fi_ref":
            return float(info(a, b, ref[None, :]).sum(axis=0).mean())
        if kind == "err_reduction_ref":
            return float(np.mean((theta0 - ref) ** 2))
        if kind == "neg_sq_err_ref":
            return float(-np.mean(((hist - ref[None, :]) ** 2).sum(axis=0)))
        raise ValueError(kind)

    def _update(self, rng, n_stored, buf_s, buf_a, buf_r, buf_s2, buf_done, buf_mask2):
        cfg = self.cfg
        idx = rng.integers(0, n_stored, size=cfg.batch_size)
        s = torch.from_numpy(buf_s[idx]).to(self.device)
        a = torch.from_numpy(buf_a[idx]).to(self.device)
        r = torch.from_numpy(buf_r[idx]).to(self.device)
        s2 = torch.from_numpy(buf_s2[idx]).to(self.device)
        done = torch.from_numpy(buf_done[idx]).to(self.device)
        mask2 = torch.from_numpy(buf_mask2[idx]).to(self.device)

        q_sa = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s2).masked_fill(mask2, -torch.inf).max(dim=1).values
            target = r + cfg.gamma * (1.0 - done) * q_next
        loss = nn.functional.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self._clamp()
