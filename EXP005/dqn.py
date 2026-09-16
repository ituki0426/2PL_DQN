"""EXP001 DQN architecture and updates, trained in epochs on real responses."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .irt import info, mle, posterior
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
    n_epochs: int = 5
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5
    eval_every: int = 0  # optional interval in completed training respondents
    eval_batch_size: int = 256
    train_seed: int = 0
    shuffle_seed: int = 20260904


STATE_DIM = {"theta": 1, "theta_step": 2, "belief": 3}
REWARDS = ("prec_gain", "var_reduction", "fi_ref", "fi_hat_prev", "fi_hat_post",
           "err_reduction_ref", "neg_sq_err_ref")
DEFERRED = ("fi_ref", "err_reduction_ref", "neg_sq_err_ref")


class QNet(nn.Module):
    def __init__(self, n_in, h1, h2, n_items):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, n_items))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, bank, test_length, cfg: DQNConfig):
        self.bank, self.L, self.cfg = np.asarray(bank, dtype=float), test_length, cfg
        if cfg.reward not in REWARDS:
            raise ValueError(f"unknown reward: {cfg.reward}")
        if cfg.state not in STATE_DIM or cfg.positive not in ("none", "all"):
            raise ValueError("unknown state or constraint")
        for name in ("hidden", "batch_size", "buffer_size", "target_every", "n_env",
                     "n_epochs", "eval_batch_size"):
            if getattr(cfg, name) < 1:
                raise ValueError(f"{name} must be positive")
        if cfg.buffer_size < max(cfg.n_env, cfg.batch_size):
            raise ValueError("buffer_size must cover n_env and batch_size")
        if cfg.eval_every < 0 or cfg.eps_decay_frac <= 0:
            raise ValueError("invalid evaluation interval or epsilon decay")
        if not 0 <= cfg.eps_end <= cfg.eps_start <= 1 or not 0 <= cfg.gamma <= 1:
            raise ValueError("invalid epsilon or gamma")
        if self.bank.ndim != 2 or self.bank.shape[1] != 2 or not np.isfinite(self.bank).all():
            raise ValueError("bank must have finite a,b columns")
        if np.any(self.bank[:, 0] <= 0) or not 1 <= test_length <= len(self.bank):
            raise ValueError("bank requires positive discrimination and enough items")
        self.n_items = self.bank.shape[0]
        self.n_in = STATE_DIM[cfg.state]
        self.deferred = cfg.reward in DEFERRED
        h2 = cfg.hidden2 or cfg.hidden
        torch.manual_seed(cfg.train_seed)
        self.q = QNet(self.n_in, cfg.hidden, h2, self.n_items)
        self._clamp()  # constrain before the target copy so both nets start identical
        self.q_target = QNet(self.n_in, cfg.hidden, h2, self.n_items)
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
            qv = self.q(torch.from_numpy(feats)).numpy()
        qv[mask] = -np.inf
        return qv.argmax(axis=1)

    def __call__(self, theta_hat, step, mask, items, resp):  # policy interface for run_cat
        log_var = np.log(posterior(self.bank, items, resp)[1]) if self.cfg.state == "belief" else None
        return self.greedy_from_features(self.features(theta_hat, step, log_var), mask)

    def _clamp(self):
        if self.cfg.positive == "all":
            with torch.no_grad():
                for p in self.q.parameters():
                    p.clamp_(min=0.0)

    def _reward(self, a, th_prev, th_post, log_var_prev, log_var_post, th_ref=None):
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
        if th_ref is None:
            raise ValueError(f"{kind} requires the end-of-episode MLE")
        if kind == "fi_ref":
            return info(a_i, b_i, th_ref)
        if kind == "err_reduction_ref":
            return (th_prev - th_ref) ** 2 - (th_post - th_ref) ** 2
        if kind == "neg_sq_err_ref":
            return -((th_post - th_ref) ** 2)
        raise ValueError(f"unknown reward: {kind}")

    # ---- training --------------------------------------------------------
    def train(self, train_responses, validation_responses, theta_reference_val=None, log=print):
        """Train on each training respondent exactly once per epoch.

        Only the optional validation diagnostic receives reference scores. Training
        features, actions, answers, rewards and model selection never use them.
        """
        cfg, L, n_items = self.cfg, self.L, self.n_items
        for values in (train_responses, validation_responses):
            if (values.ndim != 2 or values.shape[1] != n_items or not len(values)
                    or not np.isin(values, [0, 1]).all()):
                raise ValueError("training/validation responses must be nonempty aligned 0/1 matrices")
        if theta_reference_val is not None:
            theta_reference_val = np.asarray(theta_reference_val)
            if theta_reference_val.shape != (len(validation_responses),) or not np.isfinite(theta_reference_val).all():
                raise ValueError("invalid validation theta_reference")
        rng = np.random.default_rng([cfg.train_seed, 10])
        shuffle_rng = np.random.default_rng(cfg.shuffle_seed)
        total_episodes = len(train_responses) * cfg.n_epochs
        B = cfg.buffer_size
        buf_s = np.zeros((B, self.n_in), np.float32)
        buf_a = np.zeros(B, np.int64)
        buf_r = np.zeros(B, np.float32)
        buf_s2 = np.zeros((B, self.n_in), np.float32)
        buf_done = np.zeros(B, np.float32)
        buf_mask2 = np.zeros((B, n_items), bool)
        n_stored, ptr, grad_steps = 0, 0, 0
        best_score, best_state, episodes = -np.inf, None, 0
        self.validation_log = []
        self.epoch_orders = []
        self.transitions = 0
        next_eval = cfg.eval_every if cfg.eval_every else np.inf
        empty = np.empty((1, 0), dtype=np.int64)
        initial_log_var = float(np.log(posterior(self.bank, empty, empty)[1][0]))
        for epoch in range(1, cfg.n_epochs + 1):
            order = shuffle_rng.permutation(len(train_responses))
            self.epoch_orders.append(order.copy())
            for start in range(0, len(order), cfg.n_env):
                respondent_indices = order[start:start + cfg.n_env]
                size = len(respondent_indices)
                rows = np.arange(size)
                frac = min(1.0, episodes / (cfg.eps_decay_frac * total_episodes))
                eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
                theta_hat = np.zeros(size)
                log_var = np.full(size, initial_log_var)
                mask = np.zeros((size, n_items), bool)
                items = np.zeros((size, L), np.int64)
                resp = np.zeros((size, L), np.int64)
                episode = []
                for t in range(L):
                    s = self.features(theta_hat, t, log_var)
                    a = self.greedy_from_features(s, mask)
                    explore = rng.random(size) < eps
                    if explore.any():
                        a[explore] = np.array([rng.choice(np.flatnonzero(~m)) for m in mask[explore]])
                    u = train_responses[respondent_indices, a]
                    items[:, t], resp[:, t] = a, u
                    mask[rows, a] = True
                    th_prev, log_var_prev = theta_hat, log_var
                    theta_hat = mle(self.bank, items[:, :t + 1], resp[:, :t + 1], theta_hat)
                    log_var = np.log(posterior(self.bank, items[:, :t + 1], resp[:, :t + 1])[1])
                    s2 = self.features(theta_hat, t + 1, log_var)
                    done = float(t == L - 1)
                    episode.append((s, a.copy(), th_prev, theta_hat, log_var_prev,
                                    log_var, s2, done, mask.copy()))
                    if not self.deferred:
                        pending = (episode[-1],)
                    elif done:
                        pending = episode
                    else:
                        pending = ()
                    for s_, a_, thp, thq, lvp, lvq, s2_, d_, mask_ in pending:
                        reward = self._reward(a_, thp, thq, lvp, lvq, th_ref=theta_hat)
                        idx = (ptr + rows) % B
                        buf_s[idx], buf_a[idx], buf_r[idx], buf_s2[idx] = s_, a_, reward, s2_
                        buf_done[idx], buf_mask2[idx] = d_, mask_
                        ptr = (ptr + size) % B
                        n_stored = min(n_stored + size, B)
                        self.transitions += size
                    if n_stored >= cfg.batch_size:
                        self._update(rng, n_stored, buf_s, buf_a, buf_r, buf_s2, buf_done, buf_mask2)
                        grad_steps += 1
                        if grad_steps % cfg.target_every == 0:
                            self.q_target.load_state_dict(self.q.state_dict())
                episodes += size
                epoch_end = start + size == len(order)
                if episodes >= next_eval or epoch_end:
                    if episodes >= next_eval:
                        next_eval = (episodes // cfg.eval_every + 1) * cfg.eval_every
                    hist, items_v, resp_v, theta0_v = run_cat(
                        self, self.bank, validation_responses, L, batch_size=cfg.eval_batch_size)
                    score = self.episode_return(items_v, resp_v, hist, theta0_v)
                    if not np.isfinite(score):
                        raise FloatingPointError("nonfinite validation return")
                    diagnostic = (float(rmse_by_step(hist, theta_reference_val)[-1])
                                  if theta_reference_val is not None else float("nan"))
                    self.validation_log.append(dict(epoch=epoch, epoch_end=epoch_end,
                        episodes=episodes, transitions=self.transitions, grad_steps=grad_steps,
                        epsilon=eps, val_return=score, val_rmse_reference=diagnostic))
                    log(f"  epoch {epoch}/{cfg.n_epochs} episodes {episodes:6d} eps {eps:.2f} "
                        f"val return {score:.3f} (val RMSE vs theta_reference {diagnostic:.3f}; diagnostic only)")
                    if score > best_score:
                        best_score = score
                        best_state = {k: v.clone() for k, v in self.q.state_dict().items()}
                        self.best_epoch, self.best_episodes = epoch, episodes
        self.q.load_state_dict(best_state)
        self.q_target.load_state_dict(best_state)
        self.grad_steps, self.episodes = grad_steps, episodes
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
        ref = hist[-1]  # final MLE from the complete L-response CAT episode
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
        s = torch.from_numpy(buf_s[idx])
        a = torch.from_numpy(buf_a[idx])
        r = torch.from_numpy(buf_r[idx])
        s2 = torch.from_numpy(buf_s2[idx])
        done = torch.from_numpy(buf_done[idx])
        mask2 = torch.from_numpy(buf_mask2[idx])

        q_sa = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s2).masked_fill(mask2, -torch.inf).max(dim=1).values
            # At L == n_items, every terminal next action is masked (-inf).
            q_next = torch.where(done.bool(), torch.zeros_like(q_next), q_next)
            target = r + cfg.gamma * q_next
        loss = nn.functional.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self._clamp()
