"""Numerical and integration checks for EXP003's CPU/EAP experiment."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from EXP001.simulate import run_cat as run_mle_cat
from EXP003.analyze_selection import by_step, records, trace_run
from EXP003.benchmark_cost import history
from EXP003.dqn import DQNAgent, DQNConfig
from EXP003.irt import GRID, eap, make_bank, posterior
from EXP003.rules import MFIPolicy
from EXP003.simulate import run_cat


def direct_eap(bank, items, responses):
    """Independent likelihood-times-prior calculation for small test histories."""
    estimates = []
    for row_items, row_responses in zip(items, responses):
        weights = np.exp(-GRID ** 2 / 2)
        for item, response in zip(row_items, row_responses):
            a, b = bank[item]
            p = 1 / (1 + np.exp(-a * (GRID - b)))
            weights *= p if response else 1 - p
        estimates.append(np.sum(GRID * weights) / np.sum(weights))
    return np.array(estimates)


class EAPTests(unittest.TestCase):
    def setUp(self):
        self.bank = make_bank(12, np.random.default_rng(2))

    def test_empty_history_is_zero(self):
        empty = np.empty((4, 0), dtype=int)
        np.testing.assert_array_equal(eap(self.bank, empty, empty), np.zeros(4))

    def test_posterior_mean_matches_direct_calculation(self):
        items = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        responses = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1]])
        np.testing.assert_allclose(eap(self.bank, items, responses),
                                   direct_eap(self.bank, items, responses), atol=1e-12)

    def test_extreme_responses_remain_finite_and_symmetric(self):
        bank = np.column_stack([np.ones(8), np.zeros(8)])
        positives = []
        for length in range(1, 9):
            items = np.tile(np.arange(length), (2, 1))
            responses = np.vstack([np.ones(length, int), np.zeros(length, int)])
            estimates = eap(bank, items, responses)
            self.assertTrue(np.isfinite(estimates).all())
            self.assertTrue((np.abs(estimates) < 4).all())
            self.assertAlmostEqual(estimates[0], -estimates[1])
            positives.append(estimates[0])
        self.assertTrue((np.diff(positives) > 0).all())

    def test_simulation_uses_eap_at_every_step_and_preserves_response_stream(self):
        theta = np.array([-2., -0.5, 0.5, 2.])

        def fixed_policy(theta_hat, step, mask, items, resp):
            return np.full(len(theta_hat), step, dtype=int)

        for guessing in (0.0, 0.2):
            with self.subTest(guessing=guessing):
                bank_true = np.column_stack([self.bank, np.full(len(self.bank), guessing)])
                hist, items, resp, initial = run_cat(fixed_policy, self.bank, theta, 4, 10, bank_true)
                _, old_items, old_resp, _ = run_mle_cat(fixed_policy, self.bank, theta, 4, 10, bank_true)
                np.testing.assert_array_equal(initial, np.zeros(len(theta)))
                np.testing.assert_array_equal(items, old_items)
                np.testing.assert_array_equal(resp, old_resp)
                for t in range(1, 5):
                    np.testing.assert_allclose(hist[t - 1], direct_eap(self.bank, items[:, :t], resp[:, :t]), atol=1e-12)

    def test_analysis_handles_constant_initial_estimates(self):
        theta = np.array([-2., -1., 0., 1., 2.])
        hist, items, resp, prev = trace_run(MFIPolicy(self.bank), self.bank, theta, 3, 10)
        summary = by_step(records("MFI", self.bank, theta, hist, items, resp, prev))
        np.testing.assert_array_equal(prev[0], np.zeros(len(theta)))
        self.assertTrue(np.isnan(summary.loc[summary.step == 1, "slope_b_on_prev"]).all())
        self.assertTrue(np.isfinite(summary.rmse_post).all())

    def test_benchmark_history_uses_eap(self):
        items, resp, estimates, _ = history(self.bank, np.array([-1., 0., 1.]), 3, np.random.default_rng(10))
        np.testing.assert_allclose(estimates, direct_eap(self.bank, items, resp), atol=1e-12)

    def test_training_and_rewards_use_cpu_eap(self):
        settings = dict(hidden=8, hidden2=8, batch_size=4, buffer_size=32,
                        n_env=4, n_episodes=8, n_val=4, eval_every=4, target_every=2, seed=12)
        rewards = ["prec_gain", "var_reduction", "fi_hat_prev", "fi_hat_post",
                   "fi_ref", "err_reduction_ref", "neg_sq_err_ref"]
        for index, reward in enumerate(rewards):
            with self.subTest(reward=reward):
                cfg = DQNConfig(**settings, reward=reward,
                                state=["theta", "theta_step", "belief"][index % 3],
                                positive="all" if reward == "fi_hat_prev" else "none")
                agent = DQNAgent(self.bank, 3, cfg)
                with patch.object(agent, "features", wraps=agent.features) as features:
                    score = agent.train(log=lambda _: None)
                self.assertTrue(np.isfinite(score))
                self.assertGreater(agent.grad_steps, 0)
                np.testing.assert_array_equal(features.call_args_list[0].args[0], np.zeros(cfg.n_env))
                empty = np.empty((1, 0), dtype=int)
                expected_log_var = np.log(posterior(self.bank, empty, empty)[1][0])
                np.testing.assert_allclose(features.call_args_list[0].args[2], expected_log_var)
                for net in (agent.q, agent.q_target):
                    self.assertTrue(all(p.device.type == "cpu" for p in net.parameters()))
                    self.assertTrue(all(torch.isfinite(p).all() for p in net.parameters()))

                hist, items, resp, theta0 = run_cat(agent, self.bank, np.array([-1., 0., 1.]), 3, 20)
                ref = direct_eap(self.bank, items, resp)
                np.testing.assert_allclose(hist[-1], ref, atol=1e-12)
                returns = np.zeros(len(theta0))
                for t in range(3):
                    before = theta0 if t == 0 else hist[t - 1]
                    log_v0 = np.log(posterior(self.bank, items[:, :t], resp[:, :t])[1])
                    log_v1 = np.log(posterior(self.bank, items[:, :t + 1], resp[:, :t + 1])[1])
                    returns += agent._reward(items[:, t], before, hist[t], log_v0, log_v1, ref)
                self.assertAlmostEqual(agent.episode_return(items, resp, hist, theta0), returns.mean())


if __name__ == "__main__":
    unittest.main()
