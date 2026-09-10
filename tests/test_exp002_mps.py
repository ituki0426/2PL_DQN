"""Device integration checks: CPU regression and real MPS training when available."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from EXP001.dqn import DQNAgent as CPUAgent, DQNConfig as CPUConfig
from EXP002.dqn import DQNAgent, DQNConfig
from EXP002.irt import make_bank
from EXP002.simulate import run_cat


class DQNDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        # Adam lazily imports backend introspection code; load it before mocking MPS.
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.bank = make_bank(12, np.random.default_rng(1))
        self.settings = dict(hidden=8, hidden2=8, batch_size=4, buffer_size=32,
                             n_env=4, n_episodes=12, n_val=4, eval_every=4,
                             target_every=2, seed=12)

    def test_cpu_fallback_preserves_training_results(self):
        for state, reward, positive in [("belief", "prec_gain", "none"),
                                         ("theta", "fi_hat_prev", "all"),
                                         ("theta_step", "fi_ref", "none")]:
            with self.subTest(state=state, reward=reward):
                settings = dict(self.settings, state=state, reward=reward, positive=positive)
                with patch("torch.backends.mps.is_available", return_value=False):
                    agent = DQNAgent(self.bank, 3, DQNConfig(**settings))
                    self.assertEqual(agent.device.type, "cpu")
                    score = agent.train(log=lambda _: None)
                    baseline = CPUAgent(self.bank, 3, CPUConfig(**settings))
                    baseline_score = baseline.train(log=lambda _: None)
                self.assertEqual(score, baseline_score)
                self.assertGreater(agent.grad_steps, 0)
                self.assertEqual(agent.grad_steps, baseline.grad_steps)
                for name, value in agent.q.state_dict().items():
                    torch.testing.assert_close(value, baseline.q.state_dict()[name], rtol=0, atol=0)

    def check_action_mask(self, agent):
        # Equal Q values must select the first available item, including on MPS.
        with torch.no_grad():
            for parameter in agent.q.parameters():
                parameter.zero_()
        feats = np.zeros((2, agent.n_in), dtype=np.float32)
        mask = np.zeros((2, len(self.bank)), dtype=bool)
        mask[0, :2] = True
        mask[1, :5] = True
        original_mask = mask.copy()
        actions = agent.greedy_from_features(feats, mask)
        self.assertIsInstance(actions, np.ndarray)
        np.testing.assert_array_equal(actions, [2, 5])
        np.testing.assert_array_equal(mask, original_mask)

    def test_cpu_action_mask(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            self.check_action_mask(DQNAgent(self.bank, 3, DQNConfig(**self.settings)))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS hardware is unavailable")
    def test_mps_training_and_inference(self):
        for state, reward, positive in [("belief", "prec_gain", "none"),
                                         ("theta", "fi_hat_prev", "all"),
                                         ("theta_step", "fi_ref", "none")]:
            with self.subTest(state=state, reward=reward):
                settings = dict(self.settings, state=state, reward=reward, positive=positive)
                agent = DQNAgent(self.bank, 3, DQNConfig(**settings))
                self.assertEqual(agent.device.type, "mps")
                initial = {name: value.detach().cpu().clone() for name, value in agent.q.state_dict().items()}
                score = agent.train(log=lambda _: None)
                self.assertTrue(np.isfinite(score))
                self.assertGreater(agent.grad_steps, 0)
                self.assertTrue(any(not torch.equal(value.cpu(), initial[name])
                                    for name, value in agent.q.state_dict().items()))
                for net in (agent.q, agent.q_target):
                    for parameter in net.parameters():
                        self.assertEqual(parameter.device.type, "mps")
                        self.assertTrue(torch.isfinite(parameter).all().item())
                        if positive == "all":
                            self.assertTrue((parameter >= 0).all().item())
                hist, items, _, _ = run_cat(agent, self.bank, np.array([-1., 0., 1.]), 3, seed=20)
                self.assertTrue(np.isfinite(hist).all())
                self.assertTrue(all(len(set(row)) == 3 for row in items))
                self.check_action_mask(agent)


if __name__ == "__main__":
    unittest.main()
