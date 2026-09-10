"""FP64 device integration checks on CPU and real CUDA when available."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from EXP004.dqn import DQNAgent, DQNConfig
from EXP004.irt import make_bank
from EXP004.simulate import run_cat


class DQNDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        # Load Adam's lazy backend introspection before mocking CUDA availability.
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.bank = make_bank(12, np.random.default_rng(1))
        self.settings = dict(hidden=8, hidden2=8, batch_size=4, buffer_size=32,
                             n_env=4, n_episodes=12, n_val=4, eval_every=4,
                             target_every=2, seed=12)

    def check_training(self, device):
        for state, reward, positive in [("belief", "prec_gain", "none"),
                                       ("theta", "fi_hat_prev", "all"),
                                       ("theta_step", "fi_ref", "none")]:
            with self.subTest(state=state, reward=reward):
                settings = dict(self.settings, state=state, reward=reward, positive=positive)
                default_dtype = torch.get_default_dtype()
                agent = DQNAgent(self.bank, 3, DQNConfig(**settings))
                self.assertEqual(torch.get_default_dtype(), default_dtype)
                self.assertEqual(agent.device.type, device)
                initial = {k: v.detach().cpu().clone() for k, v in agent.q.state_dict().items()}
                with patch.object(agent, '_update', wraps=agent._update) as update, \
                     patch('torch.nn.functional.mse_loss', wraps=torch.nn.functional.mse_loss) as loss:
                    score = agent.train(log=lambda _: None)
                self.assertTrue(np.isfinite(score))
                self.assertGreater(agent.grad_steps, 0)
                self.assertEqual(update.call_count, agent.grad_steps)
                for call in update.call_args_list:
                    # Replay data must retain FP64 before transfer to the device.
                    for value, dtype in zip(call.args[2:],
                                            [np.float64, np.int64, np.float64,
                                             np.float64, np.float64, np.bool_]):
                        self.assertEqual(value.dtype, dtype)
                self.assertEqual(loss.call_count, agent.grad_steps)
                for call in loss.call_args_list:
                    for tensor in call.args:
                        self.assertEqual(tensor.dtype, torch.float64)
                        self.assertEqual(tensor.device.type, device)
                        self.assertTrue(torch.isfinite(tensor).all().item())
                self.assertTrue(any(not torch.equal(v.cpu(), initial[k])
                                    for k, v in agent.q.state_dict().items()))
                for net in (agent.q, agent.q_target):
                    for parameter in net.parameters():
                        self.assertEqual(parameter.device.type, device)
                        self.assertEqual(parameter.dtype, torch.float64)
                        self.assertTrue(torch.isfinite(parameter).all().item())
                        if positive == "all":
                            self.assertTrue((parameter >= 0).all().item())
                for parameter in agent.q.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(parameter.grad.dtype, torch.float64)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    for key in ['exp_avg', 'exp_avg_sq']:
                        moment = agent.opt.state[parameter][key]
                        self.assertEqual(moment.dtype, torch.float64)
                        self.assertEqual(moment.device.type, device)
                hist, items, _, _ = run_cat(agent, self.bank, np.array([-1., 0., 1.]), 3, seed=20)
                self.assertTrue(np.isfinite(hist).all())
                self.assertTrue(all(len(set(row)) == 3 for row in items))
                self.check_action_mask(agent)

    def test_cpu_fallback_fp64_training_and_inference(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.check_training('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware is unavailable')
    def test_cuda_fp64_training_and_inference(self):
        self.check_training('cuda')

    def test_features_preserve_fp64_precision(self):
        with patch('torch.cuda.is_available', return_value=False):
            agent = DQNAgent(self.bank, 3, DQNConfig(**self.settings))
        # These values would lose information if rounded to FP32.
        theta = np.array([1.0 + 2**-40, -1.0 - 2**-40])
        log_var = np.array([-0.5 + 2**-40, -0.5 - 2**-40])
        feats = agent.features(theta, 1, log_var)
        self.assertEqual(feats.dtype, np.float64)
        np.testing.assert_array_equal(feats[:, 0], theta)
        np.testing.assert_array_equal(feats[:, 2], log_var)

    def check_action_mask(self, agent):
        with torch.no_grad():
            for parameter in agent.q.parameters():
                parameter.zero_()
        mask = np.zeros((2, len(self.bank)), dtype=bool)
        mask[0, :2] = True
        mask[1, :5] = True
        original_mask = mask.copy()
        for dtype in [np.float32, np.float64]:
            actions = agent.greedy_from_features(np.zeros((2, agent.n_in), dtype=dtype), mask)
            self.assertIsInstance(actions, np.ndarray)
            self.assertEqual(actions.dtype, np.int64)
            np.testing.assert_array_equal(actions, [2, 5])
            np.testing.assert_array_equal(mask, original_mask)


if __name__ == '__main__':
    unittest.main()
