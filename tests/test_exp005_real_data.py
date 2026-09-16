"""Data integrity, leakage prevention, epoch accounting and CLI integration."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from EXP001.irt import mle as original_mle
from EXP005.data import DATASETS, load_dataset, split_respondents
from EXP005.dqn import DQNAgent, DQNConfig
from EXP005.irt import posterior
from EXP005.rules import MFIPolicy
from EXP005.run_grid import aggregate, main
from EXP005.simulate import metrics_by_step, run_cat


def fixture(root):
    directory = root / DATASETS[0]
    directory.mkdir()
    pd.DataFrame({"item": ["03", "01", "02", "04"], "a": [1.3, 0.7, -0.5, 1.1],
                  "b": [1., -1., 0., 0.]}).to_csv(directory / "item_parameters.csv", index=False)
    ids = [f"{i:04}" for i in range(100)]
    pd.DataFrame({"id": ids[::-1], "theta_EAP": np.linspace(-2, 2, 100)[::-1]}).to_csv(
        directory / "person_scores.csv", index=False)
    rng = np.random.default_rng(12)
    values = rng.integers(0, 2, size=(100, 4))
    frame = pd.DataFrame(values, columns=["04", "02", "01", "03"])
    frame.insert(0, "id", ids)
    frame.to_csv(directory / "responses.csv", index=False)
    return directory, frame


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory, self.responses = fixture(self.root)

    def load(self):
        return load_dataset(DATASETS[0], self.root, test_length=3)

    def test_ids_alignment_and_filtering(self):
        data = self.load()
        self.assertEqual(data.respondent_ids[0], "0000")
        self.assertEqual(data.item_ids.tolist(), ["01", "03", "04"])
        self.assertEqual(data.excluded_item_ids, ["02"])
        np.testing.assert_array_equal(data.responses, self.responses[["01", "03", "04"]].to_numpy())
        np.testing.assert_allclose(data.theta_reference, np.linspace(-2, 2, 100))
        with self.assertRaisesRegex(ValueError, "usable items"):
            load_dataset(DATASETS[0], self.root, test_length=4)

    def test_duplicate_or_mismatched_ids_and_invalid_responses(self):
        path = self.directory / "responses.csv"
        original = path.read_text()
        variants = [original.replace("id,04,02,01,03", "id,04,02,01,01"),
                    original.replace("0001,", "0000,"),
                    original.replace("0001,", "xxxx,")]
        for changed in variants:
            with self.subTest(changed=changed[:40]):
                path.write_text(changed)
                with self.assertRaises(ValueError):
                    self.load()
        for invalid in ("", "NA", "2", "inf", "-1"):
            with self.subTest(invalid=invalid):
                frame = self.responses.astype(str).copy()
                # Even excluded items must have valid responses.
                frame.loc[0, "02"] = invalid
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    self.load()

    def test_invalid_parameters_scores_and_item_ids(self):
        for filename, column, invalid in (("item_parameters.csv", "a", "inf"),
                ("person_scores.csv", "theta_EAP", "NA"),
                ("item_parameters.csv", "item", "unknown"),
                ("person_scores.csv", "id", "")):
            path = self.directory / filename
            original = path.read_text()
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame.loc[0, column] = invalid
            frame.to_csv(path, index=False)
            with self.subTest(filename=filename, column=column), self.assertRaises(ValueError):
                self.load()
            path.write_text(original)

    def test_quantile_split_is_fixed_disjoint_and_handles_ties(self):
        data = self.load()
        split = split_respondents(data)
        pd.testing.assert_frame_equal(split, split_respondents(data))
        self.assertEqual(split.split.value_counts().to_dict(), {"train": 70, "test": 20, "validation": 10})
        for _, group in split.groupby("stratum"):
            self.assertEqual(group.split.value_counts().to_dict(), {"train": 7, "test": 2, "validation": 1})
        self.assertEqual(split.id.nunique(), len(data.respondent_ids))
        self.assertFalse(split.equals(split_respondents(data, 17)))
        data.theta_reference = np.zeros_like(data.theta_reference)
        tied = split_respondents(data)
        self.assertEqual(tied.stratum.nunique(), 1)
        self.assertEqual(tied.split.value_counts().to_dict(), split.split.value_counts().to_dict())


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.bank = np.array([[0.8, -1.], [1.3, 0.], [1., 1.]])
        self.responses = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0]])

    def cfg(self, **kw):
        return DQNConfig(**dict(dict(hidden=8, batch_size=2, buffer_size=8, n_env=2,
            n_epochs=2, target_every=2, train_seed=12, shuffle_seed=15), **kw))

    def test_real_responses_initial_zero_mle_dodd_and_no_repeated_items(self):
        outputs = [run_cat(MFIPolicy(self.bank), self.bank, self.responses, 3, batch_size=b)
                   for b in (2, 5)]
        for actual, expected in zip(*outputs):
            np.testing.assert_allclose(actual, expected)
        hist, items, resp, initial = outputs[0]
        np.testing.assert_array_equal(initial, np.zeros(5))
        np.testing.assert_array_equal(resp, self.responses[np.arange(5)[:, None], items])
        self.assertTrue(all(len(np.unique(row)) == 3 for row in items))
        previous = initial
        for t in range(3):
            previous = original_mle(self.bank, items[:, :t + 1], resp[:, :t + 1], previous)
            np.testing.assert_allclose(hist[t], previous)
        with self.assertRaisesRegex(ValueError, "already administered"):
            run_cat(lambda th, *args: np.zeros(len(th), dtype=int), self.bank, self.responses, 3)

    def test_training_consumes_real_answers_and_partial_batches(self):
        import EXP005.dqn as module
        for state, reward, positive in (("belief", "prec_gain", "none"), ("theta", "fi_hat_prev", "all")):
            with self.subTest(state=state):
                agent = DQNAgent(self.bank, 3, self.cfg(state=state, reward=reward, positive=positive))
                with patch.object(module, "mle", wraps=module.mle) as mle:
                    best = agent.train(self.responses, self.responses[:2], np.array([-1., 1.]), log=lambda _: None)
                self.assertEqual(agent.episodes, 10)
                self.assertEqual(agent.transitions, 30)
                self.assertEqual(agent.grad_steps, 18)
                self.assertEqual(len(agent.validation_log), 2)
                self.assertEqual(best, max(row["val_return"] for row in agent.validation_log))
                calls = iter(mle.call_args_list)
                for order in agent.epoch_orders:
                    np.testing.assert_array_equal(np.sort(order), np.arange(5))
                    for start in range(0, 5, 2):
                        ids = order[start:start + 2]
                        for t in range(3):
                            _, items, observed, previous = next(calls).args
                            np.testing.assert_array_equal(observed, self.responses[ids[:, None], items])
                            if t == 0:
                                np.testing.assert_array_equal(previous, np.zeros(len(ids)))
                for net in (agent.q, agent.q_target):
                    self.assertTrue(all(torch.isfinite(p).all() for p in net.parameters()))
                    if positive == "all":
                        self.assertTrue(all((p >= 0).all() for p in net.parameters()))

    def test_reference_scores_do_not_change_training_or_model_selection(self):
        agents = []
        for reference in (np.array([-1., 1.]), np.array([100., -100.])):
            agent = DQNAgent(self.bank, 3, self.cfg(eval_every=3))
            agent.train(self.responses, self.responses[:2], reference, log=lambda _: None)
            agents.append(agent)
        for name, value in agents[0].q.state_dict().items():
            torch.testing.assert_close(value, agents[1].q.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(agents[0].best_episodes, agents[1].best_episodes)
        self.assertEqual(sum(row["epoch_end"] for row in agents[0].validation_log), 2)
        self.assertNotEqual(agents[0].validation_log[0]["val_rmse_reference"],
                            agents[1].validation_log[0]["val_rmse_reference"])
        other = DQNAgent(self.bank, 3, self.cfg(n_env=1, train_seed=99))
        other.train(self.responses, self.responses[:2], log=lambda _: None)
        for a, b in zip(agents[0].epoch_orders, other.epoch_orders):
            np.testing.assert_array_equal(a, b)
        deferred_agents = []
        for reference in (np.array([-1., 1.]), np.array([100., -100.])):
            agent = DQNAgent(self.bank, 3, self.cfg(reward="fi_ref", train_seed=27))
            agent.train(self.responses, self.responses[:2], reference, log=lambda _: None)
            deferred_agents.append(agent)
            self.assertEqual(agent.transitions, 30)
        for name, value in deferred_agents[0].q.state_dict().items():
            torch.testing.assert_close(value, deferred_agents[1].q.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(deferred_agents[0].best_episodes, deferred_agents[1].best_episodes)
        for reward in ("fi_ref", "err_reduction_ref", "neg_sq_err_ref"):
            self.assertTrue(DQNAgent(self.bank, 3, self.cfg(reward=reward)).deferred)

    def test_episode_return_matches_actual_rewards(self):
        rewards = ("prec_gain", "var_reduction", "fi_ref", "fi_hat_prev", "fi_hat_post",
                   "err_reduction_ref", "neg_sq_err_ref")
        for reward in rewards:
            agent = DQNAgent(self.bank, 3, self.cfg(reward=reward))
            hist, items, resp, initial = run_cat(agent, self.bank, self.responses, 3)
            total = np.zeros(5)
            reference = hist[-1]
            for t in range(3):
                before = initial if t == 0 else hist[t - 1]
                lv0 = np.log(posterior(self.bank, items[:, :t], resp[:, :t])[1])
                lv1 = np.log(posterior(self.bank, items[:, :t + 1], resp[:, :t + 1])[1])
                total += agent._reward(items[:, t], before, hist[t], lv0, lv1,
                                       th_ref=reference)
            self.assertAlmostEqual(agent.episode_return(items, resp, hist, initial), total.mean())

    def test_metrics(self):
        values = metrics_by_step(np.array([[1., 3.], [1., 1.]]), np.array([0., 2.]))
        np.testing.assert_allclose(values["rmse"], [1, 1])
        np.testing.assert_allclose(values["mae"], [1, 1])
        np.testing.assert_allclose(values["bias"], [1, 0])
        self.assertAlmostEqual(values["correlation"][0], 1)
        self.assertTrue(np.isnan(values["correlation"][1]))


class CLITests(unittest.TestCase):
    def test_sensitivity_grid_uses_all_seven_rewards_and_separate_output(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            fixture(root)
            args = ["--dataset", DATASETS[0], "--data-root", str(root),
                    "--out", str(root / "out"), "--grid", "sensitivity",
                    "--test-length", "3", "--n-epochs", "1", "--threads", "1",
                    "--conditions", "prec_gain_g0.0,fi_hat_prev_g0.5,fi_ref_g0.9",
                    "--rep", "1"]
            main(args)
            out = root / "out" / "EXP005" / DATASETS[0] / "sensitivity"
            frame = pd.read_csv(out / "rep1.csv")
            self.assertEqual(set(frame.condition),
                             {"MFI", "FIWL", "MPWI", "MEPV",
                              "prec_gain_g0.0", "fi_hat_prev_g0.5", "fi_ref_g0.9"})
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertEqual(metadata["planned_dqn_replications"], 3)
            self.assertEqual(set(metadata["configs"]),
                             {f"{reward}_g{gamma}"
                              for reward in ("prec_gain", "var_reduction", "fi_ref",
                                             "fi_hat_prev", "fi_hat_post",
                                             "err_reduction_ref", "neg_sq_err_ref")
                              for gamma in (0.0, 0.5, 0.9, 1.0)})

    def test_rep_10_upgrades_planned_replication_metadata(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            fixture(root)
            args = ["--dataset", DATASETS[0], "--data-root", str(root), "--out", str(root / "out"),
                    "--test-length", "3", "--n-epochs", "1", "--threads", "1",
                    "--conditions", "proposed"]
            main(args + ["--rep", "1"])
            out = root / "out" / "EXP005" / DATASETS[0] / "main"
            manifest = out / "metadata.json"
            metadata = json.loads(manifest.read_text())
            metadata["planned_dqn_replications"] = 5
            manifest.write_text(json.dumps(metadata))

            main(args + ["--rep", "10"])

            self.assertTrue((out / "rep10.csv").is_file())
            self.assertEqual(json.loads(manifest.read_text())["planned_dqn_replications"], 10)

    def test_run_per_rep_split_aggregate_and_settings_guard(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            fixture(root)
            args = ["--dataset", DATASETS[0], "--data-root", str(root), "--out", str(root / "out"),
                    "--test-length", "3", "--n-epochs", "1", "--threads", "1"]
            main(args + ["--rep", "1"])
            out = root / "out" / "EXP005" / DATASETS[0] / "main"
            self.assertFalse((out / "analytic.csv").exists())
            self.assertFalse((out / "splits.csv").exists())
            self.assertTrue((out / "splits_rep1.csv").is_file())
            rep1 = pd.read_csv(out / "rep1.csv")
            self.assertEqual(set(rep1.condition),
                             {"MFI", "FIWL", "MPWI", "MEPV", "existing", "proposed"})
            main(args + ["--rep", "2"])
            self.assertTrue((out / "splits_rep2.csv").is_file())
            s1 = pd.read_csv(out / "splits_rep1.csv")
            s2 = pd.read_csv(out / "splits_rep2.csv")
            self.assertFalse(s1.split.equals(s2.split))
            rep2 = pd.read_csv(out / "rep2.csv")
            for rule in ("MFI", "FIWL", "MPWI", "MEPV"):
                self.assertNotEqual(rep1[rep1.condition == rule].rmse.tolist(),
                                    rep2[rep2.condition == rule].rmse.tolist())
            agg = aggregate(out)
            self.assertEqual(agg[agg.condition == "MFI"].n_rep.tolist(), [2, 2, 2])
            self.assertEqual(agg[agg.condition == "proposed"].n_rep.tolist(), [2, 2, 2])
            self.assertTrue((out / "mean.png").is_file())
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["split_seed_scheme"], "split_seed + rep - 1")
            self.assertEqual(metadata["split_seed"], 20260904)
            dqn = rep1[rep1.condition.isin({"existing", "proposed"})]
            self.assertTrue((dqn.episodes == 70).all())
            self.assertTrue((dqn.transitions == 210).all())
            checkpoint = torch.load(out / "model_proposed_rep1.pt", weights_only=True)
            self.assertEqual(checkpoint["item_ids"], ["01", "03", "04"])
            with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                main(args + ["--rep", "3", "--n-epochs", "2"])

    def test_v1_layout_is_dropped_on_first_run(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            fixture(root)
            out = root / "out" / "EXP005" / DATASETS[0] / "main"
            out.mkdir(parents=True)
            (out / "analytic.csv").write_text("legacy\n")
            (out / "splits.csv").write_text("legacy\n")
            (out / "metadata.json").write_text(json.dumps({"schema_version": 1}))
            (out / "rep1.csv").write_text("legacy\n")
            args = ["--dataset", DATASETS[0], "--data-root", str(root), "--out", str(root / "out"),
                    "--test-length", "3", "--n-epochs", "1", "--threads", "1", "--rep", "1"]
            main(args)
            self.assertFalse((out / "analytic.csv").exists())
            self.assertFalse((out / "splits.csv").exists())
            self.assertTrue((out / "splits_rep1.csv").is_file())
            self.assertEqual(json.loads((out / "metadata.json").read_text())["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
