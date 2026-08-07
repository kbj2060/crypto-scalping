import unittest

import pandas as pd

from pipeline.btc_trajectory_teacher import (
    TeacherConfig,
    build_state_conditioned_teacher_labels,
    build_teacher_path,
    first_action_utilities,
)


class BtcTrajectoryTeacherTest(unittest.TestCase):
    def test_builds_bounded_margin_and_next_bar_execution_labels(self):
        timestamps = pd.date_range("2026-01-01", periods=80, freq="5min")
        frame = pd.DataFrame({"timestamp": timestamps, "open": range(100, 180), "close": range(101, 181)})
        labels = build_teacher_path(frame, TeacherConfig(horizon_bars=8))
        self.assertFalse(labels.empty)
        self.assertLessEqual(labels["hard_target_margin_fraction"].abs().max(), .30)
        self.assertTrue(((labels["teacher_short_probability"] + labels["teacher_flat_probability"] + labels["teacher_long_probability"] - 1.0).abs() < 1e-8).all())
        self.assertTrue(labels["teacher_margin_fraction"].between(0.0, .30).all())
        self.assertNotIn("teacher_exit_now_binary", labels.columns)
        self.assertEqual(labels["execution_timestamp"].iloc[0], timestamps[1].tz_localize("UTC"))
        self.assertTrue(set(labels["exit_label"]).issubset({"hold", "exit_full", "enter", "reverse", "exit_partial", "increase"}))

    def test_state_conditioned_labels_expose_only_a_causal_current_margin_state(self):
        timestamps = pd.date_range("2026-01-01", periods=30, freq="5min")
        frame = pd.DataFrame({"timestamp": timestamps, "open": range(100, 130), "close": range(101, 131)})
        config = TeacherConfig(margin_step=.15, horizon_bars=8)
        labels = build_state_conditioned_teacher_labels(frame, config)
        self.assertEqual(len(labels), (len(frame) - config.horizon_bars - 1) * len(config.actions))
        self.assertEqual(set(labels["current_margin_fraction"].unique()), set(config.actions))
        probability_columns = [f"teacher_action_{action:+.2f}_probability" for action in config.actions]
        self.assertTrue(((labels[probability_columns].sum(axis=1) - 1.0).abs() < 1e-8).all())
        self.assertTrue((labels["teacher_switch_advantage"] >= 0.0).all())
        first_state = labels.loc[labels["current_margin_fraction"] == config.actions[0]].iloc[0]
        future_returns = (
            frame["close"].iloc[1 : 1 + config.horizon_bars].to_numpy()
            / frame["open"].iloc[1 : 1 + config.horizon_bars].to_numpy() - 1.0
        )
        expected = first_action_utilities(float(config.actions[0]), future_returns, config)
        actual = [first_state[f"teacher_action_{action:+.2f}_utility"] for action in config.actions]
        self.assertTrue(((expected - actual) < 1e-10).all())
