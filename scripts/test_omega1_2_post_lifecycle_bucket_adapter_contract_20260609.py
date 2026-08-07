#!/usr/bin/env python3
from __future__ import annotations

import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import train_eval_omega1_2_post_lifecycle_bucket_adapter_20260605 as adapter


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega1_2_post_lifecycle_bucket_adapter_20260605_hgb_base_lev5_eff_cap150_comp_tpup_voltarget_trainall_c96_replayk2_s260726"
)


def configure_s260726_contract() -> None:
    adapter._set_bucket_preset("base")
    adapter.LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    adapter.ENTRY_BUCKETS = np.asarray(
        [
            (t, s, n, l)
            for t in range(len(adapter.TP_BUCKETS))
            for s in range(len(adapter.SL_BUCKETS))
            for n in range(len(adapter.NOTIONAL_BUCKETS))
            for l in range(len(adapter.LEVERAGE_BUCKETS))
        ],
        dtype=np.int64,
    )
    adapter.NOTIONAL_MULT = 1.0
    adapter.NOTIONAL_CAP = 1.5
    adapter.USE_LEVERAGE_EXPOSURE = True
    adapter.COMPENSATE_SLTP_BY_NOTIONAL = True
    adapter.COMPENSATE_REF_NOTIONAL = 0.45


class Omega12PostLifecycleAdapterContractTest(unittest.TestCase):
    def setUp(self) -> None:
        configure_s260726_contract()

    def test_s260726_risk_ids_use_true_leverage_exposure_and_compensated_sltp(self) -> None:
        risk = adapter._risk_from_ids((2, 2, 4, 2))

        self.assertAlmostEqual(risk["margin_notional"], 0.55, places=6)
        self.assertAlmostEqual(risk["leverage"], 3.0, places=6)
        self.assertAlmostEqual(risk["notional"], 1.5, places=6)
        self.assertAlmostEqual(risk["tp"], 0.026 / 0.45 * 1.5, places=6)
        self.assertAlmostEqual(risk["sl"], 0.012 / 0.45 * 1.5, places=6)

    def test_s260726_leverage_cap_reduces_only_leverage_bucket(self) -> None:
        medium, reason_medium = adapter._apply_vol_leverage_cap(
            np.asarray([2, 2, 4, 4], dtype=np.int64),
            0.006,
            enabled=True,
            high=0.008,
            medium=0.005,
        )
        high, reason_high = adapter._apply_vol_leverage_cap(
            np.asarray([2, 2, 4, 4], dtype=np.int64),
            0.009,
            enabled=True,
            high=0.008,
            medium=0.005,
        )

        self.assertEqual(reason_medium, "vol_cap_medium")
        self.assertEqual(tuple(medium), (2, 2, 4, 1))
        self.assertEqual(reason_high, "vol_cap_high")
        self.assertEqual(tuple(high), (2, 2, 4, 0))

    def test_adapter_feature_row_rejects_forbidden_features(self) -> None:
        row = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-01")],
                "safe_feature": [1.0],
                "clean_regime4_state": [0.2],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "forbidden post adapter features"):
            adapter._adapter_feature_row(row, adapter.lifecycle.ENTER_BASE)

    def test_normalizer_is_fail_fast_on_column_contract_mismatch(self) -> None:
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        _x, norm = adapter._fit_norm(train)

        with self.assertRaisesRegex(RuntimeError, "column contract mismatch"):
            adapter._apply_norm(pd.DataFrame({"b": [5.0], "a": [2.0]}), norm)

    def test_s260726_artifact_matches_expected_bucket_contract(self) -> None:
        artifact_path = REPORT_DIR / "post_bucket_adapter.pkl"
        self.assertTrue(artifact_path.exists(), str(artifact_path))

        with artifact_path.open("rb") as f:
            artifact = pickle.load(f)

        self.assertEqual(len(artifact["tp_buckets"]), 5)
        self.assertEqual(len(artifact["sl_buckets"]), 5)
        self.assertEqual(len(artifact["notional_buckets"]), 5)
        self.assertEqual(len(artifact["leverage_buckets"]), 5)
        self.assertEqual(artifact["entry_buckets"].shape[0], 625)
        self.assertIn("tp", artifact["models"])
        self.assertIn("sl", artifact["models"])
        self.assertIn("notional", artifact["models"])
        self.assertIn("leverage", artifact["models"])


if __name__ == "__main__":
    unittest.main()
