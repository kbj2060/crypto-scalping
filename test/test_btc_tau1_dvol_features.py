from __future__ import annotations

import unittest

from scripts.build_btc_tau1_dvol_features_20260804 import TAU1_BTC_FEATURE_COLS


class Tau1BtcDvolContractTest(unittest.TestCase):
    def test_tau1_contract_has_38_features_plus_dvol(self):
        self.assertEqual(len(TAU1_BTC_FEATURE_COLS), 39)
        self.assertEqual(TAU1_BTC_FEATURE_COLS[-1], "dvol_btc")
        self.assertNotIn("btc_logret_1", TAU1_BTC_FEATURE_COLS)
        self.assertIn("eth_logret_1", TAU1_BTC_FEATURE_COLS)
        self.assertIn("btc_eth_spread_6", TAU1_BTC_FEATURE_COLS)

