from __future__ import annotations

import unittest

import torch

from scripts.train_eval_rho2_crosssymbol_causal_20260804 import (
    QUANTILES,
    Rho2CrossSymbol,
    WINDOW_L,
)


class Rho2CrossSymbolArchitectureTests(unittest.TestCase):
    def test_forward_shapes_and_monotone_quantiles(self) -> None:
        model = Rho2CrossSymbol(
            n_temporal_features=4, n_snapshot_features=3, n_symbols=5, btc_id=0, d_model=16
        )
        btc_window = torch.randn(2, WINDOW_L, 4)
        panel_snapshot = torch.randn(2, 4, 3)
        symbol_ids = torch.tensor([[0, 1, 2, 3], [4, 2, 0, 1]])

        direction_logit, rank, quantiles = model(btc_window, panel_snapshot, symbol_ids)

        self.assertEqual(direction_logit.shape, (2,))
        self.assertEqual(rank.shape, (2,))
        self.assertEqual(quantiles.shape, (2, len(QUANTILES)))
        self.assertTrue(torch.all(quantiles[:, 1:] >= quantiles[:, :-1]))


if __name__ == "__main__":
    unittest.main()
