from __future__ import annotations

import unittest

import torch

from core.multiasset_tft import MultiAssetTFT


class MultiAssetTFTTests(unittest.TestCase):
    def test_forecast_is_monotone_and_explainable(self) -> None:
        model = MultiAssetTFT(
            n_asset_features=5, n_global_features=3, n_assets=8, quantile_count=7,
            d_model=16, n_heads=4,
        )
        output = model(
            torch.randn(2, 12, 4, 5), torch.tensor([[0, 1, 2, 3], [4, 5, 2, 7]]),
            torch.tensor([0, 2]), torch.randn(2, 12, 3),
        )
        self.assertEqual(output.quantiles.shape, (2, 7))
        self.assertEqual(output.entry_logits.shape, (2, 3))
        self.assertEqual(output.regime_logits.shape, (2, 3))
        self.assertEqual(output.exit_logits.shape, (2, 3))
        self.assertTrue(torch.all(output.quantiles[:, 1:] >= output.quantiles[:, :-1]))
        self.assertEqual(output.asset_variable_weights.shape, (2, 12, 4, 5))
        self.assertTrue(torch.allclose(output.asset_variable_weights.sum(dim=-1), torch.ones(2, 12, 4)))
        self.assertEqual(output.target_asset_attention.shape, (2, 4))
        self.assertTrue(torch.allclose(output.target_asset_attention.sum(dim=-1), torch.ones(2), atol=1e-5))
        self.assertEqual(output.global_variable_weights.shape, (2, 12, 3))

    def test_rejects_missing_required_global_history(self) -> None:
        model = MultiAssetTFT(n_asset_features=2, n_global_features=1, n_assets=3, quantile_count=3, d_model=8, n_heads=2)
        with self.assertRaisesRegex(ValueError, "global_history"):
            model(torch.randn(1, 4, 2, 2), torch.tensor([[0, 1]]), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
