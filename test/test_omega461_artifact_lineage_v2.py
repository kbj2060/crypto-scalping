from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trading_bot_modules.omega4_6_1_runtime_contract import validate_sidecar_lineage


class Omega461ArtifactLineageTests(unittest.TestCase):
    def _fixture(self) -> tuple[Path, Path, Path]:
        root = Path(tempfile.mkdtemp())
        prediction_dir = root / "parent"
        sidecar_dir = root / "sidecar"
        prediction_dir.mkdir()
        sidecar_dir.mkdir()
        bundle = prediction_dir / "true_3head_tabm_bundle.pt"
        sidecar = sidecar_dir / "risk_sidecar.pkl"
        bundle.write_bytes(b"bundle")
        sidecar.write_bytes(b"sidecar")
        for split in ("train", "validation", "oos"):
            (prediction_dir / f"{split}_predictions_q060.csv").write_text(
                "timestamp,prediction\n", encoding="utf-8"
            )
        report = {
            "risk_model": {
                "selection_scope": "validation_only",
                "precomputed_prediction_dir": str(prediction_dir),
                "precomputed_prediction_tag": "q060",
            },
            "contract": {"quality_threshold": 0.60},
        }
        (sidecar_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return root, bundle, sidecar

    def test_exact_validation_only_lineage_passes(self) -> None:
        root, bundle, sidecar = self._fixture()
        result = validate_sidecar_lineage(
            repo_root=root,
            bundle_path=bundle,
            sidecar_path=sidecar,
            quality_threshold=0.60,
        )
        self.assertEqual(result["prediction_tag"], "q060")

    def test_oos_guard_selection_fails(self) -> None:
        root, bundle, sidecar = self._fixture()
        report_path = sidecar.parent / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["risk_model"]["selection_scope"] = "validation_oos_guard"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "validation_only"):
            validate_sidecar_lineage(
                repo_root=root,
                bundle_path=bundle,
                sidecar_path=sidecar,
                quality_threshold=0.60,
            )

    def test_wrong_threshold_tag_fails(self) -> None:
        root, bundle, sidecar = self._fixture()
        report_path = sidecar.parent / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["risk_model"]["precomputed_prediction_tag"] = "q055"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "prediction tag"):
            validate_sidecar_lineage(
                repo_root=root,
                bundle_path=bundle,
                sidecar_path=sidecar,
                quality_threshold=0.60,
            )

    def test_missing_exact_prediction_file_fails(self) -> None:
        root, bundle, sidecar = self._fixture()
        (bundle.parent / "oos_predictions_q060.csv").unlink()

        with self.assertRaisesRegex(ValueError, "missing exact prediction"):
            validate_sidecar_lineage(
                repo_root=root,
                bundle_path=bundle,
                sidecar_path=sidecar,
                quality_threshold=0.60,
            )


if __name__ == "__main__":
    unittest.main()
