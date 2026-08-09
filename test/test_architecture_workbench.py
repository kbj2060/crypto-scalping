import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import pipeline.architecture_workbench as workbench


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArchitectureWorkbenchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=workbench.ROOT / "tmp")
        self.root = Path(self.temp.name)
        self.dataset = self.root / "features.csv"
        pd.DataFrame({
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00"],
            "close": [100.0, 101.0, 102.0], "volume": [2.0, 3.0, 4.0], "constant": [1.0, 1.0, 1.0],
        }).to_csv(self.dataset, index=False)
        self.manifest = self.root / "DATASET_MANIFEST.json"
        self.manifest.write_text(json.dumps({"files": {self.dataset.relative_to(workbench.ROOT).as_posix(): {"sha256": sha256(self.dataset)}}}))
        self.manifest_patch = patch.object(workbench, "DATASET_MANIFEST_PATH", self.manifest)
        self.manifest_patch.start()

    def tearDown(self):
        self.manifest_patch.stop()
        self.temp.cleanup()

    def contract(self):
        return {
            "schema_version": workbench.SCHEMA_VERSION, "experiment_id": "btc-new-v1", "hypothesis": "Test a causal feature.",
            "research": {"line_id": "new-line", "related_prior_line_ids": [], "prior_failure_reassessment": "", "retest_design": ""},
            "market": {"symbols": ["BTCUSDT"], "bar_interval": "5m"},
            "data": {"feature_dataset": self.dataset.relative_to(workbench.ROOT).as_posix(), "raw_sources": ["klines"]},
            "features": {"groups": ["price"], "timestamp_column": "timestamp", "analysis_target": "close"},
            "higher_timeframe": {"enabled": False, "availability_artifact": "", "decision_timestamp_column": "decision_timestamp", "source_available_at_column": "source_available_at"},
            "label": {"type": "triple_barrier", "horizon_bars": 48, "timeout_handling": "explicit_class"},
            "splits": dict(workbench.DEFAULT_SPLITS),
            "model": {"cheap_gate_family": "lightgbm", "candidate_architecture": "tabm", "seeds": ["270705"], "seed_ensemble_claim": False},
            "selection": {
                "minimum_trades_per_split": 15, "validation_pass_criteria": "positive net PnL",
                "effect_size_gate": {
                    "min_abs_t": 2.0, "min_permutation_percentile": 0.90,
                    "risk_channel_tested": False, "premise_checked_in_selection_window": False,
                    "falsification_audit_required": False,
                },
            },
            "execution": {"entry_timing": "next_bar_open", "cost_model": "conservative", "sizing_contract": "margin_fraction_times_leverage"},
            "evaluation": {"candidate_selection_scope": "validation_only", "final_evaluation": "fresh_forward_oos_only"},
        }

    def test_preflight_records_verified_dataset(self):
        evidence = workbench.run_preflight(self.contract())
        self.assertTrue(evidence["pass"])
        self.assertEqual(evidence["dataset"]["sha256"], sha256(self.dataset))

    def test_preflight_rejects_unregistered_or_drifted_dataset(self):
        self.manifest.write_text(json.dumps({"files": {}}))
        with self.assertRaisesRegex(ValueError, "not registered"):
            workbench.run_preflight(self.contract())

    def test_preflight_rejects_retired_regime4_column(self):
        df = pd.read_csv(self.dataset)
        df["clean_regime4_state24_chop_prob"] = 0.0
        df.to_csv(self.dataset, index=False)
        self.manifest.write_text(json.dumps({"files": {self.dataset.relative_to(workbench.ROOT).as_posix(): {"sha256": sha256(self.dataset)}}}))
        with self.assertRaisesRegex(ValueError, "Forbidden feature columns"):
            workbench.run_preflight(self.contract())

    def test_preflight_rejects_non_target_label_column(self):
        df = pd.read_csv(self.dataset)
        df["label_triple_barrier"] = 0
        df.to_csv(self.dataset, index=False)
        self.manifest.write_text(json.dumps({"files": {self.dataset.relative_to(workbench.ROOT).as_posix(): {"sha256": sha256(self.dataset)}}}))
        with self.assertRaisesRegex(ValueError, "Forbidden feature columns"):
            workbench.run_preflight(self.contract())

    def test_preflight_allows_declared_analysis_target_with_label_prefix(self):
        df = pd.read_csv(self.dataset)
        df["label_triple_barrier"] = [0.0, 1.0, 0.0]
        df.to_csv(self.dataset, index=False)
        self.manifest.write_text(json.dumps({"files": {self.dataset.relative_to(workbench.ROOT).as_posix(): {"sha256": sha256(self.dataset)}}}))
        contract = self.contract()
        contract["features"]["analysis_target"] = "label_triple_barrier"
        evidence = workbench.run_preflight(contract)
        self.assertTrue(evidence["pass"])
        stats, pairs, summary = workbench.analyze_features(contract, evidence)
        self.assertIn("spearman_to_target", stats.columns)

    def test_preflight_rejects_unfinished_hourly_bar(self):
        availability = self.root / "availability.csv"
        pd.DataFrame({
            "decision_timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00"],
            "source_available_at": ["2025-12-31 23:00:00", "2025-12-31 23:00:00", "2026-01-01 01:00:00"],
        }).to_csv(availability, index=False)
        contract = self.contract()
        contract["higher_timeframe"].update({"enabled": True, "availability_artifact": availability.relative_to(workbench.ROOT).as_posix()})
        with self.assertRaisesRegex(ValueError, "Higher-timeframe lookahead"):
            workbench.run_preflight(contract)

    def test_preflight_rejects_availability_artifact_that_does_not_cover_dataset(self):
        availability = self.root / "availability.csv"
        pd.DataFrame({"decision_timestamp": ["2026-01-01 00:55:00"], "source_available_at": ["2026-01-01 00:00:00"]}).to_csv(availability, index=False)
        contract = self.contract()
        contract["higher_timeframe"].update({"enabled": True, "availability_artifact": availability.relative_to(workbench.ROOT).as_posix()})
        with self.assertRaisesRegex(ValueError, "does not cover"):
            workbench.run_preflight(contract)

    def test_feature_analysis_requires_matching_preflight_and_writes_all_statistics(self):
        contract = self.contract()
        evidence = workbench.run_preflight(contract)
        stats, pairs, summary = workbench.analyze_features(contract, evidence)
        paths = workbench.write_feature_analysis(stats, pairs, summary, self.root / "analysis")
        self.assertTrue(paths["stats"].is_file())
        self.assertIn("constant", set(stats.loc[stats["is_constant"], "feature"]))
        self.assertIn("spearman_to_target", stats.columns)

    def test_refuses_to_overwrite_contract_artifact(self):
        output = self.root / "contract.json"
        workbench._write_new_json({"ok": True}, output)
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            workbench._write_new_json({"ok": False}, output)

    def test_prior_line_is_retestable_with_rationale(self):
        contract = self.contract()
        contract["research"] = {
            "line_id": "btc_tpfirst_three_way_label", "related_prior_line_ids": [],
            "prior_failure_reassessment": "Reproduce under matching cost accounting.",
            "retest_design": "Use a frozen dataset and untouched fresh-forward window.",
        }
        workbench.validate_contract(contract)

    def test_seed_ensemble_claim_requires_at_least_five_seeds(self):
        contract = self.contract()
        contract["model"]["seed_ensemble_claim"] = True
        contract["model"]["seeds"] = ["1", "2", "3"]
        with self.assertRaisesRegex(ValueError, "at least 5 seeds"):
            workbench.validate_contract(contract)

    def test_seed_ensemble_claim_rejects_fixed_increment_cluster(self):
        contract = self.contract()
        contract["model"]["seed_ensemble_claim"] = True
        contract["model"]["seeds"] = ["270705", "270710", "270715", "270720", "270725"]
        with self.assertRaisesRegex(ValueError, "fixed-increment cluster"):
            workbench.validate_contract(contract)

    def test_seed_ensemble_claim_accepts_genuinely_diverse_seeds(self):
        contract = self.contract()
        contract["model"]["seed_ensemble_claim"] = True
        contract["model"]["seeds"] = ["4821", "193042", "77", "1000003", "58291"]
        workbench.validate_contract(contract)

    def test_seeds_without_ensemble_claim_are_unrestricted(self):
        contract = self.contract()
        contract["model"]["seed_ensemble_claim"] = False
        contract["model"]["seeds"] = ["270705"]
        workbench.validate_contract(contract)

    def test_revise_rejects_noop_revision(self):
        contract = self.contract()
        source = self.root / "contract.json"
        workbench._write_new_json(contract, source)
        with self.assertRaisesRegex(ValueError, "identical"):
            workbench.write_revision(source, self.contract(), self.root / "revised.json")

    def test_revise_accepts_changed_contract(self):
        contract = self.contract()
        source = self.root / "contract.json"
        workbench._write_new_json(contract, source)
        changed = self.contract()
        changed["hypothesis"] = "A materially different hypothesis."
        output = self.root / "revised.json"
        workbench.write_revision(source, changed, output)
        written = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(written["revision_of"], str(source))
        self.assertEqual(written["hypothesis"], "A materially different hypothesis.")


class EffectSizeGateFalsificationTest(unittest.TestCase):
    """assert_effect_size_gate's falsification-audit wiring, isolated from the (currently
    stale, pre-existing) full-contract fixture above -- this only needs selection.effect_size_gate.
    """

    def contract(self, **gate_overrides):
        gate = {
            "min_abs_t": 2.0, "min_permutation_percentile": 0.90,
            "risk_channel_tested": False, "premise_checked_in_selection_window": False,
            "falsification_audit_required": True, "min_falsification_percentile": 0.95,
        }
        gate.update(gate_overrides)
        return {"selection": {"effect_size_gate": gate}}

    def passing_report(self):
        return {"welch_t": 3.0, "mean_diff": 1.0}

    def test_requires_a_falsification_report_when_declared_required(self):
        with self.assertRaisesRegex(ValueError, "falsification_audit_required declared but no"):
            workbench.assert_effect_size_gate(self.contract(), self.passing_report())

    def test_rejects_a_falsification_report_below_the_required_percentile(self):
        falsification = {"zero_predictability_percentile": 0.99, "microstructure_placebo_percentile": 0.40}
        with self.assertRaisesRegex(ValueError, "microstructure_placebo_percentile 0.400 < required 0.95"):
            workbench.assert_effect_size_gate(self.contract(), self.passing_report(), falsification=falsification)

    def test_passes_a_falsification_report_above_the_required_percentile(self):
        falsification = {"zero_predictability_percentile": 0.99, "microstructure_placebo_percentile": 0.98}
        workbench.assert_effect_size_gate(self.contract(), self.passing_report(), falsification=falsification)

    def test_falsification_audit_not_enforced_when_gate_does_not_require_it(self):
        workbench.assert_effect_size_gate(self.contract(falsification_audit_required=False), self.passing_report())

    def test_uses_the_gates_own_percentile_threshold(self):
        falsification = {"zero_predictability_percentile": 0.80, "microstructure_placebo_percentile": 0.80}
        workbench.assert_effect_size_gate(
            self.contract(min_falsification_percentile=0.75), self.passing_report(), falsification=falsification
        )
