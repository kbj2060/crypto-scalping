"""Contract, preflight, and feature-analysis workbench for new experiments.

The contract captures research intent.  A separate preflight inspects the actual
dataset and writes immutable evidence; feature analysis consumes that evidence.
Neither stage accepts a user-provided boolean as proof of causal correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "architecture_experiment_contract_v2"
PREFLIGHT_SCHEMA_VERSION = "architecture_preflight_v1"
FEATURE_ANALYSIS_SCHEMA_VERSION = "architecture_feature_analysis_v1"
PRIOR_RESEARCH_REGISTRY_PATH = ROOT / "docs/model_contracts/research_line_registry.json"
DATASET_MANIFEST_PATH = ROOT / "data/splits/DATASET_MANIFEST.json"
DEFAULT_SPLITS = {
    "train_end": "2025-08-31", "validation_start": "2025-09-01",
    "validation_end": "2025-12-31", "oos_start": "2026-01-01", "oos_end": "2026-03-31",
}
RETIRED_FEATURE_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_", "regime4_pred_")
LEAKAGE_FEATURE_PREFIXES = ("future_", "label_", "target_", "exit_")
FORBIDDEN_FEATURE_PREFIXES = RETIRED_FEATURE_PREFIXES + LEAKAGE_FEATURE_PREFIXES


def _ask(prompt: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or (default or "")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _yes_no(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "true", "1", "예", "네", "사용"}:
        return True
    if normalized in {"no", "n", "false", "0", "아니오", "아니요", "미사용"}:
        return False
    raise ValueError(f"Expected yes/no (or 예/아니오), got: {value!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_hash(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode()).hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _relative_to_root(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path must be inside the repository: {path}") from exc


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported feature dataset format: {path.suffix}")


def load_prior_research_registry() -> dict[str, Any]:
    return json.loads(PRIOR_RESEARCH_REGISTRY_PATH.read_text(encoding="utf-8"))


def prior_research_line_ids() -> set[str]:
    return {entry["id"] for entry in load_prior_research_registry()["prior_lines"]}


def assert_safe_feature_columns(columns: Any, *, exempt: set[str] | None = None) -> None:
    """Retired-surface columns are always forbidden. Leakage-prefixed columns (future_/label_/
    target_/exit_) may be exempted only for the one column the contract declared as its
    analysis target -- that column is read for correlation, never used as a model input."""
    exempt = exempt or set()
    bad: list[str] = []
    for column in columns:
        name = str(column)
        if name.startswith(RETIRED_FEATURE_PREFIXES):
            bad.append(name)
        elif name in exempt:
            continue
        elif name.startswith(LEAKAGE_FEATURE_PREFIXES):
            bad.append(name)
    if bad:
        raise ValueError(f"Forbidden feature columns: {bad[:20]}")


def _is_clustered_seed_list(seeds: list[int]) -> bool:
    """A sorted seed list where every consecutive gap is identical is a fixed-increment
    draw (e.g. base, base+5, base+10, ...), not genuinely diverse random seeds."""
    if len(seeds) < 3:
        return False
    ordered = sorted(seeds)
    diffs = {b - a for a, b in zip(ordered, ordered[1:])}
    return len(diffs) == 1


def assert_higher_timeframe_availability(decision_timestamps: Any, source_available_at: Any) -> None:
    decision = pd.to_datetime(pd.Series(decision_timestamps), errors="raise", utc=True)
    available = pd.to_datetime(pd.Series(source_available_at), errors="raise", utc=True)
    if len(decision) != len(available):
        raise ValueError("higher-timeframe availability check needs equal-length inputs")
    violations = available > decision
    if violations.any():
        raise ValueError(f"Higher-timeframe lookahead: {int(violations.sum())} rows use unfinished future bars")


def _write_new_json(payload: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}. Create a revision instead.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_interactive_contract() -> dict[str, Any]:
    print("새 아키텍처 계약을 만듭니다. 실제 검사는 preflight 단계에서 수행됩니다.")
    use_higher_timeframe = _yes_no(_ask("1h/4h 등 상위 시간봉 피처 사용 여부 (yes/no 또는 예/아니오)", default="no"))
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": _ask("실험 ID (영문/숫자/_/-)"),
        "hypothesis": _ask("검증할 한 문장 가설"),
        "research": {
            "line_id": _ask("연구 라인 ID"),
            "related_prior_line_ids": _csv(_ask("관련된 과거 연구 라인 ID (없으면 비움)")),
            "prior_failure_reassessment": _ask("과거 실패가 이번에는 왜 달라질 수 있는지"),
            "retest_design": _ask("과거 결과와 구분되는 재검증 설계"),
        },
        "market": {"symbols": _csv(_ask("대상 심볼", default="BTCUSDT")), "bar_interval": _ask("bar 주기", default="5m")},
        "data": {"feature_dataset": _ask("고정된 feature dataset 경로"), "raw_sources": _csv(_ask("원천 데이터 종류"))},
        "features": {"groups": _csv(_ask("사용할 피처 그룹")), "timestamp_column": _ask("timestamp 컬럼", default="timestamp"), "analysis_target": _ask("상관 분석 target 컬럼 (없으면 비움)")},
        "higher_timeframe": {
            "enabled": use_higher_timeframe,
            "availability_artifact": _ask("1h availability CSV/Parquet 경로", default="") if use_higher_timeframe else "",
            "decision_timestamp_column": "decision_timestamp",
            "source_available_at_column": "source_available_at",
        },
        "label": {"type": _ask("라벨 유형"), "horizon_bars": _ask("라벨 horizon (bar)", default="48"), "timeout_handling": _ask("timeout 처리", default="explicit_class")},
        "splits": {key: _ask(key, default=value) for key, value in DEFAULT_SPLITS.items()},
        "model": {
            "cheap_gate_family": _ask("첫 cheap-falsification 모델", default="lightgbm"),
            "candidate_architecture": _ask("후보 아키텍처"),
            "seeds": _csv(_ask("학습 seed", default="270705")),
            "seed_ensemble_claim": _yes_no(_ask("복수 시드 평균/배깅으로 승격을 주장합니까 (yes/no 또는 예/아니오)", default="no")),
        },
        "selection": {"minimum_trades_per_split": _ask("split별 최소 거래 수", default="15"), "validation_pass_criteria": _ask("Validation 통과 기준")},
        "execution": {"entry_timing": _ask("체결 규칙", default="next_bar_open"), "cost_model": _ask("비용 모델 식별자"), "sizing_contract": "margin_fraction_times_leverage"},
        "evaluation": {"candidate_selection_scope": "validation_only", "final_evaluation": "fresh_forward_oos_only"},
    }


def validate_contract(contract: dict[str, Any]) -> None:
    errors: list[str] = []
    if contract.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(contract.get("experiment_id", ""))):
        errors.append("experiment_id must contain only letters, digits, _ or -")
    for path, value in (("hypothesis", contract.get("hypothesis")), ("research.line_id", contract.get("research", {}).get("line_id")), ("data.feature_dataset", contract.get("data", {}).get("feature_dataset")), ("model.cheap_gate_family", contract.get("model", {}).get("cheap_gate_family")), ("model.candidate_architecture", contract.get("model", {}).get("candidate_architecture")), ("execution.cost_model", contract.get("execution", {}).get("cost_model")), ("selection.validation_pass_criteria", contract.get("selection", {}).get("validation_pass_criteria"))):
        if not str(value or "").strip():
            errors.append(f"missing {path}")
    research = contract.get("research", {})
    prior_ids = prior_research_line_ids()
    related = set(research.get("related_prior_line_ids", []))
    if unknown := related - prior_ids:
        errors.append(f"research.related_prior_line_ids has unknown IDs: {sorted(unknown)}")
    if str(research.get("line_id", "")) in prior_ids or related:
        for field in ("prior_failure_reassessment", "retest_design"):
            if not str(research.get(field, "")).strip():
                errors.append(f"research.{field} is required when retesting a prior line")
    if not contract.get("market", {}).get("symbols") or not contract.get("market", {}).get("bar_interval"):
        errors.append("market.symbols and market.bar_interval are required")
    if int(contract.get("label", {}).get("horizon_bars", 0)) <= 0:
        errors.append("label.horizon_bars must be > 0")
    if contract.get("label", {}).get("timeout_handling") not in {"explicit_class", "mark_to_market", "not_applicable"}:
        errors.append("label.timeout_handling is invalid")
    if int(contract.get("selection", {}).get("minimum_trades_per_split", 0)) <= 0:
        errors.append("selection.minimum_trades_per_split must be > 0")
    model = contract.get("model", {})
    if model.get("seed_ensemble_claim"):
        try:
            seeds_int = [int(seed) for seed in model.get("seeds", [])]
        except (TypeError, ValueError):
            errors.append("model.seeds must all be integers when seed_ensemble_claim is true")
            seeds_int = []
        if seeds_int:
            if len(seeds_int) < 5:
                errors.append("model.seed_ensemble_claim requires at least 5 seeds (seed-diversity gate)")
            elif _is_clustered_seed_list(seeds_int):
                errors.append("model.seeds is a fixed-increment cluster, not genuinely diverse random seeds (seed-diversity gate)")
    if contract.get("execution", {}).get("sizing_contract") != "margin_fraction_times_leverage":
        errors.append("execution.sizing_contract must be margin_fraction_times_leverage")
    if contract.get("evaluation", {}).get("candidate_selection_scope") != "validation_only":
        errors.append("candidate selection must be validation_only")
    if contract.get("evaluation", {}).get("final_evaluation") != "fresh_forward_oos_only":
        errors.append("final evaluation must be fresh_forward_oos_only")
    higher = contract.get("higher_timeframe", {})
    if higher.get("enabled") and not str(higher.get("availability_artifact", "")).strip():
        errors.append("higher_timeframe.availability_artifact is required when enabled")
    if errors:
        raise ValueError("Invalid architecture experiment contract:\n- " + "\n- ".join(errors))


def run_preflight(contract: dict[str, Any]) -> dict[str, Any]:
    """Inspect the actual pinned frame and return evidence for later stages."""
    validate_contract(contract)
    path = _resolve_path(contract["data"]["feature_dataset"])
    if not path.is_file():
        raise FileNotFoundError(f"Feature dataset does not exist: {path}")
    rel = _relative_to_root(path)
    manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = manifest.get("files", {}).get(rel)
    if entry is None:
        raise ValueError(f"Feature dataset is not registered in DATASET_MANIFEST: {rel}")
    digest = _sha256_file(path)
    if digest != entry.get("sha256"):
        raise ValueError(f"Feature dataset hash drift: manifest={entry.get('sha256')} current={digest}")
    frame = _read_frame(path)
    timestamp_col = contract["features"]["timestamp_column"]
    if timestamp_col not in frame.columns:
        raise ValueError(f"Feature dataset missing timestamp column: {timestamp_col}")
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Feature timestamps must be valid, unique, and monotonically increasing")
    feature_columns = [str(c) for c in frame.columns if c != timestamp_col]
    target_col = str(contract["features"].get("analysis_target") or "")
    assert_safe_feature_columns(feature_columns, exempt={target_col} if target_col else None)
    higher_result: dict[str, Any] = {"enabled": bool(contract["higher_timeframe"].get("enabled")), "rows_checked": 0}
    if higher_result["enabled"]:
        availability_path = _resolve_path(contract["higher_timeframe"]["availability_artifact"])
        availability = _read_frame(availability_path)
        decision_col = contract["higher_timeframe"]["decision_timestamp_column"]
        source_col = contract["higher_timeframe"]["source_available_at_column"]
        if decision_col not in availability or source_col not in availability:
            raise ValueError(f"Availability artifact requires {decision_col} and {source_col}")
        assert_higher_timeframe_availability(availability[decision_col], availability[source_col])
        availability_decisions = set(pd.to_datetime(availability[decision_col], errors="raise", utc=True))
        dataset_decisions = set(timestamps)
        uncovered = dataset_decisions - availability_decisions
        if uncovered:
            raise ValueError(
                f"Higher-timeframe availability artifact does not cover {len(uncovered)} of the pinned "
                "dataset's own decision timestamps; it must be built from this dataset's actual decision "
                "points, not a standalone sample"
            )
        higher_result.update({"availability_artifact": _relative_to_root(availability_path), "rows_checked": int(len(availability))})
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION, "pass": True, "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": contract["experiment_id"],
        "dataset": {"path": rel, "sha256": digest, "manifest_sha256": entry["sha256"], "rows": int(len(frame)), "schema_hash": _schema_hash([str(c) for c in frame.columns]), "columns": [str(c) for c in frame.columns]},
        "checks": {"forbidden_feature_columns": [], "timestamp_valid_unique_sorted": True, "higher_timeframe": higher_result},
    }


def validate_preflight_artifact(contract: dict[str, Any], preflight_path: Path) -> dict[str, Any]:
    evidence = json.loads(preflight_path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != PREFLIGHT_SCHEMA_VERSION or evidence.get("pass") is not True:
        raise ValueError("Invalid or failed preflight artifact")
    if evidence.get("experiment_id") != contract.get("experiment_id"):
        raise ValueError("Preflight experiment_id does not match contract")
    path = _resolve_path(contract["data"]["feature_dataset"])
    if evidence.get("dataset", {}).get("sha256") != _sha256_file(path):
        raise ValueError("Dataset changed after preflight; rerun preflight")
    return evidence


def analyze_features(contract: dict[str, Any], preflight: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return statistics and every numeric-pair correlation from preflighted data."""
    path = _resolve_path(contract["data"]["feature_dataset"])
    if preflight.get("dataset", {}).get("sha256") != _sha256_file(path):
        raise ValueError("Dataset changed after preflight; rerun preflight before feature analysis")
    frame = _read_frame(path)
    timestamp_col = contract["features"]["timestamp_column"]
    feature_cols = [c for c in frame.columns if c != timestamp_col]
    target = contract["features"].get("analysis_target", "")
    assert_safe_feature_columns(feature_cols, exempt={target} if target else None)
    rows: list[dict[str, Any]] = []
    numeric: dict[str, pd.Series] = {}
    for col in feature_cols:
        raw = frame[col]
        values = pd.to_numeric(raw, errors="coerce")
        numeric_col = pd.api.types.is_numeric_dtype(raw)
        finite = values[np.isfinite(values)] if numeric_col else pd.Series(dtype=float)
        row = {"feature": col, "dtype": str(raw.dtype), "rows": int(len(raw)), "missing_count": int(raw.isna().sum()), "missing_ratio": float(raw.isna().mean()), "unique_count": int(raw.nunique(dropna=True)), "is_numeric": bool(numeric_col), "infinite_count": int(np.isinf(values).sum()) if numeric_col else 0, "is_constant": bool(raw.nunique(dropna=True) <= 1)}
        if numeric_col and not finite.empty:
            row.update({"mean": float(finite.mean()), "std": float(finite.std(ddof=0)), "min": float(finite.min()), "q01": float(finite.quantile(.01)), "q50": float(finite.quantile(.50)), "q99": float(finite.quantile(.99)), "max": float(finite.max())})
            numeric[str(col)] = values.replace([np.inf, -np.inf], np.nan)
        rows.append(row)
    stats = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    numeric_frame = pd.DataFrame(numeric)
    non_constant_numeric = [col for col in numeric_frame if numeric_frame[col].nunique(dropna=True) > 1]
    correlations = numeric_frame[non_constant_numeric].corr(method="spearman", min_periods=2)
    pairs = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "spearman_correlation"]
    pairs["abs_spearman_correlation"] = pairs["spearman_correlation"].abs()
    pairs = pairs.sort_values("abs_spearman_correlation", ascending=False).reset_index(drop=True)
    if target:
        if target not in numeric_frame:
            raise ValueError(f"analysis_target must be a numeric feature column: {target}")
        target_corr = numeric_frame[non_constant_numeric].corrwith(numeric_frame[target], method="spearman").rename("spearman_to_target")
        stats = stats.merge(target_corr, left_on="feature", right_index=True, how="left")
    duplicate_pairs: list[list[str]] = []
    seen: dict[str, str] = {}
    for col in numeric_frame:
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(numeric_frame[col], index=False).values.tobytes()).hexdigest()
        if fingerprint in seen and numeric_frame[col].equals(numeric_frame[seen[fingerprint]]):
            duplicate_pairs.append([seen[fingerprint], col])
        else:
            seen[fingerprint] = col
    summary = {"schema_version": FEATURE_ANALYSIS_SCHEMA_VERSION, "experiment_id": contract["experiment_id"], "preflight_dataset_sha256": preflight["dataset"]["sha256"], "feature_count": int(len(feature_cols)), "numeric_feature_count": int(len(numeric_frame.columns)), "constant_feature_count": int(stats["is_constant"].sum()), "exact_duplicate_numeric_feature_pairs": duplicate_pairs, "high_correlation_pairs_abs_ge_0_98": int((pairs["abs_spearman_correlation"] >= .98).sum())}
    return stats, pairs, summary


def write_feature_analysis(stats: pd.DataFrame, pairs: pd.DataFrame, summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = summary["experiment_id"] + "_feature_analysis"
    paths = {"stats": output_dir / f"{stem}_stats.csv", "correlations": output_dir / f"{stem}_correlations.csv", "summary": output_dir / f"{stem}.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite feature-analysis artifacts; create a revision directory")
    stats.to_csv(paths["stats"], index=False)
    pairs.to_csv(paths["correlations"], index=False)
    _write_new_json(summary, paths["summary"])
    return paths


def write_revision(original_path: Path, revised: dict[str, Any], output: Path) -> None:
    original = json.loads(original_path.read_text(encoding="utf-8"))
    comparable_original = {k: v for k, v in original.items() if k != "revision_of"}
    comparable_revised = {k: v for k, v in revised.items() if k != "revision_of"}
    if comparable_original == comparable_revised:
        raise ValueError("Revision is identical to the original contract; no changes were made")
    payload = dict(revised)
    payload["revision_of"] = str(original_path)
    _write_new_json(payload, output)


def main() -> int:
    parser = argparse.ArgumentParser(description="New architecture contract, preflight, and feature analysis.")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init"); init.add_argument("--output", type=Path, required=True)
    revise = commands.add_parser("revise"); revise.add_argument("contract", type=Path); revise.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate"); validate.add_argument("contract", type=Path); validate.add_argument("--preflight", type=Path)
    preflight = commands.add_parser("preflight"); preflight.add_argument("contract", type=Path); preflight.add_argument("--output", type=Path, required=True)
    analysis = commands.add_parser("analyze-features"); analysis.add_argument("contract", type=Path); analysis.add_argument("--preflight", type=Path, required=True); analysis.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        contract = build_interactive_contract(); validate_contract(contract); _write_new_json(contract, args.output); print(f"[PASS] contract written: {args.output}"); return 0
    contract = json.loads(args.contract.read_text(encoding="utf-8")); validate_contract(contract)
    if args.command == "revise":
        write_revision(args.contract, contract, args.output); print(f"[PASS] revision written: {args.output}"); return 0
    if args.command == "validate":
        if args.preflight:
            validate_preflight_artifact(contract, args.preflight)
        else:
            print("[INFO] no --preflight given; only contract fields were validated, not the dataset")
        print(f"[PASS] valid contract: {args.contract}"); return 0
    if args.command == "preflight":
        _write_new_json(run_preflight(contract), args.output); print(f"[PASS] preflight written: {args.output}"); return 0
    evidence = validate_preflight_artifact(contract, args.preflight)
    stats, pairs, summary = analyze_features(contract, evidence)
    paths = write_feature_analysis(stats, pairs, summary, args.output_dir)
    print(f"[PASS] feature analysis written: {paths['summary']}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
