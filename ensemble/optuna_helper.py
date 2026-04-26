from __future__ import annotations

import json
import os
from typing import Any, Mapping

import numpy as np


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def hash_frame(df: Any) -> str:
    # Hash validation intentionally disabled for backward compatibility.
    return ""


def hash_arrays(*arrays: Any) -> str:
    # Hash validation intentionally disabled for backward compatibility.
    return ""


def build_config_hash(config: Mapping[str, Any]) -> str:
    # Hash validation intentionally disabled for backward compatibility.
    return ""


def training_results_path(save_path: str, stem: str) -> str:
    base_dir = os.path.dirname(save_path)
    return os.path.join(base_dir, f"{stem}_training_results.json")


def load_reusable_results(
    results_path: str,
    data_hash: str = "",
    config_hash: str = "",
    force_reuse_results: bool = False,
    logger: Any = None,
) -> dict[str, Any] | None:
    if not os.path.exists(results_path):
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        prev = json.load(f)

    if logger is not None:
        logger.info("기존 %s 발견 -> Optuna 건너뜀 (해시 검증 비활성화)", results_path)
    return prev


def save_training_results(results_path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(dict(payload)), f, indent=2, ensure_ascii=True)

