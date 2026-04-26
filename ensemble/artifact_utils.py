from __future__ import annotations

import json
import os
import pickle
from typing import Any


def resolve_model_meta_paths(save_path: str, model_ext: str = ".pkl") -> tuple[str, str]:
    lower = save_path.lower()
    if lower.endswith(".json"):
        meta_path = save_path
        model_path = os.path.splitext(save_path)[0] + model_ext
        return model_path, meta_path
    if lower.endswith(model_ext):
        model_path = save_path
        meta_path = os.path.splitext(save_path)[0] + ".json"
        return model_path, meta_path
    prefix = os.path.splitext(save_path)[0]
    return prefix + model_ext, prefix + ".json"


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_best_params_from_meta(
    meta_path: str,
    score_keys: list[str] | None = None,
) -> tuple[dict[str, Any] | None, float | None]:
    data = load_json(meta_path)
    if not data:
        return None, None
    meta = data.get("meta", data)
    best_params = meta.get("best_params")
    if not isinstance(best_params, dict):
        return None, None

    best_score = None
    for key in score_keys or []:
        if key in meta:
            best_score = float(meta[key])
            break
    return dict(best_params), best_score

