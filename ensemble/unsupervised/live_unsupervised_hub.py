from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


class UnsupervisedRegimeHub:
    """비지도 학습 모델 아티팩트를 한 곳에서 관리하는 허브."""

    DEFAULT_ARTIFACTS = {
        "gmm_volatility": "data/ensemble/unsupervised/gmm_volatility.pkl",
        "hdbscan_regime": "data/ensemble/unsupervised/hdbscan_regime.pkl",
        "isolation_forest": "data/ensemble/unsupervised/isolation_forest.pkl",
        "vae_anomaly": "data/ensemble/unsupervised/vae_anomaly.pkl",
    }

    def __init__(self, artifact_paths: dict[str, str] | None = None):
        self.artifact_paths = dict(self.DEFAULT_ARTIFACTS)
        if artifact_paths:
            self.artifact_paths.update(artifact_paths)

    def status(self) -> dict:
        models = {}
        loaded = 0
        for name, path in self.artifact_paths.items():
            exists = os.path.exists(path)
            meta_path = os.path.splitext(path)[0] + ".json"
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}
            models[name] = {
                "path": path,
                "exists": exists,
                "meta_path": meta_path if os.path.exists(meta_path) else None,
                "meta": meta,
            }
            if exists:
                loaded += 1

        return {
            "available_count": loaded,
            "total_count": len(self.artifact_paths),
            "models": models,
        }

    def summary_line(self) -> str:
        st = self.status()
        return f"UnsupervisedRegimeHub artifacts: {st['available_count']}/{st['total_count']} available"
