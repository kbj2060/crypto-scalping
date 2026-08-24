#!/usr/bin/env python3
"""θ를 올린 DC 라벨의 피쳐 정보량 검증 -- "θ=0.004는 chance(p=0.380)였는데, 더 크고 깨끗한
스윙(θ 상향)은 정보량이 있는가?"라는, 어제 경제성 스윕(θ 상향의 경제성 동기는 착시로 판명)과는
독립적인 별개 가설을 직접 테스트한다.

`eth_dc_feature_set_information_content_20260820.py`(원본, θ=0.004 대상)와 정확히 같은 방법론
-- 실제라벨∩예측 대신 여기서는 "라벨 자체와 피쳐" 관계라 학습된 모델이 필요없음: 이벤트bar
(zigzag_action != CASH)에서 LONG=1/SHORT=0 이진타겟에 대해 각 피쳐 단독 AUC(방향무관,
max(auc,1-auc)) + permutation null(라벨셔플 200회)로 "158개 중 최고AUC가 우연보다 유의한가"를
판정한다. 라벨은 새로 생성(θ 상향 스윕과 동일 엔진 호출)하되, DC/TabM 학습이 실제로 쓰는
프레임(omega._load_omega_frames(), 2025=train/2026=eval, 오버레이 포함, 158 base 피쳐)에만
merge한다 -- 원본 θ=0.004 검증과 동일 모집단이라야 직접 비교가 성립한다.

θ 후보: 0.015(경제성 스윕에서 가장 극단값, "큰 스윙" 가설을 가장 강하게 테스트) 우선.
theta=0.004 참고값(원본 스크립트 결과, 재계산 안 함): mean_auc=0.5038, max_auc=0.5141,
empirical_p=0.380."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")

BASE_158 = json.loads((SCRATCH / "dc_base_158_cols.json").read_text())

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega

THETA_CANDIDATES = [0.015]  # 우선 가장 극단값 하나만; 필요시 리스트에 추가해 재실행
RNG = np.random.default_rng(20260820)
N_PERM = 200
REF_THETA_004 = {"mean_auc": 0.5038, "max_auc": 0.5141, "empirical_p": 0.380}


def _load_engine():
    spec = importlib.util.spec_from_file_location("event_label_engine", ROOT / "core" / "event_label_engine.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # numba(cache=True)가 캐시 재로드 시 모듈명을 재해석하려면 필요
    spec.loader.exec_module(mod)
    return mod


def _build_labels_for_theta(engine, theta: float) -> pd.DataFrame:
    """eth_dc_theta_sweep_20260820.py와 동일 로직(전체 concat으로 이벤트+배리어 해석 후
    event_time 기준 연도분리) -- 2025/2026만 반환(우리 canonical train/eval 프레임과 매칭)."""
    req = {"timestamp", "open", "high", "low", "close", "volume"}
    frames = {}
    for year, path in ((2024, ROOT / "data/splits/year_oos/training_features_2024.csv"),
                        (2025, ROOT / "data/splits/year_oos/training_features_2025.csv"),
                        (2026, ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")):
        f = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        f = f[sorted(req)].dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        frames[year] = f
    full = pd.concat([frames[2024], frames[2025], frames[2026]], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    vol = engine.ewma_volatility(full["close"], span=100)
    event_idx, _ = engine.directional_change_events(full["close"].to_numpy(), theta)
    best_barrier = engine.calibrate_barriers(full, event_idx, vol, pt_mult_grid=(1.0, 1.5, 2.0, 3.0),
                                              sl_mult_grid=(1.0, 1.5, 2.0, 3.0), max_hold_grid=(24, 48, 96),
                                              target_balance=0.30, min_events=200)
    cfg = engine.LabelEngineConfig(event_method="directional_change", dc_theta=theta, vol_method="ewma",
                                    vol_span=100, barrier=best_barrier)
    labels = engine.generate_labels(full, cfg)
    labels = labels.copy()
    labels["zigzag_action"] = labels["label"].map({1: 1, -1: 2, 0: 0}).astype("int64")
    labels["timestamp"] = pd.to_datetime(labels["event_time"])
    labels["_year"] = labels["timestamp"].dt.year
    out = labels[labels["_year"].isin([2025, 2026])][["timestamp", "zigzag_action"]].reset_index(drop=True)
    print(f"  theta={theta}: pt_mult={best_barrier.pt_mult} sl_mult={best_barrier.sl_mult} "
          f"2025+2026 이벤트={len(out):,}개", flush=True)
    return out


def auc_dir_agnostic(y: np.ndarray, x: np.ndarray) -> float:
    valid = ~np.isnan(x)
    if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
        return float("nan")
    auc = roc_auc_score(y[valid], x[valid])
    return max(auc, 1.0 - auc)


def main() -> None:
    engine = _load_engine()
    train, eval_df = omega._load_omega_frames()[:2]
    feat_frames = []
    for feat in (train, eval_df):
        f = feat[["timestamp", *BASE_158]].copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        feat_frames.append(f)
    feat_all = pd.concat(feat_frames, ignore_index=True)
    print(f"피쳐 프레임(2025+2026, 오버레이 포함): {len(feat_all):,}행, {len(BASE_158)}개 base 피쳐\n", flush=True)

    print(f"참고 -- theta=0.004(원본, 이미 계산됨): mean_auc={REF_THETA_004['mean_auc']:.4f} "
          f"max_auc={REF_THETA_004['max_auc']:.4f} empirical_p={REF_THETA_004['empirical_p']:.3f}\n", flush=True)

    report = {"theta_0.004_reference": REF_THETA_004}
    for theta in THETA_CANDIDATES:
        print(f"=== theta={theta} ===", flush=True)
        lbl = _build_labels_for_theta(engine, theta)
        data = feat_all.merge(lbl, on="timestamp", how="inner")
        y = (data["zigzag_action"] == 1).to_numpy().astype(np.int64)
        n_long, n_short = int(y.sum()), int((1 - y).sum())
        print(f"  피쳐프레임과 매칭된 이벤트: {len(data):,}개 (LONG={n_long} SHORT={n_short})", flush=True)

        X = {}
        real_auc = {}
        for c in BASE_158:
            x = pd.to_numeric(data[c], errors="coerce").to_numpy(dtype=np.float64)
            X[c] = x
            real_auc[c] = auc_dir_agnostic(y, x)
        vals = {k: v for k, v in real_auc.items() if not np.isnan(v)}
        arr = np.array(list(vals.values()))
        top5 = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)[:5]
        mean_auc, max_auc = float(arr.mean()), float(arr.max())
        print(f"  mean_auc={mean_auc:.4f} max_auc={max_auc:.4f}", flush=True)
        for feat, a in top5:
            print(f"    top: {feat:50s} auc={a:.4f}", flush=True)

        Xmat = np.column_stack([X[c] for c in BASE_158])
        null_max = []
        for _ in range(N_PERM):
            y_perm = RNG.permutation(y)
            aucs = []
            for j in range(Xmat.shape[1]):
                xj = Xmat[:, j]
                valid = ~np.isnan(xj)
                if valid.sum() < 30:
                    continue
                a = roc_auc_score(y_perm[valid], xj[valid])
                aucs.append(max(a, 1 - a))
            null_max.append(max(aucs) if aucs else float("nan"))
        null_arr = np.array(null_max)
        p95 = float(np.nanpercentile(null_arr, 95))
        empirical_p = float((null_arr >= max_auc).mean())
        print(f"  permutation null: 95th={p95:.4f} empirical_p={empirical_p:.3f} "
              f"({'유의' if empirical_p < 0.05 else '비유의(chance)'})\n", flush=True)

        report[f"theta_{theta}"] = {
            "n_events_matched": int(len(data)), "n_long": n_long, "n_short": n_short,
            "mean_auc": mean_auc, "max_auc": max_auc, "top5": [(f, float(a)) for f, a in top5],
            "null_p95": p95, "empirical_p": empirical_p,
        }

    out_path = ROOT / "tmp/eth_dc_theta_raised_feature_information_content_20260820.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[report] {out_path}")


if __name__ == "__main__":
    main()
