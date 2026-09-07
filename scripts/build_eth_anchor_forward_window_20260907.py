#!/usr/bin/env python3
"""전방 구간(2026-06-30~07-31) 앵커의 창 텐서 (2026-09-07).

`build_eth_anchor_window_tensor_20260907.py` 와 **같은 채널·같은 K·같은 정렬**로,
앵커 목록만 `features154`(2026-06-29 까지) 대신 학습셋(2026-07-31 까지)에서 가져온다.
전방 156 앵커 중 y_bin 유효 84 개가 전방 확인 대상이다.

⚠️구성 코드는 원 빌더를 import 해 재사용한다 -- 채널 정의가 갈라지면 전방 확인이 무의미하다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SET = ROOT / "tmp/eth_anchor_training_set_20260907/train_set_P1_H48.parquet"
SEEN = ROOT / "tmp/eth_anchor_features154_20260907/features154.parquet"
OUT = ROOT / "tmp/eth_anchor_forward_window_20260907"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    WB = _load("winbuild", ROOT / "scripts/build_eth_anchor_window_tensor_20260907.py")
    B = _load("anchor_builder", ROOT / "scripts/build_eth_anchor_label_dataset_20260907.py")
    OC = _load("osc_cores", ROOT / "scripts/build_eth_anchor_oscillator_cores_20260907.py")

    T = pd.read_parquet(TRAIN_SET)
    if "anchor" in T.columns:
        T = T[T.anchor == "any3/Wc3"]
    cut = pd.read_parquet(SEEN, columns=["timestamp"]).timestamp.max()
    D = T[T.timestamp > cut].sort_values("timestamp").reset_index(drop=True)
    print(f"[1/4] 전방 앵커 {len(D)} · {D.timestamp.min()} ~ {D.timestamp.max()} (컷 {cut})", flush=True)

    eth = B._load_kl(B.ETH_KL); fund = B._load_funding(); btc = B._load_kl(B.BTC_KL)
    tmax = min(eth["timestamp"].max(), btc["timestamp"].max(), fund["calc_time"].max())
    eth = eth[eth["timestamp"] <= tmax].reset_index(drop=True)
    sig = B.compute_signals(eth, btc_df=btc[btc["timestamp"] <= tmax],
                            funding_df=fund[fund["calc_time"] <= tmax])
    C = OC.compute_cores(sig, B)
    print(f"[2/4] 신호 프레임 {len(sig):,} (상한 {tmax})", flush=True)

    K, CH = WB.K, WB.CHANNELS
    close = sig["close"].to_numpy(float); high = sig["high"].to_numpy(float)
    low = sig["low"].to_numpy(float); atr = sig["atr_pct"].to_numpy(float)
    bar = {"logret": np.concatenate([[np.nan], np.diff(np.log(close))]), "close": close,
           "dem14": sig["dem"].to_numpy(float), "p_fast": C["p_fast"].to_numpy(float),
           "delta_z": C["delta_z"].to_numpy(float), "ret3_z": C["ret3_z"].to_numpy(float),
           "kalman_dev_z": sig["kalman_dev_z"].to_numpy(float),
           "hl_range": (high - low) / np.maximum(close, 1e-12)}
    ts = pd.to_datetime(sig["timestamp"].to_numpy())
    idx = pd.Series(np.arange(len(ts)), index=ts).reindex(D["timestamp"]).to_numpy()
    assert not np.isnan(idx).any(), "전방 앵커가 봉 프레임에 없음"
    idx = idx.astype(int)

    n = len(D); X = np.full((n, len(CH), K), np.nan, np.float32); valid = np.zeros(n, bool)
    cs = np.where(D["side"].to_numpy() == "top", 1.0, -1.0)
    tsv = ts.to_numpy(); offs = np.arange(K - 1, -1, -1) * np.timedelta64(5, "m")
    n_gap = 0
    for i in range(n):
        s = idx[i] - K + 1
        if s < 0:
            continue
        w = slice(s, idx[i] + 1)
        if not np.array_equal(tsv[w], D["timestamp"].to_numpy()[i] - offs):
            n_gap += 1; continue
        cl = bar["close"][w]; a = atr[idx[i]]
        for c, ch in enumerate(CH):
            v = (cl - cl[-1]) / max(a * cl[-1], 1e-12) if ch == "path_atr" else bar[ch][w]
            if ch in WB.DIRECTIONAL:
                v = v * cs[i]
            elif ch in WB.CENTERED:
                v = 0.5 + (v - 0.5) * cs[i]
            X[i, c, :] = v
        valid[i] = np.isfinite(X[i]).all()
    print(f"[3/4] 결측봉 제외 {n_gap} · 유효 {int(valid.sum())}/{n}", flush=True)
    bad = sum(1 for i in np.flatnonzero(valid)
              if abs(float(X[i, CH.index("path_atr"), -1])) > 1e-9)
    print(f"      창 끝=앵커봉 트립와이어: {bad}/{int(valid.sum())} → {'PASS' if bad == 0 else '🔴FAIL'}", flush=True)

    np.save(OUT / "X.npy", X); np.save(OUT / "valid.npy", valid)
    keep = [c for c in ("timestamp", "side", "split", "y3", "y_bin", "is_clean", "range_pct", "atr_pct")
            if c in D.columns]
    D[keep].assign(valid=valid).to_parquet(OUT / "index.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps(
        {"K": K, "channels": CH, "n": n, "n_valid": int(valid.sum()), "n_gap": n_gap,
         "cut": str(cut), "range": [str(D.timestamp.min()), str(D.timestamp.max())],
         "y_bin_valid": int(np.isfinite(D["y_bin"].to_numpy()).sum())}, indent=1, ensure_ascii=False))
    print(f"[4/4] 저장: {OUT} · X {X.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
