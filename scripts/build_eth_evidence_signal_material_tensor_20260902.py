#!/usr/bin/env python3
"""증거신호 8종 -> 매 봉 재료 텐서 (2026-09-02, 재료화 2단계).

WHY
---
사용자 재구성: "증거신호로 매매를 진행할 필요가 없다. 재료로 보고 있고, 앞으로 딥러닝이나
강화학습 모델에 필요한 재료가 될 것이다."

전략 등급과 재료 등급은 요구사항이 다르다. 재료는:
  - 추론 시점 인과성 (raw 트리거는 이미 만족)
  - **학습 모집단 = 추론 모집단** (1단계에서 수정: 앵커 -> raw 트리거)
  - **매 봉 값 존재** -- 희소 이벤트가 아니라 조밀한 텐서 (이 스크립트)
  - lag/horizon 명시

현재 칩은 발동 봉에서만 값이 있다. RL/DL은 매 봉 상태벡터를 원한다.

WHAT
----
신호마다 4개 열을 매 봉에 만든다 (부호 규약: +가 롱-우호/bottom, -가 숏-우호/top):
  <sig>_fire        발동 봉에서 +1/-1, 그 외 0
  <sig>_proba       인과 모집단 메타라벨 확률. 발동 봉부터 그 신호 자신의 HORIZON 동안 유지,
                    이후 0. 유지 중 새 발동이 나면 덮어쓴다.
  <sig>_signed      proba * 방향 (RL이 바로 먹는 형태)
  <sig>_age         발동 후 경과봉 / HORIZON, [0,1] 클립. 1이면 만료(=신호 없음).
덤으로 레짐 재료 2열: regime_eth / regime_btc (clean-cutoff 재훈련본, 0 bull / 1 bear / 2 chop).

⚠️인과 규약: 인덱스 i의 값은 **i번째 봉의 종가까지의 정보만** 쓴다. 매매에 쓸 때는 i+1 시가
진입이 이 저장소의 표준이다. 이 텐서 자체에는 미래 정보가 없다.
⚠️HOLDOUT 구간에도 값을 채운다. 그건 재료를 쓸 수 있게 만드는 추론이지 평가가 아니다 --
홀드아웃 성능 판정은 하류 사용자 몫이다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OUT_DIR = ROOT / "data/materials/eth_evidence_signal_tensor_20260902"
ETH_REGIME = ROOT / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"
BTC_REGIME = ROOT / "tmp/btc_regime_s24k3_clean_20260902/predictions.parquet"


def log(m): print(f"[material] {m}", flush=True)


def main() -> int:
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines

    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    kl = load_klines()[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    kl = kl.sort_values("timestamp").reset_index(drop=True)
    n = len(kl)
    pos_of = {t: i for i, t in enumerate(pd.DatetimeIndex(kl["timestamp"]))}
    out = kl[["timestamp"]].copy()
    log(f"기준 {n:,}봉 {kl.timestamp.min()} ~ {kl.timestamp.max()}")

    stats = []
    for name, c in cfg.items():
        H = int(c["horizon"])
        f = SRC / f"{name}_causal_proba.csv"
        if not f.exists():
            log(f"⚠️ {name}: proba 없음 -- 건너뜀"); continue
        d = pd.read_csv(f, parse_dates=["timestamp"])
        d = d[d["timestamp"].isin(pos_of)].copy()
        d["i"] = [pos_of[t] for t in d["timestamp"]]
        d["dir"] = np.where(d["is_bottom"] == 1, 1.0, -1.0)
        d = d.sort_values(["i", "proba"]).drop_duplicates("i", keep="last")  # 동시발동시 확률 큰 쪽

        fire = np.zeros(n); proba = np.zeros(n); age = np.ones(n)
        dirs = np.zeros(n)
        fire[d["i"].to_numpy()] = d["dir"].to_numpy()
        # 발동 봉부터 H봉 동안 유지 (새 발동이 덮어씀) -- 전부 뒤를 보지 않는 전방 채움
        last_p = last_d = 0.0; last_i = -10**9
        pi = {int(i): (float(p), float(dd)) for i, p, dd in
              zip(d["i"], d["proba"], d["dir"])}
        for i in range(n):
            if i in pi:
                last_p, last_d = pi[i]; last_i = i
            elapsed = i - last_i
            if elapsed < H:
                proba[i] = last_p; dirs[i] = last_d; age[i] = elapsed / H
            else:
                proba[i] = 0.0; dirs[i] = 0.0; age[i] = 1.0
        out[f"{name}_fire"] = fire
        out[f"{name}_proba"] = proba
        out[f"{name}_signed"] = proba * dirs
        out[f"{name}_age"] = age
        cov = float((proba > 0).mean())
        stats.append({"signal": name, "horizon": H, "n_fires": int(len(d)),
                      "bar_coverage": round(cov, 4),
                      "proba_mean_active": round(float(proba[proba > 0].mean()), 4) if cov else 0.0})
        log(f"{name:26s} H={H:<3} 발동 {len(d):6,} | 봉 커버리지 {cov:.1%}")

    for tag, p in (("eth", ETH_REGIME), ("btc", BTC_REGIME)):
        if p.exists():
            r = pd.read_parquet(p)
            out = out.merge(r.rename(columns={"regime": f"regime_{tag}"}), on="timestamp", how="left")
            out[f"regime_{tag}"] = out[f"regime_{tag}"].ffill().fillna(-1).astype(int)
            log(f"regime_{tag} 병합: chop 비중 {float((out[f'regime_{tag}']==2).mean()):.3f}")

    # ---- 인과성 검증 ----
    log("\n=== 인과성/정합성 검증 ===")
    bad = 0
    for name, c in cfg.items():
        col = f"{name}_proba"
        if col not in out: continue
        fcol = out[f"{name}_fire"].to_numpy()
        pcol = out[col].to_numpy()
        # proba>0인데 직전 H봉 안에 발동이 없으면 오류
        H = int(c["horizon"])
        act = np.flatnonzero(pcol > 0)
        for i in act[:: max(1, len(act) // 2000)]:
            lo = max(0, i - H + 1)
            if not np.any(fcol[lo:i + 1] != 0):
                bad += 1; break
    log(f"  유지창 밖 활성값: {bad}건 (0이어야 정상)")
    log(f"  결측: {int(out.isna().sum().sum())}개")
    same = out.drop(columns=["timestamp"]).abs().sum(axis=1) > 0
    log(f"  최소 1개 신호가 살아있는 봉 비율: {float(same.mean()):.1%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "eth_evidence_material_5m.parquet", index=False)
    pd.DataFrame(stats).to_csv(OUT_DIR / "coverage.csv", index=False)
    (OUT_DIR / "README.json").write_text(json.dumps({
        "rows": int(len(out)), "cols": int(out.shape[1]),
        "range": [str(out.timestamp.min()), str(out.timestamp.max())],
        "columns_per_signal": ["<sig>_fire (+1 bottom / -1 top / 0)",
                               "<sig>_proba (causal-population metalabel, held for the signal's own HORIZON)",
                               "<sig>_signed (proba * direction)",
                               "<sig>_age (elapsed/HORIZON, 1.0 = expired)"],
        "extra": ["regime_eth", "regime_btc  (0 bull / 1 bear / 2 chop, clean-cutoff TRAIN<=2025-08-31)"],
        "causality": "value at index i uses information through bar i's CLOSE only; standard use is entry at i+1 open",
        "population_fix": "metalabels retrained on RAW TRIGGER population (no cluster_dedup) so train==inference",
        "holdout": "values are filled over HOLDOUT (inference, not evaluation); holdout AUC deliberately not computed",
        "signals": {k: v for k, v in cfg.items()},
    }, indent=2, ensure_ascii=False))
    log(f"\n산출: {OUT_DIR} ({len(out):,}행 x {out.shape[1]}열)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
