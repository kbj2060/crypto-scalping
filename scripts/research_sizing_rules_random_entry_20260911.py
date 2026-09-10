#!/usr/bin/env python3
"""사이징 규칙 비교 — **무작위 진입 · 대표본** (2026-09-11).

사용자: *"아무 봉이나 선택해서 랜덤으로 크기를 얼마나 잘 맞추나 테스트해보면 되잖아.
랜덤이나 내 선택이나 비슷해."*

## 맞는 지적이다 — 그리고 왜 맞는가
실계좌 12왕복에서 진입 타이밍은 **무작위와 구분되지 않았다**(승률 58.3% p=0.387 · t=+0.15 ·
CI[−37,+29] · MDE +51.6bp). 그러면 **사이징 규칙 자체**는 무작위 진입으로 검정할 수 있고,
표본이 12 가 아니라 **수만 건**이 된다.

## 단, 하나를 분리해야 한다
무작위 진입의 기대손익은 방향 실력이 0 이므로 **비용만큼 음수**다. 사이징은 그 평균을 못 바꾼다
(크기를 줄이면 손실도 줄 뿐이다). ⇒ *"사이징이 돈을 번다"* 는 무작위 진입으로 증명할 수 없다.
**증명할 수 있는 것은 두 가지다:**
 ① **위험 통제**: 같은 평균 노출에서 손실 꼬리·청산 빈도·산포를 얼마나 줄이는가 (실력 무관)
 ② **실력 조건부 성장**: 진짜 실력이 e bp 라고 **가정**했을 때 규칙별 로그성장 차이
    (e 를 가정하는 것이지 e 를 주장하는 게 아니다 — 여러 e 에 대해 전부 보고한다)

## 규칙 (사전 지정)
`fixed`     고정 명목 — 대조군
`invvol`    1/atr_pct — 배포 봇이 쓰는 형태(`notional=0.004/(5·atr_pct)`)
`invmae`    1/MAE_hat(q) — 오늘 보정 검증된 추정치. q=0.9
            ⚠️`MAE_hat = k·atr_pct·√H` 이므로 **1/MAE_hat ∝ 1/atr_pct** — `invvol` 과 항등이다.
              오늘의 보정 검증은 새 규칙을 만든 게 아니라 **기존 규칙의 눈금이 맞다**는 확인이었다.
`invmae_fc` invmae 인데 변동성 전망 「위험」이면 절반 — 전방 정보를 얹은 형태
`userlike`  실계좌 수량 분포에서 부트스트랩 — 재량 사이징의 대리
**모든 규칙은 평균 명목이 같도록 정규화한다.** 그래야 "크게 걸어서 더 벌었다"가 섞이지 않는다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path(__file__).resolve().parents[1] / "tmp/sizing_rules_20260911"
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
HOLD = 48                      # 4시간 보유(실계좌 왕복 중앙 간격에 가깝다)
COST_BP = 10.0
LONG_SHARE = 0.857             # 실계좌 관측 롱 비율
EDGES_BP = (0.0, 5.0, 10.0, 20.0, 30.0)
LEV = 50.0                     # 실계좌 관측 레버리지 — 청산 근접 빈도 계산용
SEED = 20260911


def log(m):
    print(f"[size {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(KL, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    c = d["close"].to_numpy(float); hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
    n = len(c)
    tr = np.abs(np.diff(c, prepend=c[0]))
    atr_pct = (pd.Series(tr).rolling(288, min_periods=200).mean().to_numpy() / np.maximum(c, 1e-9))
    log(f"5분봉 {n:,} · {d.index[0]} → {d.index[-1]}")

    idx = np.arange(600, n - HOLD - 1, 6)                  # 30분 간격 무작위 진입 후보
    idx = idx[np.isfinite(atr_pct[idx]) & (atr_pct[idx] > 0)]
    side = np.where(rng.random(len(idx)) < LONG_SHARE, 1.0, -1.0)
    ret = (c[idx + HOLD] / c[idx] - 1.0) * side * 1e4       # 단위당 왕복 수익(bp), 비용 전
    # 최대불리이탈(진입 방향 기준)
    mae = np.array([((c[i] - lo[i + 1:i + 1 + HOLD].min()) if s > 0
                     else (hi[i + 1:i + 1 + HOLD].max() - c[i])) / c[i]
                    for i, s in zip(idx, side)]) * 1e4
    log(f"진입 표본 {len(idx):,} · 롱 {(side>0).mean()*100:.1f}% · 평균 MAE {mae.mean():.0f}bp")

    ap = atr_pct[idx]
    k90 = float(np.quantile(mae / (ap * 1e4 * np.sqrt(HOLD)), 0.9))
    mae_hat = k90 * ap * 1e4 * np.sqrt(HOLD)
    # 전방 변동성 전망 대용: rv48/rv288 비율이 상위 20% 면 「확대 예상」으로 본다
    r5 = np.diff(np.log(c), prepend=0.0)
    rv48 = pd.Series(r5).rolling(48, min_periods=24).std().to_numpy()
    rv288 = pd.Series(r5).rolling(288, min_periods=150).std().to_numpy()
    with np.errstate(all="ignore"):
        vr = (rv48 / np.maximum(rv288, 1e-12))[idx]
    warn = vr >= np.nanquantile(vr, 0.8)

    user_q = np.array([2.727, 1.418, 1.359, 1.427, 4.476, 3.701, 2.727, 11.546,
                       8.245, 2.283, 2.352, 4.761])       # 실계좌 관측 수량
    rules = {
        "fixed": np.ones(len(idx)),
        "invvol": 1.0 / ap,
        "invmae": 1.0 / np.maximum(mae_hat, 1e-9),
        "invmae_fc": (1.0 / np.maximum(mae_hat, 1e-9)) * np.where(warn, 0.5, 1.0),
        "userlike": rng.choice(user_q, len(idx)),
    }
    rep = {"hold_bars": HOLD, "cost_bp": COST_BP, "n": int(len(idx)),
           "long_share": float(side.mean()), "edges_bp": EDGES_BP,
           "note": "모든 규칙은 평균 명목을 1로 정규화 — 크기를 키워서 번 효과를 배제", "rules": {}}
    for name, w in rules.items():
        w = w / w.mean()                                  # ⭐평균 노출 동일
        pnl0 = w * (ret - COST_BP)                        # 실력 0 일 때 건당 손익(단위 명목당 bp)
        loss = w * mae                                    # 불리이탈 노출
        cell = {"weight_cv": float(w.std(ddof=1) / w.mean()),
                "mean_bp_edge0": float(pnl0.mean()), "sd_bp": float(pnl0.std(ddof=1)),
                "p05_bp": float(np.percentile(pnl0, 5)), "p01_bp": float(np.percentile(pnl0, 1)),
                "mae_exposure_mean": float(loss.mean()),
                "mae_exposure_p99": float(np.percentile(loss, 99)),
                # 50배에서 청산선(2% 역행)에 닿는 비율 — 명목 가중
                "stopout_50x": float(np.mean(w * mae > 1e4 / LEV)),
                "growth": {}}
        for e in EDGES_BP:
            pnl = w * (ret + e - COST_BP)
            g = np.log1p(np.clip(pnl / 1e4, -0.99, None))  # 로그성장(건당)
            cell["growth"][str(e)] = {"mean_bp": float(pnl.mean()),
                                      "log_growth_per_trade": float(g.mean()),
                                      "sharpe_per_trade": float(pnl.mean() / pnl.std(ddof=1))}
        rep["rules"][name] = cell

    log("=" * 112)
    log(f"{'규칙':>11} {'가중CV':>7} {'실력0 평균':>9} {'표준편차':>8} {'하위1%':>9} "
        f"{'MAE노출 평균':>11} {'99분위':>9} {'50배 청산율':>10}")
    for k, v in rep["rules"].items():
        log(f"{k:>11} {v['weight_cv']:>7.2f} {v['mean_bp_edge0']:>+9.2f} {v['sd_bp']:>8.1f} "
            f"{v['p01_bp']:>+9.1f} {v['mae_exposure_mean']:>11.1f} {v['mae_exposure_p99']:>9.1f} "
            f"{v['stopout_50x']:>9.2%}")
    log("=" * 112)
    log("⭐실력 가정별 **건당 로그성장** (평균 노출 동일 · 높을수록 좋다)")
    log(f"{'규칙':>11}" + "".join(f"{'e='+str(int(e))+'bp':>12}" for e in EDGES_BP))
    for k, v in rep["rules"].items():
        log(f"{k:>11}" + "".join(f"{v['growth'][str(e)]['log_growth_per_trade']*1e4:>12.2f}"
                                 for e in EDGES_BP))
    log("   (단위 1e-4 ≈ bp)")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
