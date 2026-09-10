#!/usr/bin/env python3
"""사용자 **실제 진입**의 분포 프로파일 (2026-09-11). 읽기 전용 조회. 모델 학습 없음.

사용자: *"내 진입을 가상으로 만들어서 백테스트 못해?"*

## 무엇이 되고 무엇이 안 되는가
🔴**판단은 못 만든다.** 규칙으로 진입을 생성하면 그건 사용자가 아니라 **그 규칙**을 검정하는 것이다.
   청산 감시자의 합성 포지션 14만 건이 정확히 그랬고(모든 봉 × 양측 ≈ 균등 무작위) 진입 실력을 못 담았다.
✅**분포는 닮게 만들 수 있다.** 실제 진입이 특정 시각·레짐·직전움직임에 몰려 있다면,
   그 조건부 분포에서 합성 진입을 뽑아 "사용자스러운 진입 위에서 청산 모델이 값을 하는가"를 물을 수 있다.
   이건 기존 검정(균등 무작위)과 **다른 질문**이다.

## 이 스크립트의 범위 — 1단계뿐
**진입이 어디에 몰려 있는지만 잰다.** 몰림이 없으면 조건부 합성은 균등 무작위와 같아지고,
그러면 애초에 다시 돌릴 이유가 없다. 그 판단을 먼저 내리기 위한 관문이다.

⚠️표본이 매우 작다(왕복 ~19건). 프로파일 자체의 불확실성을 반드시 같이 낸다 —
   각 축에서 **균등 귀무 대비 얼마나 벗어났는지**를 순열 검정으로 병기한다.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path(__file__).resolve().parents[1] / "tmp/user_entry_profile_20260911"
SYMBOLS = ("ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT")
B_PERM = 5000


def log(m):
    print(f"[prof {time.strftime('%H:%M:%S')}] {m}", flush=True)


async def pull() -> list[dict]:
    import aiohttp
    import live_binance_account_20260910 as AC
    trips = []
    async with aiohttp.ClientSession() as s:
        a = await AC.fetch_account(s, SYMBOLS, trade_limit=1000)
        if not a.get("ok"):
            log(f"🔴 조회 실패: {a.get('error')}")
            return []
        # ⚠️왕복은 최상위 `trades` 에 평평하게 온다(심볼별 블록이 아니다).
        trips = list(a.get("trades") or [])
        if a.get("trades_truncated"):
            log(f"⚠️잘림 보고: {a['trades_truncated']} — 가장 오래된 왕복의 진입가·방향은 불신")
    return trips


def perm_p(obs_stat, values, groups, b=B_PERM, seed=11):
    """관측 통계량이 **균등 귀무**에서 얼마나 드문가.
    🔴초판 결함: `rng.permutation(groups)` 를 썼다 — **개수 분포는 순열에 불변**이라
       통계량이 매번 같고 p 가 항상 1.0 이 된다(롱 85.7% 인데 p=1.0 으로 나와서 잡았다).
    올바른 귀무는 **균등 다항 재추출**: n 개를 k 범주에서 균등하게 새로 뽑는다."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    draws = rng.integers(0, values, size=(b, n))
    stats = np.array([np.abs(np.bincount(d, minlength=values) / n - 1.0 / values).max()
                      for d in draws])
    return float((np.sum(stats >= obs_stat) + 1) / (b + 1))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    trips = asyncio.run(pull())
    if not trips:
        log("왕복이 없다 — 조회 실패이거나 거래 이력이 없다."); return 1
    df = pd.DataFrame(trips)
    log(f"왕복 {len(df)}건 · 심볼 {df['symbol'].value_counts().to_dict()}")
    tcol = next((c for c in ("entry_at", "opened_at", "open_time") if c in df.columns), None)
    if tcol is None:
        log(f"🔴 진입 시각 컬럼 없음. 사용 가능 컬럼: {list(df.columns)}"); return 1
    df["t"] = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["t"]).sort_values("t")
    log(f"진입 시각 유효 {len(df)}건 · {df.t.min()} → {df.t.max()}")
    rep = {"n_trips": int(len(df)), "span": [str(df.t.min()), str(df.t.max())],
           "note": "1단계 = 몰림 여부만. 판단은 재현 불가, 분포만 닮게 할 수 있다.", "axes": {}}

    # 축 1: 시각(UTC 6구간)
    hb = (df.t.dt.hour // 4).to_numpy()
    obs = np.abs(np.bincount(hb, minlength=6) / len(df) - 1 / 6).max()
    rep["axes"]["시각대(4시간 6구간)"] = {
        "counts": np.bincount(hb, minlength=6).tolist(),
        "max_dev_from_uniform": float(obs), "perm_p": float(perm_p(obs, 6, hb))}
    # 축 2: 방향
    if "side" in df.columns:
        sd = (df["side"].astype(str).str.upper().str.contains("LONG|BUY")).astype(int).to_numpy()
        obs2 = abs(sd.mean() - 0.5) * 2
        obs_sd = float(np.abs(np.bincount(sd, minlength=2) / len(sd) - 0.5).max())
        rep["axes"]["방향(롱 비율)"] = {"long_share": float(sd.mean()),
                                    "max_dev_from_uniform": obs_sd,
                                    "perm_p": perm_p(obs_sd, 2, sd)}
    # 축 3: 요일
    wd = df.t.dt.dayofweek.to_numpy()
    obs3 = np.abs(np.bincount(wd, minlength=7) / len(df) - 1 / 7).max()
    rep["axes"]["요일"] = {"counts": np.bincount(wd, minlength=7).tolist(),
                         "max_dev_from_uniform": float(obs3), "perm_p": float(perm_p(obs3, 7, wd))}
    # 축 4: 진입 간격(군집성)
    gaps = df.t.diff().dt.total_seconds().dropna() / 3600
    rep["axes"]["진입 간격(시간)"] = {
        "median_h": float(gaps.median()) if len(gaps) else None,
        "within_1h_share": float((gaps < 1).mean()) if len(gaps) else None,
        "n_gaps": int(len(gaps))}

    log("=" * 88)
    log(f"{'축':>22} {'관측':>34} {'균등이탈':>9} {'순열 p':>8}")
    for k, v in rep["axes"].items():
        detail = (str(v.get("counts")) if "counts" in v
                  else (f"롱 {v['long_share']:.1%}" if "long_share" in v
                        else f"중앙 {v.get('median_h')}h · 1h내 {v.get('within_1h_share')}"))
        log(f"{k:>22} {detail:>34} {v.get('max_dev_from_uniform', float('nan')):>9.3f} "
            f"{v.get('perm_p', float('nan')):>8.4f}")
    sig = [k for k, v in rep["axes"].items() if (v.get("perm_p") or 1) < 0.05]
    log("=" * 88)
    log(f"⭐균등 귀무를 벗어난 축: {len(sig)}개 {sig}")
    if not sig:
        log("   ⇒ **몰림 증거 없음.** 조건부 합성 진입은 균등 무작위와 사실상 같아지고,")
        log("      그러면 청산 모델 재검정은 이미 한 것(합성 14만 건)과 같은 실험이 된다.")
    rep["clustered_axes"] = sig
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    keep = ["t", "symbol"] + [c for c in ("side", "max_qty", "entry_price", "exit_price",
                                          "realized_pnl", "net_pnl", "closed", "fills")
                              if c in df.columns]
    df[keep].to_csv(OUT / "trips.csv", index=False)
    log(f"저장 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
