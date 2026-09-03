#!/usr/bin/env python3
"""XRP·BTC **지정가(resting limit) 진입 이식** -- 앵커 미래참조를 우회하는 유일한 검증된 경로.

## 왜

2026-09-03 감사에서 XRP 5종·BTC 5종의 경제성게이트 수치가 **지연확정 대조군에서 10/10 붕괴**했다.
원인은 성능 부족이 아니라 **앵커(`cluster_dedup` 최극단 봉) 선택이 미래참조**라는 인과성 문제다.

비용 민감도까지 재봤다: 트레일링스톱 청산은 본질적으로 taker라 현실적 하한이 **7bp**인데,
7bp에서 지연확정을 통과하는 건 10종 중 **1종**(XRP taker, VAL +0.74bp로 사실상 0)뿐이다.
⇒ **비용을 낮추는 것으로는 안 된다.**

ETH는 이 문제를 **진입 메커니즘 교체**로 풀었다(README **5.18절**):

> 지정가 주문은 '지금 결정'을 요구하지 않는다. 앵커는 결국 버스트의 극단 **가격**이므로,
> 깊은 지정가는 가격이 극단으로 갈 때 체결된다.
> **알 수 없는 축("어느 봉")을 내가 정하는 축("어느 가격")으로 바꾼다.**

ETH `short_term_return_z` 실측: 즉시 taker −4.28/+1.29/−4.16(flip ❌)
→ **지정가 3.0×ATR/3봉대기 +8.48/+5.12/+4.47(flip 3창 ✅)**.

⇒ 그 기계를 XRP·BTC로 이식한다.

## 설계 -- 재구현 금지

`research_eth_resting_limit_entry_20260902.py`의 상태기계 `run()`을 **그대로 import**한다.
재구현하면 체결 판정·트레일링·만료 규약이 조용히 달라진다.

  · 트리거: **raw**(dedup 없음). `compute_signals()`의 `bottom_*`/`top_*` -- 라이브 칩과 **같은 소스**다.
    ⭐dedup을 안 쓰는 것이 이 접근의 핵심이다(앵커 선택 자체가 사라진다).
  · 셀(sl/arm/trail/horizon): 각 자산 게이트 리포트의 ARM>=1.0 최선 셀 -- 지연확정이 무너뜨린 바로 그 셀.
  · depth [0, 0.5, 1.0, 1.5, 2.0, 3.0] x wait [3, 6, 12]. depth=0은 **즉시 taker 대조군**.
  · 비용 10bp(전 taker) / 7bp(maker 진입) 두 가정 병기.
  · **방향뒤집기 대조군**을 같은 격자 전량에 건다.
  · 1슬롯: PENDING/OPEN 중 새 트리거 무시(대기 주문도 슬롯 점유 -- 보수적).

## 판정 (실행 전 고정)

  depth>0 셀이 **VAL·OOS 둘 다 양수** AND **정방향 > 뒤집기**(양 구간)여야 후보.
  depth=0(즉시 taker) 대조군이 같은 조건을 만족하면 "지정가 덕분"이라 말할 수 없다.

⚠️HOLDOUT은 **진단으로만** 출력하고 판정에 쓰지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

_E = importlib.util.spec_from_file_location(
    "ethrest", ROOT / "scripts/research_eth_resting_limit_entry_20260902.py")
_e = importlib.util.module_from_spec(_E)
_E.loader.exec_module(_e)

from live_evidence_signal_dashboard_20260823 import compute_signals   # noqa: E402

OUT = ROOT / "data/research/xrp_btc_resting_limit_entry_v2_20260903.json"

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

# ⚠️2026-09-03 v2: 1차 실행에서 통과 셀이 거의 전부 depth=3.0(격자 상단 경계)에 몰렸다.
# 오늘 반복 확인된 격자 경계 규칙(포팅 프로토콜 §5-A)에 따라 위로 넓힌다.
DEPTHS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
WAITS = [3, 6, 12]
COSTS = {"10bp(전taker)": 0.0010, "7bp(maker진입)": 0.0007}

ASSETS = {
    # ⚠️2026-09-03 v2: `data/*_5m_1year.csv`는 **2026-02-17에 끝난다** -- OOS 창(2026-01~03)의
    # 절반만 덮고 HOLDOUT은 0봉이었다(1차 실행이 이 상태였다). 전체 구간 api CSV로 바꾼다.
    "XRP": {"klines": ROOT / "binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv",
            "partner": ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
            "funding": "research_xrp_regime_label_conditional_lift_20260903.py:load_xrp_funding_z",
            "gate": ROOT / "data/research/xrp_evidence_signals_costgate_20260903/report.json"},
    "BTC": {"klines": ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
            "partner": ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
            "funding": "research_btc_regime_label_conditional_lift_20260902.py:load_btc_funding_z",
            "gate": ROOT / "data/research/btc_evidence_signals_costgate_20260902/report.json"},
}
# 게이트 리포트의 신호명 -> compute_signals의 컬럼 접두어
COL_ALIAS = {"taker_delta_climax": "taker_delta_z_climax"}


def log(m): print(f"[rest-port] {m}", flush=True)


def load_funding(spec: str):
    rel, fn = spec.split(":")
    sp = importlib.util.spec_from_file_location(f"f_{Path(rel).stem}", ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    out = getattr(m, fn)()
    for c in out.columns:                       # ⚠️[ns]/[us] 통일(이 저장소 상습 함정)
        if str(out[c].dtype).startswith("datetime64"):
            out[c] = out[c].astype("datetime64[ns]")
    return out


def main() -> int:
    t0 = time.time()
    rep = {"depths": DEPTHS, "waits": WAITS, "costs_bp": [10.0, 7.0],
           "holdout_used_for_decision": False, "assets": {},
           "note": "raw 트리거(dedup 없음) + 지정가 진입 -- 앵커 선택 자체가 없다"}

    for asset, cfg in ASSETS.items():
        log("")
        log("#" * 76)
        log(f"{asset}")
        log("#" * 76)
        raw = pd.read_csv(cfg["klines"], parse_dates=["timestamp"])
        raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        partner = pd.read_csv(cfg["partner"], usecols=["timestamp", "high", "low"],
                              parse_dates=["timestamp"])
        funding = load_funding(cfg["funding"])
        for d in (raw, partner):
            d["timestamp"] = d["timestamp"].astype("datetime64[ns]")
        frame = compute_signals(raw, btc_df=partner, funding_df=funding)
        frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
        ts = frame["timestamp"]
        o, h, l, c = (frame[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        atr = frame["atr_pct"].to_numpy(float)
        log(f"프레임 {len(frame):,}봉 | atr 결측 {int(np.isnan(atr).sum()):,} "
            f"| {ts.min()} ~ {ts.max()}")
        for _wn, _lo, _hi in (("VAL", VAL_START, OOS_START), ("OOS", OOS_START, HOLDOUT_START),
                              ("HOLDOUT", HOLDOUT_START, ts.max())):
            _n = int(((ts >= _lo) & (ts < _hi)).sum())
            log(f"   {_wn:<8} {_n:>7,}봉  {'⚠️0봉 -- 데이터가 이 창을 안 덮는다' if _n == 0 else ''}")

        idx = pd.DatetimeIndex(ts)
        W = {"VAL": (int(idx.searchsorted(VAL_START)), int(idx.searchsorted(OOS_START))),
             "OOS": (int(idx.searchsorted(OOS_START)), int(idx.searchsorted(HOLDOUT_START))),
             "HOLDOUT": (int(idx.searchsorted(HOLDOUT_START)), len(idx))}

        gate = json.loads(cfg["gate"].read_text())["signals"]
        res = {}
        for name, v in gate.items():
            g1 = v.get("genuine_arm_ge_1") or []
            if not g1:
                log(f"  {name}: 게이트 통과 셀 없음 -- 건너뜀"); continue
            best = max(g1, key=lambda x: min(x["val_fwd_bp"], x["oos_fwd_bp"]))
            sp = {"sl": best["sl"], "arm": best["arm"], "trail": best["trail"],
                  "horizon": v["horizon_bars"]}
            col = COL_ALIAS.get(name, name)
            cb, ct = f"bottom_{col}", f"top_{col}"
            if cb not in frame.columns or ct not in frame.columns:
                log(f"  {name}: 트리거 컬럼 없음({cb}) -- 건너뜀"); continue
            tb = frame[cb].fillna(False).to_numpy(bool)
            tt = frame[ct].fillna(False).to_numpy(bool)
            log("")
            log(f"=== {name}  셀 SL={sp['sl']} ARM={sp['arm']} Trail={sp['trail']} H={sp['horizon']} ===")
            log(f"  raw 트리거 bottom {int(tb.sum()):,} / top {int(tt.sum()):,}  (dedup 없음)")
            rows = []
            for depth in DEPTHS:
                for wait in (WAITS if depth > 0 else [0]):
                    for cname, cost in COSTS.items():
                        if depth == 0.0 and cname != "10bp(전taker)":
                            continue
                        r = {"depth": depth, "wait": wait, "cost": cname}
                        okw = True
                        for wn, (lo, hi) in W.items():
                            f_ = _e.stats(_e.run(tb, tt, atr, o, h, l, c, lo, hi, depth=depth,
                                                 wait=wait, cost=cost, flip=False, **sp))
                            x_ = _e.stats(_e.run(tb, tt, atr, o, h, l, c, lo, hi, depth=depth,
                                                 wait=wait, cost=cost, flip=True, **sp))
                            r[wn] = {"fwd": f_, "flip": x_}
                            if wn in ("VAL", "OOS"):
                                okw &= (f_["mean_bp"] > 0 and f_["mean_bp"] > x_["mean_bp"])
                        r["passes_val_oos"] = bool(okw)
                        rows.append(r)
            # 요약: depth>0 통과 셀
            pas = [r for r in rows if r["passes_val_oos"] and r["depth"] > 0]
            ctrl = [r for r in rows if r["depth"] == 0.0 and r["passes_val_oos"]]
            log(f"  {'depth':>6}{'wait':>6}{'비용':>14}{'VAL bp':>9}{'OOS bp':>9}"
                f"{'HOLD bp':>9}{'n(V/O)':>12}  판정")
            for r in rows:
                mark = "✅" if r["passes_val_oos"] else "  "
                if r["depth"] == 0.0 or r["passes_val_oos"] or r["depth"] == 3.0:
                    log(f"  {r['depth']:>6.1f}{r['wait']:>6}{r['cost']:>14}"
                        f"{r['VAL']['fwd']['mean_bp']:>+9.2f}{r['OOS']['fwd']['mean_bp']:>+9.2f}"
                        f"{r['HOLDOUT']['fwd']['mean_bp']:>+9.2f}"
                        f"{str(r['VAL']['fwd']['n']) + '/' + str(r['OOS']['fwd']['n']):>12}  {mark}")
            log(f"  ⇒ depth>0 통과 {len(pas)}셀 / 즉시taker(depth=0) 통과 {len(ctrl)}셀"
                f"  {'⭐**지정가 덕분**' if pas and not ctrl else ('⚠️대조군도 통과 -- 지정가 효과 아님' if ctrl else '❌통과 없음')}")
            res[name] = {"cell": sp, "n_trig": {"bottom": int(tb.sum()), "top": int(tt.sum())},
                         "rows": rows, "n_pass_limit": len(pas), "n_pass_taker_control": len(ctrl)}
        rep["assets"][asset] = res

    log("")
    log("=" * 80)
    log("종합 -- 지정가 진입이 앵커 문제를 우회하는가")
    log("=" * 80)
    log(f"{'자산':<5}{'신호':<26}{'지정가 통과':>11}{'즉시taker 통과':>15}  판정")
    tot = 0
    for asset, res in rep["assets"].items():
        for name, v in res.items():
            good = v["n_pass_limit"] > 0 and v["n_pass_taker_control"] == 0
            tot += good
            log(f"{asset:<5}{name:<26}{v['n_pass_limit']:>11}{v['n_pass_taker_control']:>15}  "
                f"{'⭐지정가 효과' if good else ('대조군도 통과' if v['n_pass_taker_control'] else '통과 없음')}")
    log("")
    log(f"⇒ 지정가로만 통과하는 신호: **{tot}종**")
    rep["n_limit_only"] = tot
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
