#!/usr/bin/env python3
"""V자반등 **정보 칩**의 히스토리 스트립이 사이클 간에 **과거 봉을 다시 쓰는지** 직접 시험.

## 사용자 신고

*"원래 신호가 없다가 다음 신호에 과거 신호까지 뒤바뀌면서 신호가 생겼던 버그"*

2026-08-31의 `current[-1]`→`current[0]` 버그(이벤트 선택)는 2026-09-01 매 봉 스코어링 재설계로
구조 자체가 사라졌다. 하지만 **현재 코드에도 과거를 덮어쓸 수 있는 경로가 둘 있다**:

  ①`_call_end_pos()`의 `horizon_end = min(pos + BADGE_HORIZON_BARS, last_pos)` -- 끝봉이
    `last_pos`에 묶여 있어, 새 봉이 올 때마다 **기존 콜의 칠해지는 구간이 앞으로 자란다**.
  ②`fill[q] = tone`을 오래된 순으로 덮어쓴다 -- **나중 콜이 이전 콜을 덮는다.** 어떤 봉의
    proba가 사이클 간에 임계값(0.60)을 넘나들면 그 봉이 새 콜이 되어 **과거 구간의 톤을 바꾼다.**

②를 흔들 수 있는 것: TabPFN 배치 구성 의존성(오늘 실측 ~1e-4). 임계값 코앞의 봉만 해당.

## 방법

klines를 한 번 받아 **봉을 하나씩 늘려가며 연속 사이클을 흉내낸다**(`_fetch_klines`를 잘린
프레임을 돌려주도록 대체). 각 사이클의 `history`+`times`를 모아, **겹치는 타임스탬프의 톤이
사이클 간에 바뀌는지** 비교한다. 바뀌면 그게 사용자가 본 그 현상이다.

⚠️읽기 전용. 라이브 코드/아티팩트 변경 없음.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

N_CYCLES = 12
N_ANCHORS = 26          # 과거 앵커를 훑어 **콜이 실제로 난 구간**을 찾는다
OUT = ROOT / "data/research/eth_v_rebound_chip_history_stability_20260904/report.json"


def log(m): print(f"[chip] {m}", flush=True)


def _cycle_test(V, anchor, full_eth, full_btc, orig, n_cycles, log):
    """앵커 직전 n_cycles개 봉을 하나씩 늘려가며 연속 사이클을 흉내낸다."""
    snaps = []
    for k in range(n_cycles):
        cut = anchor - (n_cycles - 1 - k)

        def fake(symbol, _cut=cut):
            src = full_eth if symbol == V.SYMBOL else full_btc
            return src.iloc[max(0, _cut - V.FETCH_LIMIT):_cut].reset_index(drop=True).copy()

        V._fetch_klines = fake
        try:
            out = V.compute_eth_sweep_v_rebound_signal()
        finally:
            V._fetch_klines = orig
        if out.get("error") or not out.get("times"):
            continue
        toned = sum(1 for t in out["history"] if t not in ("neutral", "flat"))
        snaps.append({"cycle": k, "last_bar": out["times"][-1], "toned": toned,
                      "map": dict(zip(out["times"], out["history"])),
                      "badge": out.get("tone"), "proba": out.get("proba_rebound")})
        log(f"    c{k}: {str(out['times'][-1])[:16]} 배지 {out.get('tone')} "
            f"p={out.get('proba_rebound')} 유색 {toned}/48")
    return snaps


def _diff(snaps):
    """겹치는 타임스탬프의 톤이 사이클 간에 바뀐 것들."""
    seen, changes = {}, []
    for sn in snaps:
        for ts, tone in sn["map"].items():
            if ts in seen and seen[ts][1] != tone:
                changes.append({"timestamp": ts, "from_cycle": seen[ts][0],
                                "from_tone": seen[ts][1], "to_cycle": sn["cycle"],
                                "to_tone": tone})
            seen[ts] = (sn["cycle"], tone)
    return changes


def main() -> int:
    t0 = time.time()
    import live_eth_sweep_v_rebound_signal_20260829 as V

    # ⚠️라이브 API는 1,500봉(≈5일)뿐이고 지금 시장이 조용해 콜이 한 번도 안 났다(proba 0.10~0.37).
    # 콜이 없으면 span도 없어 덮어쓰기 경로가 실행되지 않는다 -- 널 테스트가 된다.
    # 그래서 **캐노니컬 CSV 전체 이력**에서 콜이 실제로 난 구간을 찾아 거기서 시험한다.
    import importlib.util

    def _load(n, r):
        sp = importlib.util.spec_from_file_location(n, ROOT / r)
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

    _pf = _load("pf_chip", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
    feas = _pf._s1._feas
    log("캐노니컬 CSV 로드...")
    full_eth = pd.read_csv(feas.ETH_CSV)
    full_btc = pd.read_csv(feas.BTC_CSV)
    for d in (full_eth, full_btc):
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        if getattr(d["timestamp"].dt, "tz", None) is None:
            d["timestamp"] = d["timestamp"].dt.tz_localize("UTC")
    full_eth = full_eth.sort_values("timestamp").reset_index(drop=True)
    full_btc = full_btc.sort_values("timestamp").reset_index(drop=True)
    if len(full_eth) < 2000:
        log("❌klines 부족"); return 1
    log(f"  ETH {len(full_eth):,}봉 {full_eth['timestamp'].iloc[0]} ~ {full_eth['timestamp'].iloc[-1]}")
    log(f"  임계값 {V.PROBA_THRESHOLD} · HISTORY_BARS {V.HISTORY_BARS} · "
        f"BADGE_HORIZON {V.BADGE_HORIZON_BARS}")

    orig = V._fetch_klines

    def run_at(cut):
        def fake(symbol, _cut=cut):
            src = full_eth if symbol == V.SYMBOL else full_btc
            return src.iloc[max(0, _cut - V.FETCH_LIMIT):_cut].reset_index(drop=True).copy()
        V._fetch_klines = fake
        try:
            return V.compute_eth_sweep_v_rebound_signal()
        finally:
            V._fetch_klines = orig

    # ⭐1단계: 콜(proba>=임계값)이 실제로 난 앵커 찾기
    log(f"\n1단계: 앵커 {N_ANCHORS}개 훑어 콜 발생 구간 탐색...")
    lo, hi = 1600, len(full_eth) - N_CYCLES - 1
    anchors = [int(lo + (hi - lo) * i / (N_ANCHORS - 1)) for i in range(N_ANCHORS)]
    hits = []
    for a in anchors:
        o_ = run_at(a)
        if o_.get("error"):
            continue
        pr = o_.get("proba_rebound")
        toned = sum(1 for t in (o_.get("history") or []) if t not in ("neutral", "flat"))
        if toned:
            hits.append((a, pr, toned, o_["times"][-1]))
        log(f"    앵커 {a:6d} {str(o_['times'][-1])[:16]} p={pr} 유색칸 {toned}/48")
    if not hits:
        log("  ❌콜이 난 앵커를 못 찾음 -- 임계값 0.60이 높아 발생이 드물다")
        return 1
    # ⭐적대적 선택: proba가 **임계값 코앞**인 앵커가 가장 뒤집히기 쉽다(TabPFN 배치 의존성
    # ~1e-4가 판정을 흔들 수 있는 유일한 구간). 유색칸이 많은 것보다 이쪽이 결정적이다.
    thr = V.PROBA_THRESHOLD
    near = [x for x in hits if x[1] is not None and x[1] >= thr]
    near.sort(key=lambda x: abs(x[1] - thr))
    picks = [x[0] for x in near[:3]] or [max(hits, key=lambda x: x[2])[0]]
    log(f"\n⭐적대적 앵커 {len(picks)}개 (임계값 {thr} 코앞): "
        + " · ".join(f"{a}(p={dict((h[0], h[1]) for h in hits)[a]})" for a in picks))

    all_changes, all_snaps = [], []
    for anchor in picks:
        log(f"\n--- 앵커 {anchor} ---")
        snaps = _cycle_test(V, anchor, full_eth, full_btc, orig, N_CYCLES, log)
        if len(snaps) >= 2:
            ch = _diff(snaps)
            all_changes += ch
            all_snaps += snaps
            log(f"    -> 과거 톤 변경 {len(ch)}건")
    snaps = all_snaps
    if False:
        pass

    if len(snaps) < 2:
        log("❌비교할 스냅샷 부족"); return 1

    # ⭐겹치는 타임스탬프의 톤이 사이클 간에 바뀌는가
    log("\n=== 과거 봉 톤 변경 검사 (적대적 앵커 전체) ===")
    changes = all_changes
    by_ts = defaultdict(list)
    for ch in changes:
        by_ts[ch["timestamp"]].append(ch)

    if not changes:
        log("  ✅과거 봉 톤 변경 **0건** -- 사이클 간 히스토리가 안정적이다")
    else:
        log(f"  ⚠️**{len(changes)}건** 변경 (고유 봉 {len(by_ts)}개) -- 사용자 신고 현상 재현")
        for ts in sorted(by_ts)[:14]:
            seq = by_ts[ts]
            path = " -> ".join([f"c{seq[0]['from_cycle']}:{seq[0]['from_tone']}"] +
                               [f"c{c['to_cycle']}:{c['to_tone']}" for c in seq])
            log(f"    {ts[:16]}  {path}")

    # 신호 없음 -> 신호 생김 (사용자가 정확히 지목한 패턴)
    NEUTRAL = {"neutral", "flat"}
    appeared = [c for c in changes if c["from_tone"] in NEUTRAL and c["to_tone"] not in NEUTRAL]
    vanished = [c for c in changes if c["from_tone"] not in NEUTRAL and c["to_tone"] in NEUTRAL]
    log(f"\n  ⭐'신호 없음 -> 신호 생김': **{len(appeared)}건**")
    log(f"    '신호 있음 -> 사라짐'   : {len(vanished)}건")
    log(f"    그 외 톤 전환          : {len(changes) - len(appeared) - len(vanished)}건")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_cycles": len(snaps), "threshold": V.PROBA_THRESHOLD,
        "history_bars": V.HISTORY_BARS, "badge_horizon": V.BADGE_HORIZON_BARS,
        "n_changes": len(changes), "n_unique_bars_changed": len(by_ts),
        "n_appeared_from_neutral": len(appeared), "n_vanished_to_neutral": len(vanished),
        "changes": changes[:200],
        "cycles": [{"cycle": s["cycle"], "last_bar": s["last_bar"], "badge": s["badge"],
                    "proba": s["proba"]} for s in snaps],
        "runtime_sec": round(time.time() - t0, 1)}, ensure_ascii=False, indent=2))
    log(f"\n산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
