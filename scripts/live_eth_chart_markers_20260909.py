#!/usr/bin/env python3
"""청산맵 차트용 **신호 마커** 페이로드 (2026-09-09, 사용자 요청).

사용자: *"청산맵에 증거신호랑 이벤트 트리거 표시가 나왔으면 좋겠어. 천장과 숏은 봉 위에 빨간색,
        바닥과 롱은 아래에 초록색"* → 설계 3안 비교 후 **C안(하이브리드)** 채택.

C안: 증거신호는 **고정 레인**(종수를 진하기로), 이벤트 트리거만 **봉 밀착 삼각형**.
근거(87일 실측): 청산맵 창은 72봉(6시간)이고 컬럼 피치가 14.5px 뿐인데, 증거신호는 6시간당
  중앙 10개·90분위 22개가 발동한다(봉의 84.4%는 빈 봉). 전부 봉에 붙이면 90분위 창에서
  2.2컬럼당 하나가 되어 캔들을 덮는다. 반면 이벤트 트리거는 6시간당 0.7~5.5개뿐이다.
  정보 등급도 다르다 -- 증거신호는 배경, 이벤트 트리거는 주장.

🔴차트가 72봉인데 각 신호 페이로드의 이력 창이 48봉으로 제각각이라 그대로는 정렬이 안 된다.
   그래서 **하나의 타임스탬프 격자**로 맞춰 내보내는 엔드포인트를 따로 둔다.
⚠️ETH 전용이다. 증거신호는 BTC/XRP 에도 있지만 그 페이로드는 각자 다른 스크립트가 만들고
   교차자산 비교 기준도 달라, 여기서 임의로 재계산하면 화면과 다른 값이 된다. 다른 코인은
   `unsupported` 로 정직하게 비운다(빈 레인은 "신호 없음"으로 오독된다).
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402
import live_eth_extreme_detector_20260909 as LD  # noqa: E402

CHART_BARS = 72          # dashboard/live/app.js 의 SNAPSHOT_CHART_MAX_CANDLES 와 같아야 한다
SUPPORTED = ("eth",)


def _iso(t) -> str:
    return pd.Timestamp(t).tz_localize("UTC").isoformat()


def _merge_history(out: dict[str, dict], payload: dict | None, kind: str, label: str) -> None:
    """다른 신호의 이력 띠를 같은 격자에 얹는다. 없으면 조용히 건너뛴다.

    톤 규약(대시보드 §2): good = 롱 기대 = **바닥 쪽**, bad = 숏 기대 = **천장 쪽**.
    flat/neutral 은 마커를 만들지 않는다.
    ⚠️신호마다 이력 필드 모양이 다르다 -- v_rebound 는 (history, times) 를 같이 주지만
      돌파/되돌림은 `tone_history` + `latest_ts_utc` 뿐이라 5분 간격으로 시각을 되짚어야 한다.
      여기서 흡수하지 않으면 화면과 격자가 한 칸씩 어긋난다."""
    if not isinstance(payload, dict):
        return
    tones = payload.get("history") or payload.get("tone_history") or []
    times = payload.get("times") or []
    if not tones:
        return
    if not times:
        end = payload.get("latest_ts_utc")
        if not end:
            return
        t1 = pd.Timestamp(str(end).replace("Z", "+00:00"))
        if t1.tzinfo is None:
            t1 = t1.tz_localize("UTC")
        times = [(t1 - pd.Timedelta(minutes=5 * (len(tones) - 1 - i))).isoformat()
                 for i in range(len(tones))]
    for tone, t in zip(tones, times):
        if tone not in ("good", "bad"):
            continue
        ts_ = pd.Timestamp(str(t).replace("Z", "+00:00"))
        key = (ts_ if ts_.tzinfo else ts_.tz_localize("UTC")).tz_convert("UTC").isoformat()
        if key not in out:
            continue
        out[key].setdefault("events", []).append(
            {"kind": kind, "label": label, "side": "bottom" if tone == "good" else "top"})


def _onset_only(grid: dict[str, dict], times: list[str]) -> list[dict]:
    """연속 봉으로 이어지는 같은 (kind, side) 이벤트는 **첫 봉만** 남긴다.

    2026-09-10 사용자 요청("이벤트 트리거 발동 시점에만 삼각형 표시"). 세 출처가 전부 봉별
    **상태**라 조건이 유지되는 동안 매 봉 이벤트가 나온다 -- 극점은 `grade != '-' & ~gated`,
    V자반등/돌파되돌림은 톤이 good/bad 인 봉. 실제로 V자반등이 09-09 15:20~15:35 네 봉 연속으로
    찍히고 있었다. 증거신호 스트립에서 같은 성질을 고친 것과 같은 이야기다(커밋 3033dec).
    ⚠️창 왼쪽 끝은 직전 봉을 볼 수 없어, 이미 진행 중이던 발동도 발동 시점으로 그린다. 6시간
      창이라 실용상 무시할 수 있고, 그 자리에 신호가 살아 있다는 것 자체는 사실이다.
    grade(극점 강/중/약)는 키에 넣지 않는다 -- 한 발동 안에서 등급이 오르내려도 발동은 하나다.
    """
    prev: set[tuple[str, str]] = set()
    out: list[dict] = []
    for t in times:
        cur: set[tuple[str, str]] = set()
        for ev in grid[t].get("events", []):
            key = (str(ev.get("kind", "")), str(ev.get("side", "")))
            cur.add(key)
            if key not in prev:
                out.append(dict(ev, t=t))
        prev = cur
    return out


def compute_chart_markers(asset: str = "eth", v_rebound: dict | None = None,
                          extreme: dict | None = None) -> dict[str, Any]:
    """청산맵 72봉에 정렬된 마커. 절대 예외를 올리지 않는다."""
    asset = (asset or "eth").lower()
    if asset not in SUPPORTED:
        return {"available": False, "unsupported": True, "asset": asset,
                "reason": "증거신호·극점 탐지기 격자를 ETH 에서만 만듭니다"}
    try:
        kl = LD._fetch(LD.SYMBOL)
        if kl is None or len(kl) < 1200:
            return {"available": False, "asset": asset, "error": "price_fetch_failed"}
        btc = LD._fetch(LD.BTC_SYMBOL)
        if btc is None:
            return {"available": False, "asset": asset, "error": "btc_fetch_failed"}
        sig = compute_signals(kl, btc_df=btc, funding_df=None)
        ts = pd.to_datetime(sig["timestamp"])
        n = len(sig)
        lo = max(n - CHART_BARS, 0)
        bot = np.zeros(n, int); top = np.zeros(n, int)
        for s in B.SIGNALS:
            bot += sig[f"bottom_{s}"].fillna(False).to_numpy(bool).astype(int)
            top += sig[f"top_{s}"].fillna(False).to_numpy(bool).astype(int)
        names_b, names_t = [], []
        for i in range(lo, n):
            names_b.append(",".join(B.ABBR.get(s, s) for s in B.SIGNALS
                                    if bool(sig[f"bottom_{s}"].fillna(False).iloc[i])))
            names_t.append(",".join(B.ABBR.get(s, s) for s in B.SIGNALS
                                    if bool(sig[f"top_{s}"].fillna(False).iloc[i])))
        grid = {_iso(t): {"t": _iso(t)} for t in ts.iloc[lo:]}
        times = list(grid.keys())

        # 🔴2026-09-10 장애 수정: 극점 탐지기를 여기서 **채점하지 않는다**. 다른 두 신호와 똑같이
        #   이미 계산된 페이로드(워커 상태 파일)를 얹기만 한다.
        #   원인: 다른 세션이 이날 01:00 극점 아티팩트를 HGB -> TabPFN 으로 올렸다
        #   (model.joblib 1.8MB -> 1.08GB, meta.model="tabpfn"). 그 세션은 같은 이유로 채점을
        #   워커로 뺐는데(4c3805d) 이 함수만 인라인으로 남아 있었다. 서버 실측:
        #     load_artifact 415.9s + 5모델 predict 71.9s = 1회 488s. 캐시 TTL 이 60초라
        #     듀티사이클 814% -- asyncio.to_thread 의 기본 풀(16스레드)이 몇 분 만에 고갈되고
        #     to_thread 를 쓰는 모든 엔드포인트(증거신호 포함)가 영원히 큐에 걸렸다.
        #     증상: /api/state 는 2ms 인데 /api/evidence-signals 는 끝나지 않음(사용자 신고
        #     "증거신호가 대시보드에 안 나온다"). klines 0.16s·compute_signals 0.09s 로 나머지는
        #     전부 무죄였다.
        #   ⚠️봉별 등급(강/중/약)과 확률은 워커 이력에 없어서 여기서 사라진다 -- 삼각형의 위치와
        #     측면은 그대로다. 등급을 되살리려면 워커가 등급 이력을 내보내야 한다(다른 세션 파일).
        _merge_history(grid, extreme, "extreme", "극점")
        _merge_history(grid, v_rebound, "v_rebound", "V자반등")
        return {
            "available": True, "asset": asset, "bars": CHART_BARS,
            "latest_ts_utc": times[-1] if times else None,
            "times": times,
            "ev_bottom": [int(x) for x in bot[lo:]], "ev_top": [int(x) for x in top[lo:]],
            "ev_bottom_names": names_b, "ev_top_names": names_t,
            "events": _onset_only(grid, times),
            "n_signals_total": len(B.SIGNALS),
        }
    except Exception as e:  # noqa: BLE001 -- 차트 렌더를 절대 깨지 않는다
        return {"available": False, "asset": asset, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import json
    d = compute_chart_markers("eth")
    print(json.dumps({k: v for k, v in d.items() if k not in ("times", "ev_bottom_names",
                                                              "ev_top_names")},
                     ensure_ascii=False, indent=1)[:1800])
