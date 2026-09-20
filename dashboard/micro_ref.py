"""«미시 참고» 카드의 계산부 (2026-09-20). 서버는 값을 모아 이 함수들에 넣고 결과를 그대로 보낸다.

근거: docs/experiments/eth_realtime_five_stream_1s_joint_analysis_20260920.md (1초 패널 4.6일).
  ① QI(최우선 큐 기울기) × OFI(10초 호가 순증감)가 같은 쪽일 때만 1~15초 방향 (+0.4bp/15s, 갈리면 0)
  ② 60초 거래량은 «같은 시간대» 분위로만 «많다/적다» (미국장 13~15 UTC 는 평소 2~8배)
  ③ 청산맵 레벨 ±15bp 는 청산 5배·앞 5분 고저폭 2배의 «위치 예보» (방향 아님)
  ④ 거래량 폭발 5초 뒤 OI 급감 = 청산 덩어리 통과 (거래량→ΔOI +5s −0.23)
  ⑤ @forceOrder 원시 이벤트(지금까지는 1분 합만 저장) → 최근 60초 롱/숏 청산 $
🔴임계값은 달러·수량 상수가 아니라 분위로 둔다(2026-09-19 규약). 이 파일의 상수는 «분위 값»이다.
ponytail: 부호 규약 — 매수·매수벽·지지쪽이 양수.
"""
from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

QI_SIDE_ABS = 0.56        # |QI| p50 (4.6일 실측 0.56) -- 이보다 기울어야 «쪽»을 말한다
NEAR_BP = 15.0            # 레벨 «근접» 판정폭 (분석과 같은 값)
BAND_SHALLOW_BP = 10.0    # 얕은 불균형 밴드
BAND_DEEP_BP = 40.0       # 깊은 불균형 밴드 -- 래스터가 ±$120(≈46bp) 라 50 대신 40
BURST_WINDOW_S = 8        # 거래량 폭발 뒤 OI 를 보는 창 (CCF 피크 +5초 + OI 갱신 3.7초)
BURST_DOI_ABS_PCT = 0.9   # 최근 10분 |ΔOI 8s| 의 이 분위를 넘어야 «급감»


def qi(bid_qty: float, ask_qty: float) -> float:
    s = bid_qty + ask_qty
    return (bid_qty - ask_qty) / s if s > 0 else 0.0


def side_of(x: float, thr: float) -> str:
    return "매수" if x >= thr else ("매도" if x <= -thr else "중립")


def raster_flow(window: dict[str, Any]) -> dict[str, Any]:
    """read_window() 결과 → 얕은/깊은 불균형(마지막 유효 초)과 10초 OFI.

    qty 는 절대 가격축(열 k 의 가격 = (bin_lo + k) * bin_size), + 매수호가 / − 매도호가.
    OFI_1s = Δ(밴드 안 매수호가 합) − Δ(밴드 안 매도호가 합). 취소·체결을 못 가르지만
    분석의 dd_ofi 와 같은 정의다(레벨 총량 변화). 밴드는 ±BAND_DEEP_BP, 격자는 절대라
    초가 바뀌어도 같은 레벨을 비교한다."""
    q = np.asarray(window.get("qty"))
    mid = np.asarray(window.get("mid"), dtype=float)
    if q.ndim != 2 or q.shape[1] == 0 or not np.isfinite(mid).any():
        return {"ok": False}
    bin_lo, bs = int(window["bin_lo"]), float(window["bin_size"])
    px = (bin_lo + np.arange(q.shape[1])) * bs
    last = int(np.flatnonzero(np.isfinite(mid))[-1])
    m = float(mid[last])

    def band_sums(row: np.ndarray, bp: float) -> tuple[float, float]:
        w = m * bp / 1e4
        sel = (px >= m - w) & (px <= m + w)
        r = row[sel]
        return float(r[r > 0].sum()), float(-r[r < 0].sum())

    b10, a10 = band_sums(q[last], BAND_SHALLOW_BP)
    b40, a40 = band_sums(q[last], BAND_DEEP_BP)
    # OFI: 마지막 10개 «유효 연속» 초의 밴드 합 변화. 무효 초(mid NaN)를 만나면 거기서 끊는다.
    sums = []
    for i in range(last, max(-1, last - 11), -1):
        if not np.isfinite(mid[i]):
            break
        sums.append(band_sums(q[i], BAND_DEEP_BP))
    ofi = 0.0
    for (b1, a1), (b0, a0) in zip(sums[:-1], sums[1:]):
        ofi += (b1 - b0) - (a1 - a0)
    return {"ok": True, "mid": m, "imb10": qi(b10, a10), "imb40": qi(b40, a40),
            "bid40": b40, "ask40": a40, "ofi10": ofi, "ofi_secs": max(0, len(sums) - 1)}


def agree_state(qi_val: float, ofi10: float, ofi_thr: float) -> str:
    q = side_of(qi_val, QI_SIDE_ABS)
    o = side_of(ofi10, ofi_thr)
    if q == "중립" or o == "중립":
        return "중립"
    return "동조매수" if q == o == "매수" else ("동조매도" if q == o == "매도" else "갈림")


def pct_rank(x: float, sorted_vals: list[float] | np.ndarray) -> float | None:
    """x 가 기준 분포에서 몇 분위인가(0~1). 기준이 비면 None."""
    a = np.asarray(sorted_vals, dtype=float)
    if a.size < 20 or not np.isfinite(x):
        return None
    return float(np.searchsorted(a, x, side="right") / a.size)


def act_label(p: float | None) -> str:
    if p is None:
        return "기준없음"
    return "활발" if p >= 0.8 else ("조용" if p <= 0.2 else "보통")


def sr_context(levels: dict[str, Any] | None, mid: float, imb40: float | None) -> dict[str, Any]:
    """청산맵 payload(support_levels/resistance_levels, 가까운 순) + 현재 mid → 근접 상태.
    deep_wall: 깊은 불균형이 «가까운 레벨 쪽»이면 «레벨쪽»(버팀 후보), 반대면 «반대쪽»(돌파 후보)."""
    out: dict[str, Any] = {"sup": None, "res": None, "sup_bp": None, "res_bp": None, "near": "없음", "deep_wall": None}
    if not levels or not (mid > 0):
        return out
    s = (levels.get("support_levels") or [None])[0]
    r = (levels.get("resistance_levels") or [None])[0]
    if s:
        out["sup"] = float(s["price"]); out["sup_bp"] = (mid - out["sup"]) / mid * 1e4
    if r:
        out["res"] = float(r["price"]); out["res_bp"] = (out["res"] - mid) / mid * 1e4
    cands = [(abs(out["sup_bp"]), "지지근접", +1) if out["sup_bp"] is not None else None,
             (abs(out["res_bp"]), "저항근접", -1) if out["res_bp"] is not None else None]
    cands = [c for c in cands if c]
    if cands:
        d, name, sign = min(cands)
        if d <= NEAR_BP:
            out["near"] = name
            if imb40 is not None:
                out["deep_wall"] = "레벨쪽" if imb40 * sign > 0 else "반대쪽"
    return out


def burst_state(vol1s: list[tuple[int, float]], oi_ring: dict[int, float], vol1s_p99: float | None,
                now_sec: int) -> dict[str, Any]:
    """최근 60초에서 «1초 거래량 ≥ 시간대 p99» 인 초를 찾아, 그 뒤 BURST_WINDOW_S 초의 ΔOI 를 본다.
    ΔOI 임계 = 링에 있는 8초 ΔOI 절대값의 p90 (자기 보정). 급감이면 «청산 덩어리 통과»."""
    out: dict[str, Any] = {"at": None, "vol1s": None, "doi": None, "state": "—", "p99": vol1s_p99}
    if vol1s_p99 is None or not oi_ring:
        return out
    secs = sorted(oi_ring)
    if len(secs) < 30:
        return out
    # 8초 ΔOI 분포(자기 보정 임계)
    arr = np.array([oi_ring[s] for s in secs]); ts = np.array(secs)
    d8 = []
    for i in range(len(secs)):
        j = np.searchsorted(ts, ts[i] + BURST_WINDOW_S, side="right") - 1
        if j > i:
            d8.append(arr[j] - arr[i])
    thr = float(np.quantile(np.abs(d8), BURST_DOI_ABS_PCT)) if len(d8) >= 20 else None
    hits = [(s, v) for s, v in vol1s if v >= vol1s_p99 and now_sec - 90 <= s <= now_sec - 2]
    if not hits:
        return out
    s, v = hits[-1]
    before = [k for k in secs if k <= s]
    after = [k for k in secs if s < k <= s + BURST_WINDOW_S]
    out.update(at=s, vol1s=v)
    if not before or not after:
        out["state"] = "OI 대기"
        return out
    doi = oi_ring[after[-1]] - oi_ring[before[-1]]
    out["doi"] = doi
    if thr is None:
        out["state"] = "기준없음"
    elif doi <= -thr:
        out["state"] = "청산 덩어리 통과"
    elif doi >= thr:
        out["state"] = "신규 포지션 유입"
    else:
        out["state"] = "OI 변화 없음"
    out["thr"] = thr
    return out


# ── 기준선(같은 시간대 분위) — trade_tape.duckdb 읽기 전용 ──────────────────
def baseline_from_tape(db_path: Path, days: int = 7) -> dict[str, Any] | None:
    """UTC 시간대별 «1분 거래량» 분포와 «1초 거래량 p99». 수집기 아카이브를 읽기 전용으로 연다.
    실패(락·파일 없음)면 None -- 화면은 절대값만 보인다."""
    try:
        import duckdb  # noqa: PLC0415
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            since = int(time.time()) - days * 86400
            rows = con.execute(
                "SELECT (ts_sec // 60) * 60 AS m, sum(buy_qty + sell_qty) AS v FROM trade_tape_1s "
                "WHERE ts_sec >= ? GROUP BY 1", [since]).fetchall()
            sec = con.execute(
                "SELECT ((ts_sec % 86400) // 3600) AS h, quantile_cont(v, 0.99) FROM ("
                "SELECT ts_sec, sum(buy_qty + sell_qty) AS v FROM trade_tape_1s WHERE ts_sec >= ? GROUP BY 1) "
                "GROUP BY 1", [since]).fetchall()
        finally:
            con.close()
    except Exception:  # noqa: BLE001 -- 기준선은 장식이다. 못 읽으면 없이 간다.
        return None
    if len(rows) < 600:
        return None
    by_hour: dict[int, list[float]] = {h: [] for h in range(24)}
    for m, v in rows:
        by_hour[int((m % 86400) // 3600)].append(float(v))
    return {"minute_vol_sorted": {h: sorted(vs) for h, vs in by_hour.items()},
            "sec_vol_p99": {int(h): float(p) for h, p in sec}, "days": days, "built_at": time.time()}


def liq_prev_minute(db_path: Path) -> dict[str, Any] | None:
    """tail_risk_1m 의 마지막 완결 분 (봇이 쓰는 표, 읽기 전용)."""
    try:
        import duckdb  # noqa: PLC0415
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            row = con.execute("SELECT epoch(ts), long_usd_1m, short_usd_1m FROM tail_risk_1m ORDER BY ts DESC LIMIT 1").fetchone()
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    return {"ts": int(row[0]), "long": float(row[1] or 0.0), "short": float(row[2] or 0.0)}


if __name__ == "__main__":  # 자체점검 -- 부호 규약과 밴드 합, 근접 판정, 버스트 판정
    assert qi(3, 1) == 0.5 and qi(0, 0) == 0.0 and side_of(0.6, QI_SIDE_ABS) == "매수" and side_of(-0.2, QI_SIDE_ABS) == "중립"
    # 격자: bin_size 0.5, bin_lo 5200 → 가격 2600.0 부터. mid 2600.5. +매수 −매도.
    q = np.zeros((3, 8), np.float32)
    q[0] = [2, 2, 0, 0, -1, -1, 0, 0]; q[1] = [3, 2, 0, 0, -1, -2, 0, 0]; q[2] = [3, 3, 0, 0, -1, -1, 0, 0]
    w = {"qty": q, "mid": np.array([2600.5, 2600.5, 2600.5], np.float32), "bin_lo": 5200, "bin_size": 0.5}
    f = raster_flow(w)
    assert f["ok"] and f["bid40"] == 6.0 and f["ask40"] == 2.0 and math.isclose(f["imb40"], 0.5)
    assert math.isclose(f["ofi10"], ((5 - 4) - (3 - 2)) + ((6 - 5) - (2 - 3))) and f["ofi_secs"] == 2  # 초0→1: 매수+1 매도+1 → 0 · 초1→2: 매수+1 매도−1 → +2
    assert agree_state(0.7, 50, 10) == "동조매수" and agree_state(-0.7, 50, 10) == "갈림" and agree_state(0.1, 50, 10) == "중립"
    assert pct_rank(5, list(range(100))) == 0.06 and pct_rank(5, [1, 2]) is None and act_label(0.9) == "활발"
    lv = {"support_levels": [{"price": 2598.0}], "resistance_levels": [{"price": 2640.0}]}
    c = sr_context(lv, 2600.0, imb40=0.3)
    assert c["near"] == "지지근접" and c["deep_wall"] == "레벨쪽" and round(c["sup_bp"], 2) == 7.69
    assert sr_context(lv, 2600.0, imb40=-0.3)["deep_wall"] == "반대쪽" and sr_context(lv, 2620.0, 0.0)["near"] == "없음"
    now = 1_000_000
    ring = {now - 120 + i: 1_000_000.0 - (300.0 if i >= 100 else 0.0) for i in range(0, 121, 4)}  # 20초 전에 −300
    b = burst_state([(now - 24, 900.0)], ring, vol1s_p99=500.0, now_sec=now)
    assert b["state"] == "청산 덩어리 통과" and b["doi"] == -300.0, b
    assert burst_state([(now - 24, 100.0)], ring, 500.0, now)["state"] == "—"
    print("micro_ref selftest ok")
