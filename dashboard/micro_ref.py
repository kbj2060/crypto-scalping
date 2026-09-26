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
# 2026-09-20 «계속 바뀐다»(사용자). 깜빡임은 두 종류이고 하나만 공짜로 없앨 수 있다:
#   ①경계 채터 -- 값이 임계선 근처에서 떨어 라벨만 오간다. 슈미트 트리거로 제거(정보 손실 0).
#     실측: 시간당 변경 713->568(-20%) · 직접 반전 4.75->2.76(-42%) · edge15 +1.10->+1.13(SE 안).
#   ②신호 자체가 빠름 -- 이건 못 없앤다. 값의 **전부가 발동한 그 초**에 있다:
#     마지막 동조 이후 나이별 방향맞춘 앞 15초 수익 = 0초 +0.553 · 1~2초 +0.072 · 3초 이후 0(SE 안).
#     그래서 늦추는 장치는 늦춘 만큼 정확히 잃는다(3초 확인 -42% · 30초 점수 -74%).
#     🔴«최소 체류»는 쓰면 안 된다 -- edge -81% 인데 **직접 반전이 4배로 늘어난다**(4.75->17.75/시간).
#       얼려 두는 동안 원 상태가 반대로 가 있어서, 풀리는 순간 중립을 안 거치고 건너뛴다.
#   ⇒ 화면은 «지금 상태»가 아니라 «마지막 방아쇠 + 나이»를 보여준다(라벨은 새 방아쇠에서만 바뀐다).
QI_ENTER, QI_EXIT = 0.75, 0.45   # 슈미트 트리거. 들어가는 문턱은 높고 나오는 문턱은 낮다
TRIGGER_LIVE_S = 2               # 나이 이 이하만 «지금». 그 위는 지나간 것으로 흐리게 그린다
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


def qi_side_hyst(qi_val: float, prev: str = "중립") -> str:
    """|QI| >= QI_ENTER 에서 그 쪽으로 들어가고, |QI| < QI_EXIT 이 되어야 나온다.
    반대쪽으로 가려면 반대쪽 ENTER 를 넘어야 한다 -- 경계에서 떠는 값이 라벨을 못 흔든다."""
    if not np.isfinite(qi_val):
        return prev
    if prev == "매수":
        return "매도" if qi_val <= -QI_ENTER else ("매수" if qi_val >= QI_EXIT else "중립")
    if prev == "매도":
        return "매수" if qi_val >= QI_ENTER else ("매도" if qi_val <= -QI_EXIT else "중립")
    return "매수" if qi_val >= QI_ENTER else ("매도" if qi_val <= -QI_ENTER else "중립")


def agree_state(qi_val: float, ofi10: float, ofi_thr: float, qi_prev: str = "중립") -> tuple[str, str]:
    """(동조 상태, 이번 QI 쪽). QI 쪽은 다음 호출에 qi_prev 로 돌려줘야 히스테리시스가 이어진다."""
    q = qi_side_hyst(qi_val, qi_prev)
    o = side_of(ofi10, ofi_thr)
    if q == "중립" or o == "중립":
        return "중립", q
    return ("동조매수" if q == o == "매수" else "동조매도" if q == o == "매도" else "갈림"), q


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
# 2026-09-26 잠금 재시도: 수집기가 ~4.6초마다 파일 잠금을 쥔다(중앙 0.49초·최대 1.1초·시간의 10.6%,
#   F_GETLK 비침습 실측). duckdb 는 read_only 연결도 그동안 거부하므로 무작위 시도의 ~10%가 실패했다.
#   최대 점유보다 긴 간격으로 몇 번 다시 열면 실패는 ~0.1% 이하로 내려간다.
BASELINE_LOCK_RETRIES = 4
BASELINE_LOCK_WAIT_S = 1.2


def _connect_read_only_retry(db_path: Path, retries: int = BASELINE_LOCK_RETRIES, wait_s: float = BASELINE_LOCK_WAIT_S):
    import duckdb  # noqa: PLC0415
    for i in range(retries + 1):
        try:
            return duckdb.connect(str(db_path), read_only=True)
        except duckdb.IOException as exc:
            if "could not set lock" not in str(exc).lower() or i == retries:   # 경로에 «lock» 이 든 다른 오류까지 기다리지 않는다
                raise
            time.sleep(wait_s)


def baseline_from_tape(db_path: Path, days: int = 7) -> dict[str, Any] | None:
    """UTC 시간대별 «1분 거래량» 분포와 «1초 거래량 p99». 수집기 아카이브를 읽기 전용으로 연다.
    실패(락·파일 없음)면 None -- 화면은 절대값만 보인다."""
    try:
        con = _connect_read_only_retry(db_path)
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
    except Exception as exc:  # noqa: BLE001 -- 기준선은 장식이다. 못 읽으면 없이 가되, 왜인지는 올린다.
        raise RuntimeError(f"baseline_from_tape({db_path.name}): {exc!r}") from exc
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


def deriv_from_ring(ring: dict[int, tuple[float, float, float]], horizon_s: int, pct: float,
                    step_s: int = 30, min_samples: int = 60) -> dict[str, Any]:
    """markPrice@1s 링 {sec: (mark, index, funding)} → 펀딩·베이시스(bp)·지평 전 대비 Δ베이시스·그 임계.
    임계는 상수가 아니라 링 안 |Δ베이시스| 의 분위(pct) -- 표본이 min_samples 미만이면 None(판정 보류)."""
    out: dict[str, Any] = {"funding": None, "basis_bp": None, "basis_d_bp": None, "basis_thr_bp": None}
    if not ring:
        return out
    bas = lambda s: (ring[s][0] - ring[s][1]) / ring[s][1] * 1e4   # noqa: E731
    last = max(ring)
    out["funding"], out["basis_bp"] = ring[last][2], bas(last)
    prev = max((s for s in ring if s <= last - horizon_s), default=None)
    if prev is not None:
        out["basis_d_bp"] = bas(last) - bas(prev)
    ds = sorted(abs(bas(s) - bas(s - horizon_s)) for s in sorted(ring)[::step_s] if s - horizon_s in ring)
    if len(ds) >= min_samples:
        out["basis_thr_bp"] = ds[min(len(ds) - 1, int(len(ds) * pct))]
    return out


if __name__ == "__main__":  # 자체점검 -- 부호 규약과 밴드 합, 근접 판정, 버스트 판정
    assert qi(3, 1) == 0.5 and qi(0, 0) == 0.0 and side_of(0.6, QI_SIDE_ABS) == "매수" and side_of(-0.2, QI_SIDE_ABS) == "중립"
    # 격자: bin_size 0.5, bin_lo 5200 → 가격 2600.0 부터. mid 2600.5. +매수 −매도.
    q = np.zeros((3, 8), np.float32)
    q[0] = [2, 2, 0, 0, -1, -1, 0, 0]; q[1] = [3, 2, 0, 0, -1, -2, 0, 0]; q[2] = [3, 3, 0, 0, -1, -1, 0, 0]
    w = {"qty": q, "mid": np.array([2600.5, 2600.5, 2600.5], np.float32), "bin_lo": 5200, "bin_size": 0.5}
    f = raster_flow(w)
    assert f["ok"] and f["bid40"] == 6.0 and f["ask40"] == 2.0 and math.isclose(f["imb40"], 0.5)
    assert math.isclose(f["ofi10"], ((5 - 4) - (3 - 2)) + ((6 - 5) - (2 - 3))) and f["ofi_secs"] == 2  # 초0→1: 매수+1 매도+1 → 0 · 초1→2: 매수+1 매도−1 → +2
    assert agree_state(0.8, 50, 10)[0] == "동조매수" and agree_state(-0.8, 50, 10)[0] == "갈림" and agree_state(0.1, 50, 10)[0] == "중립"
    # 히스테리시스: ENTER 를 넘어야 들어가고, EXIT 밑으로 내려가야 나오며, 반대는 반대쪽 ENTER 가 필요하다
    assert qi_side_hyst(0.70, "중립") == "중립" and qi_side_hyst(0.80, "중립") == "매수"
    assert qi_side_hyst(0.50, "매수") == "매수" and qi_side_hyst(0.40, "매수") == "중립"
    assert qi_side_hyst(-0.50, "매수") == "중립" and qi_side_hyst(-0.80, "매수") == "매도"
    assert qi_side_hyst(float("nan"), "매도") == "매도"      # 값이 없으면 라벨을 흔들지 않는다
    # 경계에서 떠는 값이 라벨을 못 흔든다 -- 같은 입력열에서 고정 임계는 5번 바뀌고 히스테리시스는 1번
    seq = [0.80, 0.55, 0.60, 0.50, 0.58, 0.52]
    fixed = [side_of(v, QI_SIDE_ABS) for v in seq]
    hyst, prev = [], "중립"
    for v in seq:
        prev = qi_side_hyst(v, prev); hyst.append(prev)
    assert sum(a != b for a, b in zip(fixed, fixed[1:])) == 5 and sum(a != b for a, b in zip(hyst, hyst[1:])) == 0, (fixed, hyst)
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
    # 베이시스 파생 -- 프리미엄이 지평 동안 1bp→3bp 로 벌어지면 Δ=+2, 임계는 링 분위
    ring = {i: (2600.0 * (1 + (1.0 + 2.0 * (i / 7200)) / 1e4), 2600.0, -0.0002) for i in range(0, 7201)}
    dv = deriv_from_ring(ring, horizon_s=1800, pct=0.75)
    assert dv["funding"] == -0.0002 and abs(dv["basis_bp"] - 3.0) < 1e-6 and abs(dv["basis_d_bp"] - 0.5) < 1e-6, dv
    assert dv["basis_thr_bp"] is not None and abs(dv["basis_thr_bp"] - 0.5) < 1e-6, dv
    assert deriv_from_ring({0: (2601.0, 2600.0, 0.0)}, 1800, 0.75)["basis_d_bp"] is None
    assert deriv_from_ring({}, 1800, 0.75)["funding"] is None
    print("micro_ref selftest ok")
