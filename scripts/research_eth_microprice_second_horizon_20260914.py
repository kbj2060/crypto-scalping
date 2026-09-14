"""**초 단위 호가 불균형의 예측력** — 문헌을 우리 거래소·우리 자산에서 재현 (2026-09-14).

사용자 *"실시간 호가창이나 빠른 데이터로 잡는 전략"* → 문헌 조사에서 나온 두 결과를 직접 잰다:
  · Cont·Kukanov·Stoikov(2013): 단기 가격변화 ≈ 주문흐름 불균형(OFI), 기울기 ∝ 1/깊이
  · Stoikov(2018) micro-price: 큐 불균형으로 보정한 공정가가 중간가보다 다음 가격을 잘 맞힌다

같은 날 1분봉에서 잰 결과는 **동시 ρ +0.62 · 전방 IC −0.021**(= 정보는 봉 안에서 소진)이었다.
여기서는 그 «봉 안»을 **초 단위**로 열어본다. 재료는 오늘 가동된 호가 래스터
(`live_orderflow_raster_collector_20260914`): 1초 1행, 가격대별 잔량, 부호가 방향(+매수/−매도).

지표 셋:
  QI   = (최우선 매수잔량 − 최우선 매도잔량) / 합           ← 큐 불균형
  MP   = 큐가중 공정가 = (bid·Qa + ask·Qb)/(Qb+Qa)          ← Stoikov micro-price
  dMP  = MP − mid, 틱 단위                                   ← 「중간가가 얼마나 치우쳤나」
  OFI  = 최우선 잔량 변화의 부호합(Cont 정의의 래스터 근사)

⚠️표본이 몇 시간뿐이라 **탐색적**이다. 결론이 아니라 「이 해상도에 신호가 있나」의 첫 읽기다.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import struct
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
RASTER = ROOT / "data/live/orderflow/raster/ETHUSDT"
OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
HDR = struct.Struct("<4sHHfIqII")          # magic, ver, n_bins, bin_size, flags, hour_ms, dt_ms, rsv


def read_hour(p: pathlib.Path):
    raw = p.read_bytes()
    magic, ver, n_bins, bin_size, flags, hour_ms, dt_ms, _ = HDR.unpack_from(raw, 0)
    assert magic == b"FLWR", f"{p.name}: magic {magic!r}"
    row = 8 + 4 + 4 + 4 * n_bins
    n = (len(raw) - HDR.size) // row
    buf = np.frombuffer(raw, dtype=np.uint8, offset=HDR.size, count=n * row).reshape(n, row)
    ts = buf[:, 0:8].copy().view("<i8").ravel()
    blo = buf[:, 8:12].copy().view("<i4").ravel()
    mid = buf[:, 12:16].copy().view("<f4").ravel().astype(np.float64)
    qty = buf[:, 16:].copy().view("<f4").reshape(n, n_bins).astype(np.float64)
    return {"ts": ts, "bin_lo": blo, "mid": mid, "qty": qty, "bin_size": float(bin_size),
            "n_bins": int(n_bins), "ver": int(ver)}


def best_levels(q: np.ndarray):
    """행마다 최우선 매수(+ 중 가장 높은 빈)·최우선 매도(− 중 가장 낮은 빈)의 (빈, 잔량)."""
    n, nb = q.shape
    bid_i = np.full(n, -1); ask_i = np.full(n, -1)
    pos = q > 0; neg = q < 0
    has_b = pos.any(1); has_a = neg.any(1)
    bid_i[has_b] = nb - 1 - np.argmax(pos[has_b][:, ::-1], axis=1)     # 가장 높은 양수 빈
    ask_i[has_a] = np.argmax(neg[has_a], axis=1)                        # 가장 낮은 음수 빈
    qb = np.where(bid_i >= 0, q[np.arange(n), np.clip(bid_i, 0, nb - 1)], np.nan)
    qa = np.where(ask_i >= 0, -q[np.arange(n), np.clip(ask_i, 0, nb - 1)], np.nan)
    return bid_i, ask_i, qb, qa


# ── bookTicker(.bt) 리더 — 진짜 최우선 호가 (2026-09-14 추가) ────────────────────
BT_ROOT = ROOT / "data/live/orderflow/bookticker/ETHUSDT"
BT_HDR = struct.Struct("<4sHHqQII")        # magic"BTKR", ver, rowsz, hour_ms, first_u, flags, rsv
BT_ROW = struct.Struct("<qdfdf")           # ts_ms, bid_px, bid_qty, ask_px, ask_qty


def read_bt(p: pathlib.Path) -> dict:
    """시각 파일 하나를 읽는다. `.bt.gz` 도 그대로 받는다(지난 시각은 수집기가 압축한다).

    ⚠️수집기가 WS 재연결 때 같은 파일에 이어붙이므로 **헤더는 파일당 하나**다. 행은 불규칙
    시각이고 여기서 리샘플하지 않는다 — 이벤트 해상도가 이 파일의 존재 이유다."""
    raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    magic, ver, rowsz, _hour_ms, _first_u, _f, _r = BT_HDR.unpack_from(raw, 0)
    assert magic == b"BTKR", f"{p.name}: magic {magic!r}"
    assert rowsz == BT_ROW.size, f"{p.name}: rowsz {rowsz}"
    n = (len(raw) - BT_HDR.size) // rowsz
    buf = np.frombuffer(raw, dtype=np.uint8, offset=BT_HDR.size, count=n * rowsz).reshape(n, rowsz)
    return {"ts": buf[:, 0:8].copy().view("<i8").ravel(),
            "bid_px": buf[:, 8:16].copy().view("<f8").ravel(),
            "bid_qty": buf[:, 16:20].copy().view("<f4").ravel().astype(np.float64),
            "ask_px": buf[:, 20:28].copy().view("<f8").ravel(),
            "ask_qty": buf[:, 28:32].copy().view("<f4").ravel().astype(np.float64),
            "ver": int(ver)}


def load_bt(glob: str = "*.bt*"):
    files = sorted(BT_ROOT.glob(glob))
    assert files, f"bookTicker 파일이 없다: {BT_ROOT}"
    parts = [read_bt(p) for p in files]
    ts = np.concatenate([x["ts"] for x in parts])
    bid = np.concatenate([x["bid_px"] for x in parts])
    ask = np.concatenate([x["ask_px"] for x in parts])
    qb = np.concatenate([x["bid_qty"] for x in parts])
    qa = np.concatenate([x["ask_qty"] for x in parts])
    o = np.argsort(ts, kind="stable")          # 재연결 구간이 섞일 수 있다
    ts, bid, ask, qb, qa = ts[o], bid[o], ask[o], qb[o], qa[o]
    span = (ts[-1] - ts[0]) / 1000
    print(f"bookTicker {len(files)}파일 · {len(ts):,}행 · {span/60:.1f}분 · {len(ts)/max(span,1e-9):,.0f}행/초")
    return ts, bid, ask, qb, qa


def cont_ofi(bid, ask, qb, qa):
    """Cont·Kukanov·Stoikov(2013) 식 (3) 의 주문흐름 불균형, **이벤트 단위**.

    e_n = 1[b_n≥b_{n-1}]·Qb_n − 1[b_n≤b_{n-1}]·Qb_{n-1}
        − 1[a_n≤a_{n-1}]·Qa_n + 1[a_n≥a_{n-1}]·Qa_{n-1}
    가격이 오르면 그 잔량 전부가 새 매수압, 내리면 사라진 잔량 전부가 매도압이다 — 잔량 차분
    (`dqb−dqa`, 래스터판이 쓴 근사)은 **가격이 움직인 이벤트에서 부호가 틀린다**."""
    b0, a0, qb0, qa0 = bid[:-1], ask[:-1], qb[:-1], qa[:-1]
    b1, a1, qb1, qa1 = bid[1:], ask[1:], qb[1:], qa[1:]
    e = (np.where(b1 >= b0, qb1, 0.0) - np.where(b1 <= b0, qb0, 0.0)
         - np.where(a1 <= a0, qa1, 0.0) + np.where(a1 >= a0, qa0, 0.0))
    return np.concatenate([[np.nan], e])


def trailing_sum(ts, x, win_ms):
    """[t−win, t] 구간 합. 누적합 + searchsorted — 불규칙 시각에 맞는 유일한 방식."""
    c = np.concatenate([[0.0], np.nancumsum(np.nan_to_num(x))])
    lo = np.searchsorted(ts, ts - win_ms, side="left")
    return c[np.arange(len(ts)) + 1] - c[lo]


def forward_return(ts, lm, h_ms):
    """**시각 기준** 전방 로그수익. 이벤트 데이터에서 행 shift 는 지평이 아니다 --
    676행/초 구간의 10행과 정체 구간의 10행은 전혀 다른 시간이다.

    🔴수집 구멍을 뛰어넘지 않는다: 착지 시각이 목표를 h_ms 넘게 지나치면(=실경과 2H 초과)
    버린다. 래스터에 43초 구멍이 6개 있어 1초 E|r| 이 0.62 → 14.6bp 로 부풀었던 자리다.
    bookTicker 는 지금 최대 359ms 라 영향이 없지만, WS 재연결이 쌓이면 같은 함정이 생긴다."""
    j = np.searchsorted(ts, ts + h_ms, side="left")
    ok = j < len(ts)
    fwd = np.full(len(ts), np.nan)
    fwd[ok] = lm[j[ok]] - lm[ok]
    over = np.full(len(ts), np.inf)
    over[ok] = ts[j[ok]] - (ts[ok] + h_ms)
    return np.where(over <= h_ms, fwd, np.nan)


def nonoverlap_idx(ts, h_ms):
    """겹치지 않는 결정만 고른다. 이벤트 IC 의 t 는 겹침 때문에 몇 배로 부푼다."""
    out, i, n = [], 0, len(ts)
    while i < n:
        out.append(i)
        j = int(np.searchsorted(ts, ts[i] + h_ms, side="left"))
        i = j if j > i else i + 1
    return np.array(out, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["raster", "bt"], default="bt",
                    help="raster=$0.50 빈(인공물 기록) · bt=최우선 호가 원천")
    ap.add_argument("--horizons", default="0.1,0.25,0.5,1,2,5,10,30,60")
    ap.add_argument("--cost-bp", type=float, default=5.88, help="왕복 비용(진입 2.95+peg 청산 2.93)")
    ap.add_argument("--tag", default="microprice_bt")
    a = ap.parse_args()

    if a.source == "bt":
        ts, bid_px, ask_px, qb, qa = load_bt()
        mid = (bid_px + ask_px) / 2.0
        spread = ask_px - bid_px
        good = (np.isfinite(mid) & (mid > 0) & (spread > 0) & (qb > 0) & (qa > 0))
        tick = 0.01
        print(f"  스프레드 중앙 ${np.median(spread[good]):.4f} "
              f"({1e4*np.median(spread[good]/mid[good]):.3f}bp) · {np.median(spread[good])/tick:.1f}틱 · "
              f"1틱 비율 {np.mean(np.isclose(spread[good], tick)):.1%}")
    else:
        files = sorted(RASTER.glob("*.f32"))
        assert files, f"래스터 파일이 없다: {RASTER}"
        parts = [read_hour(p) for p in files]
        ts = np.concatenate([x["ts"] for x in parts])
        mid = np.concatenate([x["mid"] for x in parts])
        blo = np.concatenate([x["bin_lo"] for x in parts])
        qty = np.vstack([x["qty"] for x in parts])
        bs = parts[0]["bin_size"]
        # 🔴래스터에는 ts=0 행이 섞여 있다(2026-09-14 실측 8,836행 중 228행). 시각기준 전방수익이
        # 그 구멍을 뛰어넘어 1초 E|r| 을 0.8 → 14.6bp 로 부풀렸다. 원본 수집기 쪽 결함이다.
        keep = ts > 0
        ts, mid, blo, qty = ts[keep], mid[keep], blo[keep], qty[keep]
        bi, ai, qb, qa = best_levels(qty)
        bid_px = (blo + bi) * bs
        ask_px = (blo + ai) * bs
        spread = ask_px - bid_px
        good = (np.isfinite(mid) & (mid > 0) & (bi >= 0) & (ai >= 0)
                & np.isfinite(qb) & np.isfinite(qa) & (qb + qa > 0) & (spread > 0) & (spread < 5 * bs))
        print(f"래스터 {len(files)}시간 · {len(ts):,}초 · 최우선 양측 {good.mean():.1%} · 빈 {bs}")

    QI = np.where(good, (qb - qa) / (qb + qa), np.nan)
    MP = np.where(good, (bid_px * qa + ask_px * qb) / (qb + qa), np.nan)
    dMP = np.where(good, (MP - mid) / np.maximum(spread, 1e-9), np.nan)
    OFI = np.where(good, cont_ofi(bid_px, ask_px, qb, qa), np.nan)

    # ⭐자체점검: 진짜 최우선 호가면 (MP−mid)/spread ≡ QI/2 다(대수 항등식). 래스터는 이게
    #   깨져서 dMP +0.279 / QI −0.029 로 갈렸고, 그 격차 자체가 빈격자 잔차였다.
    gap = float(np.nanmax(np.abs(dMP - QI / 2)))
    print(f"\n항등식 |dMP − QI/2| 최대 = {gap:.3e}  → "
          + ("✅원천이 정확하다(두 IC 는 반드시 일치)" if gap < 1e-9
             else f"🔴격자 잔차 {gap:.4f} — dMP 의 초과 IC 는 신호가 아니다"))
    if a.source == "bt":
        assert gap < 1e-9, f"bookTicker 인데 항등식이 깨졌다: {gap}"

    lm = np.log(np.maximum(mid, 1e-9))
    hs = [float(v) for v in a.horizons.split(",")]
    rep: dict = {}
    print(f"\n{'지표':>10} {'지평':>7} {'전방IC':>9} {'n겹침':>10} {'n비겹':>7} "
          f"{'E|r|bp':>7} {'총bp':>7} {'블록t':>7} {'순bp':>8}")
    for name, x in {"QI": QI, "OFI": OFI}.items():
        rep[name] = {}
        for H in hs:
            h_ms = int(round(H * 1000))
            xx = trailing_sum(ts, x, h_ms) if name == "OFI" else x
            xx = np.where(good, xx, np.nan)
            fwd = forward_return(ts, lm, h_ms)
            m = np.isfinite(xx) & np.isfinite(fwd)
            if m.sum() < 500:
                continue
            ic = float(spearmanr(xx[m], fwd[m]).statistic)
            e = float(np.mean(np.abs(fwd[m])) * 1e4)
            k = nonoverlap_idx(ts, h_ms)
            k = k[m[k]]
            s = np.sign(xx[k]); s[s == 0] = 1
            r = s * fwd[k] * 1e4
            t = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 else np.nan
            net = float(r.mean() - a.cost_bp)
            rep[name][H] = {"ic": ic, "n": int(m.sum()), "n_nonoverlap": int(len(r)),
                            "e_abs_bp": e, "gross_bp": float(r.mean()), "t_block": t, "net_bp": net}
            print(f"{name:>10} {H:>6}초 {ic:>+9.4f} {m.sum():>10,} {len(r):>7,} "
                  f"{e:>7.3f} {r.mean():>+7.3f} {t:>+7.2f} {net:>+8.3f}")
        print()
    print(f"dMP 는 QI 의 단조변환이라 IC 가 동일하다 — 따로 싣지 않는다(항등식 점검 위 참조).")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(
        {"source": a.source, "identity_gap": gap, "cost_bp": a.cost_bp, "rows": int(len(ts)),
         "span_sec": float((ts[-1] - ts[0]) / 1000), "metrics": rep}, indent=1, ensure_ascii=False))
    print(f"저장: {OUT/(a.tag+'.json')}")
    return 0


def _selfcheck() -> None:
    """항등식과 OFI 부호를 손으로 계산한 예로 확인한다(프레임워크 없이)."""
    b = np.array([100.0, 100.0, 100.01]); a_ = np.array([100.01, 100.01, 100.02])
    qb_ = np.array([10.0, 30.0, 5.0]); qa_ = np.array([10.0, 10.0, 10.0])
    mid = (b + a_) / 2; sp = a_ - b
    qi = (qb_ - qa_) / (qb_ + qa_)
    mp = (b * qa_ + a_ * qb_) / (qb_ + qa_)
    assert np.allclose((mp - mid) / sp, qi / 2, atol=1e-12), "항등식이 깨졌다"
    e = cont_ofi(b, a_, qb_, qa_)
    assert e[1] == 20.0, f"매수잔량만 +20 인데 {e[1]}"          # 가격 그대로, 매수 10→30
    # 매수·매도호가 둘 다 상승: 새 매수잔량 +5, 옛 매수잔량은 차감 안 함(소진=매수압),
    # 매도호가가 올라갔으니 사라진 옛 매도잔량 +10 → +15
    assert e[2] == 15.0, f"양측 호가 상승 이벤트: {e[2]}"
    ts = np.array([0, 100, 250, 400], dtype=np.int64)
    assert np.allclose(trailing_sum(ts, np.ones(4), 200), [1, 2, 2, 2]), "구간합"
    lm = np.log(np.array([100.0, 101.0, 102.0, 103.0]))
    f = forward_return(ts, lm, 200)
    assert np.isclose(f[0], lm[2] - lm[0]) and np.isnan(f[3]), "전방수익 시각기준"
    # 구멍 건너뛰기 금지: 0ms 다음 유효행이 5,000ms 면 200ms 지평은 답이 없다
    g = forward_return(np.array([0, 5000, 5100], dtype=np.int64), np.log(np.array([1.0, 2.0, 3.0])), 200)
    assert np.isnan(g[0]), f"43초 구멍을 뛰어넘었다: {g[0]}"
    assert list(nonoverlap_idx(ts, 200)) == [0, 2], "비겹침 선택"
    print("자체점검 통과")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    raise SystemExit(main())
