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
import json
import pathlib
import struct

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,2,3,5,10,20,30,60,120,300")
    ap.add_argument("--tag", default="microprice_second")
    a = ap.parse_args()
    files = sorted(RASTER.glob("*.f32"))
    assert files, f"래스터 파일이 없다: {RASTER}"
    parts = [read_hour(p) for p in files]
    ts = np.concatenate([x["ts"] for x in parts])
    mid = np.concatenate([x["mid"] for x in parts])
    blo = np.concatenate([x["bin_lo"] for x in parts])
    qty = np.vstack([x["qty"] for x in parts])
    bs = parts[0]["bin_size"]
    ok = np.isfinite(mid) & (mid > 0)
    print(f"래스터 {len(files)}시간 · {len(ts):,}초 · 유효 {ok.mean():.1%} · 빈 {parts[0]['n_bins']} × {bs}")
    bi, ai, qb, qa = best_levels(qty)
    bid_px = (blo + bi) * bs; ask_px = (blo + ai) * bs
    good = ok & (bi >= 0) & (ai >= 0) & np.isfinite(qb) & np.isfinite(qa) & (qb + qa > 0)
    spread = ask_px - bid_px
    good &= (spread > 0) & (spread < 5 * bs)      # 비정상 스프레드 제외
    print(f"  최우선 양측 존재 {good.mean():.1%} · 스프레드 중앙 {np.nanmedian(spread[good]):.3f}달러 "
          f"({1e4*np.nanmedian(spread[good]/mid[good]):.2f}bp)")

    QI = np.where(good, (qb - qa) / (qb + qa), np.nan)
    MP = np.where(good, (bid_px * qa + ask_px * qb) / (qb + qa), np.nan)
    dMP = np.where(good, (MP - mid) / np.maximum(spread, 1e-9), np.nan)   # 스프레드 단위
    dqb = np.diff(qb, prepend=np.nan); dqa = np.diff(qa, prepend=np.nan)
    OFI = np.where(good, dqb - dqa, np.nan)
    lm = np.log(np.maximum(mid, 1e-9))
    preds = {"QI 큐불균형": QI, "dMP 마이크로프라이스편차": dMP, "OFI 잔량변화": OFI}
    print(f"\n{'지표':>22} {'지평':>5} {'전방 IC':>9} {'t':>7} {'E|r|bp':>8} {'손익분기IC':>10} {'관측/필요':>8}")
    rep = {}
    for name, x in preds.items():
        rep[name] = {}
        for H in [int(v) for v in a.horizons.split(",")]:
            fwd = np.full(len(lm), np.nan); fwd[:-H] = lm[H:] - lm[:-H]
            m = np.isfinite(x) & np.isfinite(fwd)
            if m.sum() < 500:
                continue
            ic = float(spearmanr(x[m], fwd[m]).statistic)
            e = float(np.mean(np.abs(fwd[m])) * 1e4)
            be = 5.88 / (e * 0.7979) if e > 0 else np.nan
            s = np.sign(x[m]); s[s == 0] = 1
            r = s * fwd[m] * 1e4
            t = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 else np.nan
            rep[name][H] = {"ic": ic, "e_abs_bp": e, "breakeven_ic": be, "t": t, "n": int(m.sum())}
            print(f"{name:>22} {H:>4}초 {ic:>+9.4f} {t:>+7.2f} {e:>8.2f} {be:>10.3f} "
                  f"{ic/be if be > 0 else np.nan:>7.0%}")
        print()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"저장: {OUT/(a.tag+'.json')}  ⚠️표본 {len(files)}시간 — 탐색적")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
