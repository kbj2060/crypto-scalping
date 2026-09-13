"""**분할 진입의 천장** — 물타기·불타기를 오라클로 상한 지어 본다 (2026-09-14, 사용자 요청).

사용자: *"분할 진입의 목적은 물타기와 불타기 2개의 목적을 갖고 있어. 언제 새로 진입할지는
모델이 직접 정하게 해줘."*

## 모델을 짜기 전에 천장을 재는 이유
이 저장소는 「완벽 지식도 비용을 못 넘는다」로 축을 닫은 전례가 있다
([[eth_entry_exit_axes_closed_by_oracle_upper_bound_20260911]]: 완벽 변동성 지식 +2.19bp
< peg 7.8bp). **오라클이 못 넘으면 어떤 모델도 못 넘는다** -- 먼저 그걸 본다.

## 두 목적은 다른 축이다
총 명목이 상한에 묶이고 청산 시점이 같으면 PnL = Σ size·(청산/진입 − 1) 이므로
**더 비싸게 사는 건 산술적으로 손해**다. 즉 「같은 총량」 비교에서 불타기는 물타기에 진다.
불타기가 뜻을 가지려면 비교가 달라야 한다 -- **지는 거래에서 덜 쓰기**(조건부 노출).
  · 물타기 오라클 = 창 안의 **가장 싼 k−1 지점**에 추가(롱). 평단 개선의 상한.
  · 불타기 오라클 = **이길 거래에서만** 상한까지 채운다(부호를 미리 안다). 조건부 노출의 상한.
불타기 오라클은 사실상 **방향을 아는 것**이라, 이 축이 통과해도 그건 방향 예측 문제로
환원된다(이 저장소가 .50~.53 으로 닫은 축).

## 대조군
  · 단일: 진입 시점에 전량
  · 균등 시간분할: 창을 k 등분해 기계적으로 추가(09-13 에 기각된 그 팔)
⚠️의도 명목은 **전 팔 동일**하다. 실현 노출은 다르므로 «의도명목당」과 «노출당」을 같이 낸다.

⚠️오라클은 **상한**이지 전략이 아니다. 통과해도 그건 «모델을 만들 가치가 있다」까지다.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

KL = pathlib.Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
MAKER_BP, PEG_BP, EXIT_BP = 2.0, 2.95, 2.93
SEED = 20260914
WINDOWS = {"VAL(2025-09~12)⚠": ("2025-09-01", "2025-12-31"),
           "OOS(2026-01~03)": ("2026-01-01", "2026-03-31"),
           "TEST(2026-04~09)": ("2026-04-01", "2026-09-10")}


def run(c, hi, lo, idx, sides, *, k: int, w: int, mode: str) -> dict:
    """의도 명목 1 단위당 bp. mode: single / twap / avgdown_oracle / pyramid_oracle."""
    net, expo = [], []
    for i, s in zip(idx, sides):
        e0, exit_px = c[i], c[i + w]
        ret = s * (exit_px / e0 - 1.0)                      # 단일 진입의 수익(비용 전)
        if mode == "single":
            pnl = ret - (PEG_BP + EXIT_BP) / 1e4
            net.append(1e4 * pnl); expo.append(1.0); continue
        seg = c[i + 1:i + 1 + w]
        size = 1.0 / k
        # 칸 0 은 항상 진입 시점
        px = [e0]; tt = [0]
        if mode == "twap":                                  # 창을 k 등분해 기계적으로
            for j in range(1, k):
                t = int(round(j * w / k)) - 1
                px.append(float(seg[t])); tt.append(t)
        elif mode == "avgdown_oracle":
            # 🔴**안 넣는 선택도 오라클의 권한**이다. 첫 판은 k−1 칸을 강제로 쓰게 짰는데,
            # 단조 상승 경로에서는 진입가가 이미 최저가라 나중 추가가 평단을 **나쁘게** 한다
            # (자체점검이 그걸로 실패했다). 진입가보다 유리한 지점만 고르고, 없으면 안 넣는다.
            better = np.where(seg < e0 if s > 0 else seg > e0)[0]
            if len(better):
                rank = better[np.argsort(seg[better] if s > 0 else -seg[better])][:k - 1]
                for t in np.sort(rank):
                    px.append(float(seg[t])); tt.append(int(t))
        elif mode == "pyramid_oracle":
            # 이길 거래에서만 상한까지 채운다. 질 거래는 칸 0 만 들고 끝낸다.
            if ret > 0:
                for j in range(1, k):                       # 즉시 전량(가장 유리한 형태)
                    px.append(e0); tt.append(0)
        else:
            raise ValueError(mode)
        pnl = sum(size * (s * (exit_px / p - 1.0) - ((PEG_BP if j == 0 else MAKER_BP)
                                                     + EXIT_BP) / 1e4)
                  for j, p in enumerate(px))
        net.append(1e4 * pnl)
        expo.append(float(sum(size * (w - t) / w for t in tt)))
    net = np.array(net); expo = np.array(expo)
    return {"net_bp": float(net.mean()), "expo": float(expo.mean()),
            "per_expo": float(net.mean() / expo.mean()), "worst": float(net.min())}


def _self_check() -> None:
    w, k = 48, 5
    c = np.full(300, 100.0); hi = c.copy(); lo = c.copy()
    idx, sd = np.array([10]), np.array([1.0])
    # 평평하면 모든 팔의 수익이 같고 비용만 남는다. 단일은 peg+exit, 사다리는 섞인다.
    a = run(c, hi, lo, idx, sd, k=1, w=w, mode="single")
    assert abs(a["net_bp"] + PEG_BP + EXIT_BP) < 1e-6, a
    # 오라클 물타기는 **절대** 단일보다 나쁠 수 없다(같은 가격이면 수수료만 싸진다)
    b = run(c, hi, lo, idx, sd, k=k, w=w, mode="avgdown_oracle")
    assert b["net_bp"] > a["net_bp"] - 1e-9, (a, b)
    # 🔴**단조 상승**에서는 진입가가 최저가라 추가할 «더 싼 곳»이 없다 -> 오라클은 안 넣고
    # 단일과 같아진다(칸 0 만). 첫 판은 «오라클이 반드시 이긴다»고 단정했다가 실패했다.
    c2 = np.concatenate([np.full(11, 100.0), np.linspace(100, 110, 289)])
    d = run(c2, c2, c2, idx, sd, k=k, w=w, mode="avgdown_oracle")
    e = run(c2, c2, c2, idx, sd, k=1, w=w, mode="single")
    assert abs(d["expo"] - 1.0 / k) < 1e-9, f"더 싼 곳이 없는데 넣었다: {d}"
    # V 자(내려갔다 회복)에서는 진짜로 더 싼 지점이 있으므로 오라클이 단일을 이긴다
    c4 = np.concatenate([np.full(11, 100.0), np.linspace(100, 95, 24),
                         np.linspace(95, 105, 265)])
    g = run(c4, c4, c4, idx, sd, k=k, w=w, mode="avgdown_oracle")
    h = run(c4, c4, c4, idx, sd, k=1, w=w, mode="single")
    assert g["net_bp"] > h["net_bp"], (g, h)
    # 불타기 오라클: 지는 거래(하락 롱)에서는 칸 0 만 -> 노출이 1/k
    c3 = np.concatenate([np.full(11, 100.0), np.linspace(100, 90, 289)])
    f = run(c3, c3, c3, idx, sd, k=k, w=w, mode="pyramid_oracle")
    assert abs(f["expo"] - 1.0 / k) < 1e-9, f
    print("통과 — 비용 항등 · 오라클 물타기 지배 · 지는 거래는 칸0 만")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hold-bars", type=int, default=48)
    ap.add_argument("--acc", default="0.60,0.50")
    ap.add_argument("--every", type=int, default=24)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check(); return 0

    d = pd.read_csv(KL, usecols=["timestamp", "high", "low", "close"],
                    parse_dates=["timestamp"]).dropna().reset_index(drop=True)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    ts = d.timestamp.to_numpy()
    w = a.hold_bars
    print(f"5분봉 {len(d):,} · 보유 {w*5}분 · k={a.k} · {a.every*5}분마다 표집")
    print("⚠️오라클은 **상한**이다 -- 실현 가능한 전략이 아니라 «모델이 최대로 벌 수 있는 값»\n")
    rng = np.random.default_rng(SEED)
    for acc in [float(x) for x in a.acc.split(",")]:
        print(f"=== 정확도 {acc} ===")
        print(f"{'창':>18} {'팔':>18} {'의도명목당bp':>12} {'노출':>6} {'노출당':>9} {'최악':>9}")
        for wname, (w0, w1) in WINDOWS.items():
            lo_i = max(int(np.searchsorted(ts, np.datetime64(w0))), 300)
            hi_i = int(np.searchsorted(ts, np.datetime64(w1 + "T23:59:59"))) - w - 1
            if hi_i - lo_i < 1000:
                continue
            idx = np.arange(lo_i, hi_i, a.every)
            truth = np.where(c[idx + w] >= c[idx], 1.0, -1.0)
            sides = np.where(rng.random(len(idx)) < acc, truth, -truth)
            for lab, mode, kk in (("단일", "single", 1),
                                  ("균등 시간분할", "twap", a.k),
                                  ("물타기 **오라클**", "avgdown_oracle", a.k),
                                  ("불타기 **오라클**", "pyramid_oracle", a.k)):
                r = run(c, hi, lo, idx, sides, k=kk, w=w, mode=mode)
                print(f"{wname:>18} {lab:>18} {r['net_bp']:>12.2f} {r['expo']:>6.2f} "
                      f"{r['per_expo']:>9.2f} {r['worst']:>9.1f}")
        print()
    print("⚠️오라클이 단일을 못 넘으면 **어떤 모델도 못 넘는다** -- 그 축은 거기서 닫힌다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
