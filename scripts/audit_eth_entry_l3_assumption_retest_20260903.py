#!/usr/bin/env python3
"""L0 위에 세워진 **가정 전수 재시험** -- 정직한 L3 라벨에서 (2026-09-03).

사용자 요청: "L0 위에 올려진 가정들은 모두 재시험해줘."

L0(체결 봉의 체결-이전 고가를 크레딧)가 미래참조로 확정됐고, 오염 크기가 **깊이에 비례**함이
확인됐다(depth 2.0→3.5에서 22.0→31.3bp). 따라서 L0에서 고른 **모든 구조 결정**이 편향된
비교 위에 있다. 여기서 기계 수준 가정을 전부 다시 훑는다(모델 불필요, 따라서 값싸다):

  ① 깊이 × 대기        -- 별도 확인 완료(L3에서 전 셀 0 근처, HOLDOUT 전부 음수)
  ② 슬롯 수            -- {1,2,3,4,6,무제한}. 4는 L0에서 골랐다
  ③ 팔 구성            -- {신호방향만, 역방향만, 양팔}. "양쪽 다 걸기가 방향 오라클을 이긴다"는
                          핵심 결론이 L0 위에 있었다
  ④ 청산 파라미터      -- SL/ARM/Trail 격자. ⭐L0에서는 **무장이 사실상 공짜**였으므로
                          ARM 값이 무의미했다. L3에서는 진짜로 1 ATR을 벌어야 무장한다
  ⑤ 신호별 기여        -- 8종 각각의 단독 성과. 무작위 봉 대조군이 실패했으므로
                          "어떤 신호라도 기여하는가"를 직접 본다

⚠️여기서 나오는 최선 조합을 **채택 근거로 쓰면 안 된다.** VAL/OOS/HOLDOUT은 이미 여러 번
소진됐다. 이건 "L0에서 고른 것이 L3에서도 최선인가"를 묻는 진단이다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import KLINES_PATH  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402
from research_eth_entry_direction_oracle_ceiling_20260903 import NOTIONAL, COST  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
L3D = ROOT / "tmp/eth_entry_1m_resolved_20260903/labels_1m_all.csv"
M1P = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = ROOT / "tmp/eth_entry_l3_assumptions_20260903"
DEPTH, WAIT = 3.0, 6
W3 = ("VAL", "OOS", "HOLDOUT")


def log(m): print(f"[assume] {m}", flush=True)


def trail_g(side, e, a, hi, lo, cl, sl, arm, trl):
    """청산 파라미터를 인자로 받는 트레일링."""
    if side > 0:
        stop = e * (1 - sl * a); peak = e; armed = False
        for k in range(len(cl)):
            if lo[k] <= stop: return stop / e - 1.0
            if hi[k] > peak:
                peak = hi[k]
                if not armed and (peak - e) / e >= arm * a: armed = True
                if armed: stop = max(stop, peak * (1 - trl * a))
        return cl[-1] / e - 1.0
    stop = e * (1 + sl * a); peak = e; armed = False
    for k in range(len(cl)):
        if hi[k] >= stop: return 1.0 - stop / e
        if lo[k] < peak:
            peak = lo[k]
            if not armed and (e - peak) / e >= arm * a: armed = True
            if armed: stop = min(stop, peak * (1 + trl * a))
    return 1.0 - cl[-1] / e


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values(
        "timestamp").reset_index(drop=True)
    h5, l5, c5 = (kl[x].to_numpy(float) for x in ("high", "low", "close"))
    ts5 = pd.DatetimeIndex(kl["timestamp"]); n5 = len(kl)
    m1 = pd.read_csv(M1P, parse_dates=["timestamp"]).sort_values(
        "timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    m1["b5"] = m1["timestamp"].dt.floor("5min")
    m1h, m1l = m1["high"].to_numpy(float), m1["low"].to_numpy(float)
    f0 = m1.groupby("b5", sort=True).apply(lambda d: d.index[0])
    cn = m1.groupby("b5", sort=True).size()
    IDX0 = {k: int(v) for k, v in f0.items()}; IDXN = {k: int(v) for k, v in cn.items()}

    LAB = pd.read_csv(L3D, parse_dates=["timestamp"])
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    A = D.merge(LAB[["timestamp", "signal", "arm", "depth", "btf", "y_L3"]],
                on=["timestamp", "signal", "arm", "depth", "btf"], how="left")
    A = A[np.isfinite(A.y_L3)].reset_index(drop=True)
    S = A[(A.depth == DEPTH) & (A.btf <= WAIT)].reset_index(drop=True)
    log(f"모집단 {len(A):,} · 기준 셀(depth{DEPTH}/wait{WAIT}) {len(S):,}")

    def run(d, ns, col="y_L3"):
        if not len(d): return np.nan, 0
        t = slotN(d.assign(y=d[col]), ns) if ns else d[col].to_numpy()
        return (float(np.mean(t) * 1e4) if len(t) else 0.0), int(len(t))

    # ---- ② 슬롯 수 ----
    print(f"\n=== ② 슬롯 수 (양팔·무필터·L3) ===")
    print(f"{'슬롯':>6s}" + "".join(f"{w:>16s}" for w in W3))
    for ns in (1, 2, 3, 4, 6, 0):
        r = [run(S[S.split == w], ns) for w in W3]
        tag = "무제한" if ns == 0 else str(ns)
        star = " ←배포" if ns == 4 else ""
        print(f"{tag:>6s}" + "".join(f"{v:+9.2f}(n{n:5d})" for v, n in r) + star)

    # ---- ③ 팔 구성 ----
    print(f"\n=== ③ 팔 구성 (4슬롯·무필터·L3) ===")
    print(f"{'구성':>14s}" + "".join(f"{w:>16s}" for w in W3))
    for tag, m in (("신호방향만(arm1)", S.arm == 1), ("역방향만(arm0)", S.arm == 0),
                   ("양팔", np.ones(len(S), bool))):
        r = [run(S[m & (S.split == w)], 4) for w in W3]
        star = " ←배포" if tag == "양팔" else ""
        print(f"{tag:>14s}" + "".join(f"{v:+9.2f}(n{n:4d})" for v, n in r) + star)

    # ---- ⑤ 신호별 ----
    print(f"\n=== ⑤ 신호별 단독 (4슬롯·양팔·무필터·L3) ===")
    print(f"{'신호':>26s}" + "".join(f"{w:>15s}" for w in W3))
    for sg in sorted(S.signal.unique()):
        r = [run(S[(S.signal == sg) & (S.split == w)], 4) for w in W3]
        print(f"{sg:>26s}" + "".join(f"{v:+9.2f}(n{n:3d})" for v, n in r))

    # ---- ④ 청산 파라미터 (L3 재시뮬) ----
    log("\n④ 청산 파라미터 격자 (L3 재시뮬)...")
    i0 = S.i.to_numpy().astype(int); e0 = S.lim.to_numpy(float)
    a0 = S.atr_pct.to_numpy(float); sd0 = S.sd.to_numpy(int)
    fi0 = S.fi.to_numpy().astype(int); hz0 = (S.ei - S.fi).to_numpy().astype(int)
    # 체결 봉의 사후(고가,저가)를 한 번만 계산해 재사용
    post = np.full((len(S), 2), np.nan)
    for j in range(len(S)):
        bt = ts5[fi0[j]]
        if bt not in IDX0: continue
        s0, nn = IDX0[bt], IDXN[bt]
        sh, sl_ = m1h[s0:s0 + nn], m1l[s0:s0 + nn]
        hit = np.flatnonzero(sl_ <= e0[j]) if sd0[j] > 0 else np.flatnonzero(sh >= e0[j])
        if not len(hit): continue
        k0 = int(hit[0]); ph, pl = sh[k0 + 1:], sl_[k0 + 1:]
        post[j] = [float(ph.max()) if len(ph) else e0[j],
                   float(pl.min()) if len(pl) else e0[j]]
    ok = np.isfinite(post[:, 0])
    log(f"  사후 폭 계산 완료 ({int(ok.sum()):,}/{len(S):,})")

    GRID = [(3.0, 1.0, 0.1), (3.0, 0.5, 0.1), (3.0, 1.5, 0.1), (3.0, 2.0, 0.1),
            (3.0, 1.0, 0.3), (3.0, 1.0, 0.5), (2.0, 1.0, 0.1), (4.0, 1.0, 0.1),
            (99.0, 99.0, 99.0)]      # 마지막 = 시간청산만(스톱/트레일 무력화)
    print(f"\n=== ④ 청산 파라미터 (4슬롯·양팔·무필터·L3) ===")
    print(f"{'SL':>6s}{'ARM':>6s}{'Trail':>7s}" + "".join(f"{w:>15s}" for w in W3))
    for sl, arm, trl in GRID:
        yy = np.full(len(S), np.nan)
        for j in np.flatnonzero(ok):
            f_, hz_ = int(fi0[j]), int(hz0[j])
            if hz_ <= 0 or f_ + hz_ > n5: continue
            H = np.concatenate([[post[j, 0]], h5[f_ + 1:f_ + hz_]])
            L = np.concatenate([[post[j, 1]], l5[f_ + 1:f_ + hz_]])
            C = np.concatenate([[c5[f_]], c5[f_ + 1:f_ + hz_]])
            yy[j] = trail_g(sd0[j], e0[j], a0[j], H, L, C, sl, arm, trl) * NOTIONAL - COST * NOTIONAL
        T = S.assign(yg=yy)
        T = T[np.isfinite(T.yg)]
        r = [run(T[T.split == w], 4, "yg") for w in W3]
        tag = "시간청산만" if sl > 50 else ""
        star = " ←배포" if (sl, arm, trl) == (3.0, 1.0, 0.1) else ""
        print(f"{sl:6.1f}{arm:6.1f}{trl:7.1f}" + "".join(f"{v:+9.2f}(n{n:4d})" for v, n in r)
              + star + tag)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
