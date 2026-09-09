#!/usr/bin/env python3
"""증거신호 **합의(겹침)의 실측 잣대** -- 공통 스케일 통합용 (2026-09-10).

대시보드는 "겹칠수록 신뢰도가 실제로 높아짐"이라고 쓰고 있다. 09-09 스크린이 손익 기준으로는
반대(3종+ 동시발동이 가장 나쁘다)임을 보였는데, 그건 **귀무 없는 원시 gross** 였다.
여기서는 순환이동 귀무(발동 군집·개수 보존)로 **초과분**을 다시 잰다. 이 표가 통합 점수의
가중치가 된다 -- 17개 칩을 하나의 잣대("귀무 대비 초과 bp")로 환산하는 근거다.

측정
  A. 신호 단위 초과분 (H=12/48) -- 칩마다 "이게 켜지면 몇 bp 인가"
  B. 동시발동 개수별 초과분     -- 겹침이 실제로 값을 더하는가
  C. 측면별                     -- 바닥/천장 따로 (거울상이면 잔존 베타다)
⚠️전건을 센다. 커버리지 병기. 비용선(테이커 10bp · 메이커 7.8bp)과 함께 읽는다.
"""
from __future__ import annotations
import argparse, json, os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

OUT = ROOT / "tmp/eth_edge_scaling_20260910"
HS = (12, 48)
NSHIFT = 600
RNG = np.random.default_rng(20260910)
WARM = 900
COSTS = {"테이커": 10.0, "메이커": 7.8}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2024-06-01")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    kl = B._load_kl(B.ETH_KL); kl = kl[kl["timestamp"] >= pd.Timestamp(a.start)].reset_index(drop=True)
    btc = B._load_kl(B.BTC_KL); fund = B._load_funding()
    tmax = kl["timestamp"].max()
    sig = compute_signals(kl, btc_df=btc[btc["timestamp"] <= tmax],
                          funding_df=fund[fund["calc_time"] <= tmax])
    ts = pd.to_datetime(sig["timestamp"].to_numpy())
    op = sig["open"].to_numpy(float); cl = sig["close"].to_numpy(float); n = len(sig)
    lo_i, hi_i = WARM, n - max(HS) - 2
    span = hi_i - lo_i
    days = (ts[hi_i] - ts[lo_i]).total_seconds() / 86400
    print(f"평가 {ts[lo_i]:%Y-%m-%d} ~ {ts[hi_i]:%Y-%m-%d} ({days:.0f}일)\n")
    valid = np.zeros(n, bool); valid[lo_i:hi_i] = True

    def ret_bp(idx, H, long):
        r = (cl[idx + H] - op[idx + 1]) / op[idx + 1] * 1e4
        return r if long else -r

    def excess(idx, H, long):
        g = float(np.mean(ret_bp(idx, H, long)))
        nul = np.empty(NSHIFT)
        for b in range(NSHIFT):
            sh = RNG.integers(500, span - 500)
            nul[b] = np.mean(ret_bp(lo_i + ((idx - lo_i + sh) % span), H, long))
        return g, float(nul.mean()), g - float(nul.mean()), float((nul >= g).mean())

    fires = {s: {sd: (sig[f"{sd}_{s}"].fillna(False).to_numpy(bool) & valid)
                 for sd in ("bottom", "top")} for s in B.SIGNALS}
    rows = []
    print("=" * 104)
    print("A. 신호 단위 초과분 (귀무 = 순환이동, B=600) -- 이게 통합 점수의 칩별 가중치다")
    print("=" * 104)
    print(f"{'신호':<26}{'측면':<7}{'건/일':>7}" + "".join(f"{'H'+str(h)+' 초과':>11}{'p':>7}" for h in HS))
    for s in B.SIGNALS:
        for sd in ("bottom", "top"):
            idx = np.flatnonzero(fires[s][sd])
            if len(idx) < 100: continue
            cells = [excess(idx, H, sd == "bottom") for H in HS]
            rows.append(dict(kind="signal", key=s, side=sd, n=len(idx), per_day=len(idx)/days,
                             **{f"ex{H}": c[2] for H, c in zip(HS, cells)},
                             **{f"p{H}": c[3] for H, c in zip(HS, cells)}))
            print(f"{s:<26}{sd:<7}{len(idx)/days:>7.2f}" +
                  "".join(f"{c[2]:>11.2f}{c[3]:>7.3f}" for c in cells))

    print("\n" + "=" * 104)
    print("B. 동시발동 개수별 초과분 -- 화면 문구 '겹칠수록 신뢰도↑' 를 손익으로 직접 검정")
    print("=" * 104)
    print(f"{'측면':<7}{'동시발동':<9}{'건/일':>7}{'커버':>7}" +
          "".join(f"{'H'+str(h)+' 초과':>11}{'p':>7}" for h in HS))
    for sd in ("bottom", "top"):
        cntm = np.sum([fires[s][sd] for s in B.SIGNALS], axis=0)
        tot = int((cntm >= 1).sum())
        for k, lab in ((1, "1종"), (2, "2종"), (3, "3종+")):
            m = (cntm == k) if k < 3 else (cntm >= 3)
            idx = np.flatnonzero(m & valid)
            if len(idx) < 60: continue
            cells = [excess(idx, H, sd == "bottom") for H in HS]
            rows.append(dict(kind="confluence", key=lab, side=sd, n=len(idx), per_day=len(idx)/days,
                             **{f"ex{H}": c[2] for H, c in zip(HS, cells)},
                             **{f"p{H}": c[3] for H, c in zip(HS, cells)}))
            print(f"{sd:<7}{lab:<9}{len(idx)/days:>7.2f}{len(idx)/tot:>7.1%}" +
                  "".join(f"{c[2]:>11.2f}{c[3]:>7.3f}" for c in cells))

    d = pd.DataFrame(rows); d.to_csv(OUT / "confluence_null.csv", index=False)
    print("\n" + "=" * 104)
    print("비용선 대비 -- 초과분이 비용을 넘는 셀")
    print("=" * 104)
    for cname, c in COSTS.items():
        ok = d[(d.ex12 > c) | (d.ex48 > c)]
        print(f"  {cname} {c}bp: {len(ok)}/{len(d)}셀" +
              ("  " + ", ".join(f"{r.key}/{r.side}" for r in ok.itertuples()) if len(ok) else ""))
    print(f"\n최대 초과분 H12 {d.ex12.max():.2f}bp · H48 {d.ex48.max():.2f}bp")
    print(f"저장 {OUT/'confluence_null.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
