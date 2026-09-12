"""호가 재검정 — 앞선 검정의 결함 3개를 고친다.

앞선 검정: (1) 30초 패널을 5분 `.last()` 로 리샘플해 10개 중 9개를 버렸고,
(2) 누적 밴드 up0..up5 를 그냥 합산해 중복 계산했으며, (3) 레벨만 보고 **변화**를 안 봤다.
호가의 교과서적 돌파 선행 신호는 유동성 **인출**(주문 취소로 얇아짐)인데 그걸 안 쟀다.

여기서는 30초 해상도를 유지하고, 밴드를 분리하고, 얇아짐/변화를 잰다.
사건은 5분봉 기준이므로 30초 격자에 매핑해 -60분~+10분 프로파일을 본다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
BOOK = ROOT / "data/research/eth_trend_signals_v1_screen_20260904/bookdepth_wide.parquet"
COMPRESS, EXPAND, BACK = 0.7, 1.8, 72
LAGS_S = [-3600, -1800, -900, -600, -300, -180, -60, 0, 120, 300]   # 초 단위
B_NULL, SEED = 300, 615372041


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    c = d.c.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    n = len(d)
    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(volexp < COMPRESS).rolling(BACK).max().shift(12).to_numpy() == 1
    ev5 = np.flatnonzero(cross & was)
    ev5 = ev5[(ev5 > 300) & (ev5 < n - 20)]
    ev_ts = d.timestamp.to_numpy()[ev5]

    bk = pd.read_parquet(BOOK)
    bk["ts"] = pd.to_datetime(bk["ts"])
    bk = bk[(bk.ts >= d.timestamp.min()) & (bk.ts <= d.timestamp.max())].sort_values("ts")
    bk = bk.set_index("ts").resample("30s").last()          # 균등 30초 격자
    print(f"[호가] {len(bk):,}행 30초 격자  {bk.index.min()} ~ {bk.index.max()}")
    ev_ts = ev_ts[(ev_ts >= np.datetime64(bk.index.min() + pd.Timedelta(hours=2)))
                  & (ev_ts <= np.datetime64(bk.index.max() - pd.Timedelta(minutes=20)))]
    print(f"[사건] 겹침 구간 {len(ev_ts)}건", flush=True)

    W = 120                                                  # 1시간 = 120 × 30초
    f = {}
    for b in (0, 1, 2, 5):
        up, dn = pd.to_numeric(bk[f"up{b}"]), pd.to_numeric(bk[f"dn{b}"])
        tot = up + dn
        f[f"밴드{b} 총량 z"] = ((tot - tot.rolling(W).mean()) / tot.rolling(W).std())
        f[f"밴드{b} 얇아짐"] = np.log(tot) - np.log(tot.rolling(W).median())
        f[f"밴드{b} Δ1분"] = np.log(tot) - np.log(tot.shift(2))
        imb = (up - dn) / tot.replace(0, np.nan)
        f[f"밴드{b} |불균형|"] = imb.abs()
        f[f"밴드{b} |불균형|Δ"] = imb.abs() - imb.abs().rolling(W).median()
    # 밴드0/밴드5 비율 = 근접 유동성의 상대적 인출
    t0 = pd.to_numeric(bk["up0"]) + pd.to_numeric(bk["dn0"])
    t5 = pd.to_numeric(bk["up5"]) + pd.to_numeric(bk["dn5"])
    r = t0 / t5.replace(0, np.nan)
    f["근접/광역 비율 얇아짐"] = np.log(r) - np.log(r.rolling(W).median())

    idx = bk.index.to_numpy()
    pos = np.searchsorted(idx, ev_ts)
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(500, len(idx) - 500, size=B_NULL)
    print(f"\n{'지표':22s}" + "".join(f"{l//60:+5d}m" for l in LAGS_S) + "   판정")
    rows = []
    for nm, s in f.items():
        v = s.to_numpy(dtype=float)
        line, sig = f"{nm:22s}", []
        for lag in LAGS_S:
            k = np.clip(pos + lag // 30, 0, len(v) - 1)
            obs = float(np.nanmean(v[k]))
            null = np.array([np.nanmean(v[np.clip((pos + lag // 30 + sh) % len(v), 0, len(v) - 1)])
                             for sh in shifts])
            null = null[np.isfinite(null)]
            p2 = float(min((null >= obs).mean(), (null <= obs).mean()) * 2)
            line += f"{obs:+5.2f}{'*' if p2 <= 0.01 else ('.' if p2 <= 0.05 else ' ')}"
            rows.append({"지표": nm, "lag_s": lag, "obs": obs, "p": p2})
            if lag < 0 and p2 <= 0.01:
                sig.append(lag)
        print(line + ("   선행 %+dm" % (min(sig) // 60) if sig else "   선행 없음"), flush=True)
    pd.DataFrame(rows).to_csv(D / "breakout_orderbook_redo.csv", index=False)
    print(f"\n(* p<=0.01 양측, . p<=0.05 · 순환이동 귀무 B={B_NULL} · 30초 해상도)")
    print("앞선 검정은 5분 .last() 리샘플 + 누적밴드 합산 + 레벨만 봤다 — 그게 신호를 지웠는지 여기서 갈린다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
