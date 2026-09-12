"""돌파 감지기 검증 3종 — 임계값 견고성 · 방어 규칙 백테스트 · 호가 추가 기여.

① 견고성: 압축/확장/소급 정의를 격자로 흔들어 lift 가 유지되는지.
② 방어: "압축 중 체결속도 z 급등이면 청산"을 실계좌 14건 + 배포 원장 41건에 적용.
   손익은 **가격변동 %** 로 낸다 — 실계좌는 물타기로 평단이 움직여 USDT 환산이 부정확하고,
   물타기를 아예 안 하게 되는 이득은 여기 안 잡히므로 **보수적**이다.
③ 호가: bookDepth 패널(2024-04~2026-03)이 겹치는 구간에서 체결속도보다 앞서는지.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
BOOK = ROOT / "data/research/eth_trend_signals_v1_screen_20260904/bookdepth_wide.parquet"
B_NULL, SEED = 300, 615372041


def _z(s, w=288):
    return ((s - s.rolling(w).mean()) / s.rolling(w).std()).to_numpy()


def _events(volexp, comp, exp, back, n):
    cross = (volexp >= exp) & np.r_[False, volexp[:-1] < exp]
    was = pd.Series(volexp < comp).rolling(back).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    return ev[(ev > 300) & (ev < n - 20)]


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c = d.c.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    nz = _z(d.n)
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(300, n - 300, size=B_NULL)

    print("=== ① 임계값 견고성 (z>=1.0, 압축 봉 한정) ===")
    print(f"{'압축<':>6s} {'확장>=':>6s} {'소급':>5s} {'사건':>5s} {'압축봉':>8s} {'기저':>7s} "
          f"{'적중':>7s} {'lift':>6s} {'귀무q95':>8s} {'p':>6s}")
    grid = []
    for comp in (0.5, 0.6, 0.7, 0.8):
        for exp in (1.5, 1.8, 2.2, 2.6):
            for back in (36, 72, 144):
                ev = _events(volexp, comp, exp, back, n)
                if len(ev) < 60:
                    continue
                fut = pd.Series(volexp).rolling(12).max().shift(-12).to_numpy()
                uni = (volexp < comp) & np.isfinite(nz) & np.isfinite(fut)
                base = float(np.mean(fut[uni] >= exp))
                m = uni & (nz >= 1.0)
                if m.sum() < 50:
                    continue
                hit = float(np.mean(fut[m] >= exp))
                null = []
                for s in shifts[:150]:
                    k = uni & (np.roll(nz, int(s)) >= 1.0)
                    if k.sum() >= 30:
                        null.append(float(np.mean(fut[k] >= exp)))
                null = np.asarray(null)
                p = float((null >= hit).mean()) if len(null) else np.nan
                grid.append({"comp": comp, "exp": exp, "back": back, "ev": len(ev),
                             "base": base, "hit": hit, "lift": hit / max(base, 1e-9), "p": p})
                print(f"{comp:6.1f} {exp:6.1f} {back:5d} {len(ev):5d} {int(uni.sum()):8,d} "
                      f"{base*100:6.2f}% {hit*100:6.2f}% {hit/max(base,1e-9):5.2f}x "
                      f"{np.quantile(null,.95)*100:7.2f}% {p:6.3f}", flush=True)
    g = pd.DataFrame(grid)
    g.to_csv(D / "breakout_robustness_grid.csv", index=False)
    print(f"\n  격자 {len(g)}셀 · lift 중앙 {g.lift.median():.2f}x · lift>2 인 셀 "
          f"{int((g.lift>2).sum())}/{len(g)} · p<=0.01 {int((g.p<=0.01).sum())}/{len(g)}")

    print("\n=== ② 방어 규칙 백테스트 — '압축 중 z 급등이면 청산' ===")
    ts = d.timestamp.to_numpy()
    trips = pd.DataFrame(json.load(open(D / "real_account_trips.json")))
    trips = trips.dropna(subset=["exit_time"]).copy()
    trips["e0"] = pd.to_datetime(trips.entry_time, unit="ms")
    trips["e1"] = pd.to_datetime(trips.exit_time, unit="ms")
    trips["sgn"] = np.where(trips.side == "LONG", 1, -1)
    for thr in (1.0, 1.5, 2.0, 3.0):
        rows = []
        for r in trips.itertuples():
            m = (ts >= np.datetime64(r.e0)) & (ts <= np.datetime64(r.e1))
            k = np.flatnonzero(m & (nz >= thr) & np.isfinite(nz))
            act = (r.exit_price - r.entry_price) / r.entry_price * r.sgn * 100
            if len(k):
                dfd = (c[k[0]] - r.entry_price) / r.entry_price * r.sgn * 100
                rows.append({"act": act, "def": dfd, "fired": 1})
            else:
                rows.append({"act": act, "def": act, "fired": 0})
        q = pd.DataFrame(rows)
        print(f"  z>={thr:.1f}  발동 {int(q.fired.sum())}/{len(q)}건  "
              f"실제 합계 {q.act.sum():+7.2f}%  방어 합계 {q['def'].sum():+7.2f}%  "
              f"개선 {q['def'].sum()-q.act.sum():+7.2f}%p  "
              f"최악 {q.act.min():+6.2f}% → {q['def'].min():+6.2f}%", flush=True)

    lgp = D / "replay_ledger_A.csv"
    if lgp.exists():
        lg = pd.read_csv(lgp)
        lg["e0"] = pd.to_datetime(lg.entry_timestamp)
        lg["e1"] = pd.to_datetime(lg.exit_timestamp)
        px = pd.DataFrame({"ts": ts, "c": c})
        print("\n  [배포 원장 41건]")
        for thr in (1.0, 1.5, 2.0, 3.0):
            rows = []
            for r in lg.itertuples():
                m = (ts >= np.datetime64(r.e0)) & (ts <= np.datetime64(r.e1))
                k = np.flatnonzero(m & (nz >= thr) & np.isfinite(nz))
                e = np.flatnonzero(ts >= np.datetime64(r.e0))
                if not len(e):
                    continue
                ep = c[e[0]]
                act = r.trade_return * 100
                if len(k):
                    dfd = (c[k[0]] - ep) / ep * r.side * 100 * float(r.notional)
                    rows.append({"act": act, "def": dfd, "fired": 1})
                else:
                    rows.append({"act": act, "def": act, "fired": 0})
            q = pd.DataFrame(rows)
            print(f"  z>={thr:.1f}  발동 {int(q.fired.sum())}/{len(q)}건  "
                  f"실제 합계 {q.act.sum():+8.2f}%  방어 합계 {q['def'].sum():+8.2f}%  "
                  f"개선 {q['def'].sum()-q.act.sum():+8.2f}%p  "
                  f"최악 {q.act.min():+6.2f}% → {q['def'].min():+6.2f}%", flush=True)

    print("\n=== ③ 호가 추가 기여 (bookDepth 2024-04~2026-03 겹침 구간) ===")
    if not BOOK.exists():
        print("  패널 없음 — 건너뜀")
        return 0
    bk = pd.read_parquet(BOOK)
    bk["ts"] = pd.to_datetime(bk["ts"])
    bk = bk[(bk.ts >= d.timestamp.min()) & (bk.ts <= d.timestamp.max())]
    if bk.empty:
        print("  겹침 구간 없음")
        return 0
    up = bk[[f"up{i}" for i in range(6)]].sum(axis=1)
    dn = bk[[f"dn{i}" for i in range(6)]].sum(axis=1)
    bk = bk.assign(tot=up + dn, imb=(up - dn) / (up + dn).replace(0, np.nan))
    b5 = bk.set_index("ts")[["tot", "imb"]].resample("5min").last().reset_index()
    mg = d.merge(b5, left_on="timestamp", right_on="ts", how="left")
    ok = mg.tot.notna()
    print(f"  겹침 {int(ok.sum()):,}봉 ({mg.timestamp[ok].min()} ~ {mg.timestamp[ok].max()})")
    sub = np.flatnonzero(ok.to_numpy())
    cands = {"체결속도 n": nz, "호가 총량": _z(mg.tot), "호가 |불균형|": _z(mg.imb.abs())}
    ev = _events(volexp, 0.7, 1.8, 72, n)
    ev = ev[np.isin(ev, sub)]
    print(f"  겹침 구간 사건 {len(ev)}건")
    if len(ev) >= 30:
        print(f"  {'지표':14s}" + "".join(f"{f't{l:+d}':>9s}" for l in (-12, -6, -3, -1, 0)))
        for nm, z in cands.items():
            line = f"  {nm:14s}"
            for lag in (-12, -6, -3, -1, 0):
                k = np.clip(ev + lag, 0, n - 1)
                obs = float(np.nanmean(z[k]))
                null = np.array([np.nanmean(z[np.clip((ev + lag + s) % n, 0, n - 1)]) for s in shifts[:150]])
                p = float((null >= obs).mean())
                line += f"{obs:+8.2f}{'*' if p <= 0.01 else ' '}"
            print(line, flush=True)
    print("\n단일 자산 · 2026년 · 실계좌 표본 13건(청산분) — 방향 판단용이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
