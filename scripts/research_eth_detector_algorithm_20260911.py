"""탐지·예측 알고리즘 확정 — 지정 피쳐 3+3의 최적 지평/임계 탐색 후 조합.

예측 후보: 체결속도 n · 체결속도 3봉지속 · 거래대금 qv
탐지 후보: 거래대금 qv · 체결속도 n · volexp 자체

예측 지표 = lift(앞 H봉 실현변동성 상위 5% 적중), 타깃은 앞만 본다(순환 배제).
탐지 지표 = 진행률(감지 시점 이동폭 / 전체 이동폭) · 포착률 · 오탐/h.
조합은 OR(놓침↓ 오탐↑)/AND(오탐↓ 놓침↑) 둘 다 잰다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXPAND, BACK, FULL, TOPQ = 0.7, 1.8, 72, 144, 0.95
HS = [(2, "10분"), (3, "15분"), (4, "20분"), (6, "30분"), (9, "45분"), (12, "1시간"), (24, "2시간")]
WS = [288, 864, 2016]
B_NULL, SEED = 120, 615372041


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c = d.c.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    rv288 = pd.Series(lr).rolling(288).std().to_numpy()
    volexp = pd.Series(lr).rolling(12).std().to_numpy() / rv288
    comp = (volexp < COMPRESS) & np.isfinite(volexp)
    zf = lambda s, W: ((s - s.rolling(W).mean()) / s.rolling(W).std()).to_numpy()
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(900, n - 900, size=B_NULL)

    # ── 예측
    print("=== 예측 (앞 H봉 실현변동성 상위 5%, 발동 q99) ===")
    print(f"{'피쳐':16s} {'창':>5s} {'지평':>6s} {'적중률':>7s} {'lift':>6s} {'p':>6s}")
    P = {}
    for nm in ("체결속도 n", "체결속도 3봉지속", "거래대금 qv"):
        best = None
        for W in WS:
            base = zf(d.n, W) if nm.startswith("체결속도") else zf(d.qv, W)
            x = pd.Series(base).rolling(3).min().to_numpy() if "3봉" in nm else base
            for H, hn in HS:
                fwd = pd.Series(lr).rolling(H).std().shift(-H).to_numpy()
                v = comp & np.isfinite(fwd) & np.isfinite(x)
                tgt = fwd >= np.nanquantile(fwd[v], TOPQ)
                thr = float(np.nanquantile(x[v], 0.99))
                m = v & (x >= thr)
                if m.sum() < 50:
                    continue
                hit = float(np.mean(tgt[m]))
                b = float(np.mean(tgt[v]))
                null = [float(np.mean(tgt[v & (np.roll(x, int(s)) >= thr)]))
                        for s in shifts if (v & (np.roll(x, int(s)) >= thr)).sum() >= 30]
                p = float((np.asarray(null) >= hit).mean()) if null else np.nan
                r = dict(feat=nm, W=W, H=H, hn=hn, hit=hit, lift=hit / max(b, 1e-9), p=p, thr=thr)
                if best is None or (r["lift"] > best["lift"] and r["p"] <= 0.05):
                    best = r
        P[nm] = best
        print(f"{nm:16s} {best['W']:5d} {best['hn']:>6s} {best['hit']*100:6.2f}% "
              f"{best['lift']:5.2f}x {best['p']:6.3f}", flush=True)

    # ── 탐지
    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(volexp < COMPRESS).rolling(BACK).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    ev = ev[(ev > 900) & (ev < n - FULL - 2)]
    starts, fulls = [], []
    for e in ev:
        w = np.flatnonzero(comp[max(e - BACK, 0):e])
        s = (max(e - BACK, 0) + w[-1]) if len(w) else e - 12
        starts.append(s)
        seg = c[s:min(e + FULL, n)]
        fulls.append(float(np.max(np.abs(seg - c[s])) / c[s] * 100))
    starts, fulls = np.asarray(starts), np.asarray(fulls)
    keep = fulls > 0.2
    ev, starts, fulls = ev[keep], starts[keep], fulls[keep]
    hours = float(comp.sum()) * 5 / 60

    def score(fired):
        dly, prog, hit = [], [], 0
        for e, s, fu in zip(ev, starts, fulls):
            k = np.flatnonzero(fired[s:min(e + 24, n)])
            if len(k):
                t = s + k[0]
                hit += 1
                dly.append(t - s)
                prog.append(abs(c[t] - c[s]) / c[s] * 100 / fu * 100)
        if hit < 20:
            return None
        return dict(recall=hit / len(ev), delay=float(np.median(dly)),
                    prog=float(np.median(prog)), fa=(int((fired & comp).sum()) - hit) / hours)

    print(f"\n=== 탐지 ({len(ev)}건 · 오탐 1.0/h 근처에서 진행률 최소) ===")
    print(f"{'피쳐':16s} {'창':>5s} {'분위':>6s} {'포착률':>7s} {'지연':>5s} {'진행률':>7s} {'오탐/h':>7s}")
    T, FIRED = {}, {}
    for nm in ("거래대금 qv", "체결속도 n", "volexp 자체"):
        best = None
        for W in ([0] if nm == "volexp 자체" else WS):
            x = volexp if nm == "volexp 자체" else zf(d.n if nm.startswith("체결") else d.qv, W)
            for q in (0.80, 0.85, 0.90, 0.95):
                thr = float(np.nanquantile(x[comp & np.isfinite(x)], q))
                fired = np.isfinite(x) & (x >= thr)
                sc = score(fired)
                if sc is None or sc["fa"] > 1.3:
                    continue
                r = dict(feat=nm, W=W, q=q, thr=thr, **sc)
                if best is None or r["prog"] < best["prog"]:
                    best = r; bf = fired
        T[nm], FIRED[nm] = best, bf
        print(f"{nm:16s} {best['W']:5d} {best['q']:6.2f} {best['recall']*100:6.1f}% "
              f"{best['delay']:4.0f}봉 {best['prog']:6.2f}% {best['fa']:7.3f}", flush=True)

    print(f"\n=== 탐지 조합 ===")
    ks = list(FIRED)
    for nm2, f in (("OR (셋 중 하나)", FIRED[ks[0]] | FIRED[ks[1]] | FIRED[ks[2]]),
                   ("AND (셋 모두)", FIRED[ks[0]] & FIRED[ks[1]] & FIRED[ks[2]]),
                   ("2/3 다수결", (FIRED[ks[0]].astype(int) + FIRED[ks[1]].astype(int)
                                 + FIRED[ks[2]].astype(int)) >= 2)):
        sc = score(f)
        if sc:
            print(f"{nm2:16s} {'':5s} {'':6s} {sc['recall']*100:6.1f}% {sc['delay']:4.0f}봉 "
                  f"{sc['prog']:6.2f}% {sc['fa']:7.3f}")
    pd.DataFrame(list(P.values()) + list(T.values())).to_csv(D / "detector_algorithm.csv", index=False)

    print(f"\n{'='*72}\n[알고리즘 사양]")
    bp = max(P.values(), key=lambda r: r["lift"])
    bt = min(T.values(), key=lambda r: r["prog"])
    print(f"경보(예측)  {bp['feat']} · z{bp['W']} >= {bp['thr']:.3f}(q99)")
    print(f"            앞 {bp['hn']} 실현변동성 상위 5% 적중 {bp['hit']*100:.2f}% (기저 5%, "
          f"lift {bp['lift']:.2f}x, p={bp['p']:.3f})")
    print(f"청산(탐지)  {bt['feat']} · " + (f"z{bt['W']} " if bt['W'] else "")
          + f">= {bt['thr']:.3f}(q{bt['q']:.2f})")
    print(f"            포착률 {bt['recall']*100:.1f}% · 지연 {bt['delay']:.0f}봉({bt['delay']*5:.0f}분) "
          f"· 진행률 {bt['prog']:.2f}% · 오탐 {bt['fa']:.2f}/h")
    print("두 규칙 모두 압축 구간(volexp<0.7)에서만 감시한다. 입력은 공개 kline 의 n·quote_volume 뿐이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
