#!/usr/bin/env python3
"""Zeus 라벨·게이트 축 (2026-09-18). 결론: **전부 기각, v4 동결 유지.**

기록: docs/experiments/zeus_label_and_gate_axis_20260918.md

서브명령
  build  <variant>  라벨 생성 -> tmp/.../zeus_double_barrier_labels_20260917/<name>.parquet
                    (학습은 `research_zeus_routing_and_side_experts_20260917.py --dirlabel=<name>`)
  sweep             지그재그 반전폭 스윕 -- «라벨을 그대로 매매하면 하루 몇 건인가»
  grid              팔별 배리어 격자 (각 라벨에 «자기 최적 배리어» 를 준다)
  pick              후보 칸을 **개별 시드** CI 하한·시드폭으로 고른다
  ab                v4 대비 A/B + 극점 거리별 진단(막차·관성)
  gate              인과 게이트 스윕 (학습 0, 채점만)

🔴판정 규칙(이 세션에서 두 번 걸렸다): 6시드 **앙상블 평균만으로 팔을 비교하지 말 것.**
시드가 흩어진 팔일수록 앙상블 이득이 커서 헤드라인이 «분산 감소»를 신호로 보이게 한다.
`pick` 이 개별 시드로 다시 재는 이유다.
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_zeus_fold_replay_20260918 as R           # noqa: E402
import build_wave3_action_labels_20260531 as W           # noqa: E402

LBLDIR = R.BASE / "zeus_double_barrier_labels_20260917"
GENK = dict(min_wave_bars=8, atr_window=14, atr_multiplier=0.0, mae_penalty=1.25,
            softmax_temperature=1.75, min_risk_floor=0.0010)
FO = ["F1", "F2", "F3", "CAND", "F4", "F5"]
DAYS = {f: (pd.Timestamp(R.FOLDS[f][1]) - pd.Timestamp(R.FOLDS[f][0])).days + 1 for f in R.FOLDS}
# 라벨 변형. (반전폭, 버퍼, soft argmax 사용 여부)
VARIANTS = {"zigzag_rev016": (0.016, 2, False),      # 파동을 키운다        -> 기각
            "zigzag_softargmax": (0.010, 2, True),   # 끝자락·작은파동 CASH -> 기각
            "zigzag_b0soft": (0.010, 0, True)}       # + 극점 봉 복원       -> 기각


def raw_frame():
    df = pd.read_parquet(R.BASE / "features_with_regime_2022_2026_realfunding.parquet",
                         columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df.timestamp).dt.tz_localize(None)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def cmd_build(name):
    pct, buf, soft = VARIANTS[name]
    df = raw_frame()
    L = W.build_zigzag_action_labels(df, min_reversal_pct=pct, transition_buffer=buf, **GENK)
    if soft:                      # soft argmax: «경제성이 나쁜 봉» 을 CASH 로 내린다(방향은 안 뒤집힌다)
        a = L[["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]].to_numpy().argmax(1)
    else:
        a = L.zigzag_action.to_numpy()
    out = pd.DataFrame({"timestamp": df.timestamp, "tb_action": a.astype("int8")})
    p = LBLDIR / f"{name}.parquet"; out.to_parquet(p, index=False)
    sh = np.bincount(a, minlength=3) / len(a)
    mfe = L.zigzag_path_mfe.to_numpy(); m = a != 0
    print(f"{name}: CASH/LONG/SHORT {sh.round(4)} · 방향봉 {100*m.mean():.1f}% · "
          f"그 봉들 TP{R.TP*100:g}% 도달 {100*(mfe[m] >= R.TP).mean():.1f}%")
    print("저장", p)


def waves(te, pct, minbars=8):
    """피벗 -> (진입봉, 방향, 파동수익). 생성기 함수를 그대로 쓴다(오라클)."""
    pv = W._filter_alternating(W._zigzag_pivots(te, min_reversal_pct=pct, atr_window=14,
                                                atr_multiplier=0.0))
    i = np.array([p[0] for p in pv]); px = np.array([p[1] for p in pv])
    k = np.array([p[2] for p in pv])
    m = np.diff(i) >= minbars
    return i[:-1][m], np.where(k[:-1] == "L", 1.0, -1.0)[m]


def cmd_sweep():
    te = R.panel(); days = (te.timestamp.max() - te.timestamp.min()).days
    print(f"창 {days}일 · TP{R.TP*100:g}%/SL{R.SL*100:g}% · 1슬롯 · 라벨을 그대로 매매")
    print(f"{'반전%':>7}{'파동/일':>9}{'체결/일':>9}{'건당bp':>9}{'TP율':>7}{'보유중앙':>9}")
    for pct in (0.004, 0.006, 0.008, 0.010, 0.013, 0.016, 0.020):
        ent, sd = waves(te, pct)
        side = np.zeros(len(te)); side[ent] = sd
        tr, _ = R.sequential(te, side)
        print(f"{pct*100:>7.1f}{len(ent)/days:>9.2f}{len(tr)/days:>9.2f}{tr.bp.mean():>+9.2f}"
              f"{(tr.why=='TP').mean()*100:>6.1f}%{np.median(tr.hold)*5/60:>8.1f}h")


def _cands(df, vers):
    out = {}
    for ver, sc in vers:
        for f in FO:
            out[(ver, sc, f)] = R.candidates(df, f, ver, score=sc)
    return out


def cmd_grid(vers):
    df = R.panel(); C = _cands(df, vers)
    for ver, sc in vers:
        print(f"\n■ {ver}({sc})  건당bp / 체결일 / 부호폴드")
        print("  TP\\SL " + "".join(f"{s*100:>21.1f}%" for s in (0.005, 0.007, 0.010, 0.013)))
        for tp in (0.010, 0.015, 0.020, 0.025, 0.030):
            row = f"{tp*100:>6.1f}%"
            for sl in (0.005, 0.007, 0.010, 0.013):
                B, n, d, pos = [], 0, 0, 0
                for f in FO:
                    te, side = C[(ver, sc, f)]
                    tr, _ = R.sequential(te, side, tp, sl)
                    B.append(tr.bp.to_numpy()); n += len(tr); d += DAYS[f]; pos += tr.bp.mean() > 0
                b = np.concatenate(B)
                row += f"{b.mean():>+9.2f}/{n/d:>5.2f}/{pos}/6"
            print(row)


def cmd_pick(cells):
    """개별 시드로 CI 하한·시드폭을 잰다. 앙상블 평균이 감추는 분산을 드러낸다."""
    df = R.panel(); CA = {}
    for ver, sc, _tp, _sl in cells:
        cache, _ = R.SPEC[ver]; z = np.load(cache, allow_pickle=True)
        for f in FO:
            v0, v1 = R.FOLDS[f]
            te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
            hlc = tuple(pd.to_numeric(te[c]).to_numpy(float) for c in ("high", "low", "close"))
            for si, s in enumerate(R.SEEDS):
                c = dict(z[f"{f}|N1s{s}"].item())
                da, qf = R.gate_score(c["D"], c["Q"], sc)
                m = da != 0; t = R.thr_seq(qf[m])
                ok = np.zeros(len(da), bool); ok[np.where(m)[0]] = np.isfinite(t) & (qf[m] >= t)
                CA[(ver, sc, f, si)] = (np.where((da == 1) & ok, 1.0, np.where((da == 2) & ok, -1.0, 0.0)),
                                        hlc, te.timestamp.to_numpy())
    print(f"{'구성':>24}{'체결/일':>9}{'건당bp':>9}{'CI95(날짜블록)':>22}{'독립일':>7}{'시드폭':>8}{'부호':>7}")
    for ver, sc, tp, sl in cells:
        ms, PB, PD = [], [], []
        for si in range(len(R.SEEDS)):
            bs, ds, fm = [], [], []
            for f in FO:
                side, (hi, lo, cl), ts = CA[(ver, sc, f, si)]
                te = pd.DataFrame({"timestamp": ts, "high": hi, "low": lo, "close": cl})
                tr, _ = R.sequential(te, side, tp, sl)
                b = tr.bp.to_numpy()
                bs.append(b); ds.append(tr.ts.to_numpy().astype("datetime64[D]")); fm.append(b.mean())
            B = np.concatenate(bs)
            ms.append((B.mean(), len(B), int(np.sum(np.array(fm) > 0))))
            PB.append(B); PD.append(np.concatenate(ds))
        B = np.concatenate(PB); D = np.concatenate(PD)
        g = pd.DataFrame({"b": B, "d": D}).groupby("d").b.mean(); se = g.std() / np.sqrt(len(g))
        mm = [x[0] for x in ms]
        print(f"{f'{ver}({sc}) TP{tp*100:g}/SL{sl*100:g}':>24}"
              f"{np.mean([x[1] for x in ms])/sum(DAYS[f] for f in FO):>9.2f}{np.mean(mm):>+9.2f}"
              f"   [{g.mean()-1.96*se:+7.2f},{g.mean()+1.96*se:+7.2f}]{len(g):>9}"
              f"{max(mm)-min(mm):>8.2f}{int(np.median([x[2] for x in ms])):>5}/6")


def _pivots_ts():
    lab = pd.concat([pd.read_csv(p, usecols=["timestamp", "zigzag_transition_buffer"])
                     for p in sorted(glob.glob(str(R.BASE / "zigzag_labels_full/*.csv")))])
    lab["timestamp"] = pd.to_datetime(lab.timestamp).dt.tz_localize(None)
    lab = lab.drop_duplicates("timestamp").sort_values("timestamp")
    return np.sort(lab.timestamp.to_numpy()[lab.zigzag_transition_buffer.to_numpy() == 1])


def cmd_ab(vers):
    """A/B + 극점 거리별 진단. 「막차」=다음 극점까지, 「관성」=직전 극점 이후."""
    df = R.panel(); piv = _pivots_ts()
    lab = pd.concat([pd.read_csv(p, usecols=["timestamp", "zigzag_action"])
                     for p in sorted(glob.glob(str(R.BASE / "zigzag_labels_full/*.csv")))])
    lab["timestamp"] = pd.to_datetime(lab.timestamp).dt.tz_localize(None)
    lab = lab.drop_duplicates("timestamp")
    A = dict(zip(lab.timestamp.to_numpy(), lab.zigzag_action.to_numpy()))
    SUM = {}
    for ver, sc in vers:
        rows, tot = [], 0
        for f in FO:
            _, tr, _ = R.trades(df, f, ver, score=sc)
            ts = pd.to_datetime(tr.ts).to_numpy(); j = np.searchsorted(piv, ts)
            la = np.array([A.get(np.datetime64(pd.Timestamp(t)), -1) for t in ts])
            s = tr.side.to_numpy()
            rows.append(pd.DataFrame({
                "bp": tr.bp.to_numpy(), "h": ((s > 0) & (la == 1)) | ((s < 0) & (la == 2)),
                "prev": np.where(j > 0, (ts - piv[np.clip(j - 1, 0, len(piv) - 1)]) / np.timedelta64(5, "m"), 9e9),
                "nxt": np.where(j < len(piv), (piv[np.clip(j, 0, len(piv) - 1)] - ts) / np.timedelta64(5, "m"), 9e9),
                "d": pd.to_datetime(tr.ts).dt.floor("D").to_numpy()}))
            tot += DAYS[f]
        t = pd.concat(rows); g = t.groupby("d").bp.mean(); se = g.std() / np.sqrt(len(g))
        print(f"{ver}({sc}) {len(t)/tot:.2f}건/일 {t.bp.mean():+.2f}bp · 라벨일치 {t.h.mean()*100:.1f}%"
              f" · 독립일 {len(g)} · CI95 [{g.mean()-1.96*se:+.2f},{g.mean()+1.96*se:+.2f}]")
        SUM[f"{ver}({sc})"] = t
    for nm, col in (("막차 — 다음 극점까지", "nxt"), ("관성 — 직전 극점 이후", "prev")):
        print(f"\n=== {nm}")
        for k, t in SUM.items():
            b = pd.cut(t[col], [-1, 5, 12, 24, 48, 10 ** 9], labels=["≤5봉", "6-12", "13-24", "25-48", ">48"])
            g = t.groupby(b, observed=True).agg(n=("bp", "size"), hit=("h", "mean"), bp=("bp", "mean"))
            print(f" {k}: " + " | ".join(f"{i} n{int(r.n)} {r.hit*100:.0f}% {r.bp:+.1f}bp" for i, r in g.iterrows()))


def cmd_gate():
    """인과 게이트 스윕. 🔴피벗 확정지연 중앙 11봉이라 «극점 직후 차단» 은 성립하지 않는다."""
    df = R.panel(); ST = R.causal_state(df); ST["ts"] = df.timestamp.to_numpy()
    L = ST.lag[ST.lag >= 0]
    print(f"전환 확정 {len(L):,}회 · 확정지연 분위 {np.percentile(L,[10,25,50,75,90]).astype(int)}봉 "
          f"(중앙 {np.median(L):.0f}봉 = {np.median(L)*5/60:.1f}h)")
    C = {}
    for f in FO:
        te, side = R.candidates(df, f, "v4")
        s = ST[(ST.ts >= te.timestamp.iloc[0]) & (ST.ts <= te.timestamp.iloc[-1])].reset_index(drop=True)
        assert len(s) == len(te)
        C[f] = (te, side, s)

    def run(keep, nm):
        B, n, d, pos = [], 0, 0, 0
        for f in FO:
            te, side, s = C[f]
            sd = side.copy(); i0 = np.where(sd != 0)[0]
            sd[i0[~keep(s, sd, i0)]] = 0.0
            tr, _ = R.sequential(te, sd)
            b = tr.bp.to_numpy(); B.append(b); n += len(tr); d += DAYS[f]; pos += b.mean() > 0
        b = np.concatenate(B)
        print(f"{nm:30s}{n/d:>8.2f}{b.mean():>+9.2f}{(b.mean()-1.02)*n/d:>9.1f}{pos:>5}/6")

    print(f"\n{'규칙':30s}{'체결/일':>8}{'건당bp':>9}{'순/일':>9}{'부호':>7}")
    run(lambda s, sd, i: np.ones(len(i), bool), "무필터 (v4 현행)")
    for K in (2, 3, 5, 8, 12):
        run(lambda s, sd, i, K=K: s.age.to_numpy()[i] > K, f"R1 age > {K}")
    for fr in (0.3, 0.5, 0.7):
        run(lambda s, sd, i, fr=fr: s.rtr.to_numpy()[i] < fr * s.thr.to_numpy()[i], f"R2 rtr < {fr}·thr")
    print("🔴R1 은 단조가 아니고(age>2·3·5 는 악화) age6-8 대비 우위의 폴드 부호가 3/3 동전이다 "
          "-- 메커니즘이 아니라 스캔 인공물. 실험 문서 §4 참조.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "sweep", "grid", "pick", "ab", "gate"])
    ap.add_argument("--name", default="zigzag_b0soft", help="build 용 라벨 이름")
    ap.add_argument("--vers", default="v4:edge,v7:dq", help="<라벨>:<게이트점수> 쉼표 구분")
    ap.add_argument("--cells", default="v4:edge:0.015:0.007,v7:dq:0.015:0.007")
    a = ap.parse_args()
    vers = [tuple(x.split(":")) for x in a.vers.split(",")]
    if a.cmd == "build": cmd_build(a.name)
    elif a.cmd == "sweep": cmd_sweep()
    elif a.cmd == "grid": cmd_grid(vers)
    elif a.cmd == "ab": cmd_ab(vers)
    elif a.cmd == "gate": cmd_gate()
    else:
        cmd_pick([(v, s, float(t), float(l)) for v, s, t, l in
                  (x.split(":") for x in a.cells.split(","))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
