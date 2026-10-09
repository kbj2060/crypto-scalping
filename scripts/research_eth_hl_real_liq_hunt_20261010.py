"""«청산이 많은 쪽으로 마켓 메이커가 청산을 먹으러 간다» — 실제 청산가(Hyperliquid 공개 포지션)로 검정 (2026-10-10).
사용자 «그럼 청산이 많은 쪽으로 마켓 메이커가 선물 시장에서 청산을 먹으러 간다라는 가설도 검증해줘».
앞선 자석 검정(10-08 R5·10-09 먼저 닿기·10-10 5분)은 전부 «추정» 청산맵이었다. 여기는 실제 계정의 실제 청산가.

원천: lake/hl/positions(상위 300 지갑, ~152초 바퀴, 09-24~10-07) · lake/hl/asset_ctx(HL 마크 ~1초 — HL 청산 판정 가격).
매 5분 t: 지갑마다 ex_ms ∈ (t−240초, t] 마지막 스냅샷 → 청산가 ×|szi| = 청산 명목$. 롱은 현재가 아래·숏은 위(반대쪽 = 낡은 행, 버림).
칸 k = 현재 마크 대비 거리 ((k−1)/10, k/10]% , k = 3~20(0.2~2%). 칸 중앙가에 닿음 = HL 마크 1분 고/저가 닿음.

사전등록(결과 전 고정)
- H1 «먹으러 간다»: 칸이 «덩어리»(명목 ≥ 0 아닌 칸 전체 90분위) vs «빈 칸»(자신·양옆 0). 셀 = (쪽, k, 직전 1h 변동성 3분위,
  칸이 직전 24h 고/저 밖인가)로 맞춰 덩어리 닿음률 − 빈 칸 닿음률, 덩어리 수 가중. 지평 1h·4h. 일 블록 부트스트랩.
  통과 = 4h 차 CI 하한 > 0 이고 1h 차 > 0.
- H2 «많은 쪽으로»: A = ln((위 2% 안 숏 청산$ + 1e5)/(아래 2% 안 롱 청산$ + 1e5)). 진입 t+1분 마크, 15m·1h·4h 수익 IC.
  과거 1h·24h 수익 순위로 잔차화한 부분 IC 도. 통과 = 4h 부분 IC CI 하한 > 0.
한계(결과 전 기록): 14일 · 상위 300 지갑(HL OI 의 일부) · 같은 덩어리가 여러 시각에 반복 → 실효 표본 작음, CI 넓다.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · future_rows_used_for_entry=false.
"""
import glob, json, os, sys
import numpy as np, pandas as pd

HL = os.path.expanduser("~/backups/crypto-scalping-server/data/lake/hl")
OUT = "tmp/hl_real_liq_hunt_20261010"
STEP, WIN, KS = 300_000, 240_000, np.arange(3, 21)
HORS = {"1h": 60, "4h": 240}


def lake(stream, cols):
    return pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{HL}/{stream}/coin=ETH/date=*/part.parquet"))], ignore_index=True)


def profile(snap, m):
    """스냅샷 → (위 칸 배열, 아래 칸 배열) 길이 21(인덱스 = k), 단위 $. 반대쪽 청산가(낡음)는 버림."""
    up, dn = np.zeros(21), np.zeros(21)
    r = (snap.liq_px.to_numpy() / m - 1) * 1000                                 # 0.1% 단위
    usd = np.abs(snap.szi.to_numpy()) * snap.liq_px.to_numpy()
    lng = snap.szi.to_numpy() > 0
    k = np.ceil(np.abs(r) - 1e-9).astype(int)                                  # 경계값(정확히 0.5% 등) 부동소수 오차
    ok = (k >= 1) & (k <= 20)
    np.add.at(up, k[ok & ~lng & (r > 0)], usd[ok & ~lng & (r > 0)])
    np.add.at(dn, k[ok & lng & (r < 0)], usd[ok & lng & (r < 0)])
    return up, dn


def build():
    P = lake("positions", ["ex_ms", "user", "szi", "liq_px"])
    P = P[np.isfinite(P.liq_px) & (P.liq_px > 0) & (P.szi != 0)].sort_values("ex_ms").reset_index(drop=True)
    C = lake("asset_ctx", ["recv_ms", "mark_px"]).sort_values("recv_ms")
    mn = C.assign(mi=C.recv_ms // 60_000).groupby("mi").mark_px.agg(["max", "min", "last"])
    mn = mn.reindex(np.arange(mn.index.min(), mn.index.max() + 1)).ffill()
    mi0, hi, lo, la = int(mn.index[0]), mn["max"].to_numpy(), mn["min"].to_numpy(), mn["last"].to_numpy()
    lr = np.r_[0, np.diff(np.log(la))]
    rows = []
    ex = P.ex_ms.to_numpy()
    for t in range((int(ex[0]) // STEP + 2) * STEP, (mi0 + len(la) - 241) * 60_000, STEP):
        i = t // 60_000 - mi0                                                    # 분 i = [t, t+1분) — t 에 이미 끝난 분은 i−1
        if i < 1441:
            continue
        a, b = np.searchsorted(ex, t - WIN, "right"), np.searchsorted(ex, t, "right")
        snap = P.iloc[a:b].groupby("user").tail(1)
        if len(snap) < 20:
            continue
        m = la[i - 1]
        up, dn = profile(snap, m)
        rv = float(np.std(lr[i - 60:i]))
        h24, l24 = hi[i - 1440:i].max(), lo[i - 1440:i].min()
        for s, arr in ((1, up), (-1, dn)):
            for k in KS:
                lvl = m * (1 + s * (k - 0.5) / 1000)
                r = {"t": t, "side": s, "k": int(k), "A": arr[k], "empty": bool(arr[k - 1] == 0 and arr[k] == 0 and arr[min(k + 1, 20)] == 0),
                     "rv": rv, "beyond": bool(lvl > h24 if s > 0 else lvl < l24)}
                for nm, H in HORS.items():
                    r[f"reach_{nm}"] = bool(hi[i:i + H].max() >= lvl if s > 0 else lo[i:i + H].min() <= lvl)
                rows.append(r)
        # H2 피쳐·목표(진입 = t+1분 마크 = 분 i 의 마지막)
        rows.append({"t": t, "side": 0, "A_up2": up[1:21].sum(), "A_dn2": dn[1:21].sum(), "m": m,
                     "r1h": np.log(m / la[i - 61]) * 1e4, "r24h": np.log(m / la[i - 1441]) * 1e4,
                     **{f"y{H}": np.log(la[i + H] / la[i]) * 1e4 for H in (15, 60, 240)}})
    D = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    D.to_parquet(f"{OUT}/panel.parquet")
    return D


def boot_days(day, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ud = np.unique(day); g = {u: np.flatnonzero(day == u) for u in ud}
    v = [fn(np.concatenate([g[u] for u in rng.choice(ud, len(ud))])) for _ in range(n)]
    return [round(float(x), 4) for x in np.nanquantile(v, [.025, .975])]


def matched_diff(L, nm):
    """셀 맞춤 (덩어리 닿음률 − 빈 칸 닿음률), 덩어리 수 가중."""
    g = L.groupby(["side", "k", "rvT", "beyond", "cl"])[f"reach_{nm}"].agg(["mean", "size"]).unstack("cl").dropna()
    if not len(g) or ("mean", True) not in g.columns or ("mean", False) not in g.columns:   # 부트스트랩 표본에 덩어리/빈 칸 없음
        return np.nan
    return float(((g[("mean", True)] - g[("mean", False)]) * g[("size", True)]).sum() / g[("size", True)].sum())


def run():
    D = build() if "--rebuild" in sys.argv or not os.path.exists(f"{OUT}/panel.parquet") else pd.read_parquet(f"{OUT}/panel.parquet")
    L = D[D.side != 0].copy()
    q90 = float(L.loc[L.A > 0, "A"].quantile(.9))
    L["rvT"] = pd.qcut(L.rv, 3, labels=False)
    L = L[(L.A >= q90) | L["empty"].astype(bool)].copy()
    L["cl"] = L.A >= q90
    L["day"] = L.t // 86_400_000
    out = {"days": int(L.day.nunique()), "times": int(D.t.nunique()), "cluster_usd_q90": round(q90), "H1": {}, "H2": {},
           "flags": {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "future_rows_used_for_entry": False}}
    day = L.day.to_numpy()
    for nm in HORS:
        d = matched_diff(L, nm)
        ci = boot_days(day, lambda ix: matched_diff(L.iloc[ix], nm), n=500)
        raw = L.groupby("cl")[f"reach_{nm}"].mean()
        out["H1"][nm] = {"diff_pp": round(d * 100, 2), "ci_pp": [round(x * 100, 2) for x in ci], "raw_cluster": round(float(raw[True]), 3),
                         "raw_empty": round(float(raw[False]), 3), "n_cluster": int(L.cl.sum()), "n_empty": int((~L.cl).sum())}
        print("H1", nm, out["H1"][nm], flush=True)
    by_k = L.groupby([pd.cut(L.k, [2, 5, 10, 20]), "cl"], observed=True).reach_4h.mean().unstack()
    out["H1"]["by_distance_4h_[empty,cluster]"] = {str(i): [round(float(v), 3) for v in r] for i, r in by_k.iterrows()}
    out["H1"]["pass"] = bool(out["H1"]["4h"]["ci_pp"][0] > 0 and out["H1"]["1h"]["diff_pp"] > 0)
    S = D[D.side == 0].dropna(subset=["y240", "r24h"]).reset_index(drop=True)
    A = np.log((S.A_up2 + 1e5) / (S.A_dn2 + 1e5)).to_numpy()
    rk = lambda v: pd.Series(v).rank().to_numpy()
    Z = np.c_[np.ones(len(S)), rk(S.r1h), rk(S.r24h)]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    sday = (S.t // 86_400_000).to_numpy()
    rA = res(rk(A))
    for H in (15, 60, 240):
        y = S[f"y{H}"].to_numpy(); ry = res(rk(y))
        ic = lambda ix: float(np.corrcoef(rk(A[ix]), rk(y[ix]))[0, 1])
        pic = lambda ix: float(np.corrcoef(rA[ix], ry[ix])[0, 1])
        out["H2"][H] = {"ic": round(ic(np.arange(len(S))), 4), "ic_ci": boot_days(sday, ic, n=500),
                        "partial_ic": round(pic(np.arange(len(S))), 4), "partial_ci": boot_days(sday, pic, n=500)}
        print("H2", H, out["H2"][H], flush=True)
    out["H2"]["share_up_gt_dn"] = round(float((S.A_up2 > S.A_dn2).mean()), 3)
    out["H2"]["pass"] = bool(out["H2"][240]["partial_ci"][0] > 0)
    json.dump(out, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("days", "times", "cluster_usd_q90")}), "| by_k", out["H1"]["by_distance_4h_[empty,cluster]"],
          "| H1 pass", out["H1"]["pass"], "| H2 pass", out["H2"]["pass"])


def selftest():
    snap = pd.DataFrame({"szi": [10.0, -5.0, 2.0], "liq_px": [995.0, 1012.0, 1003.0]})               # 롱 −0.5% · 숏 +1.2% · 낡은 롱(위)
    up, dn = profile(snap, 1000.0)
    assert dn[5] == 9950.0 and up[12] == 5060.0 and up.sum() == 5060.0 and dn.sum() == 9950.0
    L = pd.DataFrame({"side": 1, "k": 3, "rvT": 0, "beyond": False, "cl": [True, True, False, False], "reach_1h": [True, True, True, False]})
    assert abs(matched_diff(L, "1h") - 0.5) < 1e-12
    print("selftest OK")


if __name__ == "__main__":
    {"--run": run, "--selftest": selftest}[sys.argv[1]]()
