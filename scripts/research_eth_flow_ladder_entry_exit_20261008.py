"""청산·OI·거래량·풋프린트만으로 진입·청산 — 대시보드 4분할 기계 위에서 (2026-10-08).
사용자 «지금 내가 생각하는 핵심은 청산, OI, 거래량, 풋프린트만 가지고 진입과 청산을 다시 연구해줘.
기본 전제 규칙은 현재 대시보드에 있는 4분할 매수와 손절라인과 변동성 진입크기야.»

사전등록 (결과 전 고정)
- 기계(고정): 첫 진입 + 44bp 마다 3번 더(균등 4분할, 지정가 0bp) · 첫 진입가 ∓3σ24 손절 · 크기 = vol_mult(σ24)(대시보드 «크기∝σ»)
  · 최대 24h · 손절/시간/신호 청산 테이커 5bp(단위당). 손익 bp 는 4분할 전체 명목 대비.
- 데이터: 5분 패널(2022~2026-09, 봉 마감에 앎) — 거래량 v, 풋프린트 = 봉 델타(2·테이커매수−v)/v, OI = 1봉 보수 지연.
  청산 = OI 급감 + 큰 가격 이동 + 거래량 폭증의 대리(실측 청산은 2026-07-19~ 뿐이라 대리 사용).
  z 는 직전 7일(2016봉) 분위 기반 강건 z((x−중앙)/(IQR/1.349), 그 봉 제외).
- 사건(봉 k 마감, 다음 1분 종가 진입, 같은 규칙·방향 1h 쿨다운):
  롱청산 대리 FL: 15분 수익 ≤ −1.5·σ15 · zOI15 ≤ −2 · 15분 거래량 ≥ 평소 2배   / 숏청산 대리 SQ: 거울(+1.5·σ15).
  E1 청산 페이드: FL → 롱, SQ → 숏.
  E2 청산 + 흡수: E1 이면서 최근 2봉 안에 «반대 델타 z ≤ −1.5(롱 기준 매도 공격) 인데 종가 위치 ≥ 0.6» 봉.
  E4 OI 쌓임 추종: 1h 수익 ≥ +σ60 · zOI60 ≥ +2 · z델타60 ≥ +1 → 롱(거울 숏).
  E5 OI 쌓임 역행(연료): 1h 수익 ≤ −σ60 · zOI60 ≥ +2 · z델타60 ≤ −1 → 롱(밀어붙이는 신규 숏 = 쥐어짤 연료, 거울 숏).
- 진입 판정 청산 규칙 둘: X0 평단 +20bp 지정가 익절 · XH 신호·익절 없이 손절/24h.
  비교선 = 같은 구간·같은 방향·같은 σ24 십분위(AUX 경계)의 무작위 5분 마감 진입 5배(변동성 맞춘 위약).
- 청산 규칙(신호는 5분 마감에 앎 → 다음 1분 종가 테이커, 익절 없음, 손절·24h 유지):
  X1 반대쪽 청산 대리(롱이면 SQ) · X2 풋프린트 반전: 15분 z델타가 포지션 반대 ≥2(이익일 때만) · X3 순방향 거래량 절정: 봉 거래량 z ≥3 & 봉 수익 순방향(이익일 때만)
  · X4 OI 역행 쌓임(손절 전 정리): 15분 역행 ≥ σ15 & zOI15 ≥ +2 · X5 = X1|X2|X3(이익일 때만).
  진입 집합 = 사용자식 핑퐁 페이드 후보(AD.inputs) · 무작위. 비교선 = 같은 진입·같은 보유시간 분포를 섞어 시간 청산(«신호냐, 보유 길이냐»).
- 지표: 크기 가중 건당 bp(vol_mult × 손익) · 순차 한 포지션 복리 로그성장(λ = AUX 손절 손실 중앙 20%, 팔마다).
- 구간: AUX 2022-01~2024-12 · MAIN 2025-01~2026-09-12(주 판정, 최근 우선). 일 블록 부트 1000회.
- 통과: MAIN (팔 − 비교선) 크기 가중 bp CI 하한 > 0 · AUX 같은 부호.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · future_rows_used_for_entry=false.
ponytail: 연구 스크린 — 실측 청산 재확인은 07-19~ 표본이 쌓인 뒤.
"""
import json, os, sys
import numba
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import research_eth_rule_ladder_upgrade_20261008 as U  # noqa: E402
import research_eth_story_composite_20261008 as SC  # noqa: E402

ST, AD = U.ST, U.AD
OUT = "tmp/flow_ladder_20261008"
SPL = {"AUX": ("2022-01-08", "2025-01-01"), "MAIN": ("2025-01-01", "2026-09-12")}
W, D, K, HOR, FEE, B = np.ones(4), 44.0, 3.0, 1440, U.FEE, 0.20


@numba.njit(cache=False)
def simx(h, l, c, ei, side, S, H, TP, exl, exs, need_profit):
    """U.sim(균등 4분할) + 신호 청산: exl/exs[j] 가 참이면(분 j 마감에 앎) c[j+1] 테이커. 종류 1 익절 2 시간 3 손절 4 신호."""
    n = len(ei); nw = 4
    pnl = np.zeros(n); xi = np.zeros(n, np.int64); kind = np.zeros(n, np.int64)
    for k in range(n):
        i = ei[k]; s = side[k]; p0 = c[i]
        q = 1.0; cost = p0; nadd = 0
        end = min(i + H[k], len(c) - 2)
        exit_px = c[end]; kind[k] = 2; xi[k] = end
        for j in range(i + 1, end + 1):
            added = False
            while nadd < nw - 1:
                lvl = p0 * (1 - s * D * (nadd + 1) / 1e4)
                if (s > 0 and l[j] <= lvl) or (s < 0 and h[j] >= lvl):
                    nadd += 1; q += 1.0; cost += lvl; added = True
                else:
                    break
            stop = p0 * (1 - s * S[k] / 1e4)
            if (s > 0 and l[j] <= stop) or (s < 0 and h[j] >= stop):
                gap = (s > 0 and h[j] < stop) or (s < 0 and l[j] > stop)
                exit_px = c[j] if gap else stop; kind[k] = 3; xi[k] = j
                break
            if TP > 0 and not added:
                tgt = cost / q * (1 + s * TP / 1e4)
                if (s > 0 and h[j] >= tgt) or (s < 0 and l[j] <= tgt):
                    exit_px = tgt; kind[k] = 1; xi[k] = j
                    break
            sig = exl[j] if s > 0 else exs[j]
            if sig and ((not need_profit) or s * (c[j] * q - cost) > 0):
                exit_px = c[j + 1]; kind[k] = 4; xi[k] = j + 1
                break
        fee = FEE * q if kind[k] >= 2 else 0.0
        pnl[k] = (s * (exit_px * q - cost) / p0 * 1e4 - fee) / nw
    return pnl, xi, kind


def rz(x, w=2016):
    s = pd.Series(x)
    q = s.rolling(w, min_periods=w // 2).quantile
    q25, q50, q75 = q(.25).shift(1), q(.5).shift(1), q(.75).shift(1)
    return ((s - q50) / ((q75 - q25) / 1.349).replace(0, np.nan)).to_numpy()


def features():
    p = pd.read_parquet(f"{SC.BV}/panel/ETHUSDT.parquet").sort_values("timestamp").reset_index(drop=True)
    t5 = p.timestamp.astype("int64").to_numpy() // 10**6
    c, h, l, v, tb = (p[x].to_numpy() for x in ("close", "high", "low", "volume", "taker_buy_base"))
    oi = p.sum_open_interest.shift(1)
    r5 = np.diff(np.log(c), prepend=np.nan) * 1e4
    sig24 = np.sqrt(pd.Series(r5 ** 2).rolling(288, min_periods=280).sum().to_numpy())
    s15, s60 = sig24 / np.sqrt(96), sig24 / np.sqrt(24)
    lc = pd.Series(np.log(c))
    ret15, ret60 = ((lc - lc.shift(3)) * 1e4).to_numpy(), ((lc - lc.shift(12)) * 1e4).to_numpy()
    imb, vs = pd.Series(2 * tb - v), pd.Series(v)
    dl = (imb / vs).to_numpy()
    dl15, dl60 = (imb.rolling(3).sum() / vs.rolling(3).sum()).to_numpy(), (imb.rolling(12).sum() / vs.rolling(12).sum()).to_numpy()
    vavg = vs.rolling(288, min_periods=200).mean().shift(1).to_numpy()
    v15x = vs.rolling(3).sum().to_numpy() / (3 * vavg)
    loi = np.log(oi)
    zoi15, zoi60 = rz((loi - loi.shift(3)).to_numpy()), rz((loi - loi.shift(12)).to_numpy())
    zdl, zdl15, zdl60, zv = rz(dl), rz(dl15), rz(dl60), rz(np.log(v + 1e-9))
    clv = (c - l) / np.maximum(h - l, 1e-9)
    F = dict(t5=t5, c=c, sig24=sig24, s15=s15, s60=s60, ret15=ret15, ret60=ret60, r5=r5, zdl=zdl, zdl15=zdl15, zdl60=zdl60, zv=zv,
             v15x=v15x, zoi15=zoi15, zoi60=zoi60, clv=clv)
    F["FL"] = (ret15 <= -1.5 * s15) & (zoi15 <= -2) & (v15x >= 2)
    F["SQ"] = (ret15 >= 1.5 * s15) & (zoi15 <= -2) & (v15x >= 2)
    ab_l = (zdl <= -1.5) & (clv >= 0.6); ab_s = (zdl >= 1.5) & (clv <= 0.4)
    ab_l2 = ab_l | np.r_[False, ab_l[:-1]]; ab_s2 = ab_s | np.r_[False, ab_s[:-1]]
    F["entries"] = {
        "E1 청산 페이드": (F["FL"], F["SQ"]),
        "E2 청산+흡수": (F["FL"] & ab_l2, F["SQ"] & ab_s2),
        "E4 OI 쌓임 추종": ((ret60 >= s60) & (zoi60 >= 2) & (zdl60 >= 1), (ret60 <= -s60) & (zoi60 >= 2) & (zdl60 <= -1)),
        "E5 OI 쌓임 역행": ((ret60 <= -s60) & (zoi60 >= 2) & (zdl60 <= -1), (ret60 >= s60) & (zoi60 >= 2) & (zdl60 >= 1)),
    }
    return F


def split_of(tms):
    t = pd.to_datetime(tms, unit="ms")
    return np.select([(t >= pd.Timestamp(a)) & (t < pd.Timestamp(b)) for a, b in SPL.values()], list(SPL), "-")


def to_1m(t1, t5, mask):
    """5분 봉 k 의 신호 → 그 봉 마지막 분(마감에 앎) 인덱스에 참."""
    out = np.zeros(len(t1), np.bool_)
    want = t5[mask] + 240_000
    j = np.searchsorted(t1, want)
    ok = j < len(t1)
    j, want = j[ok], want[ok]
    out[j[t1[j] == want]] = True
    return out


def events(F, t1, long_m, short_m):
    rows = []
    for s, m in ((1, long_m), (-1, short_m)):
        last = -10**9
        for k in np.flatnonzero(np.nan_to_num(m).astype(bool) & np.isfinite(F["sig24"])):
            if k - last >= 12:
                rows.append((s, k)); last = k
    E = pd.DataFrame(rows, columns=["side", "k"])
    me = np.searchsorted(t1, F["t5"][E.k] + 240_000)
    ok = (me + 2 + HOR < len(t1)) & (t1[np.minimum(me, len(t1) - 1)] == F["t5"][E.k] + 240_000)
    E = E[ok].reset_index(drop=True); E["ei"] = me[ok] + 1
    E["sig24"] = F["sig24"][E.k]; E["sp"] = split_of(t1[E.ei])
    return E[E.sp != "-"].reset_index(drop=True)


def run(k1, E, tp=-1.0, ex=None, need_profit=False, H=None):
    h1, l1, c1 = k1
    z = np.zeros(len(c1), np.bool_)
    exl, exs = (z, z) if ex is None else ex
    H = np.full(len(E), HOR, np.int64) if H is None else H
    return simx(h1, l1, c1, E.ei.to_numpy(np.int64), E.side.to_numpy(np.int64), K * E.sig24.to_numpy(), H, tp, exl, exs, need_profit)


def growth(E, pnl, xi, lam):
    """순차 한 포지션 복리 로그성장(구간별)."""
    w = ST.vol_mult(E.sig24.to_numpy())
    out = {}
    for z in SPL:
        idx = np.flatnonzero(E.sp.to_numpy() == z)
        idx = idx[np.argsort(E.ei.to_numpy()[idx])]
        g, last = 0.0, -1
        for j in idx:
            if E.ei.iat[j] > last:
                g += np.log1p(max(lam * w[j] * pnl[j] / 1e4, -0.999)); last = xi[j]
        out[z] = round(float(g), 3)
    return out


def boot_diff(a, b, da, db, rng):
    """일 블록 부트: mean(a) − mean(b), a/b 는 각자 날짜 da/db."""
    ud = np.union1d(np.unique(da), np.unique(db))
    ga = {u: np.flatnonzero(da == u) for u in np.unique(da)}; gb = {u: np.flatnonzero(db == u) for u in np.unique(db)}
    bs = []
    for _ in range(1000):
        pick = rng.choice(ud, len(ud))
        ia = np.concatenate([ga.get(u, np.empty(0, int)) for u in pick]).astype(int)
        ib = np.concatenate([gb.get(u, np.empty(0, int)) for u in pick]).astype(int)
        if len(ia) and len(ib):
            bs.append(a[ia].mean() - b[ib].mean())
    return [round(float(np.quantile(bs, .025)), 2), round(float(np.quantile(bs, .975)), 2)]


def summarize(name, E, pnl, xi, kind, Eb, pb, rng, lam=None):
    w, wb = ST.vol_mult(E.sig24.to_numpy()), ST.vol_mult(Eb.sig24.to_numpy())
    a, b = w * pnl, wb * pb
    day, dayb = (E.ei.to_numpy() // 1440), (Eb.ei.to_numpy() // 1440)
    if lam is None:
        aux = E.sp.to_numpy() == "AUX"
        lam = B / np.median(w[aux] * U.full_stop_loss(W, D, K * E.sig24.to_numpy()[aux]))
    r = {"rule": name, "lam": round(float(lam), 3), "logg": growth(E, pnl, xi, lam)}
    for z in SPL:
        m, mb = E.sp.to_numpy() == z, Eb.sp.to_numpy() == z
        if m.sum() < 10:
            r[z] = {"n": int(m.sum())}; continue
        r[z] = {"n": int(m.sum()), "per_day": round(float(m.sum() / max(len(np.unique(day[m])), 1)), 2), "bp_w": round(float(a[m].mean()), 2),
                "base_bp_w": round(float(b[mb].mean()), 2), "diff_ci": boot_diff(a[m], b[mb], day[m], dayb[mb], rng),
                "stop%": round(float((kind[m] == 3).mean()), 3), "hold_min": int(np.median(xi[m] - E.ei.to_numpy()[m]))}
    mm, aa = r.get("MAIN", {}), r.get("AUX", {})
    r["pass"] = bool("diff_ci" in mm and mm["diff_ci"][0] > 0 and aa.get("bp_w", 0) > aa.get("base_bp_w", 0))
    print(json.dumps(r, ensure_ascii=False), flush=True)
    return r, lam


def matched_random(E, F, t1, rng, reps=5):
    """같은 구간·방향·σ24 십분위(AUX 경계) 무작위 5분 마감."""
    sig = F["sig24"]; ok = np.isfinite(sig)
    aux_k = np.flatnonzero(ok & (split_of(F["t5"]) == "AUX"))
    edges = np.quantile(sig[aux_k], np.linspace(0, 1, 11)[1:-1])
    dec_all = np.digitize(sig, edges); sp_all = split_of(F["t5"])
    rows = []
    for (sp, dc, s), g in E.assign(dec=np.digitize(E.sig24, edges)).groupby(["sp", "dec", "side"]):
        pool = np.flatnonzero(ok & (sp_all == sp) & (dec_all == dc))
        if len(pool):
            rows += [(s, kk) for kk in rng.choice(pool, reps * len(g))]
    R = pd.DataFrame(rows, columns=["side", "k"])
    me = np.searchsorted(t1, F["t5"][R.k] + 240_000)
    good = (me + 2 + HOR < len(t1)) & (t1[np.minimum(me, len(t1) - 1)] == F["t5"][R.k] + 240_000)
    R = R[good].reset_index(drop=True); R["ei"] = me[good] + 1
    R["sig24"] = F["sig24"][R.k]; R["sp"] = split_of(t1[R.ei])
    return R[R.sp != "-"].reset_index(drop=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261008)
    k = ST.load_1m(); t1 = k.t.to_numpy(); k1 = tuple(k[x].to_numpy() for x in ("h", "l", "c"))
    F = features()
    rep = {"entry": [], "exit": []}
    # 1) 진입: 사건 vs σ 맞춘 무작위, 청산 X0 / XH
    for name, (lm, sm) in F["entries"].items():
        E = events(F, t1, lm, sm); R = matched_random(E, F, t1, rng)
        for xn, tp in (("X0 +20bp", 20.0), ("XH 보유", -1.0)):
            pnl, xi, kind = run(k1, E, tp); pb, _, _ = run(k1, R, tp)
            rep["entry"].append(summarize(f"{name} · {xn}", E, pnl, xi, kind, R, pb, rng)[0])
    # 2) 청산: 사용자식 핑퐁 후보 + 무작위 진입, 신호 청산 vs 같은 보유시간 섞은 시간 청산
    _, _, _, ei, side, sg, _, _, _, _, tdec = AD.inputs()
    Eu = pd.DataFrame({"side": side, "ei": ei, "sig24": sg}); Eu["sp"] = split_of(t1[Eu.ei])
    Eu = Eu[(Eu.sp != "-") & (Eu.ei + HOR + 2 < len(t1))].reset_index(drop=True)
    nb = len(F["t5"]); kr = rng.choice(np.flatnonzero(np.isfinite(F["sig24"]) & (split_of(F["t5"]) != "-")), 20000, replace=False)
    Er = events(F, t1, np.isin(np.arange(nb), kr[: 10000]), np.isin(np.arange(nb), kr[10000:]))
    m1 = lambda x: to_1m(t1, F["t5"], np.nan_to_num(x).astype(bool))  # noqa: E731
    s15, ret15, zoi15, zdl15 = F["s15"], F["ret15"], F["zoi15"], F["zdl15"]
    X = {"X1 반대쪽 청산": ((m1(F["SQ"]), m1(F["FL"])), False),
         "X2 풋프린트 반전": ((m1(zdl15 <= -2), m1(zdl15 >= 2)), True),
         "X3 순방향 거래량 절정": ((m1((F["zv"] >= 3) & (F["r5"] > 0)), m1((F["zv"] >= 3) & (F["r5"] < 0))), True),
         "X4 OI 역행 쌓임": ((m1((ret15 <= -s15) & (zoi15 >= 2)), m1((ret15 >= s15) & (zoi15 >= 2))), False)}
    X["X5 X1|X2|X3"] = ((X["X1 반대쪽 청산"][0][0] | X["X2 풋프린트 반전"][0][0] | X["X3 순방향 거래량 절정"][0][0],
                        X["X1 반대쪽 청산"][0][1] | X["X2 풋프린트 반전"][0][1] | X["X3 순방향 거래량 절정"][0][1]), True)
    for en, E in (("핑퐁 후보", Eu), ("무작위 진입", Er)):
        p0, x0, k0 = run(k1, E, 20.0)
        rep["exit"].append(summarize(f"{en} · X0 +20bp (기준)", E, p0, x0, k0, E, p0, rng)[0])
        w = ST.vol_mult(E.sig24.to_numpy()); mm = (E.sp == "MAIN").to_numpy(); dd = E.ei.to_numpy() // 1440
        for xn, (ex, npf) in X.items():
            pnl, xi, kind = run(k1, E, -1.0, ex, npf)
            Hp = rng.permutation(xi - E.ei.to_numpy()).astype(np.int64)
            pb, _, _ = run(k1, E, -1.0, None, False, Hp)                        # 같은 보유시간 분포, 신호 없음
            r, _ = summarize(f"{en} · {xn} vs 보유시간 섞기", E, pnl, xi, kind, E, pb, rng)
            r["signal_exit%"] = round(float((kind == 4).mean()), 3)
            r["vs_X0_MAIN"] = boot_diff(w[mm] * pnl[mm], w[mm] * p0[mm], dd[mm], dd[mm], rng)
            print("   signal_exit%", r["signal_exit%"], "vs X0 MAIN", r["vs_X0_MAIN"], flush=True)
            rep["exit"].append(r)
    json.dump(rep, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
