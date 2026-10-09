"""«가까운 쪽 먼저 닿는다» 심층 + 업그레이드 (2026-10-10).
사용자 «그럼 가까운 쪽 먼저 닿는다로 심층 연구 진행해줘. 더 나아가서 업그레이드까지 해줘».

핵심: 방향 없는 무작위 경로에서 위 a·아래 b 레벨 중 위 먼저 닿을 확률 = b/(a+b) (도박꾼 파산). 이때 «가까운 쪽 목표·먼 쪽 손절»
기대값 = p·a − (1−p)·b = 0. 그러니 «가까운 쪽 먼저»는 그 자체로 정보가 아니다 — 돈이 되는 건 이 기하에서 벗어나는 부분뿐.

데이터: 대시보드 1일 청산맵 5분 피쳐(research_eth_liqmap_5m_direction_20261010 feat5m)를 15분마다. 레벨 = 위/아래 5% 안 최대 칸
(up_topd, dn_topd, 둘 다 ≥ 0.2%). 가짜 레벨 = 다른 시각의 (위, 아래) 거리 쌍을 무작위로 가져옴(지금 지도와 무관).
결과 = T 다음 1분부터 72h 안 1분 고/저 먼저 닿음(위·아래·안 닿음·같은 분 둘 다=모호).

사전등록(결과 전 고정) — TRAIN 2022-01~2024-06 · VAL 2024H2 · TEST 2025~(주 판정)
- D1 보정: 기하 p0 = b/(a+b) 구간별 실제 위 먼저 비율(청산 레벨·가짜 레벨). 청산 레벨이 «특별»하면 두 곡선이 갈린다.
- D2 기하 모델 G: 로지스틱(ln(b/a), ln(a/σ), ln(b/σ)) 를 가짜 레벨 TRAIN 으로 적합(σ = 24h 실현변동 일간 %) → 청산 레벨 TEST AUC·로그손실.
  청산 고유 효과 = G 위에 청산 몫 비(ln(위 최대 칸 몫/아래)) 추가 시 TEST 로그손실 개선 · 계수 부호(+ = 자석).
- D3 경제: «가까운 쪽 목표·먼 쪽 손절·72h» 를 진입 시장가 4bp · 목표 지정가 0 · 손절/시간 4bp 로. 같은 분 둘 다 = 손절.
- U 업그레이드: HGB(G 피쳐 + 추세 r1h·r4h·r24h·r7d + 변동성 비 rv1h/rv24h + 24h 범위 위치 + 시간대 + 청산 lr_1·lr_3·lr_mag·몫 비)
  를 TRAIN(청산+가짜, 닿은 행) 로 적합. 진입 = 방향 s 의 모델 확률 − 손익분기 p* > m, p* = (먼 쪽 거리 + 0.08%)/(a+b).
  m 은 VAL 에서 {0, .02, .05, .1} 중 순손익 최대. 통과 = TEST 비용 후 평균 일 블록 CI 하한 > 0 · VAL > 0.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · future_rows_used_for_entry=false.
"""
import json, os, sys
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import research_eth_liqmap_5m_direction_20261010 as LQ  # noqa: E402

OUT = "tmp/first_touch_20261010"
H = 72 * 60
SPL = {"TRAIN": ("2022-01-01", "2024-07-01"), "VAL": ("2024-07-01", "2025-01-01"), "TEST": ("2025-01-01", "2027-01-01")}
C_IN, C_OUT = 4.0, 4.0


@njit(cache=True)
def first_touch(h, l, c, j0, up, dn, H):
    """분 j0 끝에서 시작, j0+1.. 경로. 반환 (코드 1=위·−1=아래·0=안 닿음·2=같은 분 둘 다, 걸린 분, 청산가)."""
    n = len(j0)
    code = np.zeros(n, np.int8); tt = np.zeros(n, np.int32); px = np.zeros(n)
    for i in range(n):
        end = min(j0[i] + H, len(c) - 1)
        code[i], tt[i], px[i] = 0, end - j0[i], c[end]
        for m in range(j0[i] + 1, end + 1):
            hu, hd = h[m] >= up[i], l[m] <= dn[i]
            if hu or hd:
                code[i] = 2 if (hu and hd) else (1 if hu else -1)
                tt[i] = m - j0[i]; px[i] = up[i] if hu and not hd else dn[i]
                break
    return code, tt, px


def build():
    os.makedirs(OUT, exist_ok=True)
    F = pd.read_parquet(f"{LQ.OUT}/feat5m.parquet", columns=["T", "cp", "up_topd", "dn_topd", "up_topw", "dn_topw", "lr_1", "lr_3", "lr_mag"])
    F = F[F["T"] % 900_000 == 0].sort_values("T").reset_index(drop=True)
    F = F[(F.up_topd >= 0.2) & (F.dn_topd >= 0.2) & (F.up_topw > 0) & (F.dn_topw > 0)].reset_index(drop=True)
    k = LQ.load_1m()
    h, l, c = k.h.to_numpy(), k.l.to_numpy(), k.c.to_numpy()
    j = ((F["T"].to_numpy() - 60_000 - int(k.t.iloc[0])) // 60_000).astype(np.int64)
    assert np.allclose(c[j], F.cp)
    lc = np.log(c)
    st = {f"r{nm}": (lc[j] - lc[j - m]) * 1e4 for nm, m in (("1h", 60), ("4h", 240), ("24h", 1440), ("7d", 10080))}
    cs2 = np.cumsum(np.r_[0, np.diff(lc)] ** 2)
    rv24 = np.sqrt(cs2[j] - cs2[j - 1440]) * 100                                  # 일간 % 변동
    st["vratio"] = np.sqrt(cs2[j] - cs2[j - 60]) * np.sqrt(24) * 100 / np.maximum(rv24, 1e-9)
    hi = pd.Series(h).rolling(1440).max().to_numpy()[j]; lo = pd.Series(l).rolling(1440).min().to_numpy()[j]
    st["pos24"] = (c[j] - lo) / np.maximum(hi - lo, 1e-9)
    st["hour"] = (F["T"].to_numpy() // 3_600_000) % 24
    rng = np.random.default_rng(20261010)
    perm = rng.permutation(len(F))
    parts = []
    for kind, a, b in (("liq", F.up_topd.to_numpy(), F.dn_topd.to_numpy()), ("fake", F.up_topd.to_numpy()[perm], F.dn_topd.to_numpy()[perm])):
        code, tt, px = first_touch(h, l, c, j, c[j] * (1 + a / 100), c[j] * (1 - b / 100), H)
        P = pd.DataFrame({"T": F["T"], "kind": kind, "a": a, "b": b, "sig": rv24, "code": code, "tt": tt, "px": px, "cp": c[j],
                          "wr": np.log(F.up_topw / F.dn_topw) if kind == "liq" else np.log(F.up_topw.to_numpy()[perm] / F.dn_topw.to_numpy()[perm]),
                          **{x: F[x] for x in ("lr_1", "lr_3", "lr_mag")}, **st})
        parts.append(P)
    D = pd.concat(parts, ignore_index=True)
    ts = pd.to_datetime(D["T"], unit="ms")
    D["split"] = np.select([(ts >= x) & (ts < y) for x, y in SPL.values()], list(SPL), "-")
    D["day"] = D["T"] // 86_400_000
    D.to_parquet(f"{OUT}/panel.parquet")
    print(D.groupby(["kind", "code"]).size().unstack().to_string())


GEO = ["g", "la", "lb"]
STATE = ["r1h", "r4h", "r24h", "r7d", "vratio", "pos24", "hour"]
LIQF = ["wr", "lr_1", "lr_3", "lr_mag"]


def feats(D):
    D = D.copy()
    D["g"] = np.log(D.b / D.a); D["la"] = np.log(D.a / D.sig); D["lb"] = np.log(D.b / D.sig)
    D["p0"] = D.b / (D.a + D.b)
    D["isliq"] = (D.kind == "liq").astype(int)
    return D


def boot(day, fn, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ud = np.unique(day); g = {u: np.flatnonzero(day == u) for u in ud}
    return [round(float(x), 4) for x in np.nanquantile([fn(np.concatenate([g[u] for u in rng.choice(ud, len(ud))])) for _ in range(n)], [.025, .975])]


def pnl(D, s):
    """방향 s(+1 위·−1 아래) 진입, 목표 = 그쪽 레벨, 손절 = 반대 레벨, 72h. bp, 비용 후. 같은 분 둘 다 = 손절."""
    tgt = np.where(s > 0, D.a, D.b) * 100                                          # bp
    stp = np.where(s > 0, D.b, D.a) * 100
    code = D.code.to_numpy()
    win = code == s
    loss = (code == -s) | (code == 2)
    to = code == 0
    tmo = s * np.log(D.px / D.cp).to_numpy() * 1e4
    return np.where(win, tgt - C_IN, np.where(loss, -stp - C_IN - C_OUT, np.where(to, tmo - C_IN - C_OUT, np.nan)))


def test():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss
    D = feats(pd.read_parquet(f"{OUT}/panel.parquet"))
    out = {"flags": {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "future_rows_used_for_entry": False}}
    T_ = D[D.code.isin([1, -1])].copy(); T_["y"] = (T_.code == 1).astype(int)
    out["outcome_share"] = {k: D[D.kind == k].code.value_counts(normalize=True).round(4).to_dict() for k in ("liq", "fake")}
    # D1 보정
    bins = [0, .2, .35, .5, .65, .8, 1]
    cal = T_[T_.split == "TEST"].groupby(["kind", pd.cut(T_.p0, bins)], observed=True).y.agg(["mean", "size"])
    out["D1_calibration_test"] = {f"{k}|{iv}": [round(float(r["mean"]), 3), int(r["size"])] for (k, iv), r in cal.iterrows()}
    out["D1_closer_first"] = {k: round(float(((T_.kind == k) & ((T_.p0 > .5) == (T_.y == 1))).sum() / (T_.kind == k).sum()), 4) for k in ("liq", "fake")}
    # D2 기하 모델(가짜 레벨로 적합)
    tr_f = (T_.split == "TRAIN") & (T_.kind == "fake")
    G = LogisticRegression(C=10.0, max_iter=1000).fit(T_.loc[tr_f, GEO], T_.loc[tr_f, "y"])
    T_["pG"] = G.predict_proba(T_[GEO])[:, 1]
    out["D2"] = {"geo_coef": [round(float(x), 3) for x in G.coef_[0]]}
    for k in ("liq", "fake"):
        m = (T_.split == "TEST") & (T_.kind == k)
        out["D2"][k] = {"auc_p0": round(float(roc_auc_score(T_.y[m], T_.p0[m])), 4), "auc_G": round(float(roc_auc_score(T_.y[m], T_.pG[m])), 4),
                        "logloss_p0": round(float(log_loss(T_.y[m], T_.p0[m].clip(.01, .99))), 4), "logloss_G": round(float(log_loss(T_.y[m], T_.pG[m])), 4),
                        "mean_y_minus_pG": round(float((T_.y[m] - T_.pG[m]).mean()), 4)}
    L = T_[T_.kind == "liq"].copy()
    tr_l, te_l = (L.split == "TRAIN").to_numpy(), (L.split == "TEST").to_numpy()
    off = np.log(L.pG / (1 - L.pG)).to_numpy()
    X1 = np.c_[off, L.wr]
    LW = LogisticRegression(C=10.0, max_iter=1000).fit(X1[tr_l], L.y[tr_l])
    p1 = LW.predict_proba(X1)[:, 1]
    yt, dte = L.y.to_numpy()[te_l], L.day.to_numpy()[te_l]
    pg, pw = L.pG.to_numpy()[te_l], p1[te_l]
    ll = lambda y, p: float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    out["D2"]["liq_size_on_top_of_geo"] = {"coef_wr": round(float(LW.coef_[0][1]), 4), "test_logloss_gain": round(ll(yt, pg) - ll(yt, pw), 5),
                                           "gain_ci": boot(dte, lambda ix: ll(yt[ix], pg[ix]) - ll(yt[ix], pw[ix]))}
    # D3 경제: 가까운 쪽
    Lall = D[D.kind == "liq"].copy()
    s_close = np.where(Lall.a < Lall.b, 1, -1)
    pl = pnl(Lall, s_close)
    out["D3_closer_trade"] = {}
    for sp in SPL:
        m = (Lall.split == sp).to_numpy() & np.isfinite(pl)
        out["D3_closer_trade"][sp] = {"n": int(m.sum()), "win": round(float((Lall.code.to_numpy()[m] == s_close[m]).mean()), 3), "net_bp": round(float(pl[m].mean()), 2),
                                      "ci": boot(Lall.day.to_numpy()[m], lambda ix: pl[m][ix].mean())}
    # U 업그레이드
    FE = GEO + STATE + LIQF + ["isliq"]
    trm = T_.split == "TRAIN"
    M = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0,
                                       random_state=0).fit(T_.loc[trm, FE], T_.loc[trm, "y"])
    out["U"] = {}
    for k in ("liq", "fake"):
        m = (T_.split == "TEST") & (T_.kind == k)
        pM = M.predict_proba(T_.loc[m, FE])[:, 1].clip(1e-4, 1 - 1e-4)
        y_, pG_ = T_.y[m].to_numpy(), T_.pG[m].to_numpy()
        out["U"][k] = {"auc_M": round(float(roc_auc_score(y_, pM)), 4), "auc_G": round(float(roc_auc_score(y_, pG_)), 4),
                       "logloss_gain_vs_G": round(ll(y_, pG_) - ll(y_, pM), 5),
                       "gain_ci": boot(T_.day[m].to_numpy(), lambda ix: ll(y_[ix], pG_[ix]) - ll(y_[ix], pM[ix]))}
    Lall["pM"] = M.predict_proba(Lall[FE])[:, 1]
    a, b, pM = Lall.a.to_numpy(), Lall.b.to_numpy(), Lall.pM.to_numpy()
    edge_up = pM - (b + 0.08) / (a + b)                                             # 위로 갈 때 손익분기 대비 여유
    edge_dn = (1 - pM) - (a + 0.08) / (a + b)
    res = {}
    for mg in (0.0, 0.02, 0.05, 0.1):
        s = np.where(edge_up > mg, 1, np.where(edge_dn > mg, -1, 0))
        pl = pnl(Lall, np.where(s == 0, 1, s))
        r = {}
        for sp in SPL:
            m = (Lall.split == sp).to_numpy() & (s != 0) & np.isfinite(pl)
            r[sp] = {"n": int(m.sum()), "net_bp": round(float(pl[m].mean()), 2) if m.any() else None,
                     "ci": boot(Lall.day.to_numpy()[m], lambda ix: pl[m][ix].mean()) if m.sum() > 50 else None}
        res[mg] = r
    best = max(res, key=lambda mg: res[mg]["VAL"]["net_bp"] if res[mg]["VAL"]["net_bp"] is not None and res[mg]["VAL"]["n"] > 50 else -1e9)
    out["U_trade"] = {"by_margin": res, "chosen_margin_on_VAL": best}
    tb = res[best]["TEST"]
    out["U_pass"] = bool(tb["ci"] is not None and tb["ci"][0] > 0 and res[best]["VAL"]["net_bp"] > 0)
    json.dump(out, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(out, ensure_ascii=False, default=str, indent=1))


def selftest():
    h = np.array([10, 10, 10.5, 12, 10]); l = np.array([10, 9.5, 9.9, 9.0, 10]); c = np.array([10, 10, 10, 10, 10.])
    code, tt, px = first_touch(h, l, c, np.array([0, 0, 0]), np.array([10.4, 11.0, 13.0]), np.array([9.4, 9.6, 8.0]), 4)
    assert list(code) == [1, -1, 0] and list(tt) == [2, 1, 4] and px[0] == 10.4 and px[1] == 9.6     # 위 먼저 · 아래 먼저 · 안 닿음
    code, _, _ = first_touch(h, l, c, np.array([2]), np.array([11.0]), np.array([9.5]), 2)
    assert code[0] == 2                                                             # 같은 분 둘 다
    D = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0], "code": [1, -1], "px": [0, 0], "cp": [1, 1]})
    assert np.allclose(pnl(D, np.array([1, 1])), [100 - 4, -200 - 8])
    print("selftest OK")


if __name__ == "__main__":
    {"--build": build, "--test": test, "--selftest": selftest}[sys.argv[1]]()
