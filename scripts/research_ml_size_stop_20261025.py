#!/usr/bin/env python3
"""크기·손절 머신러닝(24h 최대 역행폭 90% 분위 LightGBM) vs 규칙(2배·변동성 + −5%) — 사전 등록 docs/prereg/ml_size_stop_20261025.md (77634f9f).

    python3 scripts/research_ml_size_stop_20261025.py --selfcheck   # T1 라벨 재구현 · T2 경계 · T3 시드
    python3 scripts/research_ml_size_stop_20261025.py               # C1 · E1 · E2 · E3

5분봉·원장 도우미는 research_system_trend_ruleb_20261025 를 그대로 쓴다. 산출 tmp/ml_size_stop_20261025/.
"""
import argparse, json, math, pathlib, sys
import numpy as np, pandas as pd, lightgbm as lgb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import research_system_trend_ruleb_20261025 as SYS   # noqa: E402

OUT = SYS.ROOT / "tmp" / "ml_size_stop_20261025"
H, Q, BUDGET, LCAP, TAKER = 288, 0.9, 0.10, 3.0, 4e-4
TRAIN = ("2022-01-08", "2024-12-30 23:00"); CAL = ("2025-01-01", "2025-12-30 23:00"); TEST = ("2026-01-01", "2026-10-22 23:00")
FEATS = ["rv1h", "rv4h", "rv24h", "rv7d", "park24", "rv_ratio", "r1h", "r4h", "r24h", "r7d", "dist_adv", "jump24", "hour", "dow", "coin", "s"]
PARAMS = dict(objective="quantile", alpha=Q, num_leaves=31, learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.8, verbose=-1)


def bar_feats(x):
    """봉 i(시작 시각 색인) 끝까지만 쓰는 피쳐 · 방향 무관 부분. 시가는 안 쓴다(가짜 시가 함정)."""
    c, h, l = x.c, x.h, x.l; lr = np.log(c).diff()
    f = pd.DataFrame(index=x.index)
    for nm, w in (("rv1h", 12), ("rv4h", 48), ("rv24h", 288), ("rv7d", 2016)):
        f[nm] = np.sqrt((lr ** 2).rolling(w).mean() * 288)          # 일 단위로 맞춘 실현변동
    f["park24"] = np.sqrt((np.log(h / l) ** 2).rolling(288).mean() / (4 * math.log(2)) * 288)
    f["rv_ratio"] = f.rv4h / f.rv7d
    for nm, w in (("r1h", 12), ("r4h", 48), ("r24h", 288), ("r7d", 2016)):
        f[nm] = c / c.shift(w) - 1
    f["lo24"] = c / l.rolling(288).min() - 1; f["hi24"] = h.rolling(288).max() / c - 1
    f["jump24"] = lr.abs().rolling(288).max()
    end = x.index + pd.Timedelta("5min"); f["hour"] = end.hour; f["dow"] = end.dayofweek
    return f


def labels(x):
    """라벨: 봉 i 종가 진입 → 봉 i+1..i+288 의 최저 저가·최고 고가(벡터판)."""
    fmin = x.l[::-1].rolling(H).min()[::-1].shift(-1); fmax = x.h[::-1].rolling(H).max()[::-1].shift(-1)
    return pd.DataFrame({"mae_long": 1 - fmin / x.c, "mae_short": fmax / x.c - 1, "c_exit": x.c.shift(-H)}, index=x.index)


def label_loop(x, i):
    seg_l, seg_h = x.l.values[i + 1:i + 1 + H], x.h.values[i + 1:i + 1 + H]
    return 1 - seg_l.min() / x.c.values[i], seg_h.max() / x.c.values[i] - 1


def rows_for(sym, ci, x, f, lab, at=None):
    """정시(또는 at 위치) 봉의 롱·숏 두 행(롱 먼저)."""
    idx = np.flatnonzero((x.index + pd.Timedelta("5min")).minute == 0) if at is None else np.asarray(at)
    out = []
    for s in (1, -1):
        r = f.iloc[idx].copy(); r["s"] = s; r["coin"] = ci
        for k in ("r1h", "r4h", "r24h", "r7d"):
            r[k] = s * r[k]
        r["dist_adv"] = r.lo24 if s > 0 else r.hi24
        r["mae"] = (lab.mae_long if s > 0 else lab.mae_short).iloc[idx].values
        r["move"] = s * (lab.c_exit.iloc[idx].values / x.c.values[idx] - 1)
        r["t"] = x.index[idx] + pd.Timedelta("5min"); r["i"] = idx; r["sym"] = sym
        out.append(r)
    return pd.concat(out)


def rule_L(x):
    """규칙 크기 L = min(2, 2/lvl): lvl = 24h 실현변동 ÷ 직전 30개 00:00 UTC 같은 값 중앙(시스템 시험과 같은 정의)."""
    lr = np.log(x.c).diff(); rv = np.sqrt((lr ** 2).rolling(288).sum())
    end = x.index + pd.Timedelta("5min"); mid = rv[(end.hour == 0) & (end.minute == 0)]
    ref = mid.rolling(30).median().reindex(x.index).ffill()
    return np.minimum(2.0, 2.0 / (rv / ref))


def build():
    data = {s: SYS.bars5(s)[0] for s in SYS.COINS}
    parts = []
    for ci, (s, x) in enumerate(data.items()):
        r = rows_for(s, ci, x, bar_feats(x), labels(x))
        r["L_rule"] = rule_L(x).iloc[r.i.values].values
        parts.append(r)
    df = pd.concat(parts).dropna(subset=FEATS + ["mae"]).reset_index(drop=True)
    df["y"] = np.log(df.mae.clip(lower=1e-4))
    return data, df


def split(df, a):
    return df[(df.t >= a[0]) & (df.t <= a[1])]


def pinball(y, q):
    d = y - q; return float(np.mean(np.maximum(Q * d, (Q - 1) * d)))


def fit(tr, seed=0):
    return lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(tr[FEATS], tr.y), num_boost_round=500)


def ml_size_stop(qhat_log, k):
    stop = np.clip(k * np.exp(qhat_log), 0.01, 0.15); return np.minimum(LCAP, BUDGET / stop), stop


def trade_R(L, stop, mae, move):
    hit = mae > stop
    return np.where(hit, -L * stop, L * move) - 2 * TAKER * L, hit


def cvar(R, a=0.05):
    v = np.sort(R); n = max(1, int(len(v) * a)); return float(v[:n].mean())


def eff(L, R):
    return float(np.mean(L) / abs(cvar(R)))


def ledger_replay_stops(L, K, Ls, stops):
    """SYS.ledger_replay 와 같은 근사 · 손절 폭만 왕복마다 다르게."""
    T = K.t.values; Hh, Ll, Cc = K.h.values, K.l.values, K.c.values; mi = lambda ms: int(np.searchsorted(T, ms // 60000 * 60000))
    out = []
    for i, r in enumerate(L.itertuples()):
        scale = Ls[i] * r.E / r.peak; s, e, st = r.s, r.entry_price, stops[i]; i0, i1 = mi(r.entry_time), mi(r.exit_time)
        kb = r.exit_price / Cc[min(i1, len(Cc) - 1)]; px = None
        for j in range(i0 + 1, i1 + 1):
            if s > 0 and Ll[j] * kb <= e * (1 - st): px = e * (1 - st); break
            if s < 0 and Hh[j] * kb >= e * (1 + st): px = e * (1 + st); break
        out.append(r.R * scale if px is None else scale * (s * (px / e - 1) * r.Nin - 0.5 * r.commission - TAKER * r.Nin) / r.E)
    return np.array(out)


def survival(L, R, nb=2000, seed=0):
    day = pd.to_datetime(L.entry_time, unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul").dt.date
    days = pd.date_range(min(day), max(day)).date; idx = {d: np.flatnonzero(day.values == d) for d in days}
    rng = np.random.default_rng(seed); dd = []
    for _ in range(nb):
        seq = np.concatenate([R[idx[days[j]]] for j in rng.integers(0, len(days), 365)])
        eq = np.cumprod(1 + seq); dd.append(min((eq / np.maximum.accumulate(eq) - 1).min(), eq.min() - 1) if len(eq) else 0)
    return float(np.mean(np.array(dd) <= -0.3))


def selfcheck(data, df):
    x = data["ETH"]; lab = labels(x); rng = np.random.default_rng(0)
    for i in rng.integers(3000, len(x) - H - 1, 2000):
        a, b = label_loop(x, i)
        assert abs(a - lab.mae_long.iat[i]) < 1e-12 and abs(b - lab.mae_short.iat[i]) < 1e-12
    print("T1 라벨 벡터판 = 루프판 (ETH 2,000 표본)")
    # T2: 피쳐 창 끝(봉 i) < 라벨 시작(봉 i+1) — labels() 의 shift(-1) 로 보장. 가짜판 = 피쳐를 한 봉 미래(i+1)로
    tr, ca = split(df, TRAIN), split(df, CAL)
    real = pinball(ca.y, fit(tr).predict(ca[FEATS]))
    fk = pd.concat([rows_for(s, ci, x_, bar_feats(x_).shift(-1), labels(x_)) for ci, (s, x_) in enumerate(data.items())])
    fk = fk.dropna(subset=FEATS + ["mae"]); fk["y"] = np.log(fk.mae.clip(lower=1e-4))
    fp = pinball(split(fk, CAL).y, fit(split(fk, TRAIN)).predict(split(fk, CAL)[FEATS]))
    print(f"T2 2025 핀볼 실제 {real:.5f} · 한 봉 미래 피쳐 {fp:.5f} ({(fp / real - 1) * 100:+.2f}%)"); assert fp < real
    # T2b(수정 1): 24h 라벨엔 한 봉 누수가 작아 T2 가 둔감 → 봉 i 에서 자른 데이터로 피쳐를 다시 계산해 같아야 한다(인과성 재구성)
    full = bar_feats(x)
    for i in rng.integers(3000, len(x) - 1, 200):
        cut = bar_feats(x.iloc[max(0, i - 2100):i + 1]).iloc[-1]
        assert np.allclose(cut.values.astype(float), full.iloc[i].values.astype(float), rtol=1e-9, equal_nan=True), i
    print("T2b 봉 i 에서 자른 데이터로 다시 계산한 피쳐 = 전체 데이터 피쳐 (200 표본)")
    s1 = pinball(ca.y, fit(tr, seed=1).predict(ca[FEATS]))
    print(f"T3 시드 0 {real:.5f} · 시드 1 {s1:.5f}"); assert abs(s1 / real - 1) < 0.02
    print("SELFCHECK PASS")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    data, df = build()
    print("행", len(df), {k: len(split(df, v)) for k, v in (("train", TRAIN), ("cal", CAL), ("test", TEST))})
    if a.selfcheck:
        return selfcheck(data, df)
    tr, ca, te = split(df, TRAIN), split(df, CAL), split(df, TEST).copy()
    m = fit(tr)
    k = float(np.quantile(ca.mae / np.exp(m.predict(ca[FEATS])), Q))   # 2025 보정: 넘는 비율 10%
    te["L_ml"], te["stop_ml"] = ml_size_stop(m.predict(te[FEATS]), k)
    c1 = {s: float((g.mae > g.stop_ml).mean()) for s, g in te.groupby("sym")}
    # E2 무작위 진입
    Lm, Lr = te.L_ml.values, te.L_rule.values
    Rm, hm = trade_R(Lm, te.stop_ml.values, te.mae.values, te.move.values)
    Rr, hr = trade_R(Lr, np.full(len(te), 0.05), te.mae.values, te.move.values)
    day = te.t.dt.floor("D").values; ud = np.unique(day); gi = {d: np.flatnonzero(day == d) for d in ud}
    rng = np.random.default_rng(7); diffs = []
    for _ in range(2000):
        ix = np.concatenate([gi[d] for d in rng.choice(ud, len(ud))])
        diffs.append(eff(Lm[ix], Rm[ix]) - eff(Lr[ix], Rr[ix]))
    e2 = {"eff_ml": eff(Lm, Rm), "eff_rule": eff(Lr, Rr), "diff_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
          "mean_L_ml": float(Lm.mean()), "mean_L_rule": float(Lr.mean()), "cvar5_ml": cvar(Rm), "cvar5_rule": cvar(Rr),
          "stop_hit_ml": float(hm.mean()), "stop_hit_rule": float(hr.mean()), "mean_R_ml": float(Rm.mean()), "mean_R_rule": float(Rr.mean()),
          "median_stop_ml": float(np.median(te.stop_ml)), "stop_ml_p10_p90": [float(np.percentile(te.stop_ml, 10)), float(np.percentile(te.stop_ml, 90))]}
    # E1·E3 실원장 — 진입 시각 직전에 끝난 ETH 5분봉의 피쳐
    L = SYS.ledger_trips(); K = pd.read_parquet(SYS.OUT / "eth1m_ledger.parquet")
    x = data["ETH"]; ends = (x.index + pd.Timedelta("5min")).values
    at = [int(np.searchsorted(ends, np.datetime64(pd.Timestamp(t, unit="ms")), side="right") - 1) for t in L.entry_time]
    rr = rows_for("ETH", 0, x, bar_feats(x), labels(x), at=at)
    rl, rs = rr[rr.s == 1].reset_index(drop=True), rr[rr.s == -1].reset_index(drop=True)
    X = rl[FEATS].where(pd.Series(L.s.values > 0), rs[FEATS]).astype(float)
    Lml, Sml = ml_size_stop(m.predict(X), k)
    Lrule = np.minimum(2.0, 2.0 / SYS.ledger_lvl(L, K))
    Rml = ledger_replay_stops(L, K, Lml, Sml); Rrule = ledger_replay_stops(L, K, Lrule, np.full(len(L), 0.05))
    (gm, mm), (gr, mr) = SYS.growth_mdd(Rml), SYS.growth_mdd(Rrule)
    e1 = {"ml_growth": gm, "ml_mdd": mm, "rule_growth": gr, "rule_mdd": mr, "ml_L_median": float(np.median(Lml)), "rule_L_median": float(np.median(Lrule)),
          "ml_stop_median": float(np.median(Sml)), "ml_worst": float(Rml.min()), "rule_worst": float(Rrule.min())}
    e3 = {"ml_P_dd30": survival(L, Rml), "rule_P_dd30": survival(L, Rrule)}
    rep = {"prereg": "docs/prereg/ml_size_stop_20261025.md@77634f9f", "k_cal": k, "rows": len(df), "test_rows": len(te),
           "pinball_2026_ml": pinball(te.y, m.predict(te[FEATS])), "C1_hit_by_coin": c1, "E1": e1, "E2": e2, "E3": e3,
           "importance_gain": dict(sorted(zip(FEATS, m.feature_importance("gain").round(0).tolist()), key=lambda z: -z[1]))}
    rep["gates"] = {"C1": all(0.07 <= v <= 0.13 for v in c1.values()), "E1": gm >= gr and mm >= mr, "E2": e2["diff_ci"][0] > 0, "E3": e3["ml_P_dd30"] <= e3["rule_P_dd30"]}
    rep["pass"] = all(rep["gates"].values())
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); m.save_model(str(OUT / "model.txt"))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
