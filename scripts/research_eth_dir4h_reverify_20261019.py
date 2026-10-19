"""ETH 4h 방향 모델 재검증 (2026-10-19, 코드 리뷰 후속 -- 로컬 CPU 만, 서버·거래소 0).

A. 문턱 버그 크기: 라이브 옛 규칙(매달 학습 행 in-sample 예측 20/80 분위) vs 새 규칙(과거 표본 밖 예측 확장 20/80 분위).
   같은 예측(nomd 65피쳐 · 365일 · 매달 재학습 · 시드 0 = pred_ETH_combo_4h_nomd.npy)에 두 문턱만 바꿔 채점.
   옛 문턱은 라이브 train() 그대로 시드 0 재적합(X[tr][::2] 학습 · X[tr][::7] in-sample 예측)으로 다시 만든다 -- 그 달 예측 상관 ≈1 확인.
   보조: 라이브 실제(시드 0·1·2 평균 예측 + 평균 in-sample 문턱) · 새 규칙의 라이브 판(달 시작까지 표본 밖 분위, 달 안 고정).
   채점 = 연구 final_model 위상 8개(6봉 간격) 1/8 포트폴리오 일 손익 · 비용 4bp · 5일 블록 부트스트랩(block_ci) · 샤프 = 일 평균/표준편차·√365.
   결합(1.47) = replicate_combo(nomd) 의 5분봉 지정가 근사 HGB + 일봉 추세 5표 + UTC 개장 되돌림 위험 맞춤 -- 문턱만 바꿔 재계산.
B. DL 후속: v2_hyb(GRU 원시 창 + 65피쳐) vs v2_mlp(65피쳐만) vs HGB 기준(pred_base, 시드 3) --
   공통 인과 문턱 = 각 모델 자기 예측의 직전 30일 20/80 분위(리뷰어 규칙) · 같은 위상 8 채점 · 짝 차 5일 블록 CI.
   IC(피어슨, 모든 봉) 원 예측 vs ×σ(σ = P_rv1d·√48; DL 은 y/σ 를 배웠으니 ×σ = bp 단위) · hyb−mlp IC 차 5일 블록 CI(일 충분통계 재표집).

사전 판정(결과 전 고정, 주 = 2025-01~2026-09 · 보조 = 2023-02~2024-12):
  A1 짝 차(새−옛) 일 손익 CI 하한 > 0 → «버그가 해쳤다»(수정 = 이득) · 상한 < 0 → «버그가 도왔다» · 0 포함 → «구분 불가(중립)».
  A2 «+24.6bp·1.47 이 고친 라이브 규칙에 적용되나» = 새 규칙이 곧 연구 규칙이므로, 라이브 판(달 고정) − 연구 판(봉마다 확장) 짝 차
     CI 가 0 을 품으면 «적용됨». 옛 규칙 수치가 연구 수치와 다르면 «옛 화면 신호는 그 수치의 근거가 아니었다».
  B1 hyb−mlp 일 손익 짝 차 CI 하한 > 0 이거나 IC 차(원·×σ 중 하나라도) CI 하한 > 0 → «원시 창이 65피쳐 밖 정보를 더한다» = 기존 주장 기각.
     아니면 주장 유지. DL−HGB 짝 차 CI 상한 < 0 → «HGB 가 실제로 낫다».

결과(10-19 실행, 주 2025-01~2026-09 · 4bp · 위상 8 · 5일 블록 CI):
  재적합 45달 그 달 예측 상관 최소 1.000000 (라이브 train() = 저장 예측). 옛 in-sample 문턱이 더 넓다(2026-02 −46.5/+45.5 vs 표본 밖 −33.3/+37.7).
  신호 비율 옛 롱 19.2%·숏 17.1% · 새 22.6%·24.0%. 🔴달별 쏠림은 버그 탓이 아니다: 2026-02 옛 롱 60.5%/숏 1.5% → 새 65.0%/3.4%
     (2026-03 옛 1.4%/39.8% → 새 2.3%/55.0%) = 예측 자체가 달마다 한쪽으로 움직임. 버그는 신호 «빈도»만 줄였다(36% vs 47%).
  일 손익(시드 0) 옛 +17.5[+0.7,+37.0] 샤프 1.62 · 새(연구) +17.1[−1.8,+38.4] 1.46 · 새(라이브 달 고정) +17.1[−1.9,+38.4] 1.45.
     라이브 실제 시드 3: 옛 +19.1[+2.7,+37.8] 1.81 · 새 +18.7[−0.1,+39.9] 1.60.
  A1 새−옛 −0.3[−4.8,+4.3] (2025 −0.8 · 2026 +0.3 · 2023-24 −0.4[−2.6,+2.2]) · 시드 3 −0.3[−5.4,+4.9] → 중립(구분 불가).
  A2 라이브 판 − 연구 판 −0.05[−0.56,+0.44] → 적용됨. 결합 샤프(2023-03~) 연구 규칙 1.47 정확 재현 · 라이브 달 고정 1.46 · 옛 1.52.
     🔴«+24.6bp» 는 전체 피쳐 판 숫자(현재 final_model.json 4bp +20.5) -- 라이브 nomd 는 +17~19bp/일, 5일 블록 CI 하한 ≈0(−1.9~−0.1).
  B(공통 문턱 = 직전 30일 20/80): HGB +12.6[−4.5,+31.8] · hyb +5.1[−7.9,+17.5] · mlp −8.9[−24.5,+5.2].
  B1 hyb−mlp +14.0[+3.6,+25.4](시드 5/5 +, 2026 +4.9[−8.5,+19.3]) · IC 원 +.023[+.002,+.046] · ×σ +.026[+.005,+.048]
     → 사전 규칙상 «원시 창 증분 0» 주장 기각(원시 창은 같은 절차 65피쳐 MLP 보다 낫다). 단 hyb−HGB −7.5[−26.7,+8.8] · IC −.031[−.073,+.009]
     → HGB 를 넘지는 못함(구분 불가) · mlp−HGB −21.5[−39.5,−5.6] = HGB 가 확실히 낫다. ×σ 는 모두 IC 를 올림(HGB .063→.076) -- 순위 결론 불변.

    nice -n 15 python scripts/research_eth_dir4h_reverify_20261019.py           # 전부(재적합 캐시 tmp/dir4h_reverify_20261019/refit.npz)
    nice -n 15 python scripts/research_eth_dir4h_reverify_20261019.py --selfcheck
플래그: fresh_forward_bar_by_bar=true(월별 재학습 표본 밖 예측) · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_eth_dir4h_20261015 import FEATS, HGB, H  # noqa: E402
HGB = {**HGB, "early_stopping": "auto"}   # 10-19 이전 라이브(A) 설정 고정 -- 라이브는 B2(early_stopping=False)로 바뀌었다

WT = Path("/home/kbj20/crypto-scalping/.claude/worktrees")
BD = WT / "liquidation-price-feature-14eceb/tmp/best_dir_20261010"
DL = WT / "4hour-forecast-accuracy-5c6776/tmp"
OUT = ROOT / "tmp/dir4h_reverify_20261019"
COST = 4.0
PERIODS = (("2025~", "2025-01-01", "2026-09-15"), ("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", "2026-09-15"),
           ("2023-24", "2023-02-01", "2025-01-01"))


def block_ci(x: np.ndarray, b: int = 5, n: int = 4000, seed: int = 7) -> tuple[float, float]:
    """= research_eth_dir4h_other_project_feats_20261018.block_ci 그대로."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(x) / b))
    st = rng.integers(0, len(x) - b + 1, size=(n, k))
    means = x[(st[:, :, None] + np.arange(b)).reshape(n, -1)[:, :len(x)]].mean(1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def side(p, lo, hi) -> np.ndarray:
    return np.where(p >= hi, 1, np.where(p <= lo, -1, 0))   # NaN 문턱 = 0


def phase_daily(pred, lo, hi, ts, y, a, b) -> pd.Series:
    """연구 final_model: 위상 8(off 0,6,…,42) 각자 48봉 간격 비겹침 매매, 1/8 씩 → 일 합. 기간 안 모든 날(신호 없는 날 0)."""
    rows = []
    for off in range(0, H, 6):
        idx = np.arange(off, len(y), H)
        s = side(pred[idx], lo[idx], hi[idx])
        m = (ts[idx] >= np.datetime64(a)) & (ts[idx] < np.datetime64(b)) & (s != 0) & np.isfinite(y[idx])
        rows.append(pd.Series((s[m] * y[idx][m] - COST) / 8, index=pd.to_datetime(ts[idx][m])))
    g = pd.concat(rows)
    last = min(pd.Timestamp(b), pd.Timestamp(ts[np.isfinite(y)].max())) - pd.Timedelta("1D")
    return g.groupby(g.index.floor("D")).sum().reindex(pd.date_range(a, last, freq="D"), fill_value=0.0)


def summ(d: pd.Series) -> dict:
    lo, hi = block_ci(d.to_numpy())
    return {"bp_day": round(float(d.mean()), 2), "ci": [round(lo, 2), round(hi, 2)], "sharpe": round(float(d.mean() / d.std() * np.sqrt(365)), 2),
            "mdd_bp": int((d.cumsum() - d.cumsum().cummax()).min())}


def diff(d1: pd.Series, d0: pd.Series) -> dict:
    x = (d1 - d0).to_numpy()
    lo, hi = block_ci(x)
    return {"mean": round(float(x.mean()), 2), "ci": [round(lo, 2), round(hi, 2)]}


def expanding_q(pred, minp=8640):
    s = pd.Series(pred)
    return s.shift(1).expanding(minp).quantile(.2).to_numpy(), s.shift(1).expanding(minp).quantile(.8).to_numpy()


def month_fixed_q(pred, ts, months):
    """라이브 thresholds(): 달 시작 전 모든 표본 밖 예측의 20/80 분위, 달 안 고정(8640행 미만이면 없음)."""
    lo, hi = np.full(len(ts), np.nan), np.full(len(ts), np.nan)
    for m in months:
        past = pred[(ts < np.datetime64(m)) & np.isfinite(pred)]
        if len(past) >= 8640:
            te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
            lo[te], hi[te] = np.quantile(past, [.2, .8])
    return lo, hi


# ── A ────────────────────────────────────────────────────────────────────
def refit(P: pd.DataFrame, months) -> dict:
    """라이브 train() 의 적합·in-sample 문턱 부분만(시드 0·1·2). 캐시."""
    f = OUT / "refit.npz"
    if f.exists():
        return dict(np.load(f))
    from sklearn.ensemble import HistGradientBoostingRegressor
    ts, y, X = P.ts.to_numpy(), P.y_4h.to_numpy(), P[FEATS].to_numpy(dtype=np.float32)
    pred_s = np.full((3, len(ts)), np.nan)
    q = np.full((len(months), 4, 2), np.nan)                      # [달, 시드0·1·2·평균, lo·hi]
    for i, m in enumerate(months):
        cut = np.datetime64(m) - np.timedelta64(5 * H, "m")
        tr = (ts < cut) & (ts >= cut - np.timedelta64(365, "D")) & np.isfinite(y)
        te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
        pts = []
        for s in (0, 1, 2):
            mdl = HistGradientBoostingRegressor(random_state=s, **HGB).fit(X[tr][::2], y[tr][::2])
            pred_s[s, te] = mdl.predict(X[te]); pts.append(mdl.predict(X[tr][::7]))
            q[i, s] = np.quantile(pts[-1], [.2, .8])
        q[i, 3] = np.quantile(np.mean(pts, 0), [.2, .8])
        print("  재적합", m.strftime("%Y-%m"), q[i, 0].round(1), flush=True)
    np.savez(f, pred_s=pred_s, q=q)
    return {"pred_s": pred_s, "q": q}


def month_arr(q2, ts, months):
    lo, hi = np.full(len(ts), np.nan), np.full(len(ts), np.nan)
    for i, m in enumerate(months):
        te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
        lo[te], hi[te] = q2[i]
    return lo, hi


def rates(pred, lo, hi, ts, a, b) -> dict:
    m = (ts >= np.datetime64(a)) & (ts < np.datetime64(b)) & np.isfinite(pred)
    s = side(pred[m], lo[m], hi[m])
    mon = pd.DataFrame({"L": s == 1, "S": s == -1}, index=pd.to_datetime(ts[m]).to_period("M")).groupby(level=0).mean()
    gap = (mon.L - mon.S).abs().sort_values(ascending=False)
    return {"long": round(float((s == 1).mean()), 3), "short": round(float((s == -1).mean()), 3),
            "month_long_range": [round(float(mon.L.min()), 3), round(float(mon.L.max()), 3)],
            "month_short_range": [round(float(mon.S.min()), 3), round(float(mon.S.max()), 3)],
            "worst5": {str(k): [round(float(mon.L[k]), 3), round(float(mon.S[k]), 3)] for k in gap.index[:5]}}


def combo(pred, lo, hi) -> dict:
    """research replicate_combo(ETH, nomd) 의 채점부 -- 문턱만 인자로."""
    sys.path.insert(0, str(WT / "liquidation-price-feature-14eceb/scripts"))
    import research_eth_best_direction_model_20261010 as R
    ts = pd.read_parquet(BD / "panel.parquet", columns=["ts"]).ts.to_numpy()
    b = R.load_panel("ETH"); t5 = (b.timestamp.astype("int64") // 10**6).to_numpy(); h5, l5, c5 = b.high.to_numpy(), b.low.to_numpy(), b.close.ffill().to_numpy()

    def leg(T, sd):
        i = np.searchsorted(t5, T); P = c5[i - 1]; f = (l5[i] < P) if sd > 0 else (h5[i] > P)
        return (P, 0.0) if f else (c5[i], 4.0)
    rows = []
    for off in range(0, 48, 6):
        idx = np.arange(off, len(ts), 48); idx = idx[(ts[idx] >= np.datetime64("2023-02-01")) & (idx + 48 < len(ts))]
        for j in idx:
            if not np.isfinite(pred[j]):
                continue
            s = 1 if pred[j] >= hi[j] else (-1 if pred[j] <= lo[j] else 0)
            if s:
                T0 = int((ts[j] + np.timedelta64(5, "m")).astype("datetime64[ms]").astype(np.int64)); pe, ce = leg(T0, s); px, cx = leg(T0 + 14_400_000, -s)
                rows.append((pd.Timestamp(T0, unit="ms").floor("D"), (s * np.log(px / pe) * 1e4 - ce - cx) / 8))
    hgb = pd.Series([r[1] for r in rows], index=[r[0] for r in rows]).groupby(level=0).sum()
    d = pd.Series(c5, index=b.timestamp).resample("1D", label="left").last(); dv = d.to_numpy()
    votes = np.mean([np.sign(dv - np.r_[np.full(L, np.nan), dv[:-L]]) for L in (7, 14, 28, 56, 90)], axis=0); pos = np.where(np.abs(votes) >= .5, np.sign(votes), 0.0)
    tr = pd.Series(pos * np.r_[np.log(dv[1:] / dv[:-1]) * 1e4, np.nan] - np.abs(np.diff(np.r_[0, pos])) * 4, index=d.index + pd.Timedelta("1D"))
    O = pd.read_parquet(BD / "open_panel.parquet").dropna(subset=["f30", "rv1d"]); E = O[O.sym == "ETH"]
    so = np.where(E.f30.abs() > .5 * E.rv1d * np.sqrt(6), -np.sign(E.f30), 0); op = pd.Series(np.where(so != 0, so * E.n4 - 4, 0.0), index=pd.to_datetime(E.day.to_numpy()))
    B = pd.DataFrame({"4h HGB": hgb, "추세": tr, "개장": op}).reindex(pd.date_range("2023-03-02", "2026-09-12", freq="D")).fillna(0.0)
    sd = B.rolling(90, min_periods=60).std().shift(1); W = (sd.mean(1).to_numpy()[:, None] / sd).clip(upper=3.0)
    B["결합"] = (B.iloc[:, :3] * W).sum(1) / 3; B = B[W.notna().all(1)]
    sh = lambda x: round(float(x.mean() / x.std() * np.sqrt(365)), 2)
    return {c: {"sharpe": sh(B[c]), "yearly": {int(Y): sh(v) for Y, v in B[c].groupby(B.index.year)},
                "sharpe_2025~": sh(B[c][B.index >= "2025-01-01"])} for c in ("4h HGB", "결합")}


def part_a() -> dict:
    P = pd.read_parquet(BD / "panel.parquet", columns=["ts", "y_4h"] + FEATS)
    ts, y = P.ts.to_numpy(), P.y_4h.to_numpy()
    p0 = np.load(BD / "pred_ETH_combo_4h_nomd.npy")
    months = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    R = refit(P, months)
    cors = []
    for m in months:
        te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1))) & np.isfinite(p0)
        cors.append(float(np.corrcoef(R["pred_s"][0, te], p0[te])[0, 1]))
    assert min(cors) > 0.9999, f"시드 0 재적합이 저장 예측과 다르다: {min(cors)}"
    p3 = R["pred_s"].mean(0)
    rules = {"old_seed0": (p0, *month_arr(R["q"][:, 0], ts, months)), "new_research_seed0": (p0, *expanding_q(p0)),
             "new_live_seed0": (p0, *month_fixed_q(p0, ts, months)),
             "old_live3seed": (p3, *month_arr(R["q"][:, 3], ts, months)), "new_live3seed": (p3, *month_fixed_q(p3, ts, months))}
    out = {"refit_corr_min": round(min(cors), 6), "rules": {}, "diff": {}, "thresholds_seed0": {}}
    D = {}
    for nm, (p, lo, hi) in rules.items():
        r = {"rates": {lab: rates(p, lo, hi, ts, a, b) for lab, a, b in PERIODS[:1] + PERIODS[3:]}}
        for lab, a, b in PERIODS:
            D[nm, lab] = phase_daily(p, lo, hi, ts, y, a, b); r[lab] = summ(D[nm, lab])
        out["rules"][nm] = r
        print(nm, json.dumps({k: v for k, v in r.items() if k != "rates"}, ensure_ascii=False), "| 2025~ 롱·숏", r["rates"]["2025~"]["long"], r["rates"]["2025~"]["short"], flush=True)
    for a_, b_ in (("new_research_seed0", "old_seed0"), ("new_live_seed0", "old_seed0"), ("new_live_seed0", "new_research_seed0"),
                   ("new_live3seed", "old_live3seed")):
        out["diff"][f"{a_} − {b_}"] = {lab: diff(D[a_, lab], D[b_, lab]) for lab, _, _ in PERIODS}
        print(f"{a_} − {b_}", out["diff"][f"{a_} − {b_}"], flush=True)
    lo_e, hi_e = expanding_q(p0)
    for i, m in enumerate(months):   # 달 첫 봉 기준 문턱 비교
        j = int(np.argmax(ts >= np.datetime64(m)))
        out["thresholds_seed0"][m.strftime("%Y-%m")] = {"insample": R["q"][i, 0].round(2).tolist(), "oos_expanding": [round(float(lo_e[j]), 2), round(float(hi_e[j]), 2)]}
    out["combo"] = {nm: combo(*rules[nm]) for nm in ("new_research_seed0", "old_seed0", "new_live_seed0")}
    print("결합", json.dumps(out["combo"], ensure_ascii=False), flush=True)
    return out


# ── B ────────────────────────────────────────────────────────────────────
def roll30_q(pred, ts):
    s = pd.Series(pred, index=pd.DatetimeIndex(ts))
    q = lambda x: s.rolling("30D", closed="left", min_periods=50).quantile(x).to_numpy()
    lo, hi = q(.2), q(.8)
    return np.where(np.isfinite(pred), lo, np.nan), np.where(np.isfinite(pred), hi, np.nan)


def ic_stats(x, y, day):
    """일별 피어슨 충분통계(n, Σx, Σy, Σx², Σy², Σxy) → (일수, 6)."""
    df = pd.DataFrame({"n": 1.0, "x": x, "y": y, "xx": x * x, "yy": y * y, "xy": x * y}, index=day)
    return df.groupby(level=0).sum().to_numpy()


def pearson(S):
    n, sx, sy, sxx, syy, sxy = np.moveaxis(S, -1, 0)
    return (sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))


def ic_diff_ci(S1, S0, b=5, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(S1) / b))
    ix = (rng.integers(0, len(S1) - b + 1, size=(n, k))[:, :, None] + np.arange(b)).reshape(n, -1)[:, :len(S1)]
    d = pearson(S1[ix].sum(1)) - pearson(S0[ix].sum(1))
    return float(pearson(S1.sum(0)) - pearson(S0.sum(0))), [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)]


def part_b() -> dict:
    Z = np.load(DL / "dir4h_seq_v2_20261018/data.npz")
    ts, y, sig = Z["ts"].astype("datetime64[m]").astype("datetime64[ns]"), Z["y"].astype(float), Z["sig"].astype(float)
    of = DL / "dir4h_otherfeats_20261018"
    M = {"hgb": np.load(of / "pred_base.npz"), "hyb": np.load(of / "pred_v2_hyb.npz"), "mlp": np.load(of / "pred_v2_mlp.npz"),
         "mlp_v1": np.load(of / "pred_mlp.npz")}
    out = {"pnl": {}, "diff": {}, "ic": {}, "ic_diff": {}, "seed_diff_2025~": {}}
    D = {}
    for nm, z in M.items():
        p = z["pred"].astype(float); lo, hi = roll30_q(p, ts)
        out["pnl"][nm] = {}
        for lab, a, b in PERIODS:
            D[nm, lab] = phase_daily(p, lo, hi, ts, y, a, b); out["pnl"][nm][lab] = summ(D[nm, lab])
        print(nm, json.dumps(out["pnl"][nm], ensure_ascii=False), flush=True)
    for a_, b_ in (("hyb", "mlp"), ("hyb", "hgb"), ("mlp", "hgb"), ("mlp_v1", "hgb")):
        out["diff"][f"{a_} − {b_}"] = {lab: diff(D[a_, lab], D[b_, lab]) for lab, _, _ in PERIODS}
        print(f"{a_} − {b_}", out["diff"][f"{a_} − {b_}"], flush=True)
    # 시드별 hyb−mlp(같은 SEEDS 순서로 짝)
    a, b = PERIODS[0][1:]
    sd = []
    for i in range(5):
        dd = []
        for nm in ("hyb", "mlp"):
            p = M[nm]["pred_s"][i].astype(float); dd.append(phase_daily(p, *roll30_q(p, ts), ts, y, a, b))
        sd.append(round(float((dd[0] - dd[1]).mean()), 2))
    out["seed_diff_2025~"]["hyb − mlp"] = sd
    print("시드별 hyb−mlp", sd, flush=True)
    for lab, a, b in PERIODS:
        m = (ts >= np.datetime64(a)) & (ts < np.datetime64(b)) & np.isfinite(y) & np.isfinite(sig) & np.all([np.isfinite(z["pred"]) for z in M.values()], 0)
        day = pd.to_datetime(ts[m]).floor("D")
        S = {(nm, k): ic_stats(z["pred"][m].astype(float) * (sig[m] if k == "sig" else 1.0), y[m], day) for nm, z in M.items() for k in ("raw", "sig")}
        out["ic"][lab] = {f"{nm}_{k}": round(float(pearson(v.sum(0))), 4) for (nm, k), v in S.items()}
        out["ic_diff"][lab] = {f"hyb − mlp {k}": ic_diff_ci(S["hyb", k], S["mlp", k]) for k in ("raw", "sig")}
        out["ic_diff"][lab].update({f"hyb − hgb {k}": ic_diff_ci(S["hyb", k], S["hgb", k]) for k in ("raw", "sig")})
        print(lab, out["ic"][lab], out["ic_diff"][lab], flush=True)
    return out


def selfcheck() -> None:
    x = np.random.default_rng(0).normal(0, 10, 600)
    lo, hi = block_ci(x); assert lo < 0 < hi and block_ci(x + 3)[0] > 0
    assert list(side(np.array([5., 0., -5., 5.]), np.array([-1., -1., -1., np.nan]), np.array([1., 1., 1., np.nan]))) == [1, 0, -1, 0]
    # 위상 8: 매 봉 +10bp 상승 · 늘 롱 → 하루 = 8 위상 × 6 진입 × (10−4)/8 = 36bp
    n = 288 * 6; ts = pd.date_range("2025-01-01", periods=n, freq="5min").to_numpy(); y = np.full(n, 10.0)
    d = phase_daily(np.ones(n), np.zeros(n), np.full(n, .5), ts, y, "2025-01-01", "2025-01-07")
    assert np.allclose(d.to_numpy()[:5], 36.0), d.head()
    # 확장 분위 = 과거만(shift 1)
    lo_, hi_ = expanding_q(np.arange(10.0), minp=5)
    assert np.isnan(lo_[4]) and lo_[5] == np.quantile(np.arange(5.0), .2)
    # 피어슨 충분통계 = np.corrcoef · 같은 입력이면 차 0
    r = np.random.default_rng(1); a_, b_ = r.normal(size=n), r.normal(size=n)
    dd = pd.to_datetime(ts).floor("D")
    S = ic_stats(a_ + b_, b_, dd)
    assert abs(pearson(S.sum(0)) - np.corrcoef(a_ + b_, b_)[0, 1]) < 1e-9 and ic_diff_ci(S, S)[0] == 0
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck()
    if "--selfcheck" not in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        res = {"A": part_a(), "B": part_b()}
        (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("저장", OUT / "report.json")
