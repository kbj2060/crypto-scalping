"""ETH 1h 방향 모델 1단계(2026-10-19, 로컬 CPU 만 · 서버·거래소·GPU 0 · 커밋 안 함).

PRE-REGISTRATION (결과 전 고정, 지시문 그대로):
  Label y_1h = ln(close[t+12]/close[t]) bp, decision = 5m bar t close, hold 1h.
  Walk-forward monthly 2023-01..2026-09: train = prior 365 days, purge = 1h (cut = month − 12 bars), seeds 5 random-drawn constants (record them; HGB has no subsampling so seeds barely matter — still average).
  Arms: A0 HGB(live HGB params) on raw y_1h · A1 HGB on y_1h/σ1h where σ1h = P_rv1d·√12 (bp), prediction × σ1h.
  Signal rules: R20 = past-OOS expanding 20/80 quantiles (shift(1), min 8640 rows) · R10 = same at 10/90.
  Execution: TAKER = 4bp per round trip · MAKER-approx = research leg(): enter at decision close as limit, filled if next 5m bar trades through (long: low < P; short: high > P) at 0bp, else market at that bar close +4bp... use exactly the research replicate_combo leg() semantics for entry and exit (read it), document it.
  Accuracy gate (weapon list): 2025-01..2026-09 IC 5-day-block CI lower > 0 AND 2023-24 IC > 0.
  Trading gate: among 8 combos (2 arms × 2 rules × 2 exec), Holm-adjusted one-sided p (block bootstrap, 5-day) < .05 on 2025~ daily PnL, AND that combo's 2023-24 mean > 0.
  Stage-2 (DL on server GPU) condition: best combo 2025~ IC ≥ .03 AND mean gross edge per trade (before cost) ≥ 2bp. Otherwise close the 1h axis, no GPU.

구현 세부(결과 전 고정):
  · 데이터 = best_dir_20261010/panel.parquet(5분 2022-01-01~2026-09-14, 행 t = 봉 t 마감값) · 65피쳐 FEATS·HGB = live_eth_dir4h_20261015 그대로.
    y_1h·y_4h 를 vision 종가(research load_panel)로 다시 계산해 패널 값과 같음을 확인 · P_r5m[t] = 봉 t 자기 수익(한 봉 밀기 트립와이어) 확인.
  · 학습 = research wf_monthly 와 같이 X[tr][::2] · tr = ts < 달 − 60분 · ≥ 그 − 365일 → 마지막 학습 행 라벨(close[t+12], t+65분에 앎) ≤ 달 시작(assert).
    시드 = secrets.randbelow(2**31) 로 결과 전 1회 추출 = SEEDS. 시드 간 예측 최대 차를 기록.
  · 위상 = 4h 의 phase_daily 일반화: off 0..11 각자 12봉 간격 비겹침 매매, 1/12 씩 → 결정 봉 날짜로 일 합(기간 안 신호 없는 날 0).
    매매/일 = 한 장부(1/12) 기준 = 전체 거래 수/12/일수.
  · MAKER = research replicate_combo leg() 그대로: leg(T) 에서 i = T 에 열리는 봉, P = c[i−1](= 결정 봉 종가 / 청산은 t+12 봉 종가),
    체결 = 그 봉 low < P(매수) · high > P(매도) → 비용 0, 아니면 c[i](그 봉 종가) 시장가 + 4bp «다리당»(왕복 최대 8bp, TAKER 왕복 4bp 보다 보수적).
    진입 = leg(t 마감, s) · 청산 = leg(t 마감 + 1h, −s). 순 = s·ln(px/pe)·1e4 − 다리 비용. 총 우위(비용 전) = s·ln(px/pe)·1e4.
  · IC = 피어슨(ic_stats·pearson, 모든 봉) · CI = 일 충분통계 5일 블록 재표집(4000, seed 7) · 스피어먼은 참고로만.
    A1 의 IC 는 예측×σ1h(bp) 기준. IC 기간 2023-24 = 2023-01-01~ · 매매 기간 2023-24 = 2023-02-01~(문턱 예열).
  · 단측 p = (1 + #{5일 블록 부트 평균 ≤ 0}) / (4000 + 1) · Holm 8개.
  · 2단계 «최고 조합» = 8조합 중 2025~ 일 순손익 평균 최대 · IC 는 그 조합 팔의 2025~ IC · 총 우위 = 그 조합 2025~ 거래당 비용 전 평균.

결과(10-19 실행, 45달 × 2팔 × 5시드, 약 15분):
  정렬: y_1h·y_4h 종가 재계산 = 패널(오차 < 1e-6) · 한 봉 밀기 트립와이어 corr(P_r5m, 봉 t) 1.000 · (봉 t+1) −.017 → 경계 계약 OK.
  🔴사전등록 가정 «시드 거의 무관» 틀림: HGB early_stopping='auto'(행 > 1만 → 무작위 10% 검증 분할) 라 시드 간 예측 최대 차 A0 58.7bp · A1 0.71σ.
     라이브도 같은 HGB 인자 → 같은 성질. 5시드 평균은 그대로 했다.
  IC(피어슨 [5일 블록 CI]) A0: 2023 −.007 · 2024 .026 · 2025 .039 · 2026 .001 · 2025~ .025[+.003,+.047] · 2023-24 .012 → 정확도 문 통과(무기 목록).
                          A1: 2023 .011 · 2024 .033 · 2025 .038 · 2026 .000 · 2025~ .025[−.001,+.053] · 2023-24 .025 → 불통과(CI 하한 −.0007).
     🔴두 팔 모두 2026 IC ≈ 0 (우위는 2024~25 몫).
  매매(2025~ 일 bp [CI] · 샤프 · 2023-24 일 bp · 거래당 비용 전 총 우위 · 매매/일(한 장부)):
     A0 R20 TAKER −25.1[−43.5,−5.8] −2.00 · −21.6 · +1.64 · 10.7   | MAKER −1.8[−20.1,+17.5] −0.14 · −0.7 · +0.14 (양 다리 체결 92%)
     A0 R10 TAKER −14.2[−28.3,+1.2] −1.37 · −8.7 · +1.52 · 5.8    | MAKER −1.6[−15.5,+14.0] −0.16 · +1.9 · −0.03
     A1 R20 TAKER −30.6[−48.4,−11.4] −2.46 · −24.1 · +1.23 · 11.1 | MAKER −6.2[−23.8,+13.2] −0.50 · −1.2 · −0.26
     A1 R10 TAKER −17.5[−31.9,−2.5] −1.76 · −10.8 · +1.09 · 6.1   | MAKER −4.6[−18.5,+10.3] −0.47 · +1.1 · −0.49
     단측 p .57~1.00 · Holm 전부 1.0 → 매매 문 통과 0. 2026 은 8조합 모두 음수(−14~−40bp/일).
     메이커 근사: 체결된 지정가가 신호 총 우위(+1.1~1.6bp)를 거의 다 지움(역선택: 다음 봉이 관통할 때만 체결) → 총 우위 ≈ 0.
  2단계 조건: 최고 조합 A0_R10_MAKER = IC .025 < .03 · 총 우위 −0.03bp < 2bp → 불충족(테이커 총 우위 최대 1.64bp 로도 미달) → 1h 축 종료, GPU 안 씀.

    nice -n 15 python scripts/research_eth_dir1h_stage1_20261019.py              # 전부(예측 캐시 tmp/dir1h_stage1_20261019/pred_A?.npz)
    nice -n 15 python scripts/research_eth_dir1h_stage1_20261019.py --selfcheck
플래그: fresh_forward_bar_by_bar=true(매달 재학습 표본 밖 예측 · 봉마다 그 시점까지의 표본 밖 예측 분위로 결정) · trade_ledgers_used_as_input=false ·
       saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")   # 10-10 교훈: 과다 스레드 = 무진행
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_eth_dir4h_20261015 import FEATS, HGB  # noqa: E402
HGB = {**HGB, "early_stopping": "auto"}   # 10-19 이전 라이브(A) 설정 고정 -- 라이브는 B2(early_stopping=False)로 바뀌었다
from research_eth_dir4h_reverify_20261019 import BD, WT, block_ci, expanding_q, ic_stats, pearson, rates, side, summ  # noqa: E402

OUT = ROOT / "tmp/dir1h_stage1_20261019"
H1 = 12
COST = 4.0
SEEDS = (1537962059, 2000446311, 185433809, 1587955666, 1505140787)   # secrets.randbelow(2**31), 결과 전 추출
RULES = {"R20": .2, "R10": .1}
PNL_PERIODS = (("2025~", "2025-01-01", "2026-09-15"), ("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", "2026-09-15"),
               ("2023-24", "2023-02-01", "2025-01-01"), ("2023", "2023-02-01", "2024-01-01"), ("2024", "2024-01-01", "2025-01-01"))
IC_PERIODS = (("2023", "2023-01-01", "2024-01-01"), ("2024", "2024-01-01", "2025-01-01"), ("2025", "2025-01-01", "2026-01-01"),
              ("2026", "2026-01-01", "2026-09-15"), ("2025~", "2025-01-01", "2026-09-15"), ("2023-24", "2023-01-01", "2025-01-01"))


def expq(pred, q, minp=8640):
    s = pd.Series(pred).shift(1).expanding(minp)
    return s.quantile(q).to_numpy(), s.quantile(1 - q).to_numpy()


def train_mask(ts, m, h=H1, days=365):
    cut = np.datetime64(m) - np.timedelta64(5 * h, "m")
    return (ts < cut) & (ts >= cut - np.timedelta64(days, "D"))


def maker_pnl(s, c, hi, lo, h=H1):
    """research leg() 벡터판. 행 j = 결정 봉(kline 인덱스 = 패널 인덱스). 반환 (총 우위, 비용) -- 거래 없는 봉 NaN."""
    n = len(c); j = np.flatnonzero((s != 0) & (np.arange(n) + h + 1 < n)); sj = s[j]
    def leg(i, sd):   # i = 열리는 봉, P = c[i−1]
        P = c[i - 1]; f = np.where(sd > 0, lo[i] < P, hi[i] > P)
        return np.where(f, P, c[i]), np.where(f, 0.0, 4.0)
    pe, ce = leg(j + 1, sj); px, cx = leg(j + 1 + h, -sj)
    g, f = np.full(n, np.nan), np.full(n, np.nan)
    g[j] = sj * np.log(px / pe) * 1e4; f[j] = ce + cx
    return g, f


def phase_daily12(pnl, ts, a, b, last_ts, h=H1) -> pd.Series:
    """위상 h 개(off 0..h−1) 각자 h 봉 간격 비겹침, 1/h 씩 → 결정 봉 날짜로 일 합. 기간 안 모든 날(거래 없는 날 0)."""
    rows = []
    for off in range(h):
        idx = np.arange(off, len(ts), h)
        m = (ts[idx] >= np.datetime64(a)) & (ts[idx] < np.datetime64(b)) & np.isfinite(pnl[idx])
        rows.append(pd.Series(pnl[idx][m] / h, index=pd.to_datetime(ts[idx][m])))
    g = pd.concat(rows)
    last = min(pd.Timestamp(b), pd.Timestamp(last_ts)) - pd.Timedelta("1D")
    return g.groupby(g.index.floor("D")).sum().reindex(pd.date_range(a, last, freq="D"), fill_value=0.0)


def boot_means(x, b=5, n=4000, seed=7):
    """= block_ci 의 재표집 그대로(평균 배열 반환)."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(x) / b))
    st = rng.integers(0, len(x) - b + 1, size=(n, k))
    return x[(st[:, :, None] + np.arange(b)).reshape(n, -1)[:, :len(x)]].mean(1)


def p_one(x):
    bm = boot_means(x)
    return float((1 + (bm <= 0).sum()) / (len(bm) + 1))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o]).clip(max=1.0)
    out = np.empty(m); out[o] = adj
    return out


def ic_ci(S, b=5, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(S) / b))
    ix = (rng.integers(0, len(S) - b + 1, size=(n, k))[:, :, None] + np.arange(b)).reshape(n, -1)[:, :len(S)]
    r = pearson(S[ix].sum(1))
    return [round(float(np.quantile(r, .025)), 4), round(float(np.quantile(r, .975)), 4)]


# ── 실행 ─────────────────────────────────────────────────────────────────
def load():
    P = pd.read_parquet(BD / "panel.parquet", columns=["ts", "y_1h", "y_4h"] + FEATS)
    P[FEATS] = P[FEATS].astype(np.float32)
    sys.path.insert(0, str(WT / "liquidation-price-feature-14eceb/scripts"))
    import research_eth_best_direction_model_20261010 as R
    b = R.load_panel("ETH")[["timestamp", "high", "low", "close"]]
    ts = P.ts.to_numpy()
    assert len(b) == len(P) and (b.timestamp.to_numpy() == ts).all(), "패널·kline 격자 불일치"
    c = b.close.ffill().to_numpy(); lc = np.log(c)
    for nm, h in (("y_1h", 12), ("y_4h", 48)):   # 라벨 정렬: 같은 종가로 재계산 = 패널
        yy = np.r_[(lc[h:] - lc[:-h]) * 1e4, np.full(h, np.nan)]; y0 = P[nm].to_numpy()
        assert np.array_equal(np.isnan(yy), np.isnan(y0)) and np.nanmax(np.abs(yy - y0)) < 1e-6, nm
    # 사건 라벨 경계: 행 t 피쳐 = 봉 t 마감까지(P_r5m = 봉 t 자기 수익), 라벨 = 봉 t+1..t+12
    r1 = np.r_[np.nan, np.diff(lc)] * 1e4; f = P.P_r5m.to_numpy(float); m = np.isfinite(f) & np.isfinite(r1)
    assert np.max(np.abs(f[m] - r1[m])) < 1e-2, "P_r5m ≠ 봉 t 수익"
    r_next = np.r_[r1[1:], np.nan]; mm = m & np.isfinite(r_next)
    tw = {"corr_P_r5m_bar_t": round(float(np.corrcoef(f[mm], r1[mm])[0, 1]), 4), "corr_P_r5m_bar_t+1": round(float(np.corrcoef(f[mm], r_next[mm])[0, 1]), 4)}
    assert tw["corr_P_r5m_bar_t"] > .999 and abs(tw["corr_P_r5m_bar_t+1"]) < .2, tw
    return P, c, b.high.to_numpy(), b.low.to_numpy(), tw


def fit(P, arm, months):
    f = OUT / f"pred_{arm}.npz"
    if f.exists():
        z = np.load(f); return z["pred"], json.loads(str(z["meta"]))
    from sklearn.ensemble import HistGradientBoostingRegressor
    ts, y, X = P.ts.to_numpy(), P.y_1h.to_numpy(), P[FEATS].to_numpy(dtype=np.float32)
    sig = P.P_rv1d.to_numpy(float) * np.sqrt(H1)
    tgt = y if arm == "A0" else y / sig
    pred = np.full(len(ts), np.nan); meta = {"seed_maxdiff": {}, "n_train": {}}
    for m in months:
        tr = train_mask(ts, m) & np.isfinite(tgt)
        assert ts[tr].max() + np.timedelta64(5 * (H1 + 1), "m") <= np.datetime64(m), "정화 실패"
        te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
        ps = np.array([HistGradientBoostingRegressor(random_state=s, **HGB).fit(X[tr][::2], tgt[tr][::2]).predict(X[te]) for s in SEEDS])
        pred[te] = ps.mean(0) * (sig[te] if arm == "A1" else 1.0)
        k = m.strftime("%Y-%m"); meta["seed_maxdiff"][k] = float(np.abs(ps - ps[0]).max()); meta["n_train"][k] = int(tr.sum())
        print(f"  {arm} {k} 학습 {int(tr.sum())} 시드차 {meta['seed_maxdiff'][k]:.2e}", flush=True)
    np.savez(f, pred=pred, meta=json.dumps(meta))
    return pred, meta


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    P, c, hi5, lo5, tw = load()
    ts, y = P.ts.to_numpy(), P.y_1h.to_numpy()
    last_ts = ts[np.isfinite(y)].max()
    months = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    out = {"seeds": SEEDS, "tripwire": tw, "flags": {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
           "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}, "arms": {}, "combos": {}}
    preds = {}
    for arm in ("A0", "A1"):
        preds[arm], meta = fit(P, arm, months)
        a = {"seed_maxdiff_max": max(meta["seed_maxdiff"].values()), "ic": {}}
        for lab, s_, e_ in IC_PERIODS:
            m = (ts >= np.datetime64(s_)) & (ts < np.datetime64(e_)) & np.isfinite(preds[arm]) & np.isfinite(y)
            S = ic_stats(preds[arm][m], y[m], pd.to_datetime(ts[m]).floor("D"))
            a["ic"][lab] = {"pearson": round(float(pearson(S.sum(0))), 4), "ci": ic_ci(S),
                            "spearman": round(float(pd.Series(preds[arm][m]).rank().corr(pd.Series(y[m]).rank())), 4)}
        a["accuracy_gate"] = bool(a["ic"]["2025~"]["ci"][0] > 0 and a["ic"]["2023-24"]["pearson"] > 0)
        out["arms"][arm] = a
        print(arm, json.dumps(a["ic"], ensure_ascii=False), "정확도 문", a["accuracy_gate"], flush=True)
    D = {}
    for arm, p in preds.items():
        for rn, q in RULES.items():
            lo, hi = (expanding_q(p) if q == .2 else expq(p, q))
            s = side(p, lo, hi) * np.isfinite(y)
            g_m, f_m = maker_pnl(s, c, hi5, lo5)
            g_t = np.where(s != 0, s * y, np.nan)
            for ex, g, fee in (("TAKER", g_t, np.where(s != 0, COST, np.nan)), ("MAKER", g_m, f_m)):
                key = f"{arm}_{rn}_{ex}"; r = {"rates": rates(p, lo, hi, ts, "2025-01-01", "2026-09-15")}
                for lab, a_, b_ in PNL_PERIODS:
                    D[key, lab] = phase_daily12(g - fee, ts, a_, b_, last_ts)
                    mm = (ts >= np.datetime64(a_)) & (ts < np.datetime64(b_)) & np.isfinite(g)
                    r[lab] = {**summ(D[key, lab]), "n_trades": int(mm.sum()), "trades_per_day": round(mm.sum() / H1 / len(D[key, lab]), 2),
                              "gross_bp_trade": round(float(g[mm].mean()), 2), "fee_bp_trade": round(float(fee[mm].mean()), 2)}
                    if ex == "MAKER":
                        r[lab]["both_legs_filled"] = round(float((fee[mm] == 0).mean()), 3)
                r["p_one_2025~"] = p_one(D[key, "2025~"].to_numpy())
                out["combos"][key] = r
                print(key, json.dumps({k: v for k, v in r.items() if k in ("2025~", "2023-24", "p_one_2025~")}, ensure_ascii=False), flush=True)
    keys = list(out["combos"]); adj = holm([out["combos"][k]["p_one_2025~"] for k in keys])
    for k, pa in zip(keys, adj):
        r = out["combos"][k]; r["p_holm"] = round(float(pa), 4)
        r["trading_gate"] = bool(pa < .05 and r["2023-24"]["bp_day"] > 0)
    best = max(keys, key=lambda k: out["combos"][k]["2025~"]["bp_day"])
    ic_b = out["arms"][best[:2]]["ic"]["2025~"]["pearson"]; ge_b = out["combos"][best]["2025~"]["gross_bp_trade"]
    out["verdict"] = {"accuracy_gate": {a: out["arms"][a]["accuracy_gate"] for a in preds},
                      "trading_gate_pass": [k for k in keys if out["combos"][k]["trading_gate"]],
                      "best_combo": best, "best_ic_2025~": ic_b, "best_gross_bp_trade_2025~": ge_b,
                      "stage2": bool(ic_b >= .03 and ge_b >= 2.0)}
    print("판정", json.dumps(out["verdict"], ensure_ascii=False), flush=True)
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    pd.DataFrame({k[0]: v for k, v in D.items() if k[1] == "2025~"}).to_parquet(OUT / "daily_2025~.parquet")
    return out


def selfcheck() -> None:
    # 확장 분위: 과거만(shift 1) · R20 = reverify expanding_q
    p = np.random.default_rng(0).normal(size=200)
    lo, hi = expq(p, .2, minp=20); lo2, hi2 = expanding_q(p, minp=20)
    assert np.allclose(lo, lo2, equal_nan=True) and np.allclose(hi, hi2, equal_nan=True)
    q = p.copy(); q[100:] = 99.0; assert np.allclose(expq(q, .1, 20)[0][:101], expq(p, .1, 20)[0][:101], equal_nan=True), "미래 예측이 문턱에 샘"
    assert np.isnan(lo[19]) and lo[20] == np.quantile(p[:20], .2)
    # 정화: 학습 마지막 행 라벨(t+12 봉 마감 = t+65분) ≤ 달 시작
    ts = pd.date_range("2024-12-01", "2025-01-10", freq="5min").to_numpy(); m = pd.Timestamp("2025-01-01")
    tr = train_mask(ts, m); assert ts[tr].max() == np.datetime64("2024-12-31T22:55") and ts[tr].max() + np.timedelta64(65, "m") <= np.datetime64(m)
    # 라벨 정렬: y[t] = lc[t+12] − lc[t]
    lc = np.log(np.linspace(100, 110, 50)); yy = np.r_[(lc[12:] - lc[:-12]) * 1e4, np.full(12, np.nan)]
    assert abs(yy[3] - (lc[15] - lc[3]) * 1e4) < 1e-9 and np.isnan(yy[-12]) and np.isfinite(yy[-13])
    # 메이커: 둘 다 관통 → 총 = s·y · 비용 0 / 관통 없음 → 다음 봉 종가 + 4bp 다리당
    n = 30; c = np.full(n, 100.0); c[13:] = 101.0; hi = c + .5; lo = c - .5
    s = np.zeros(n, int); s[0] = 1
    g, f = maker_pnl(s, c, hi, lo, h=12)
    assert f[0] == 0   # 청산 봉 = j+13: P = c[12] = 100, high 101.5 > 100 → 체결가 100(봉 13 상승분은 못 먹음)
    assert abs(g[0] - 0.0) < 1e-9 and np.isnan(g[1])
    c2 = np.full(n, 100.0); c2[12:] = 101.0
    g, f = maker_pnl(s, c2, c2 + .5, c2 - .5, h=12)
    assert f[0] == 0 and abs(g[0] - np.log(101 / 100) * 1e4) < 1e-9   # 둘 다 체결 = s·ln(c[12]/c[0])
    g, f = maker_pnl(s, c2, c2, c2, h=12)                                 # 관통 없음(high = low = 종가)
    assert f[0] == 8.0 and abs(g[0] - np.log(c2[13] / c2[1]) * 1e4) < 1e-9
    s[0] = -1; g, f = maker_pnl(s, c2, c2 + .5, c2 - .5, h=12)
    assert f[0] == 0 and abs(g[0] + np.log(101 / 100) * 1e4) < 1e-9
    # 위상 12: 매 봉 +10bp · 늘 롱 · 4bp → 하루 = 288 × (10−4)/12 = 144
    t = pd.date_range("2025-01-01", periods=288 * 6, freq="5min").to_numpy(); pn = np.full(len(t), 6.0)
    d = phase_daily12(pn, t, "2025-01-01", "2025-01-07", t[-1]); assert np.allclose(d.to_numpy()[:5], 144.0), d.head()
    # Holm · 단측 p
    assert np.allclose(holm([.01, .04, .03, .005]), [.03, .06, .06, .02])
    x = np.random.default_rng(1).normal(0, 10, 600); assert p_one(x + 3) < .01 and p_one(x - 3) > .99
    lo_, hi_ = block_ci(x); assert lo_ < 0 < hi_
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck()
    if "--selfcheck" not in sys.argv:
        run()
