#!/usr/bin/env python3
"""**Haiku 가 대시보드 숫자로 ETH 1h·4h 방향을 맞추나** — 과거 400 시점 시험(2026-10-16).

왜: 사용자 «정확도를 확인하기 위해 과거 데이터로 테스트 · 방향을 맞출 수 있는지 우선 · 입력 피쳐 충분한지도 체크».
입력: 10-10 방향 모델 패널(ETH 5분, 피쳐 101, 인과 검증됨 — research_eth_best_direction_model_20261010.py)을 사람이 읽는
  이름·단위의 JSON 으로. 날짜·절대가격 없음(역사 기억 누수 차단). 결정 = 5분봉 k 마감, 정답 y_1h·y_4h = ln(close[k+h]/close[k]) bp.
표본: 2025-01-01 ~ 2026-08-21(aggTrades·bookDepth 원천이 08-22 에 끊김) 무작위 400 시점, 서로 ≥8시간(정답 창·묶음 안 겹침 차단).
비교: HGB 4h 표본외(매달 365일 재학습, pred_FINAL) · 2024 한 해 학습 HGB(1h·4h) · 모멘텀/되돌림 · 메아리(직전 수익과의 상관).
Haiku 호출 = Claude Code 하위 에이전트(model haiku, API 키 없음) — 묶음 파일을 읽고 JSON 배열을 돌려준다 → out/batch_XX.json 에 저장.
  python scripts/research_llm_haiku_direction_20261016.py build
  python scripts/research_llm_haiku_direction_20261016.py eval
  python scripts/research_llm_haiku_direction_20261016.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# ponytail: 패널은 다른 워크트리의 tmp 에만 있다(355MB). 옮기면 이 경로만 바꾼다.
SRC = Path("/home/kbj20/crypto-scalping/.claude/worktrees/liquidation-price-feature-14eceb/tmp/best_dir_20261010")
OUT = ROOT / "tmp/haiku_direction_20261016"
BDIR = ROOT / "tmp/haiku_direction_20261016_inputs"   # 🔴 에이전트가 읽는 폴더 — 정답(OUT/gold.csv)과 분리
N, BATCH, GAP_H = 400, 40, 8
START, END = "2025-01-01", "2026-08-21 20:00"

E = lambda x: math.exp(x)  # noqa: E731  로그 비율 → 비율
# (출력 이름, 패널 열, 변환, 설명) — 시간 4열 빼고 패널 피쳐 전부(시간은 hour_utc·weekday 로)
FIELDS = [
    ("ret_5m_bp", "P_r5m", None, "ETH perp return over last 5m (bp)"),
    ("ret_15m_bp", "P_r15m", None, "return last 15m (bp)"),
    ("ret_1h_bp", "P_r1h", None, "return last 1h (bp)"),
    ("ret_4h_bp", "P_r4h", None, "return last 4h (bp)"),
    ("ret_1d_bp", "P_r1d", None, "return last 24h (bp)"),
    ("ret_7d_bp", "P_r7d", None, "return last 7d (bp)"),
    ("ret_28d_bp", "P_r28d", None, "return last 28d (bp)"),
    ("rvol_5m_bar_1h_bp", "P_rv1h", None, "std of 5m returns over last 1h (bp)"),
    ("rvol_5m_bar_1d_bp", "P_rv1d", None, "std of 5m returns over last 24h (bp)"),
    ("vol_ratio_1h_vs_1d", "P_vratio", None, "1h realized vol / 24h realized vol"),
    ("rsi14_5m", "P_rsi14", None, "RSI(14) on 5m closes"),
    ("bollinger_pos", "P_pctb", None, "(close - SMA20) / (2*std20) on 5m; +1 = upper band"),
    ("pos_in_24h_range", "P_pos1d", None, "0 = 24h low, 1 = 24h high"),
    ("dist_from_12h_mean_atr", "P_veto_z", None, "(close - 12h mean) / ATR"),
    ("ret_autocorr_12h", "P_ac144", None, "lag-1 autocorrelation of 5m returns over 12h"),
    ("daily_trend_votes", "P_trend5", None, "mean sign of daily close change over 7/14/28/56/90 days (-1..1)"),
    ("perp_taker_imb_5m", "T_imb5m", None, "(taker buy - taker sell)/volume, last 5m (-1..1)"),
    ("perp_taker_imb_15m", "T_imb15m", None, "same, 15m"),
    ("perp_taker_imb_1h", "T_imb1h", None, "same, 1h"),
    ("perp_taker_imb_4h", "T_imb4h", None, "same, 4h"),
    ("perp_cvd_1h_z", "T_cvd1h_z", None, "1h net taker volume, z vs past 7d"),
    ("volume_1h_z", "T_vol_z", None, "log 1h volume, z vs past 7d"),
    ("trades_1h_z", "T_trades_z", None, "log 1h trade count, z vs past 7d"),
    ("avg_trade_size_log", "T_size", None, "log(1h volume / trade count)"),
    ("oi_chg_5m_bp", "O_doi5m", None, "Binance open interest change 5m (bp)"),
    ("oi_chg_1h_bp", "O_doi1h", None, "OI change 1h (bp)"),
    ("oi_chg_4h_bp", "O_doi4h", None, "OI change 4h (bp)"),
    ("oi_chg_1d_bp", "O_doi1d", None, "OI change 24h (bp)"),
    ("oi_chg_1h_z", "O_doi1h_z", None, "OI 1h change, z vs past 7d"),
    ("oi_chg_1h_x_price_dir", "O_doi1h_x_r", None, "OI 1h change (bp) * sign(1h price return)"),
    ("top_trader_position_ls", "O_toppos", E, "Binance top-trader long/short POSITION ratio"),
    ("top_trader_position_ls_chg1h_log", "O_toppos_d1h", None, "log change over 1h"),
    ("top_trader_position_ls_z28d", "O_toppos_z", None, "log ratio z vs past 28d"),
    ("top_trader_account_ls", "O_topacct", E, "top-trader long/short ACCOUNT ratio"),
    ("top_trader_account_ls_chg1h_log", "O_topacct_d1h", None, "log change 1h"),
    ("top_trader_account_ls_z28d", "O_topacct_z", None, "z vs past 28d"),
    ("global_account_ls", "O_glob", E, "all-accounts long/short ratio"),
    ("global_account_ls_chg1h_log", "O_glob_d1h", None, "log change 1h"),
    ("global_account_ls_z28d", "O_glob_z", None, "z vs past 28d"),
    ("taker_ls_vol_ratio", "O_takerls", E, "taker buy/sell volume ratio (Binance metrics, 5m)"),
    ("taker_ls_vol_ratio_chg1h_log", "O_takerls_d1h", None, "log change 1h"),
    ("taker_ls_vol_ratio_z28d", "O_takerls_z", None, "z vs past 28d"),
    ("top_pos_minus_global_log", "O_top_minus_glob", None, "log(top position ratio) - log(global account ratio)"),
    ("premium_index_bp", "B_prem", None, "perp premium index (mark vs spot index, bp; negative = perp discount)"),
    ("premium_chg_1h_bp", "B_prem_d1h", None, "premium change over 1h (bp)"),
    ("premium_avg_8h_bp", "B_prem_8h", None, "mean premium over last 8h (bp, drives funding)"),
    ("premium_z28d", "B_prem_z", None, "premium z vs past 28d"),
    ("spot_minus_perp_ret_5m_bp", "S_lead5m", None, "spot return minus perp return, 5m (bp)"),
    ("spot_taker_imb_5m", "S_imb5m", None, "Binance spot taker imbalance 5m (-1..1)"),
    ("spot_minus_perp_ret_15m_bp", "S_lead15m", None, "15m"),
    ("spot_taker_imb_15m", "S_imb15m", None, "15m"),
    ("spot_minus_perp_ret_1h_bp", "S_lead1h", None, "1h"),
    ("spot_taker_imb_1h", "S_imb1h", None, "1h"),
    ("spot_minus_perp_taker_imb_1h", "S_imb_minus_perp1h", None, "spot imbalance - perp imbalance, 1h"),
    ("spot_volume_share_1h", "S_vol_share", None, "spot volume / (spot + perp volume), 1h"),
    ("agg_cvd_5m", "M_cvd", None, "aggTrades net taker volume last 5m (ETH)"),
    ("whale_trade_imb_5m", "M_bigimb", None, "net taker imbalance of large trades, 5m"),
    ("retail_trade_imb_5m", "M_smlimb", None, "net taker imbalance of small trades, 5m"),
    ("trade_count_imb_5m", "M_cntimb", None, "(buy trades - sell trades)/trades, 5m"),
    ("n_trades_5m", "M_ntrade", None, "number of aggTrades in 5m"),
    ("whale_volume_share_5m", "M_bigshare", None, "share of volume from large trades, 5m"),
    ("mean_trade_size_log_5m", "M_meansz", None, "log mean trade size, 5m"),
    ("trade_sign_autocorr_5m", "M_signac", None, "autocorrelation of trade signs, 5m"),
    ("trade_run_length_5m", "M_runlen", None, "mean run length of same-side trades, 5m"),
    ("kyle_lambda_5m", "M_kyle", None, "price impact per unit signed volume, 5m"),
    ("bar_range_5m", "M_range", None, "5m high-low range (relative)"),
    ("vpin_5m", "M_vpin", None, "VPIN order-flow toxicity, 5m"),
    ("whale_trade_imb_1h", "M_bigimb_1h", None, "1h mean of whale imbalance"),
    ("retail_trade_imb_1h", "M_smlimb_1h", None, "1h mean of retail imbalance"),
    ("trade_count_imb_1h", "M_cntimb_1h", None, "1h mean"),
    ("kyle_lambda_1h", "M_kyle_1h", None, "1h mean"),
    ("vpin_1h", "M_vpin_1h", None, "1h mean"),
    ("trade_sign_autocorr_1h", "M_signac_1h", None, "1h mean"),
    ("whale_minus_retail_imb_1h", "M_whale_minus_retail_1h", None, "whale imbalance - retail imbalance, 1h"),
    ("book_imb_within_1pct", "D_imb1", None, "(bid depth - ask depth)/(sum) within +-1% of price"),
    ("book_imb_within_2pct", "D_imb2", None, "within +-2%"),
    ("book_imb_within_5pct", "D_imb5", None, "within +-5%"),
    ("book_imb_1pct_chg_1h", "D_imb1_d1h", None, "change of +-1% book imbalance over 1h"),
    ("book_depth_1pct_z", "D_depth1_z", None, "log total depth within +-1%, z vs past 7d"),
    ("btc_ret_5m_bp", "X_btc_r5m", None, "BTC return 5m (bp)"),
    ("btc_ret_15m_bp", "X_btc_r15m", None, "BTC 15m"),
    ("btc_ret_1h_bp", "X_btc_r1h", None, "BTC 1h"),
    ("btc_ret_4h_bp", "X_btc_r4h", None, "BTC 4h"),
    ("btc_taker_imb_1h", "X_btc_imb1h", None, "BTC perp taker imbalance 1h"),
    ("btc_oi_chg_1h_bp", "X_btc_doi1h", None, "BTC OI change 1h (bp)"),
    ("alt20_avg_ret_1h_bp", "X_breadth1h", None, "mean 1h return of 20 major coins (bp)"),
    ("alt20_avg_ret_1d_bp", "X_breadth1d", None, "mean 24h return of 20 coins (bp)"),
    ("eth_minus_alt20_ret_1h_bp", "X_eth_rel1h", None, "ETH 1h return - 20-coin mean"),
    ("eth_minus_alt20_ret_1d_bp", "X_eth_rel1d", None, "ETH 24h return - 20-coin mean"),
    ("eth_rank_1d_among20", "X_eth_rank1d", None, "fraction of 20 coins ETH beat over 24h (0..1)"),
    ("alt20_dispersion_1h_bp", "X_disp1h", None, "cross-sectional std of 20-coin 1h returns"),
    ("hours_to_funding", "C_to_funding", None, "hours until next 8h funding"),
    ("eth_ret_since_utc_open_bp", "U_eth_since_open", None, "ETH return since today's 00:00 UTC (bp)"),
    ("eth_first30m_ret_today_bp", "U_eth_f30_day", None, "ETH return in first 30 min after 00:00 UTC today (bp)"),
    ("btc_ret_since_utc_open_bp", "U_btc_since_open", None, "BTC return since 00:00 UTC"),
    ("btc_first30m_ret_today_bp", "U_btc_f30_day", None, "BTC first 30 min after 00:00 UTC"),
    ("minutes_since_utc_open", "U_min_since_open", None, "minutes since 00:00 UTC"),
]
TIME_COLS = ("C_hsin", "C_hcos", "C_dow", "C_weekend")


def sig(x: float, n: int = 3):
    if x is None or not np.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def load() -> pd.DataFrame:
    F = pd.read_parquet(SRC / "panel.parquet")
    p4 = np.load(SRC / "pred_FINAL_eth_4h.npy")
    F["hgb4h_oos"] = np.nanmean(p4, axis=1) if p4.ndim == 2 else p4
    return F


def pick(F: pd.DataFrame, n: int = N, seed: int = 20261016) -> np.ndarray:
    """START~END 에서 서로 ≥GAP_H 시간 떨어진 무작위 n 시점(행 번호). 정답·주요 피쳐 결측 행 제외."""
    ok = (F.ts >= START) & (F.ts <= END) & F.y_4h.notna() & F.y_1h.notna() & F.M_bigimb_1h.notna() & F.D_imb1.notna() & F.hgb4h_oos.notna()
    cand = np.flatnonzero(ok.to_numpy())
    t = F.ts.to_numpy().astype("datetime64[m]").astype(np.int64)
    chosen: list[int] = []
    for i in np.random.default_rng(seed).permutation(cand):
        if all(abs(t[i] - t[j]) >= GAP_H * 60 for j in chosen):
            chosen.append(int(i))
            if len(chosen) == n:
                break
    assert len(chosen) == n
    return np.array(chosen)


def render(row: pd.Series) -> dict:
    tt = row.ts + pd.Timedelta("5min")
    d = {"hour_utc": round(tt.hour + tt.minute / 60, 2), "weekday": tt.day_name()}
    for name, col, fn, _ in FIELDS:
        v = row[col]
        d[name] = sig(fn(v) if fn and np.isfinite(v) else v)
    return d


GLOSSARY = "\n".join(f"- {n}: {desc}" for n, _, _, desc in FIELDS)
PROMPT = """You are given {k} independent snapshots of the ETH perpetual futures market (Binance ETHUSDT), each taken at the close of a 5-minute bar.
Snapshots are from different, unrelated days (at least 8 hours apart) and are shuffled; do not compare them to each other or try to guess dates.
For EACH snapshot, estimate the probability (0-100) that the ETH price will be HIGHER than now:
  p_up_1h = 1 hour later, p_up_4h = 4 hours later.
Use your knowledge of crypto market microstructure. 50 means no edge. Be honest; extreme values only when the evidence is strong.

Field definitions (bp = basis points; z = z-score; imbalance in -1..1, positive = buyers):
- hour_utc, weekday: time of the snapshot
{glossary}

Output ONLY one JSON array, one object per snapshot, in the same order: [{{"id": "...", "p_up_1h": int, "p_up_4h": int}}, ...]

Snapshots:
"""


def build() -> None:
    F = load()
    idx = pick(F)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "out").mkdir(exist_ok=True)
    BDIR.mkdir(parents=True, exist_ok=True)
    gold = F.loc[idx, ["ts", "y_1h", "y_4h", "hgb4h_oos", "P_r1h", "P_r4h", "P_r1d", "P_trend5"]].copy()
    gold["id"] = [f"q{j:03d}" for j in range(len(idx))]
    gold["row"] = idx
    gold.to_csv(OUT / "gold.csv", index=False)
    rows = [{"id": g, **render(F.loc[i])} for g, i in zip(gold.id, idx)]
    order = np.random.default_rng(1).permutation(len(rows))
    for b in range(0, len(rows), BATCH):
        part = [rows[j] for j in order[b:b + BATCH]]
        txt = PROMPT.format(k=len(part), glossary=GLOSSARY) + "\n".join(json.dumps(r, separators=(",", ":")) for r in part)
        (BDIR / f"batch_{b // BATCH:02d}.txt").write_text(txt)
    print(f"{len(rows)} 시점 · {len(rows) // BATCH} 묶음 → {BDIR}")
    print(f"기간 {gold.ts.min()} ~ {gold.ts.max()} · 입력 칸 {len(FIELDS) + 2}(시간 2 포함)")


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos = y > 0
    r = rankdata(s)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def boot(fn, *arrs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = len(arrs[0])
    v = [fn(*(a[k] for a in arrs)) for k in (rng.integers(0, m, m) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5]).round(3).tolist()


def train_static_hgb(F: pd.DataFrame, h: str) -> pd.Series:
    """2024 한 해로 한 번 학습한 HGB(같은 피쳐 + 시간 4열) — 2025~ 시점 예측. 정답 창이 2025 로 넘는 2024 말 행은 뺀다."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    cols = [c for _, c, _, _ in FIELDS] + list(TIME_COLS)
    tr = F[(F.ts >= "2024-01-01") & (F.ts < "2024-12-30")].iloc[::3]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=0)
    m.fit(tr[cols], tr[f"y_{h}"].fillna(0).clip(-500, 500))
    return pd.Series(m.predict(F[cols]), index=F.index)


def evaluate() -> None:
    from scipy.stats import spearmanr
    gold = pd.read_csv(OUT / "gold.csv")
    ans = {}
    for f in sorted((OUT / "out").glob("batch_*.json")):
        for a in json.loads(f.read_text()):
            ans[a["id"]] = a
    g = gold[gold.id.isin(ans)].copy()
    g["llm_1h"] = [ans[i]["p_up_1h"] for i in g.id]
    g["llm_4h"] = [ans[i]["p_up_4h"] for i in g.id]
    F = load()
    for h in ("1h", "4h"):
        g[f"hgb24_{h}"] = train_static_hgb(F, h).loc[g.row].to_numpy()
    print(f"채점 {len(g)}/{len(gold)} 시점 · {g.ts.min()[:10]} ~ {g.ts.max()[:10]} · 상승 비율 1h {np.mean(g.y_1h > 0):.2f} 4h {np.mean(g.y_4h > 0):.2f}")
    res = {}
    for h in ("1h", "4h"):
        y = g[f"y_{h}"].to_numpy()
        sigs = {"Haiku": g[f"llm_{h}"].to_numpy(float), "HGB 2024 학습": g[f"hgb24_{h}"].to_numpy(),
                "모멘텀(직전 1h)": g.P_r1h.to_numpy(), "되돌림(−직전 1h)": -g.P_r1h.to_numpy(), "일봉 추세 표": g.P_trend5.to_numpy()}
        if h == "4h":
            sigs["HGB 매달 재학습(표본외)"] = g.hgb4h_oos.to_numpy()
        print(f"\n── {h} ──  {'신호':22s} AUC [95% CI]          IC      방향 쪽 평균 bp")
        for k, s in sigs.items():
            a = auc(y, s); ci = boot(auc, y, s)
            ic = spearmanr(s, y).correlation
            pnl = float(np.mean(np.sign(s - (50 if k == "Haiku" else 0)) * y))
            print(f"   {k:22s} {a:.3f} {ci}  {ic:+.3f}   {pnl:+.1f}")
            res[f"{h}:{k}"] = {"auc": a, "ci": ci, "ic": ic, "signed_bp": pnl}
    print("\n── Haiku 답의 성질 ──")
    for h in ("1h", "4h"):
        s = g[f"llm_{h}"].to_numpy(float)
        print(f"   {h}: 평균 {s.mean():.1f} · 표준편차 {s.std():.1f} · 50 비율 {np.mean(s == 50):.0%} · "
              f"ρ(직전 1h) {spearmanr(s, g.P_r1h).correlation:+.2f} · ρ(직전 4h) {spearmanr(s, g.P_r4h).correlation:+.2f} · "
              f"ρ(직전 1d) {spearmanr(s, g.P_r1d).correlation:+.2f} · ρ(일봉 추세) {spearmanr(s, g.P_trend5).correlation:+.2f} · "
              f"ρ(HGB 4h) {spearmanr(s, g.hgb4h_oos).correlation:+.2f}")
    print(f"   ρ(Haiku 1h, Haiku 4h) {spearmanr(g.llm_1h, g.llm_4h).correlation:+.2f}")
    g.to_csv(OUT / "scored.csv", index=False)
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def _selfcheck() -> None:
    F = load()
    idx = pick(F, n=60, seed=1)
    t = F.ts.to_numpy()[idx].astype("datetime64[m]").astype(np.int64)
    d = np.abs(t[:, None] - t[None, :]) + np.eye(len(t), dtype=np.int64) * 10**9
    assert d.min() >= GAP_H * 60, d.min()                        # 정답 창 안 겹침
    assert (F.ts.iloc[idx] >= START).all() and (F.ts.iloc[idx] <= END).all()
    r = render(F.loc[idx[0]])
    assert len(r) == len(FIELDS) + 2 and not ({"ts", "y_1h", "y_4h"} & set(r))   # 정답·시각 안 샘
    i = int(idx[0])
    # 정답 = 미래 48봉, ret_4h = 과거 48봉 → 4h 뒤 행의 ret_4h 가 이 행의 y_4h 와 같아야 한다(정답이 정말 미래라는 확인)
    assert abs(F.P_r4h[i + 48] - F.y_4h[i]) < 1e-6, (F.P_r4h[i + 48], F.y_4h[i])
    assert abs(F.P_r1h[i + 12] - F.y_1h[i]) < 1e-6
    assert auc(np.array([1, -1, 1, -1]), np.array([.9, .1, .8, .2])) == 1.0
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["build", "eval"])
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.cmd == "build":
        build()
    elif a.cmd == "eval":
        evaluate()
