#!/usr/bin/env python3
"""사이징 규칙 경주 — **전체 데이터**판 (2026-09-16).

사용자 *"원장을 보지 말고 전체 데이터를 보고 말해줘"*.

원장판(`research_sizing_rule_horserace_20260916.py`)은 왕복 72 · **독립일 22** 였고
①고정이 1위였다. 그 순위가 «규칙의 성질」인지 «그 22일의 성질」인지 가른다.

ETH 5분봉 **4.8년 전체**에서 무작위 진입(양측면 균형)으로 같은 규칙들을 돌린다.
선행 09-11(`sizing_rules_random_entry_validated_20260911`, 82,167건)의 설계를 따르되
① 원장판과 **같은 규칙 목록**으로 맞추고 ② 원장판이 새로 제기한 질문을 추가한다:

    ⭐**「고변동 자리가 더 번다」가 전체 데이터에서도 성립하는가?**
      (원장에서 ρ(atr14, 실현bp) = +0.243 p .0395 · ATR288 3분위 +10.7/+12.1/**+38.1bp**)

🔴**상한류(순자산×6 · 원장중앙×2)는 계좌 상태에 비례하므로 무작위 진입에서는 상수 배수가 되어
  평균명목 정규화 후 고정과 구분되지 않는다.** 그래서 이 판에는 **변동성·위험 기반 규칙만** 넣는다.
🔴무작위 진입은 실력 e=0 이라 **평균은 −비용**이다. 사이징은 이익을 만들지 않는다 —
  그래서 ①위험 분포와 ②실력 가정별 로그성장을 나눠 본다(09-11 규약 그대로).

출력: tmp/eth_sizing_fulldata_20260916/
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp/eth_sizing_fulldata_20260916"
RNG = np.random.default_rng(20260916)
HOLD = 48            # 4시간 — 원장판·09-11 과 동일
COST_BP = 5.88       # 이 세션 규약(peg 진입 + peg 청산)
N_DRAWS = 120_000
WARMUP = 900
LIQ_X = 50.0         # 50배 기준 청산 도달률(09-11 지표)


def _mod(rel: str, name: str):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp); argv, sys.argv = sys.argv, ["x"]
    try: sp.loader.exec_module(m)
    finally: sys.argv = argv
    return m


def summarize(bp: np.ndarray, w: np.ndarray, day: np.ndarray, mae: np.ndarray) -> dict:
    """평균 명목 정규화 후 위험 분포. 09-11 지표(SD·하위1%·MAE노출·청산율) + 일별 MDD."""
    wn = np.asarray(w, float); wn = wn / wn.mean()
    x = bp * wn
    g = pd.Series(x).groupby(day).sum()
    cum = np.r_[0.0, g.cumsum().to_numpy()]
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)),
                p1=float(np.percentile(x, 1)), mdd=float((np.maximum.accumulate(cum) - cum).max()),
                mae99=float(np.percentile(mae * wn, 99)),
                liq=float((mae * wn > 1e4 / LIQ_X).mean()), cv=float(wn.std() / wn.mean()))


def selftest() -> None:
    bp = np.array([10., -20., 30., 5.]); d = np.array([0, 0, 1, 1]); mae = np.array([5., 40., 3., 2.])
    a, b = summarize(bp, np.ones(4), d, mae), summarize(bp, np.full(4, 7.0), d, mae)
    assert abs(a["sd"] - b["sd"]) < 1e-9 and abs(a["cv"]) < 1e-12      # 상수 가중 ≡ 고정
    # 손실에 큰 가중 -> SD 와 하위1% 악화
    bad = summarize(bp, np.array([1., 4., 1., 1.]), d, mae)
    assert bad["sd"] > a["sd"] and bad["p1"] < a["p1"]
    # MDD: 일별 (10-20)=-10, (30+5)=35 -> 시작 0 포함 MDD 10
    assert abs(a["mdd"] - 10.0) < 1e-9
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest(); OUT.mkdir(parents=True, exist_ok=True)
    MQ = _mod("scripts/live_eth_mae_quantile_model_20260913.py", "MQ")
    SV = _mod("scripts/live_eth_sizing_vol_model_20260912.py", "SV")
    svm = MQ.svm

    kl = pd.read_csv(REPO / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                     usecols=["timestamp", "open", "high", "low", "close", "volume",
                              "quote_volume", "trades"])
    kl["timestamp"] = pd.to_datetime(kl.timestamp)
    ts = kl.timestamp
    op, c, hi, lo = (kl.open.to_numpy(float), kl.close.to_numpy(float),
                     kl.high.to_numpy(float), kl.low.to_numpy(float))
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    atr14 = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy() / np.maximum(c, 1e-12)
    atr288 = pd.Series(tr).rolling(288, min_periods=144).mean().to_numpy() / np.maximum(c, 1e-12)
    n = len(kl); lo_i, hi_i = WARMUP, n - HOLD - 2
    day = ts.dt.floor("D").astype("int64").to_numpy()
    print(f"ETH 5분봉 {n:,} · {ts.iloc[0]} ~ {ts.iloc[-1]} · 보유 {HOLD}봉 · 비용 {COST_BP}bp")

    idx = RNG.integers(lo_i, hi_i, N_DRAWS)
    side = RNG.choice([1, -1], N_DRAWS)            # 양측면 균형
    e = op[idx + 1]
    ret = (c[idx + HOLD] - e) / e * 1e4 * side - COST_BP
    # MAE(역행폭, bp) — 청산·꼬리 지표용
    mae = np.empty(N_DRAWS)
    for k, (x, s) in enumerate(zip(idx, side)):
        w = slice(x + 1, x + 1 + HOLD)
        mae[k] = ((e[k] - lo[w].min()) if s > 0 else (hi[w].max() - e[k])) / e[k] * 1e4
    mae = np.maximum(mae, 0.0)
    print(f"무작위 진입 {N_DRAWS:,}건 · 고유일 {len(np.unique(day[idx])):,} · "
          f"평균 {ret.mean():+.2f}bp (실력 0 이라 ≈ −비용)")

    # 재료
    base = svm.build_features(ts, c, kl.quote_volume.to_numpy(float),
                              kl.trades.to_numpy(float), hi, lo).replace([np.inf, -np.inf], np.nan)
    art = MQ.load_model()
    r = base.iloc[idx].copy().reset_index(drop=True)
    r["log_h"] = np.log(HOLD * 5.0); r["side"] = side
    safe = MQ.safe_mae(art["models"], r[MQ.FEATURES], art["mult"])
    try:
        sv = SV.load_model()
        Xv = SV.build_features(ts, c, kl.quote_volume.to_numpy(float),
                               kl.trades.to_numpy(float), hi, lo)
        pv = SV.predict_vol(sv["models"], Xv.iloc[idx][sv["features"]])
    except Exception as ex:
        pv = np.full(N_DRAWS, np.nan); print(f"  ⚠️배포 vol 모델 실패: {type(ex).__name__}")
    # 재량 대리: 실계좌 명목 분포에서 부트스트랩(09-11 규약)
    tl = pd.DataFrame([json.loads(l) for l in open(REPO / "data/live/account_round_trips.jsonl")])
    userq = (tl.entry_price * tl.max_qty).to_numpy(float)

    ARMS = {
        "①고정": np.ones(N_DRAWS),
        "역ATR 1/atr14": 1.0 / np.maximum(atr14[idx], 1e-9),
        "역ATR 1/atr288": 1.0 / np.maximum(atr288[idx], 1e-9),
        "역변동성(배포모델)": 1.0 / np.maximum(pv, 1e-9),
        "동일위험 1/safeMAE": 1.0 / np.maximum(safe, 1e-9),
        "재량류(실계좌 분포)": RNG.choice(userq, N_DRAWS),
    }
    ARMS = {k: v for k, v in ARMS.items() if np.isfinite(v).all()}

    rows = [dict(arm=k, **summarize(ret, w, day[idx], mae)) for k, w in ARMS.items()]
    D = pd.DataFrame(rows)
    fx = D[D.arm == "①고정"].iloc[0]
    for col, nm in (("sd", "SD"), ("p1", "하위1%"), ("mae99", "MAE99"), ("liq", "청산율")):
        D[f"d_{col}_pct"] = (D[col] / fx[col] - 1.0) * 100
    D.round(4).to_csv(OUT / "fulldata_race.csv", index=False)

    print(f"\n{'='*124}")
    print(f"■ 전체 데이터 사이징 경주 — 무작위 진입 {N_DRAWS:,}건 · 평균명목 정규화 · 위험 분포")
    print(f"{'규칙':<22}{'가중CV':>8}{'평균bp':>9}{'SD':>9}{'vs고정':>8}{'하위1%':>10}{'vs고정':>8}"
          f"{'MAE99':>9}{'50배청산율':>10}{'vs고정':>8}{'일MDD':>10}")
    for r_ in D.sort_values("sd").itertuples():
        print(f"{r_.arm:<22}{r_.cv:>8.2f}{r_.mean:>+9.2f}{r_.sd:>9.1f}{r_.d_sd_pct:>+7.1f}%"
              f"{r_.p1:>10.1f}{r_.d_p1_pct:>+7.1f}%{r_.mae99:>9.1f}{r_.liq*100:>9.2f}%"
              f"{r_.d_liq_pct:>+7.1f}%{r_.mdd:>10.0f}")
    print("  (SD·하위1%·MAE99·청산율은 **작을수록 좋다**. 평균은 실력 0 이라 전부 ≈ −비용)")

    # ⭐원장 라운드의 주장: 고변동 자리가 더 버는가?
    print(f"\n■ ⭐원장의 주장 검정 — 「고변동 자리가 더 번다」가 전체 데이터에서도 성립하는가")
    from scipy.stats import spearmanr
    print(f"  원장(72왕복·22일): ρ(atr14, 실현bp) = **+0.243 p .0395** · "
          f"ATR288 3분위 +10.7 / +12.1 / **+38.1bp**")
    for nm, v in (("atr14", atr14[idx]), ("atr288", atr288[idx]),
                  ("안전MAE", safe), ("예측변동성", pv)):
        if not np.isfinite(v).all(): continue
        rr = spearmanr(v, ret)
        print(f"  전체({N_DRAWS:,}건): ρ({nm:<8}, 실현bp) = {rr.statistic:+.4f}  p {rr.pvalue:.4f}")
    q = pd.qcut(pd.Series(atr288[idx]), 3, labels=["저변동", "중", "고변동"]).to_numpy()
    print("  전체 ATR288 3분위 실현bp: " + " · ".join(
        f"{k} {ret[q == k].mean():+.2f}(n={int((q == k).sum()):,})" for k in ["저변동", "중", "고변동"]))
    # 일 블록 부트: 고−저 차
    dd = day[idx]
    hi_m, lo_m = q == "고변동", q == "저변동"
    u = np.unique(dd)
    by_h = pd.Series(ret[hi_m]).groupby(dd[hi_m]).mean()
    by_l = pd.Series(ret[lo_m]).groupby(dd[lo_m]).mean()
    obs = by_h.mean() - by_l.mean()
    bs = np.array([by_h.sample(len(by_h), replace=True, random_state=int(RNG.integers(1e9))).mean()
                   - by_l.sample(len(by_l), replace=True, random_state=int(RNG.integers(1e9))).mean()
                   for _ in range(2000)])
    print(f"  Δ(고변동 − 저변동) = {obs:+.2f}bp · 일 부트 CI95 "
          f"[{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]  (고유일 {len(u):,})")

    # 실력 가정별 로그성장 (09-11 규약)
    print(f"\n■ 실력 e 가정별 건당 로그성장 (≈bp) — 사이징은 이익을 만들지 않는다")
    print(f"{'규칙':<22}" + "".join(f"{'e='+str(x):>10}" for x in (0, 10, 20, 30)))
    for k, w in ARMS.items():
        wn = np.asarray(w, float); wn = wn / wn.mean()
        out = []
        for eskill in (0, 10, 20, 30):
            x = (ret + eskill) * wn / 1e4
            out.append(np.log1p(np.clip(x, -0.99, None)).mean() * 1e4)
        print(f"{k:<22}" + "".join(f"{v:>+10.2f}" for v in out))
    print("=" * 124)
    print(json.dumps({"draws": N_DRAWS, "arms": len(ARMS),
                      "hi_minus_lo_bp": round(float(obs), 2)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
