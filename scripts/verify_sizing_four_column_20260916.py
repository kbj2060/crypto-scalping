"""사이징 4열 반사실 재검증 (2026-09-16) — 09-15 §5.36-S 주장 재현 여부."""
import json, importlib.util, sys, numpy as np, pandas as pd
from pathlib import Path
R = Path("/home/kbj20/crypto-scalping"); W = Path.cwd()
RNG = np.random.default_rng(20260916)

def _mod(rel, name):
    sp = importlib.util.spec_from_file_location(name, W / rel)
    m = importlib.util.module_from_spec(sp); argv, sys.argv = sys.argv, ["x"]
    try: sp.loader.exec_module(m)
    finally: sys.argv = argv
    return m
MQ = _mod("scripts/live_eth_mae_quantile_model_20260913.py", "MQ")
PO = _mod("scripts/live_eth_risk_sizing_policy_20260913.py", "PO")

kl = pd.read_csv(R / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                 usecols=["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"])
kl["timestamp"] = pd.to_datetime(kl.timestamp)
art = MQ.load_model(); models, mult = art["models"], art["mult"]
svm = MQ.svm
base = svm.build_features(kl.timestamp, kl.close.to_numpy(float), kl.quote_volume.to_numpy(float),
                          kl.trades.to_numpy(float), kl.high.to_numpy(float), kl.low.to_numpy(float))
base["_ts"] = kl.timestamp.values
base = base.replace([np.inf, -np.inf], np.nan)
print(f"피쳐 {base.shape} · 배포 보유시간 4h(48봉) 고정")

rows = [json.loads(l) for l in open(R / "data/live/account_round_trips.jsonl")]
t = pd.DataFrame(rows)
t["entry_ts"] = pd.to_datetime(t.entry_time, unit="ms")
mv = np.where(t.side == "SHORT", -1, 1)
t["bp"] = (t.exit_price - t.entry_price) / t.entry_price * mv * 1e4 - 5.88
bts = pd.to_datetime(base["_ts"]).to_numpy()
i = np.searchsorted(bts, t.entry_ts.to_numpy(), side="right") - 1
rows_ = base.iloc[np.clip(i, 0, len(base)-1)].copy().reset_index(drop=True)
rows_["log_h"] = np.log(48 * 5.0)
rows_["side"] = np.where(t.side.to_numpy() == "SHORT", -1, 1)
t["safe_mae"] = MQ.safe_mae(models, rows_[MQ.FEATURES], mult)
print(f"안전MAE 매칭 {t.safe_mae.notna().sum()}/{len(t)} · 중앙 {t.safe_mae.median():.3f}%")
t["w_D"] = [PO.entry_notional(10_000.0, float(s))["total_notional"] for s in t.safe_mae]      # D 권고 명목(자본 고정)

# E|r| 백분위 (앞 검증에서 만든 경로 재사용)
import joblib
RX = _mod("scripts/research_direction_event_expansion_20260915.py", "RX")
ART = R / "data/models/direction_4h_top5_20260915"
a2 = joblib.load(ART / "models.joblib")
raw = pd.read_parquet(R / "data/binance_vision/panel/ETHUSDT.parquet")
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
p = RX._features(raw, "ETH", False, "2000-01-01")
xx = p[a2["cols"]].to_numpy(np.float32); ok = np.isfinite(xx).all(1)
ev = pd.Series(np.nan, index=pd.to_datetime(p["timestamp"])); ev[ok] = np.exp(a2["evr"].predict(xx[ok]))
pct = ev.dropna().expanding(8064).rank(pct=True).shift(1)
j = pct.index.searchsorted(t.entry_ts.to_numpy(), side="right") - 1
t["evr_pct"] = np.where(j >= 0, pct.to_numpy()[np.clip(j, 0, len(pct)-1)], np.nan)

def curve(w):
    w = np.asarray(w, float); w = w / w.mean()                  # 평균 명목 동일
    d = pd.DataFrame({"day": t.entry_ts.dt.floor("D"), "x": t.bp * w})
    g = d.groupby("day").x.sum()
    cum = g.cumsum()
    mdd = float((cum.cummax() - cum).max())
    return float(g.mean()), mdd, g

ARMS = {"①고정": np.ones(len(t)), "②배포본D": t.w_D.to_numpy(),
        "③E|r|만": t.evr_pct.to_numpy(), "④D×E|r|": t.w_D.to_numpy() * t.evr_pct.to_numpy()}
CLAIM = {"①고정": (57.62, 156), "②배포본D": (48.96, 84), "③E|r|만": (90.98, 118), "④D×E|r|": (68.17, 94)}
print(f"\n■ 4열 반사실 (실계좌 72왕복 · 진입·청산 고정 · 사이징만 · 평균명목 동일)")
print(f"{'열':<10}{'주장 bp/일':>11}{'재검증':>9}{'주장 MDD':>10}{'재검증':>9}")
res = {}
for k, w in ARMS.items():
    m, mdd, g = curve(w); res[k] = (m, mdd, g)
    print(f"{k:<10}{CLAIM[k][0]:>+11.2f}{m:>+9.2f}{CLAIM[k][1]:>10.0f}{mdd:>9.0f}")

d4, d2 = res["④D×E|r|"][2], res["②배포본D"][2]
diff = (d4 - d2).dropna()
print(f"\nΔ(④−②) 주장 +19.21 CI[+2.50,+43.87] → 재검증 {diff.mean():+.2f}")
u = diff.index.to_numpy()
bs = np.array([diff.loc[RNG.choice(u, len(u))].mean() for _ in range(8000)])
print(f"  일 부트 CI95 [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}] · 독립일 {len(u)}")
print(f"\nΔ(②−①) = 배포된 사이징모델의 «수익» 기여: {res['②배포본D'][0]-res['①고정'][0]:+.2f} bp/일 "
      f"· MDD {res['①고정'][1]:.0f} → {res['②배포본D'][1]:.0f}")
q = pd.qcut(t.evr_pct, 3, labels=["하위", "중간", "상위"])
print("\nE|r| 3분위 (주장 −2.27/+19.76/+35.33):",
      " / ".join(f"{b} {t.bp[q==b].mean():+.2f}" for b in ["하위", "중간", "상위"]))

# ---- 철회 사유 재검정: 순차 분할 + 상위건 제거 (정책 파일 docstring 의 주장) ----
print("\n■ ⭐E|r| 배수 철회 사유 재검정 (정책 docstring: 전반 +11.01 → 후반 +0.73, 상위10건 빼면 부호반전)")
t2 = t.sort_values("entry_ts").reset_index(drop=True)
w2 = (t2.w_D.to_numpy() * t2.evr_pct.to_numpy()); w2 = w2 / w2.mean()
wD = t2.w_D.to_numpy(); wD = wD / wD.mean()
inc = t2.bp.to_numpy() * (w2 - wD)                       # 건당 증분(bp)
h = len(t2) // 2
print(f"  전체 건당 증분 {inc.mean():+.2f}bp · 전반 {inc[:h].mean():+.2f} · 후반 {inc[h:].mean():+.2f}")
for k in (5, 10):
    o = np.argsort(-np.abs(inc))[:k]
    m = np.ones(len(inc), bool); m[o] = False
    print(f"  |증분| 상위 {k}건 제거 → {inc[m].mean():+.2f}bp  (제거분이 전체의 "
          f"{100*inc[o].sum()/inc.sum():.0f}%)")
u = t2.entry_ts.dt.floor("D"); uu = u.unique()
bs = np.array([np.concatenate([inc[(u == x).to_numpy()] for x in RNG.choice(uu, len(uu))]).mean()
               for _ in range(4000)])
print(f"  건당 증분 일군집 CI95 [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}] · 독립일 {len(uu)}")
